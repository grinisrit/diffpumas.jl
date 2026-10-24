#!/usr/bin/env julia
# New read-only audit helper: same example options, plus --baseline-results PATH.
# No reconstruction or stochastic transport; one dry forward/J assembly reused.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
using DelimitedFiles

function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function audit_stability_prior()
    raw=copy(ARGS); baseline=""; empty!(ARGS)
    i=1
    while i<=length(raw)
        if raw[i]=="--baseline-results"
            baseline=raw[i+1]; i+=2
        else
            push!(ARGS,raw[i]); i+=1
        end
    end
    isempty(baseline) && error("--baseline-results PATH required")
    args=parse_commandline()
    isdir(args.output_dir) && error("audit output must be fresh")
    isempty(args.correction_artifact) && error("frozen correction artifact required")
    args.geometry=="grid" || error("this audit is specified for the full structured grid")
    saved=audit_rows(joinpath(baseline,"lvd_fit_predictions.csv"))
    fields=audit_rows(joinpath(baseline,"lvd_fit_field.csv"))
    # Read key/value solver state without interpreting diagnostic prose.
    solver=Dict{String,String}()
    for line in eachline(joinpath(baseline,"lvd_solver_status.txt"))
        parts=split(line,'=';limit=2)
        length(parts)==2 && (solver[parts[1]]=parts[2])
    end
    ph=load_tomography_physics(args.dump_path)
    idx(name)=get_material_index(ph,name)
    porous=idx("PorousWetRock")
    cfg=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,
        args.rock_density>0 ? args.rock_density : Float64(ph.tables[idx("StandardRock")].density),
        Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
    load_csda_corrections!(args.correction_artifact)
    emap=LVDTopo.build_elevation_map(); volume=create_sensitivity_volume(emap,cfg,args)
    nc=num_cells(volume); nc==600 || error("expected 600 cells")
    shallow=Bool[is_shallow_cell(volume,c) for c in 1:nc]
    site=build_site(args.primary_altitude_km*1000)
    bins=load_measured_intensity(args.match_data)
    zen=Float64[.5*(b.zen_lo+b.zen_hi) for b in bins]
    az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : .5*(b.az_lo+b.az_hi) for b in bins]
    paths=[trace_path(volume,zen[b],az[b],emap) for b in eachindex(bins)]
    observed=Float64[b.intensity for b in bins]
    energies=args.csda_quadrature=="midpoint" ? csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max) :
        sample_energy_set(args.n_samples,args.energy_min,args.energy_max,args.seed)
    dry,J=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,zeros(nc),energies;n_cells=nc,threaded=args.threaded)
    covered=findall(b->paths[b].valid && isfinite(dry[b]) && dry[b]>0 && isfinite(observed[b]) && observed[b]>0,eachindex(dry))
    covered==Int[r["bin_idx"] for r in saved] || error("baseline covered bins mismatch")
    length(covered)==1003 || error("expected 1003 covered bins")
    obs=observed[covered]; pred=dry[covered]; jac=J[covered,:]
    isapprox(pred,Float64[r["dry_csda_flux"] for r in saved];rtol=1e-11,atol=0) || error("baseline dry flux mismatch")
    logratio=log.(pred).-log.(obs); center=median(logratio)
    screen=Float64.(abs.(logratio.-center).<=log(args.papermatch_mask_ratio))
    screen==Float64[r["fit_mask"] for r in saved] || error("baseline screen mismatch")
    crossed=vec(sum(abs.(jac);dims=1)).>0
    support=Bool[cell_centroid(volume,c)[3]>args.papermatch_prior_cutoff && crossed[c] for c in 1:nc]
    support==Bool[string(r["aquifer_mask"])=="true" for r in fields] || error("baseline support mismatch")
    dem=args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)
    W=screen./(1 .+dem.^2)
    cached=w->(pred,jac.*transpose(Float64.(support)))
    savedscale=Float64(saved[1]["normalization"])
    all(r->r["normalization"]==savedscale,saved) || error("baseline scale not constant")
    autos=Dict{String,Float64}()
    for (name,normalization) in (("profile",0.),("minus5",.95savedscale),("plus5",1.05savedscale))
        operator,_=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
            weights=W,robust_scale=args.papermatch_log_scale,normalization=normalization)
        _,transformed=operator(zeros(nc))
        diagonal=vec(sum(transformed.^2;dims=1))
        positives=filter(>(0),diagonal[support])
        autos[name]=isempty(positives) ? 1. : median(positives)
    end
    target=args.papermatch_reg*autos["profile"]
    wf=Float64[r["w"] for r in fields]
    unitprior=SmoothnessPrior(cell_neighbors(volume),1.)
    graphenergy=DiffPumas.Tomography.prior_value(unitprior,wf)
    fitted=Float64[r["matched_csda_flux"] for r in saved]
    t=log.(fitted).+log(savedscale).-log.(obs)
    h=sqrt.(1 .+(t./args.papermatch_log_scale).^2)
    dataloss=sum(W.*t.^2./(h.+1))
    finalobjective=parse(Float64,solver["objective"])
    inferred=graphenergy>0 ? (finalobjective-dataloss)/graphenergy : NaN
    isapprox(finalobjective,dataloss+target*graphenergy;rtol=1e-9,atol=1e-10) || error("baseline absolute prior/objective crosscheck failed")
    report=Dict("target_absolute_prior_weight"=>target,"objective_inferred_prior_weight"=>inferred,
        "baseline_objective"=>finalobjective,"baseline_data_loss"=>dataloss,"unit_graph_prior"=>graphenergy,
        "baseline_normalization"=>savedscale,"automatic_prior_scale"=>autos,
        "papermatch_reg"=>Dict(name=>target/value for (name,value) in autos),
        "cells"=>nc,"covered_bins"=>length(covered),"support_cells"=>count(support),
        "fit_bins"=>count(>(0),screen),"physics_dump_sha256"=>bytes2hex(sha256(read(args.dump_path))),
        "correction_artifact_sha256"=>bytes2hex(sha256(read(args.correction_artifact))),
        "helper_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    mkpath(args.output_dir)
    open(io->TOML.print(io,report),joinpath(args.output_dir,"absolute_prior_audit.toml"),"w")
    println("Absolute prior audit complete: ",args.output_dir)
end
if abspath(PROGRAM_FILE)==@__FILE__; audit_stability_prior(); end
