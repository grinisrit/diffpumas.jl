#!/usr/bin/env julia
# New read-only audit helper: same example options, plus --baseline-results PATH.
# No reconstruction or stochastic transport; one dry forward/J assembly reused.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
using DelimitedFiles

function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function audit_stability_flux_gradient()
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
    covered=Int[r["bin_idx"] for r in saved]
    paths=paths[covered];obs=observed[covered]
    support=Bool[string(r["aquifer_mask"])=="true" for r in fields]
    wf=Float64[r["w"] for r in fields]
    pred,J=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,wf,energies;n_cells=nc,threaded=args.threaded)
    screen=Float64[r["fit_mask"] for r in saved]
    dem=args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)
    W=screen./(1 .+dem.^2)
    function data_objective(p)
        cached=w->(p,zeros(length(p),1))
        op,_=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
            weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
        residual,_=op([0.])
        return .5dot(residual,residual)
    end
    unitprior=SmoothnessPrior(cell_neighbors(volume),1.)
    graphenergy=DiffPumas.Tomography.prior_value(unitprior,wf)
    savedpred=Float64[r["matched_csda_flux"] for r in saved]
    savedobjective=parse(Float64,solver["objective"])
    inferred=(savedobjective-data_objective(savedpred))/graphenergy
    lambda=haskey(solver,"absolute_prior_weight") ? parse(Float64,solver["absolute_prior_weight"]) : inferred
    lambda>=0 || error("invalid inferred prior")
    prior=SmoothnessPrior(cell_neighbors(volume),lambda)
    cached=w->(pred,J.*transpose(Float64.(support)))
    op,_=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
        weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
    residual,jac=op(wf)
    grad=transpose(jac)*residual+laplacian_gradient(prior,wf)
    pg=(wf.-clamp.(wf.-grad,0.,MAX_WATER_FRACTION)).*support
    c=argmax(abs.(pg))
    affected=findall(p->any(s->s.cell_idx==c,p.segments),paths)
    objective0=data_objective(pred)+DiffPumas.Tomography.prior_value(prior,wf)
    _,profile_scale=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
        weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
    scale=profile_scale(pred)
    t=log.(pred).+log(scale).-log.(obs)
    coefficient=W.*t./sqrt.(1 .+(t./args.papermatch_log_scale).^2)./pred
    priorgradient=laplacian_gradient(prior,wf)[c]
    defaultstep=get_adjoint_fd_step()
    baseflux=Dict(k=>compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],wf,energies) for k in affected)
    fulldiff=Dict(k=>Zygote.gradient(w->compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],w,energies),wf)[1][c] for k in affected)
    rayrows=Dict{String,Any}[]; summaries=Dict{String,Any}[]
    for internal in (1e-4,1e-5,1e-6,1e-7,1e-8)
        set_adjoint_fd_step!(internal)
        derivatives=Dict{Int,Float64}()
        for k in affected
            f,cells,vals=directional_flux_and_grad_csda(ph,shallow,cfg,site,paths[k],wf,energies)
            derivatives[k]=vals[findfirst(==(c),cells)]
            push!(rayrows,Dict("kind"=>"internal_adjoint_step","bin"=>covered[k],"internal_step"=>internal,
                "flux_sparse"=>f,"flux_full"=>baseflux[k],"derivative"=>derivatives[k],
                "default_sparse_derivative"=>J[k,c],"default_fullvector_derivative"=>fulldiff[k],"objective_coefficient"=>coefficient[k]))
        end
        datagrad=sum(coefficient[k]*derivatives[k] for k in affected)
        push!(summaries,Dict("kind"=>"internal_adjoint_step","step"=>internal,"data_gradient"=>datagrad,
            "prior_gradient"=>priorgradient,"total_gradient"=>datagrad+priorgradient))
    end
    set_adjoint_fd_step!(defaultstep)
    for h in (1e-4,1e-5,1e-6,1e-7,1e-8)
        lo=max(0.,wf[c]-h);hi=min(MAX_WATER_FRACTION,wf[c]+h)
        wl=copy(wf);wh=copy(wf);wl[c]=lo;wh[c]=hi
        derivatives=Dict{Int,Float64}()
        for k in affected
            fl=compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],wl,energies)
            fh=compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],wh,energies)
            derivatives[k]=(fh-fl)/(hi-lo)
            push!(rayrows,Dict("kind"=>"flux_finite_difference","bin"=>covered[k],"outer_step"=>h,
                "flux_lo"=>fl,"flux_hi"=>fh,"derivative"=>derivatives[k],"default_sparse_derivative"=>J[k,c],
                "objective_coefficient"=>coefficient[k]))
        end
        datagrad=sum(coefficient[k]*derivatives[k] for k in affected)
        push!(summaries,Dict("kind"=>"flux_finite_difference","step"=>h,"data_gradient"=>datagrad,
            "prior_gradient"=>priorgradient,"total_gradient"=>datagrad+priorgradient))
    end
    report=Dict("cell"=>c,"water_fraction"=>wf[c],"affected_bins"=>length(affected),
        "default_ad_gradient"=>grad[c],"prior_gradient"=>priorgradient,
        "default_data_gradient"=>sum(coefficient[k]*J[k,c] for k in affected),
        "max_sparse_full_flux_relative_error"=>maximum(abs(pred[k]-baseflux[k])/max(abs(pred[k]),abs(baseflux[k]),1e-300) for k in affected),
        "max_sparse_fullvector_derivative_relative_error"=>maximum(abs(J[k,c]-fulldiff[k])/max(abs(J[k,c]),abs(fulldiff[k]),1e-300) for k in affected),
        "summaries"=>summaries,"rays"=>rayrows)
    mkpath(args.output_dir)
    open(io->TOML.print(io,report),joinpath(args.output_dir,"flux_gradient_audit.toml"),"w")
    println("Flux-gradient audit complete: ",args.output_dir)
end
if abspath(PROGRAM_FILE)==@__FILE__; audit_stability_flux_gradient(); end
