#!/usr/bin/env julia
# New read-only audit helper: same example options, plus --baseline-results PATH.
# No reconstruction or stochastic transport; one endpoint forward/J assembly reused.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
using DelimitedFiles, Random

function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function convergence_round3_diagnostic()
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
    set_adjoint_derivative_mode!(:forwarddiff)
    load_csda_corrections!(args.correction_artifact)
    @assert DiffPumas.Tomography.CSDA_MATERIAL_CORRECTION[].enabled
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
    configuration=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
    function data_parts(p)
        cached=w->(p,zeros(length(p),1))
        op,scale=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
            weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
        residual,_=op([0.])
        return .5dot(residual,residual),scale(p)
    end
    data_objective(p)=first(data_parts(p))
    unitprior=SmoothnessPrior(cell_neighbors(volume),1.)
    graphenergy=DiffPumas.Tomography.prior_value(unitprior,wf)
    savedpred=Float64[r["matched_csda_flux"] for r in saved]
    savedobjective=parse(Float64,solver["objective"])
    inferred=(savedobjective-data_objective(savedpred))/graphenergy
    lambda=Float64(configuration["absolute_prior_weight"])
    @assert lambda==parse(Float64,solver["absolute_prior_weight"])
    lambda>=0 || error("invalid fixed prior")
    prior=SmoothnessPrior(cell_neighbors(volume),lambda)
    cached=w->(pred,J.*transpose(Float64.(support)))
    op,_=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;
        weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
    residual,jac=op(wf)
    grad=transpose(jac)*residual+laplacian_gradient(prior,wf)
    pg=(wf.-clamp.(wf.-grad,0.,MAX_WATER_FRACTION)).*support
    @assert configuration["support_sha256"]==bytes2hex(sha256(join(findall(support),',')))
    @assert configuration["screen_sha256"]==bytes2hex(sha256(join(covered[screen.>0],',')))
    @assert configuration["correction_artifact_sha256"]==bytes2hex(sha256(read(args.correction_artifact)))
    @assert configuration["robust_log_scale"]==args.papermatch_log_scale
    @assert configuration["pinned_normalization"]==args.papermatch_abs_scale
    @assert configuration["absolute_prior_weight"]==lambda
    @assert all(0 .<= wf .<= MAX_WATER_FRACTION) && all(wf[.!support].==0)
    objective0=data_objective(pred)+DiffPumas.Tomography.prior_value(prior,wf)
    @assert isapprox(objective0,savedobjective;rtol=1e-12,atol=1e-10)
    @assert maximum(abs.(pred.-savedpred)./max.(abs.(savedpred),1e-300))<1e-11
    @assert isapprox(norm(pg,Inf),parse(Float64,solver["projected_gradient"]);rtol=1e-7,atol=1e-10)
    savedscale=Float64(saved[1]["normalization"])
    @assert all(Float64(r["normalization"])==savedscale for r in saved)
    @assert isapprox(last(data_parts(pred)),savedscale;rtol=1e-12)
    # Recompute the value-only endpoint independently of the AD assembly.
    repeatpred=similar(pred)
    Threads.@threads for k in eachindex(paths)
        repeatpred[k]=compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],wf,energies)
    end
    repeatobjective=data_objective(repeatpred)+DiffPumas.Tomography.prior_value(prior,wf)
    noisefloor=max(10abs(repeatobjective-objective0),100eps(objective0))
    @assert abs(repeatobjective-objective0)<=1e-10
    mkpath(args.output_dir)
    results=Dict{String,Any}[]
    directions=Dict{String,Vector{Float64}}()
    function add_direction(name,d)
        d=Float64.(d).*support
        d[(wf.<=0).&(d.<0)].=0
        d[(wf.>=MAX_WATER_FRACTION).&(d.>0)].=0
        norm(d,Inf)>0 && (directions[name]=d./norm(d,Inf))
    end
    top=sort(findall(support);by=i->-abs(pg[i]))[1:8]
    for c in top, sign in (-1,1)
        d=zeros(nc); d[c]=sign;add_direction("coordinate_$(c)_$(sign)",d)
    end
    add_direction("projected_gradient",-pg)
    active=findall(support)
    H=Matrix(transpose(jac)*jac)
    # Build the exact quadratic-prior Hessian by linearity, without changing graph weights.
    for c in active
        basis=zeros(nc);basis[c]=1
        H[:,c].+=laplacian_gradient(prior,basis)
    end
    for mu in (1e-6,1e-3,1.)
        d=zeros(nc);d[active].=-(H[active,active]+mu*I)\grad[active]
        add_direction("damped_gn_$(mu)",d)
    end
    add_direction("diagonal_gn",-grad./max.(diag(H),1e-10))
    rng=MersenneTwister(20261025)
    for k in 1:8;add_direction("random_$(k)",randn(rng,nc));end
    for k in 1:7
        d=zeros(nc);d[top[k]]=1;d[top[k+1]]=-1
        add_direction("pair_$(k)_positive",d);add_direction("pair_$(k)_negative",-d)
    end
    open(joinpath(args.output_dir,"directions.toml"),"w") do io
        TOML.print(io,Dict("directions"=>directions,"top_cells"=>top))
    end
    function objective_at(w)
        changed=Set(findall(w.!=wf));p=copy(pred)
        affected=findall(path->any(s->s.cell_idx in changed,path.segments),paths)
        Threads.@threads for j in eachindex(affected)
            k=affected[j]
            p[k]=compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],w,energies)
        end
        data,scale=data_parts(p);pv=DiffPumas.Tomography.prior_value(prior,w)
        return data+pv,length(affected),data,pv,scale
    end
    open(joinpath(args.output_dir,"directional_checks.csv"),"w") do io
        println(io,"direction,step,actual_step_inf,objective,objective_change,slope,branch_prediction,affected_bins,data_objective,prior_objective,normalization,resolved")
        @assert length(directions)*6<=258 # preregistered hard evaluation cap; no silent truncation
        for name in sort(collect(keys(directions)))
            d=directions[name]
            for h in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8)
                w=clamp.(wf.+h.*d,0.,MAX_WATER_FRACTION)
                @assert all(w[.!support].==wf[.!support]) && all(0 .<= w .<= MAX_WATER_FRACTION)
                f,n,data,pv,scale=objective_at(w);change=f-objective0
                row=Dict("direction"=>name,"step"=>h,"actual_step_inf"=>norm(w.-wf,Inf),
                    "objective"=>f,"objective_change"=>change,"slope"=>change/h,
                    "branch_prediction"=>dot(grad,w.-wf)/h,"affected_bins"=>n,
                    "data_objective"=>data,"prior_objective"=>pv,"normalization"=>scale,"resolved"=>abs(change)>noisefloor)
                push!(results,row)
                println(io,join([row[k] for k in ("direction","step","actual_step_inf","objective","objective_change","slope","branch_prediction","affected_bins","data_objective","prior_objective","normalization","resolved")],','));flush(io)
            end
        end
    end
    report=Dict("top_cells"=>top,"direction_count"=>length(directions),
        "projected_gradient_inf"=>norm(pg,Inf),"saved_projected_gradient"=>parse(Float64,solver["projected_gradient"]),
        "max_repeat_prediction_relative_discrepancy"=>maximum(abs.(repeatpred.-pred)./max.(abs.(pred),1e-300)),
        "noise_floor"=>noisefloor,"repeat_objective"=>repeatobjective,"planned_max_evaluations"=>258,
        "objective"=>objective0,"saved_objective"=>savedobjective,"absolute_prior_weight"=>lambda,
        "diagnostic_prior_from_objective"=>inferred,"directional_checks"=>results,"certificate"=>"finite sampled feasible directions only; no generalized stationarity certification",
        "physics_dump_sha256"=>bytes2hex(sha256(read(args.dump_path))),
        "correction_artifact_sha256"=>bytes2hex(sha256(read(args.correction_artifact))),
        "helper_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    mkpath(args.output_dir)
    open(io->TOML.print(io,report),joinpath(args.output_dir,"diagnostic.toml"),"w")
    println("Convergence round3 diagnostic complete: ",args.output_dir)
end
if abspath(PROGRAM_FILE)==@__FILE__; convergence_round3_diagnostic(); end
