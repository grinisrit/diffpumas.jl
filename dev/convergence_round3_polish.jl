#!/usr/bin/env julia
include(joinpath(@__DIR__,"convergence_round3_diagnostic.jl"))
include(joinpath(@__DIR__,"convergence_round3_sampled_direction.jl"))
function convergence_round3_polish()
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
    active=findall(support);w=copy(wf);currentpred=copy(pred);currentobjective=objective0
    evaluations=0;assemblies=1;accepted=0;attempts=Dict{String,Any}[];sample_records=Dict{String,Any}[]
    function evaluate_trial(candidate)
        evaluations+=1;evaluations<=1200 || error("objective budget exhausted")
        changed=Set(findall(candidate.!=w));p=copy(currentpred)
        affected=findall(path->any(s->s.cell_idx in changed,path.segments),paths)
        Threads.@threads for j in eachindex(affected)
            k=affected[j];p[k]=compute_directional_flux_csda(ph,shallow,cfg,site,paths[k],candidate,energies)
        end
        data,scale=data_parts(p);pv=DiffPumas.Tomography.prior_value(prior,candidate)
        data+pv,p,data,pv,scale
    end
    function checkpoint()
        open(joinpath(args.output_dir,"polish_state.toml"),"w") do io
            TOML.print(io,Dict("initial_objective"=>objective0,"objective"=>currentobjective,
                "accepted"=>accepted,"objective_evaluations"=>evaluations,"gradient_assemblies"=>assemblies,
                "noise_floor"=>noisefloor,"field"=>w,"attempts"=>attempts,"sample_records"=>sample_records,
                "certificate"=>"No generalized stationarity certificate; finite sampled-gradient heuristic plus finite coordinate polling"))
        end
    end
    function try_candidate(candidate,method,scale)
        @assert all(0 .<= candidate .<= MAX_WATER_FRACTION) && candidate[.!support]==wf[.!support]
        f,p,data,pv,normscale=evaluate_trial(candidate);change=f-currentobjective
        take=change < -noisefloor
        push!(attempts,Dict("method"=>method,"step"=>scale,"objective_before"=>currentobjective,
            "objective_trial"=>f,"objective_change"=>change,"accepted"=>take,"data_objective"=>data,
            "prior_objective"=>pv,"normalization"=>normscale,"actual_step_inf"=>norm(candidate.-w,Inf)))
        if take
            w=candidate;currentpred=p;currentobjective=f;accepted+=1
            open(joinpath(args.output_dir,"accepted_$(accepted).toml"),"w") do io
                TOML.print(io,Dict("field"=>w,"objective"=>f,"method"=>method,"step"=>scale))
            end
        end
        take
    end
    # Cover every supported coordinate, including directions absent from top-PG diagnostic.
    for h in (1e-4,1e-5,1e-6)
        for c in active, sign in (-1,1)
            candidate=copy(w);candidate[c]=clamp(w[c]+sign*h,0.,MAX_WATER_FRACTION)
            candidate==w && continue
            try_candidate(candidate,"coordinate_$(c)_$(sign)",h)
        end
        checkpoint()
    end
    rng=MersenneTwister(20261025)
    function exact_gradient(x)
        assemblies+=1;assemblies<=24 || error("gradient assembly budget exhausted")
        p,j=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,x,energies;n_cells=nc,threaded=args.threaded)
        model=_ -> (p,j.*transpose(Float64.(support)))
        oper,_=DiffPumas.Tomography.profiled_log_observation_operator(model,obs;
            weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
        r,jac=oper(x)
        transpose(jac)*r+laplacian_gradient(prior,x),p
    end
    for radius in (1e-5,1e-7,1e-9)
        gradients=Vector{Vector{Float64}}();sample_fields=Vector{Vector{Float64}}()
        g,_=exact_gradient(w);push!(gradients,g[active]);push!(sample_fields,copy(w))
        for k in 1:5
            delta=randn(rng,nc).*support;delta./=norm(delta,Inf)
            x=clamp.(w.+radius.*delta,0.,MAX_WATER_FRACTION)
            g,_=exact_gradient(x);push!(gradients,g[active]);push!(sample_fields,x)
        end
        G=hcat(gradients...)
        d,alpha,residual,iterations,qpchange=round3_sampled_direction(G,w,active)
        push!(sample_records,Dict("radius"=>radius,"fields"=>sample_fields,"gradients"=>gradients,
            "alpha"=>alpha,"sampled_residual_l2"=>residual,"simplex_iterations"=>iterations,
            "simplex_last_alpha_change"=>qpchange,"simplex_converged"=>qpchange<1e-12,"direction"=>d,"field_before"=>copy(w)))
        if norm(d,Inf)>0
            d./=norm(d,Inf)
            for h in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9)
                candidate=clamp.(w.+h.*d,0.,MAX_WATER_FRACTION)
                try_candidate(candidate,"sampled_gradient_radius_$(radius)",h) && break
            end
        end
        checkpoint()
    end
    # Independent final full value/J evaluation; incremental forward cache must agree.
    finalgrad,finalpred=exact_gradient(w)
    finalobjective=data_objective(finalpred)+DiffPumas.Tomography.prior_value(prior,w)
    @assert abs(finalobjective-currentobjective)<=max(noisefloor,1e-10)
    @assert maximum(abs.(finalpred.-currentpred)./max.(abs.(finalpred),1e-300))<1e-11
    finalpg=(w.-clamp.(w.-finalgrad,0.,MAX_WATER_FRACTION)).*support
    open(joinpath(args.output_dir,"final.toml"),"w") do io
        TOML.print(io,Dict("initial_objective"=>objective0,"objective"=>finalobjective,
            "objective_improvement"=>objective0-finalobjective,"initial_projected_gradient"=>norm(pg,Inf),
            "selected_branch_projected_gradient"=>norm(finalpg,Inf),"selected_branch_pg_tolerance_met"=>norm(finalpg,Inf)<=1e-6,
            "field"=>w,"predictions"=>finalpred,"normalization"=>last(data_parts(finalpred)),
            "accepted_updates"=>accepted,"objective_evaluations"=>evaluations,"gradient_assemblies"=>assemblies,
            "termination"=>"preregistered finite polling and sampling budget complete",
            "generalized_stationarity_certified"=>false,"noise_floor"=>noisefloor))
    end
    checkpoint()
end
if abspath(PROGRAM_FILE)==@__FILE__;convergence_round3_polish();end
