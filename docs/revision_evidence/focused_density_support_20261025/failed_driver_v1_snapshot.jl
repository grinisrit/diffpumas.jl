#!/usr/bin/env julia
# Five frozen-objective, finite-budget stress arms. No production source changes.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"prototype_stability_guarded_ray_cache.jl"))
include(joinpath(@__DIR__,"generated_stability_cached_flux_factory.jl"))
include(joinpath(@__DIR__,"focused_density_support_solver.jl"))
using DelimitedFiles
function focused_rows(path)
    a,h=readdlm(path,',',Any;header=true)
    [Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function focused_csv(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','));for row in rows;println(io,join(row,','));end
    end
end
function focused_atomic_toml(path,value)
    tmp=path*".tmp";open(io->TOML.print(io,value),tmp,"w");mv(tmp,path;force=true)
end
function focused_density_support_main()
    started=time();raw=copy(ARGS);empty!(ARGS)
    extra=Dict("--baseline-results"=>"","--stress-arm"=>"")
    i=1
    while i<=length(raw)
        if haskey(extra,raw[i]);extra[raw[i]]=raw[i+1];i+=2
        else;push!(ARGS,raw[i]);i+=1;end
    end
    arms=Dict("baseline"=>(2710.,500.),"rho2660"=>(2660.,500.),"rho2760"=>(2760.,500.),"cut400"=>(2710.,400.),"cut800"=>(2710.,800.))
    arm=extra["--stress-arm"];haskey(arms,arm)||error("unregistered arm");rho,cutoff=arms[arm]
    baseline=extra["--baseline-results"];isempty(baseline)&&error("baseline required")
    args=parse_commandline();ispath(args.output_dir)&&error("fresh output required")
    args.geometry=="grid" && args.papermatch_fit_mode=="stable" && args.csda_quadrature=="midpoint" || error("frozen grid/stable/midpoint required")
    args.n_samples==32 && args.rock_density==2710 && args.papermatch_prior_cutoff==500 || error("baseline geometry/options changed")
    args.papermatch_dem_unc==.15 && args.primary_altitude_km==30 && args.energy_min==1e-3 && args.energy_max==1e9 || error("fixed observation/source/quadrature controls changed")
    args.papermatch_gn_iters==10 && args.papermatch_gradient_tol==1e-6 || error("predeclared solver budget/tolerance changed")
    saved=focused_rows(joinpath(baseline,"lvd_fit_predictions.csv"));fields=focused_rows(joinpath(baseline,"lvd_fit_field.csv"))
    configuration=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
    bytes2hex(sha256(read(args.correction_artifact)))==configuration["correction_artifact_sha256"]||error("correction changed")
    args.papermatch_log_scale==configuration["robust_log_scale"] && args.papermatch_abs_scale==configuration["pinned_normalization"] || error("profile/robust changed")
    args.papermatch_absolute_prior_weight==configuration["absolute_prior_weight"]||error("lambda changed")
    set_adjoint_derivative_mode!(:forwarddiff);ph=load_tomography_physics(args.dump_path);idx(x)=get_material_index(ph,x);porous=idx("PorousWetRock")
    config(r)=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,r,Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
    basecfg=config(2710.);cfg=config(rho);cfg.porous_top_thickness==0||error("porous layer excluded")
    load_csda_corrections!(args.correction_artifact)
    # Geometry is always built from the original density/configuration; density changes transport only.
    emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,basecfg,args);nc=num_cells(volume);nc==600||error("grid changed")
    for c in 1:nc
        Int(fields[c]["cell_idx"])==c||error("cell ordering")
        isequal(Tuple(Float64(fields[c][k]) for k in ("x_m","y_m","z_m")),cell_centroid(volume,c))||error("cell coordinates")
    end
    originalsupport=Bool[string(r["aquifer_mask"])=="true" for r in fields]
    bytes2hex(sha256(join(findall(originalsupport),',')))==configuration["support_sha256"]||error("original support changed")
    wb=Float64[r["w"] for r in fields];shallow=Bool[is_shallow_cell(volume,c) for c in 1:nc]
    site=build_site(args.primary_altitude_km*1000);bins=load_measured_intensity(args.match_data);covered=Int[r["bin_idx"] for r in saved]
    length(covered)==1003||error("coverage changed")
    zen=Float64[.5*(b.zen_lo+b.zen_hi) for b in bins];az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : .5*(b.az_lo+b.az_hi) for b in bins]
    paths=[trace_path(volume,zen[b],az[b],emap) for b in covered];all(p->p.valid,paths)||error("invalid frozen path")
    obs=Float64[bins[b].intensity for b in covered];isequal(obs,Float64[r["measured_flux"] for r in saved])||error("observations changed")
    screen=Float64[r["fit_mask"] for r in saved];count(>(0),screen)==690||error("fit bins changed")
    bytes2hex(sha256(join(covered[screen.>0],',')))==configuration["screen_sha256"]||error("screen changed")
    dem=args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0);W=screen./(1 .+dem.^2)
    energies=csda_energy_quadrature(32,args.energy_min,args.energy_max)
    prior=SmoothnessPrior(cell_neighbors(volume),configuration["absolute_prior_weight"])
    function parts(p,w)
        cached=_ -> (p,zeros(length(p),1))
        op,scale=DiffPumas.Tomography.profiled_log_observation_operator(cached,obs;weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
        r,_=op([0.]);data=.5dot(r,r);pv=DiffPumas.Tomography.prior_value(prior,w)
        (residual=r,data=data,prior=pv,objective=data+pv,normalization=scale(p))
    end
    basep,baseJ=assemble_forward_and_jacobian(ph,shallow,basecfg,site,paths,wb,energies;n_cells=nc,threaded=args.threaded)
    isequal(reinterpret(UInt64,basep),reinterpret(UInt64,Float64[r["matched_csda_flux"] for r in saved]))||error("baseline full flux parity")
    baseop,_=DiffPumas.Tomography.profiled_log_observation_operator(_->(basep,baseJ.*transpose(Float64.(originalsupport))),obs;weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale)
    baser,baseR=baseop(wb);basegrad=transpose(baseR)*baser+laplacian_gradient(prior,wb)
    baselineparts=parts(basep,wb);basepg=norm((wb.-clamp.(wb.-basegrad,0.,MAX_WATER_FRACTION)).*originalsupport,Inf)
    status=Dict(split(line,'=';limit=2)[1]=>split(line,'=';limit=2)[2] for line in eachline(joinpath(baseline,"lvd_solver_status.txt")) if occursin('=',line))
    isequal(baselineparts.objective,parse(Float64,status["objective"])) && isequal(basepg,parse(Float64,status["projected_gradient"])) ||error("baseline objective/PG parity")
    all(r->isequal(baselineparts.normalization,Float64(r["normalization"])),saved)||error("baseline normalization parity")
    _,dryJ=assemble_forward_and_jacobian(ph,shallow,basecfg,site,paths,zeros(nc),energies;n_cells=nc,threaded=args.threaded)
    crossed=vec(sum(abs.(dryJ);dims=1)).>0;heights=Float64[cell_centroid(volume,c)[3] for c in 1:nc]
    geometric=falses(nc);fitcrossed=falses(nc)
    for k in eachindex(paths),segment in paths[k].segments
        segment.distance>0||continue;geometric[segment.cell_idx]=true
        W[k]>0 && (fitcrossed[segment.cell_idx]=true)
    end
    isequal(crossed,geometric)||error("covered-J and positive-length geometric crossing disagree")
    isequal((heights.>500).&crossed,originalsupport)||error("500m support parity")
    support=(heights.>cutoff).&crossed
    expected=cutoff==400 ? (205,118,87) : cutoff==800 ? (86,59,27) : (147,94,53)
    (count(support),count(support.&fitcrossed),count(support.&.!fitcrossed))==expected||error("predeclared support counts changed")
    w0=wb.*support;added=support.&.!originalsupport;removed=originalsupport.&.!support;common=support.&originalsupport
    all(w0[added].==0)&&all(w0[.!support].==0)||error("initial projection")
    model=masked_material_operator(make_csda_operator(ph,shallow,cfg,site,paths,energies;n_cells=nc,valid_bins=collect(eachindex(paths)),threaded=args.threaded),support)
    cached,_=build_exact_cached_flux_factory(ph,shallow,cfg,site,paths,energies,args,r->cfg)
    predict=w->cached(rho,Vector{Float64}(w.*support))
    lastflux=Ref(Float64[])
    tracked=w->begin;p,J=model(w);lastflux[]=p;(p,J);end
    operator,_,predict_residual=DiffPumas.Tomography.profiled_log_observation_operator(tracked,obs;weights=W,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale,predict_only=predict,return_predict_only=true)
    function make_state(w)
        all(w[.!support].==0)&&all(0 .<=w.<=MAX_WATER_FRACTION)||error("infeasible candidate")
        r,J=operator(w);p=copy(lastflux[]);part=parts(p,w)
        gradient=transpose(J)*r+laplacian_gradient(prior,w)
        pg=norm((w.-clamp.(w.-gradient,0.,MAX_WATER_FRACTION)).*support,Inf)
        merge(part,(water=copy(w),flux=p,J=J,branch_pg=pg))
    end
    initial=make_state(w0);cached0=predict(w0)
    isequal(reinterpret(UInt64,cached0),reinterpret(UInt64,initial.flux))||error("arm cache/full primal parity")
    initial_direct=flux_only_operator(ph,shallow,cfg,site,paths,energies;threaded=args.threaded)(w0)
    isequal(reinterpret(UInt64,cached0),reinterpret(UInt64,initial_direct))||error("arm cached/scalar primal parity")
    physics=rho==2710 ? merge(baselineparts,(branch_pg=basepg,)) : merge(parts(initial.flux,wb),(branch_pg=initial.branch_pg,))
    mkpath(args.output_dir)
    function state_record(state,index,outer)
        record=Dict("arm"=>arm,"accepted_index"=>index,"outer"=>outer,"elapsed_seconds"=>time()-started,
            "field"=>state.water,"predictions"=>state.flux,"residual"=>state.residual,"objective"=>state.objective,
            "data_objective"=>state.data,"prior_objective"=>state.prior,"normalization"=>state.normalization,"selected_branch_pg"=>state.branch_pg)
        focused_atomic_toml(joinpath(args.output_dir,"accepted_$(lpad(string(index),3,'0')).toml"),record)
    end
    state_record(initial,0,0)
    effect(s)=Dict("objective"=>s.objective,"data_objective"=>s.data,"prior_objective"=>s.prior,"normalization"=>s.normalization,"selected_branch_pg"=>s.branch_pg)
    effects=Dict("baseline"=>merge(effect(merge(baselineparts,(branch_pg=basepg,))),Dict("rock_density"=>2710.,"cutoff_m"=>500.)),
        "physics_at_original_field"=>merge(effect(physics),Dict("rock_density"=>rho,"support_cutoff_m"=>500.)),
        "projected_initial"=>merge(effect(initial),Dict("rock_density"=>rho,"support_cutoff_m"=>cutoff)),
        "physics_objective_change"=>physics.objective-baselineparts.objective,"projection_objective_change"=>initial.objective-physics.objective,
        "support_cells"=>count(support),"fit_crossed_support_cells"=>count(support.&fitcrossed),"unseen_support_cells"=>count(support.&.!fitcrossed),
        "added_cells"=>findall(added),"removed_cells"=>findall(removed),"common_cells"=>findall(common),
        "baseline_flux_objective_gradient_profile_bitwise"=>true,"initial_cache_full_scalar_bitwise"=>true,
        "scope"=>"Initial density-model and support-projection effects separated from subsequent finite-budget optimization; objectives from different models are not ranks of minima.")
    focused_atomic_toml(joinpath(args.output_dir,"initial_effects.toml"),effects)
    predict_trial=w->begin;p=predict(w);merge(parts(p,w),(flux=p,));end
    header=(:outer,:lm_try,:halving,:alpha,:mu,:rescue,:fallback,:fallback_reason,:objective_before,:directional,:step_inf,:trial_objective,:confirmed_objective,:data,:prior,:normalization,:accepted,:reason)
    deadline=started+900.
    result=open(joinpath(args.output_dir,"trials.csv"),"w") do io
        println(io,join(header,','));flush(io)
        logger=row->begin;println(io,join([row[k] for k in header],','));flush(io);end
        focused_bounded_gn(initial,make_state,predict_trial,prior,support;max_outer=10,deadline=deadline,
            trace_callback=logger,state_callback=state_record,tolerance=1e-6)
    end
    final=result.state
    final_direct_verified=false
    if time()+60<deadline
        direct=flux_only_operator(ph,shallow,cfg,site,paths,energies;threaded=args.threaded)(final.water)
        isequal(reinterpret(UInt64,direct),reinterpret(UInt64,final.flux))||error("final independent scalar/full parity")
        final_direct_verified=true
    end
    focused_csv(joinpath(args.output_dir,"field.csv"),("cell_idx","x_m","y_m","z_m","original_support","support","fit_crossed","new_region","removed_region","baseline_water","initial_water","final_water","optimization_delta","total_delta"),
        [(c,cell_centroid(volume,c)...,originalsupport[c],support[c],fitcrossed[c],added[c],removed[c],wb[c],w0[c],final.water[c],final.water[c]-w0[c],final.water[c]-wb[c]) for c in 1:nc])
    focused_csv(joinpath(args.output_dir,"predictions.csv"),("bin_idx","fit_mask","observation","baseline_flux","initial_flux","final_flux","initial_log_residual","final_log_residual","final_robust_residual"),
        [(covered[k],screen[k],obs[k],basep[k],initial.flux[k],final.flux[k],log(initial.normalization*initial.flux[k]/obs[k]),log(final.normalization*final.flux[k]/obs[k]),final.residual[k]) for k in eachindex(covered)])
    report=Dict("arm"=>arm,"rock_density"=>rho,"support_cutoff_m"=>cutoff,"support_cells"=>count(support),"fit_crossed_support_cells"=>count(support.&fitcrossed),"unseen_support_cells"=>count(support.&.!fitcrossed),
        "initial"=>effect(initial),"final"=>effect(final),"accepted_updates"=>result.accepted,"outer_attempts"=>result.outer_attempts,"maximum_outer_attempts"=>10,
        "termination"=>result.termination,"soft_budget_seconds"=>900.,"elapsed_seconds"=>time()-started,"optimization_objective_change"=>final.objective-initial.objective,
        "selected_branch_pg_tolerance_met"=>final.branch_pg<=1e-6,"generalized_stationarity_certified"=>false,"final_direct_scalar_parity_checked"=>final_direct_verified,
        "optimization_field_change_l2"=>norm(final.water.-w0),"optimization_field_change_inf"=>norm(final.water.-w0,Inf),
        "common_support_change_l2"=>norm((final.water.-wb)[common]),"common_support_change_inf"=>norm((final.water.-wb)[common],Inf),
        "original_primary_field_sha256"=>bytes2hex(sha256(read(joinpath(baseline,"lvd_fit_field.csv")))),"screen_sha256"=>configuration["screen_sha256"],"absolute_prior_weight"=>configuration["absolute_prior_weight"],
        "scope"=>"Matched finite-budget density/support sensitivity, not converged minima, geological confidence intervals, or identifiable groundwater inventory. Original primary result unchanged.")
    focused_atomic_toml(joinpath(args.output_dir,"report.toml"),report)
    println("Focused density/support arm complete: ",arm," ",result.termination)
end
if abspath(PROGRAM_FILE)==@__FILE__;focused_density_support_main();end
