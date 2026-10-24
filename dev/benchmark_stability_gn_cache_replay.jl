#!/usr/bin/env julia
# Isolated exact cache experiment; not imported by production.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"prototype_stability_ray_cache.jl"))
using .StabilityRayCachePrototype
using DelimitedFiles, TOML, Serialization
include(joinpath(@__DIR__,"prototype_stability_gn_replay.jl"))
include(joinpath(@__DIR__,"test_stability_ray_cache.jl"))
function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function main_cache_replay()
    args=parse_commandline()
    isdir(args.output_dir) && error("fresh output required")
    baseline=joinpath(@__DIR__,"..","examples","data","stability_v2_dual_dry_start0_20261024","results")
    saved=audit_rows(joinpath(baseline,"lvd_fit_predictions.csv"))
    fields=audit_rows(joinpath(baseline,"lvd_fit_field.csv"))
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
    other=audit_rows(joinpath(@__DIR__,"..","examples","data","stability_v2_dual_controls_20261024","dry","start015","results","lvd_fit_field.csv"))
    w15=Float64[r["w"] for r in other]
    W=Float64[r["fit_mask"] for r in saved]./(1 .+(args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)).^2)
    prior=SmoothnessPrior(cell_neighbors(volume),3.954930197858131)
    artifact=args.correction_artifact
    load_csda_corrections!(artifact)
    raw=w->[compute_directional_flux_csda(ph,shallow,cfg,site,p,w,energies) for p in paths]
    physical=w->begin
        p,J=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,w.*support,energies;n_cells=nc,threaded=false)
        p,J.*transpose(Float64.(support))
    end
    records=Any[]
    mkpath(args.output_dir)
    for (label,w) in (("dry0",wf),)
        cache=RayCache(paths,artifact)
        cached_calls=Ref(0);cached_hits=Ref(0)
        cached=w->begin
            p,h=evaluate!(cache,w, (i,x)->compute_directional_flux_csda(ph,shallow,cfg,site,paths[i],x,energies))
            cached_calls[]+=length(paths)-h;cached_hits[]+=h;p
        end
        traces=Any[];solutions=Any[];timings=Float64[]
        # Same untouched production algorithm, fresh default mu in each run.
        for (variant,predict) in (("baseline",raw),("cache",cached))
            op,scale,predict_residual=DiffPumas.Tomography.profiled_log_observation_operator(physical,obs;
                weights=W,robust_scale=.3,normalization=0.,predict_only=predict,return_predict_only=true)
            trace=Any[];diagnostics=Any[]
            start=time_ns()
            solution=DiffPumas.Tomography.replay_cache_gauss_newton_reconstruct(op,zeros(length(obs));
                w0=w,n_iter=1,prior=prior,free=support,solver_mode=:active_set,
                predict_only=predict_residual,diagnostics=diagnostics,trial_trace=trace)
            elapsed=(time_ns()-start)/1e9
            push!(traces,trace);push!(solutions,(solution,diagnostics));push!(timings,elapsed)
            serialize(joinpath(args.output_dir,label*"_"*variant*"_trace.jls"),(trace=trace,solution=solution,diagnostics=diagnostics))
            println(label," ",variant," seconds=",elapsed," trials=",length(trace));flush(stdout)
        end
        isequal(traces[1],traces[2]) || error("candidate/objective/direction/halving trace mismatch")
        isequal(solutions[1],solutions[2]) || error("accepted iterate/history/gradient diagnostic mismatch")
        push!(records,Dict("state"=>label,"baseline_seconds"=>timings[1],"cache_seconds"=>timings[2],"trials"=>length(traces[1]),"cached_ray_calls"=>cached_calls[],"cached_ray_hits"=>cached_hits[],"bitwise_trace_equal"=>true,"bitwise_solution_equal"=>true))
        open(joinpath(args.output_dir,"replay_results.toml"),"w") do io
            TOML.print(io,Dict("records"=>records,"artifact"=>artifact,"scope"=>"one outer attempt, fresh default LM damping; not original-history replay"))
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main_cache_replay()
end
