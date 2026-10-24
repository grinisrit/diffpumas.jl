#!/usr/bin/env julia
# Isolated exact cache experiment; not imported by production.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"prototype_stability_ray_cache.jl"))
using .StabilityRayCachePrototype
using DelimitedFiles, TOML, Serialization
function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function main_corrected_trial_replay()
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
    tracepath=joinpath(@__DIR__,"..","examples","data","stability_v2_gn_cache_replay_20261024","results","dry0_baseline_trace.jls")
    trace=deserialize(tracepath).trace
    artifact=joinpath(@__DIR__,"..","docs","revision_evidence","surrogate_v2","production","frozen_corrections.toml")
    load_csda_corrections!(artifact)
    cache=RayCache(paths,artifact)
    forward=(i,w)->compute_directional_flux_csda(ph,shallow,cfg,site,paths[i],w,energies)
    raw=w->[forward(i,w) for i in eachindex(paths)]
    W=Float64[r["fit_mask"] for r in saved]./(1 .+(args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)).^2)
    prior=SmoothnessPrior(cell_neighbors(volume),3.954930197858131)
    function transform(p,w)
        op,scale=DiffPumas.Tomography.profiled_log_observation_operator(_->(p,zeros(length(p),1)),obs;weights=W,robust_scale=.3,normalization=0.)
        r,_=op([0.]);r,scale(p),.5dot(r,r)+DiffPumas.Tomography.prior_value(prior,w)
    end
    # Compile/warm cache at actual starting field, no timing claims from warmup.
    a=raw(wf);b,_=evaluate!(cache,wf,forward);isequal(a,b)||error("warmup mismatch")
    records=Any[];mkpath(args.output_dir)
    for (j,row) in enumerate(trace)
        w=row.water
        t0=time_ns();a=raw(w);rawtime=(time_ns()-t0)/1e9
        t0=time_ns();b,hits=evaluate!(cache,w,forward);cachetime=(time_ns()-t0)/1e9
        reinterpret(UInt64,a)==reinterpret(UInt64,b)||error("raw mismatch")
        isequal(transform(a,w),transform(b,w))||error("profile/objective mismatch")
        push!(records,Dict("trial"=>j,"raw_seconds"=>rawtime,"cache_seconds"=>cachetime,"hits"=>hits,"changed_cells"=>count(!iszero,w.-wf),"exact_equal"=>true))
        println("corrected real-trial replay ",last(records));flush(stdout)
    end
    open(joinpath(args.output_dir,"corrected_trial_replay.toml"),"w") do io
        TOML.print(io,Dict("artifact"=>artifact,"records"=>records,"scope"=>"corrected predictions at dry real-GN full-vector candidates; no corrected optimizer acceptance claim"))
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main_corrected_trial_replay()
end
