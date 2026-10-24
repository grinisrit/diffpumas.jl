#!/usr/bin/env julia
# Isolated exact cache experiment; not imported by production.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"prototype_stability_guarded_ray_cache.jl"))
using .GuardedStabilityRayCachePrototype
using DelimitedFiles, TOML, Serialization, SHA
include(joinpath(@__DIR__,"generated_stability_cached_flux_factory.jl"))
function audit_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function main_guarded_trial_replay()
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
    artifacts=[args.correction_artifact,joinpath(@__DIR__,"..","docs","revision_evidence","surrogate_v2","production","frozen_corrections.toml")]
    mk_cfg=ρ->MaterialConfig(cfg.rock_material,cfg.water_material,cfg.air_material,cfg.porous_material,
        ρ,cfg.water_density,cfg.porous_density,cfg.porous_top_thickness)
    forward=(i,w)->compute_directional_flux_csda(ph,shallow,cfg,site,paths[i],w,energies)
    raw=w->[forward(i,w) for i in eachindex(paths)]
    W=Float64[r["fit_mask"] for r in saved]./(1 .+(args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)).^2)
    prior=SmoothnessPrior(cell_neighbors(volume),3.954930197858131)
    function transform(p,w)
        op,scale=DiffPumas.Tomography.profiled_log_observation_operator(_->(p,zeros(length(p),1)),obs;weights=W,robust_scale=.3,normalization=0.)
        r,_=op([0.]);r,scale(p),.5dot(r,r)+DiffPumas.Tomography.prior_value(prior,w)
    end
    copied_path=joinpath(@__DIR__,"prototype_stability_gn_replay.jl")
    original_path=joinpath(@__DIR__,"..","src","Tomography_inverse.jl")
    original_text=read(original_path,String)
    original_function="function gauss_newton_reconstruct("*first(split(last(split(original_text,"function gauss_newton_reconstruct(";limit=2)),"\n\"\"\"\n    make_csda_operator";limit=2))
    copied_body=last(split(read(copied_path,String),"@eval DiffPumas.Tomography begin\n";limit=2))
    normalized=chop(copied_body;tail=5)
    normalized=replace(normalized,"function replay_cache_gauss_newton_reconstruct("=>"function gauss_newton_reconstruct(",
        "diagnostic_callback = nothing,\n                                  trial_trace = Any[])"=>"diagnostic_callback = nothing)")
    trace_line="\n                push!(trial_trace,(iteration=length(history)-1,lm_try=_try,halving=_ls,alpha=alpha,mu=mu,water=copy(w_new),prediction=copy(pred_new),objective=f_new,directional=dot(grad,w_new[active].-w[active])))"
    count(trace_line,normalized)==1 || error("unexpected GN trace instrumentation")
    normalized=replace(normalized,trace_line=>"")
    normalized==original_function || error("normalized copied GN source mismatch")
    length(trace)==124 || error("expected complete124-trial stream")
    length(paths)==1003 && length(energies)==32 || error("unexpected geometry/quadrature")
    Threads.nthreads()>=2 || error("threaded gate requires --threads=2")
    affinity=strip(match(r"Cpus_allowed_list:\s*([^\n]+)",read("/proc/self/status",String)).captures[1])
    affinity=="20" || error("gate must stay on reservedCPU20")
    reference_end=joinpath(dirname(dirname(tracepath)),"end_status.json")
    metadata=TOML.parsefile(joinpath(@__DIR__,"generated_stability_cached_flux_factory.toml"))
    digest(path)=bytes2hex(sha256(read(path)))
    digest(metadata["closure_factory_path"])==metadata["closure_factory_sha256"] || error("factory changed")
    function emit_measurement(kind,cases;actual_trial=false,threaded_gate=false)
        checks=Dict("full_prediction_bitwise"=>true,"residual_bitwise"=>true,
            "profile_bitwise"=>true,"objective_bitwise"=>true,"context_guard_passed"=>true)
        threaded_gate && (checks["threaded_parity"]=true)
        actual_trial && (checks["trial_stream_complete"]=true)
        actual_trial && (checks["copied_gn_source_match"]=true)
        data=merge(Dict{String,Any}(metadata),Dict("gate_kind"=>kind,"scope"=>"exact frozen-entry callback parity on fixed recorded124 GN candidates; not a new cached optimizer solve or full-history convergence claim; fullmodel/J unchanged",
            "sample_scope"=>"deterministic CSDA quadrature nodes per ray; no MC",
            "sample_count"=>32,"case_count"=>cases,"ray_count"=>1003,"threads"=>Threads.nthreads(),"cpu_affinity"=>[20],"checks"=>checks))
        if actual_trial
            merge!(data,Dict("expected_trial_count"=>124,"observed_trial_count"=>length(trace),
                "recorded_stream_path"=>abspath(tracepath),"recorded_stream_sha256"=>digest(tracepath),
                "reference_end_path"=>abspath(reference_end),"reference_end_sha256"=>digest(reference_end),
                "copied_gn_path"=>abspath(copied_path),"copied_gn_sha256"=>digest(copied_path),
                "original_gn_source_path"=>abspath(original_path),"original_gn_source_sha256"=>digest(original_path)))
        end
        measurement=joinpath(args.output_dir,kind*"_measurement.toml")
        open(measurement,"w") do io;TOML.print(io,data);end
        producer=joinpath(@__DIR__,"stability_cache_certificate.py")
        run(`python3 $producer emit-proof --measurement $measurement --output $(joinpath(args.output_dir,kind*"_proof.json"))`)
    end
    bittransform(x,y)=reinterpret(UInt64,x[1])==reinterpret(UInt64,y[1]) && reinterpret(UInt64,x[2])==reinterpret(UInt64,y[2]) && reinterpret(UInt64,x[3])==reinterpret(UInt64,y[3])
    mkpath(args.output_dir);records=Any[]
    other=audit_rows(joinpath(@__DIR__,"..","examples","data","stability_v2_dual_controls_20261024","dry","start015","results","lvd_fit_field.csv"))
    w15=Float64[r["w"] for r in other]
    # Full real sequence, plus a whole-field transition; do not substitute
    # coordinate-only microbenchmarks for real GN directions.
    states=vcat([w15,wf],[row.water for row in trace])
    threaded_indices=unique([1,2,3,clamp(2+length(trace)÷2,1,length(states)),length(states)])
    for artifact in artifacts
        load_csda_corrections!(artifact)
        serial,cache=build_exact_cached_flux_factory(ph,shallow,cfg,site,paths,energies,
            merge(args,(threaded=false,correction_artifact=artifact)),mk_cfg)
        threaded,threadcache=build_exact_cached_flux_factory(ph,shallow,cfg,site,paths,energies,
            merge(args,(threaded=true,correction_artifact=artifact)),mk_cfg)
        a=raw(wf);b=serial(cfg.rock_density,wf);c=threaded(cfg.rock_density,wf)
        isequal(a,b)&&isequal(a,c)||error("warmup mismatch")
        for (j,w) in enumerate(states)
            hits=count(i->cache.valid[i] && all(k->cache.snapshots[i][k]==reinterpret(UInt64,w[cache.cells[i][k]]),eachindex(cache.cells[i])),eachindex(cache.cells))
            if isodd(j)
                t0=time_ns();a=raw(w);rawtime=(time_ns()-t0)/1e9
                t0=time_ns();b=serial(cfg.rock_density,w);cachetime=(time_ns()-t0)/1e9
            else
                t0=time_ns();b=serial(cfg.rock_density,w);cachetime=(time_ns()-t0)/1e9
                t0=time_ns();a=raw(w);rawtime=(time_ns()-t0)/1e9
            end
            reinterpret(UInt64,a)==reinterpret(UInt64,b)||error("raw mismatch")
            bittransform(transform(a,w),transform(b,w))||error("profile/objective mismatch")
            threaded_tested=j in threaded_indices
            if threaded_tested
                c=threaded(cfg.rock_density,w)
                reinterpret(UInt64,a)==reinterpret(UInt64,c)||error("threaded raw mismatch")
                bittransform(transform(a,w),transform(c,w))||error("threaded profile/objective mismatch")
            end
            push!(records,Dict("artifact"=>artifact,"state"=>j,"raw_seconds"=>rawtime,"cache_seconds"=>cachetime,"hits"=>hits,"changed_cells"=>count(!iszero,w.-wf),"exact_equal"=>true,"threaded_tested"=>threaded_tested))
            println("guarded real-trial replay ",last(records));flush(stdout)
            open(joinpath(args.output_dir,"guarded_trial_replay.toml"),"w") do io
                TOML.print(io,Dict("records"=>records,"threads"=>Threads.nthreads(),"scope"=>"all real dry-GN full-vector candidates evaluated in dry/corrected contexts; no corrected optimizer acceptance claim"))
            end
        end
        function reject_context(change!,restore!)
            rejected=false
            change!()
            try
                serial(cfg.rock_density,wf)
            catch err
                err isa ArgumentError && occursin("physical context changed",sprint(showerror,err)) || rethrow()
                rejected=true
            finally
                restore!()
            end
            rejected || error("changed physical context accepted")
        end
        shallow0=shallow[1]
        reject_context(()->(shallow[1]=!shallow0),()->(shallow[1]=shallow0))
        e0=energies[1]
        reject_context(()->(energies[1]=DiffPumas.Tomography.EnergySample(nextfloat(e0.energy),e0.charge,e0.weight)),()->(energies[1]=e0))
        idx=findfirst(p->!isempty(p.segments),paths);s0=paths[idx].segments[1]
        reject_context(()->(paths[idx].segments[1]=DiffPumas.Tomography.CellSegment(s0.cell_idx,nextfloat(s0.distance))),()->(paths[idx].segments[1]=s0))
        c0=get_csda_correction();hard0=c0.kappa_hard
        reject_context(()->(c0.kappa_hard=nextfloat(hard0)),()->(c0.kappa_hard=hard0))
        m0=get_csda_material_correction()
        reject_context(()->set_csda_material_correction!(enabled=m0.enabled,linear=nextfloat(m0.linear),quadratic=m0.quadratic),
            ()->set_csda_material_correction!(enabled=m0.enabled,linear=m0.linear,quadratic=m0.quadratic))
        if artifact==first(artifacts)
            emit_measurement("dry_serial",length(states))
            emit_measurement("actual_gn_trial",length(trace);actual_trial=true)
        else
            emit_measurement("corrected_serial",length(states))
            emit_measurement("corrected_threaded",length(threaded_indices);threaded_gate=true)
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main_guarded_trial_replay()
end
