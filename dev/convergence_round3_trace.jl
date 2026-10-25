#!/usr/bin/env julia
include(joinpath(@__DIR__,"convergence_round3_diagnostic.jl"))
include(joinpath(@__DIR__,"convergence_round3_trace_functions.jl"))
function convergence_round3_trace()
    raw=copy(ARGS); baseline="";cell=0; empty!(ARGS)
    i=1
    while i<=length(raw)
        if raw[i]=="--trace-cell"
            cell=parse(Int,raw[i+1]); i+=2
        elseif raw[i]=="--baseline-results"
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
    @assert cell>0 && support[cell]
    mkpath(args.output_dir)
    mod=DiffPumas.Tomography
    affected=findall(p->any(s->s.cell_idx==cell,p.segments),paths)
    reports=Dict{String,Any}[]
    for ray in affected
        basetrace=Any[];baselineflux=0.
        for delta in (0.,-1e-6,1e-6,-1e-7,1e-7,-1e-8,1e-8)
            w=copy(wf);w[cell]=clamp(w[cell]+delta,0.,MAX_WATER_FRACTION)
            mod.ROUND3_EVENTS[]=Any[]
            traced=mod.round3_trace_flux(ph,shallow,cfg,site,paths[ray],w,energies)
            actual=compute_directional_flux_csda(ph,shallow,cfg,site,paths[ray],w,energies)
            @assert isapprox(traced,actual;rtol=1e-13,atol=0)
            ev=mod.ROUND3_EVENTS[]
            if delta==0
                basetrace=ev;baselineflux=actual
            end
            index=Dict((e.sample,e.segment,e.step)=>e for e in basetrace)
            mismatches=Dict{String,Any}[]
            trialkeys=Set((e.sample,e.segment,e.step) for e in ev)
            changes=Dict("interval"=>0,"limiter"=>0,"air_floor"=>0,"new_step"=>0,"removed_step"=>0)
            changes["removed_step"]=count(key->!(key in trialkeys),keys(index))
            for e in ev
                key=(e.sample,e.segment,e.step)
                if !haskey(index,key)
                    changes["new_step"]+=1
                    continue
                end
                old=index[key]
                changed=(old.intervals!=e.intervals || old.limiter!=e.limiter || old.air_floor!=e.air_floor)
                changes["interval"]+=old.intervals!=e.intervals
                changes["limiter"]+=old.limiter!=e.limiter
                changes["air_floor"]+=old.air_floor!=e.air_floor
                if changed && length(mismatches)<30
                    push!(mismatches,Dict("sample"=>e.sample,"segment"=>e.segment,"step"=>e.step,
                        "base_energy"=>old.energy,"trial_energy"=>e.energy,"base_remaining"=>old.remaining,
                        "trial_remaining"=>e.remaining,"base_ds"=>old.ds,"trial_ds"=>e.ds,
                        "base_intervals"=>old.intervals,"trial_intervals"=>e.intervals,
                        "base_limiter"=>old.limiter,"trial_limiter"=>e.limiter,
                        "base_limiter_density"=>old.limiter_density,"trial_limiter_density"=>e.limiter_density,
                        "base_candidates"=>old.step_candidates,"trial_candidates"=>e.step_candidates,
                        "base_tied_branches"=>old.tied_branches,"trial_tied_branches"=>e.tied_branches,
                        "base_air_floor"=>old.air_floor,"trial_air_floor"=>e.air_floor))
                end
            end
            push!(reports,Dict("ray_position"=>ray,"bin_idx"=>covered[ray],"delta"=>delta,
                "path_cell_indices"=>[s.cell_idx for s in paths[ray].segments],
                "base_flux"=>baselineflux,"flux"=>actual,"flux_change"=>actual-baselineflux,
                "trace_step_count"=>length(ev),"air_floor_steps"=>count(e->e.air_floor,ev),
                "air_overshoot_steps"=>count(e->e.segment==-2 && e.ds>e.remaining,ev),"base_step_count"=>length(basetrace),
                "changes"=>changes,"first_30_changed_events"=>mismatches))
        end
        open(joinpath(args.output_dir,"trace.toml"),"w") do io
            TOML.print(io,Dict("cell"=>cell,"reports"=>reports,"affected_ray_count"=>length(affected),
                "interval_scope"=>"all physics tables including potentially unused materials; candidate events only",
                "caveat"=>"Branch transitions identify mechanisms but do not certify nonsmooth stationarity; event mismatches after changed step count may be downstream consequences."))
        end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;convergence_round3_trace();end
