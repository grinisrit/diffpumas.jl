#!/usr/bin/env julia
# Conditional terrain stress on a frozen grid/support. Never edits the baseline.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"terrain_stress_geometry.jl"))
include(joinpath(@__DIR__,"prototype_stability_guarded_ray_cache.jl"))
include(joinpath(@__DIR__,"generated_stability_cached_flux_factory.jl"))
using .TerrainStressGeometry, DelimitedFiles

function terrain_rows(path)
    data,header=readdlm(path,',',Any;header=true)
    [Dict(string(header[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function terrain_csv(path,header,rows)
    open(path,"w") do io
        println(io,join(header,','))
        for row in rows;println(io,join(row,','));end
    end
end
function terrain_stress_main()
    raw=copy(ARGS);empty!(ARGS); extra=Dict("--baseline-results"=>"","--terrain-amplitude"=>"","--terrain-screen"=>"","--terrain-forward-only"=>"false")
    i=1
    while i<=length(raw)
        if haskey(extra,raw[i]);extra[raw[i]]=raw[i+1];i+=2
        else;push!(ARGS,raw[i]);i+=1;end
    end
    baseline=extra["--baseline-results"];isempty(baseline) && error("baseline required")
    amplitude=parse(Float64,extra["--terrain-amplitude"])
    amplitude in (-.05,-.01,0.,.01,.05) || error("unregistered amplitude")
    screenmode=extra["--terrain-screen"];screenmode in ("fixed","all") || error("screen must be fixed or all")
    forward_only=extra["--terrain-forward-only"]=="true"
    abs(amplitude)==.01 && !forward_only && error("±1% predeclared as forward-only")
    args=parse_commandline(); ispath(args.output_dir) && error("fresh output directory required")
    args.geometry=="grid" && args.papermatch_fit_mode=="stable" || error("fixed grid stable mode required")
    args.csda_quadrature=="midpoint" || error("midpoint required")
    saved=terrain_rows(joinpath(baseline,"lvd_fit_predictions.csv"))
    fields=terrain_rows(joinpath(baseline,"lvd_fit_field.csv"))
    configuration=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
    bytes2hex(sha256(read(args.correction_artifact)))==configuration["correction_artifact_sha256"] || error("artifact changed")
    args.papermatch_log_scale==configuration["robust_log_scale"] || error("robust scale changed")
    args.papermatch_abs_scale==configuration["pinned_normalization"] || error("normalization mode changed")
    args.papermatch_absolute_prior_weight==configuration["absolute_prior_weight"] || error("lambda changed")
    set_adjoint_derivative_mode!(:forwarddiff)
    ph=load_tomography_physics(args.dump_path);idx(name)=get_material_index(ph,name);porous=idx("PorousWetRock")
    cfg=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,args.rock_density,
        Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
    cfg.porous_top_thickness==0 || error("stress specified without optional porous layer")
    load_csda_corrections!(args.correction_artifact)
    emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,cfg,args)
    nc=num_cells(volume);nc==600 && length(fields)==nc || error("grid mismatch")
    support=Bool[string(r["aquifer_mask"])=="true" for r in fields]
    for c in 1:nc
        r=fields[c];Int(r["cell_idx"])==c || error("cell order mismatch")
        isequal(Tuple(Float64(r[k]) for k in ("x_m","y_m","z_m")),cell_centroid(volume,c)) || error("grid coordinates changed")
    end
    bytes2hex(sha256(join(findall(support),',')))==configuration["support_sha256"] || error("support changed")
    wbaseline=Float64[r["w"] for r in fields]
    shallow=Bool[is_shallow_cell(volume,c) for c in 1:nc]
    site=build_site(args.primary_altitude_km*1000)
    bins=load_measured_intensity(args.match_data)
    covered=Int[r["bin_idx"] for r in saved];length(covered)==1003 || error("coverage changed")
    zen=Float64[.5*(b.zen_lo+b.zen_hi) for b in bins]
    az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : .5*(b.az_lo+b.az_hi) for b in bins]
    obs=Float64[bins[b].intensity for b in covered]
    isequal(obs,Float64[r["measured_flux"] for r in saved]) || error("observations changed")
    originalscreen=Float64[r["fit_mask"] for r in saved]
    bytes2hex(sha256(join(covered[findall(>(0),originalscreen)],',')))==configuration["screen_sha256"] || error("screen changed")
    screen=screenmode=="fixed" ? originalscreen : ones(length(covered))
    originalpaths=[trace_path(volume,zen[b],az[b],emap) for b in covered]
    zero=stressed_elevation_map(emap,0.,LVDTopo.DETECTOR_ELEVATION)
    zeropaths=[trace_path(volume,zen[b],az[b],zero) for b in covered]
    isequal(path_snapshot.(originalpaths),path_snapshot.(zeropaths)) || error("zero geometry differs")
    modified=stressed_elevation_map(emap,amplitude,LVDTopo.DETECTOR_ELEVATION)
    paths=[trace_path(volume,zen[b],az[b],modified) for b in covered]
    all(p->p.valid,paths) || error("stress changed coverage; refusing silent bin removal")
    energies=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
    baselinepredict=flux_only_operator(ph,shallow,cfg,site,originalpaths,energies;threaded=args.threaded)
    baselineflux=baselinepredict(wbaseline)
    isequal(reinterpret(UInt64,baselineflux),reinterpret(UInt64,Float64[r["matched_csda_flux"] for r in saved])) || error("baseline forward parity failed")
    lambda=configuration["absolute_prior_weight"];prior=SmoothnessPrior(cell_neighbors(volume),lambda)
    dem=args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)
    weights=screen./(1 .+dem.^2)
    model=masked_material_operator(make_csda_operator(ph,shallow,cfg,site,paths,energies;
        n_cells=nc,valid_bins=collect(eachindex(paths)),threaded=args.threaded),support)
    cached,_=build_exact_cached_flux_factory(ph,shallow,cfg,site,paths,energies,args,rho->cfg)
    predict=w->cached(cfg.rock_density,Vector{Float64}(w.*support))
    operator,scale,predict_residual=DiffPumas.Tomography.profiled_log_observation_operator(model,obs;
        weights=weights,robust_scale=args.papermatch_log_scale,normalization=args.papermatch_abs_scale,
        predict_only=predict,return_predict_only=true)
    fixedflux=predict(wbaseline)
    directpredict=flux_only_operator(ph,shallow,cfg,site,paths,energies;threaded=args.threaded)
    isequal(reinterpret(UInt64,fixedflux),reinterpret(UInt64,directpredict(wbaseline))) || error("new-path cache primal parity failed")
    amplitude==0 && !isequal(reinterpret(UInt64,fixedflux),reinterpret(UInt64,baselineflux)) && error("zero cached prediction parity failed")
    initial_gate=Dict{String,Any}()
    if amplitude==0 && screenmode=="fixed"
        savedstatus=Dict(split(line,'=';limit=2)[1]=>split(line,'=';limit=2)[2]
            for line in eachline(joinpath(baseline,"lvd_solver_status.txt")) if occursin('=',line))
        r0,J0=operator(wbaseline)
        f0=.5dot(r0,r0)+DiffPumas.Tomography.prior_value(prior,wbaseline)
        g0=transpose(J0)*r0+laplacian_gradient(prior,wbaseline)
        pg0=norm((wbaseline.-clamp.(wbaseline.-g0,0.,MAX_WATER_FRACTION)).*support,Inf)
        s0=scale(fixedflux)
        isequal(f0,parse(Float64,savedstatus["objective"])) || error("zero initial objective parity failed: $f0")
        isequal(pg0,parse(Float64,savedstatus["projected_gradient"])) || error("zero initial projected-gradient parity failed: $pg0")
        all(row->isequal(s0,Float64(row["normalization"])),saved) || error("zero initial normalization parity failed: $s0")
        initial_gate=Dict("passed"=>true,"objective_bitwise"=>true,"projected_gradient_bitwise"=>true,
            "normalization_bitwise"=>true,"objective"=>f0,"projected_gradient"=>pg0,"normalization"=>s0)
    end
    mkpath(args.output_dir)
    isempty(initial_gate) || open(io->TOML.print(io,initial_gate),joinpath(args.output_dir,"zero_initial_gate.toml"),"w")
    terrain_csv(joinpath(args.output_dir,"path_changes.csv"),
        ("bin_idx","zenith_deg","azimuth_deg","fit_mask","original_fit_mask","old_slant_m","new_slant_m","delta_slant_m","delta_exit_height_m","old_segment_cells","new_segment_cells","old_unique_supported_cells","new_unique_supported_cells","old_supported_length_m","new_supported_length_m"),
        [(covered[k],zen[covered[k]],az[covered[k]],screen[k],originalscreen[k],originalpaths[k].surface_distance,paths[k].surface_distance,
          paths[k].surface_distance-originalpaths[k].surface_distance,paths[k].surface_exit_z-originalpaths[k].surface_exit_z,
          length(originalpaths[k].segments),length(paths[k].segments),
          length(unique(s.cell_idx for s in originalpaths[k].segments if support[s.cell_idx])),
          length(unique(s.cell_idx for s in paths[k].segments if support[s.cell_idx])),
          sum((s.distance for s in originalpaths[k].segments if support[s.cell_idx]);init=0.),
          sum((s.distance for s in paths[k].segments if support[s.cell_idx]);init=0.)) for k in eachindex(paths)])
    history=NamedTuple[]
    open(joinpath(args.output_dir,"solver_progress.csv"),"w") do io
        println(io,"iteration,objective,projected_gradient");flush(io)
        wfinal = forward_only ? copy(wbaseline) : first(gauss_newton_reconstruct(operator,zeros(length(obs));w0=copy(wbaseline),n_iter=args.papermatch_gn_iters,
            prior=prior,free=support,solver_mode=:active_set,projected_gradient_tol=args.papermatch_gradient_tol,
            diagnostics=history,predict_only=predict_residual,
            diagnostic_callback=row->begin;println(io,join(values(row),','));flush(io);end))
        # `wfinal` stays in this block so every final artifact uses this solve.
        finalflux=predict(wfinal);normalization=scale(finalflux);fixedscale=scale(fixedflux)
        if forward_only
            residual=predict_residual(wfinal)
            pg_inf=NaN
        else
            residual,J=operator(wfinal)
            gradient=transpose(J)*residual+laplacian_gradient(prior,wfinal)
            pg=(wfinal.-clamp.(wfinal.-gradient,0.,MAX_WATER_FRACTION)).*support
            pg_inf=norm(pg,Inf)
        end
        objective=.5dot(residual,residual)+DiffPumas.Tomography.prior_value(prior,wfinal)
        terrain_csv(joinpath(args.output_dir,"fit_field.csv"),("cell_idx","support","baseline_water","water","delta_water"),
            [(c,support[c],wbaseline[c],wfinal[c],wfinal[c]-wbaseline[c]) for c in 1:nc])
        sector=[135<=az[b]<=180 && 40<=zen[b]<=55 for b in covered]
        terrain_csv(joinpath(args.output_dir,"predictions.csv"),
            ("bin_idx","sector","fit_mask","observation","baseline_flux","fixed_field_flux","refit_flux","fixed_normalization","refit_normalization","fixed_log_residual","refit_log_residual"),
            [(covered[k],sector[k],screen[k],obs[k],baselineflux[k],fixedflux[k],finalflux[k],fixedscale,normalization,
              log(max(fixedscale*fixedflux[k],1e-300)/obs[k]),log(max(normalization*finalflux[k],1e-300)/obs[k])) for k in eachindex(obs)])
        report=Dict("amplitude"=>amplitude,"screen_mode"=>screenmode,"forward_only"=>forward_only,"scope"=>"Assumed localized terrain stress; not measured DEM error, uncertainty interval, or attribution of residual cause.",
            "grid_cells"=>nc,"support_cells"=>count(support),"covered_bins"=>length(obs),"fit_bins"=>count(>(0),screen),
            "sector_bins"=>count(sector),"sector_fit_bins"=>count(sector .& (screen.>0)),"absolute_prior_weight"=>lambda,
            "objective"=>objective,"projected_gradient_inf"=>pg_inf,"stationary_at_requested_tolerance"=>pg_inf<=args.papermatch_gradient_tol,
            "iterations_recorded"=>max(length(history)-1,0),"maximum_iterations"=>args.papermatch_gn_iters,
            "field_relative_change_from_old_baseline"=>norm(wfinal-wbaseline)/max(norm(wbaseline),1e-30),
            "baseline_prediction_bitwise"=>true,"zero_geometry_bitwise"=>true,"new_path_cache_primal_bitwise"=>true,
            "max_abs_map_height_change_m"=>maximum(abs.(modified.data.-emap.data)),
            "changed_map_nodes"=>count(!iszero,modified.data.-emap.data),
            "off_sector_changed_paths"=>count(k->!sector[k] && !isequal(path_snapshot(paths[k]),path_snapshot(originalpaths[k])),eachindex(paths)),
            "off_sector_max_abs_slant_change_m"=>maximum((abs(paths[k].surface_distance-originalpaths[k].surface_distance) for k in eachindex(paths) if !sector[k]);init=0.),
            "changed_supported_cell_intersections"=>count(k->Set(s.cell_idx for s in paths[k].segments if support[s.cell_idx])!=Set(s.cell_idx for s in originalpaths[k].segments if support[s.cell_idx]),eachindex(paths)),
            "max_abs_slant_change_m"=>maximum(abs(p.surface_distance-q.surface_distance) for (p,q) in zip(paths,originalpaths)),
            "baseline_results"=>abspath(baseline),"correction_artifact_sha256"=>bytes2hex(sha256(read(args.correction_artifact))))
        open(f->TOML.print(f,report),joinpath(args.output_dir,"report.toml"),"w")
    end
    println("Terrain stress complete; inspect report.toml stationarity rather than assuming convergence.")
end
if abspath(PROGRAM_FILE)==@__FILE__;terrain_stress_main();end
