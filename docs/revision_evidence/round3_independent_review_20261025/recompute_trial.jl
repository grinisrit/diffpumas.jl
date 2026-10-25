include(joinpath(@__DIR__,"..","..","..","examples","lvd_tomography.jl"))
using DelimitedFiles
function rows3(path)
 a,h=readdlm(path,',',Any;header=true)
 [Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function review3()
 raw=copy(ARGS);diagnostic=popfirst!(raw);baseline=popfirst!(raw);empty!(ARGS);append!(ARGS,raw)
 args=parse_commandline();ispath(args.output_dir)&&error("fresh result path required")
 config=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
 fld=rows3(joinpath(baseline,"lvd_fit_field.csv"));saved=rows3(joinpath(baseline,"lvd_fit_predictions.csv"))
 w=Float64[r["w"] for r in fld];support=Bool[string(r["aquifer_mask"])=="true" for r in fld]
 trials=rows3(joinpath(diagnostic,"directional_checks.csv"));best=trials[argmin(Float64[r["objective"] for r in trials])]
 dirs=TOML.parsefile(joinpath(diagnostic,"directions.toml"))["directions"]
 candidate=clamp.(w.+Float64(best["step"]).*dirs[string(best["direction"])],0.,MAX_WATER_FRACTION)
 @assert candidate[.!support]==w[.!support]
 ph=load_tomography_physics(args.dump_path);idx(n)=get_material_index(ph,n)
 cfg=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),2710.,Float64(ph.tables[idx("Water")].density),0.,0.)
 load_csda_corrections!(args.correction_artifact);em=LVDTopo.build_elevation_map();vol=create_sensitivity_volume(em,cfg,args)
 shallow=Bool[is_shallow_cell(vol,c) for c in 1:num_cells(vol)];site=build_site(args.primary_altitude_km*1000)
 bins=load_measured_intensity(args.match_data);covered=Int[r["bin_idx"] for r in saved]
 zen=Float64[(b.zen_lo+b.zen_hi)/2 for b in bins];az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : (b.az_hi+b.az_lo)/2 for b in bins]
 paths=[trace_path(vol,zen[b],az[b],em) for b in covered];obs=Float64[bins[b].intensity for b in covered]
 weights=Float64[r["fit_mask"] for r in saved]./(1 .+(args.papermatch_dem_unc.*max.(1 ./cosd.(zen[covered]).-1,0)).^2)
 energies=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
 prior=SmoothnessPrior(cell_neighbors(vol),Float64(config["absolute_prior_weight"]))
 function full(wv)
  p=zeros(length(paths))
  Threads.@threads for j in eachindex(paths)
   p[j]=compute_directional_flux_csda(ph,shallow,cfg,site,paths[j],wv,energies)
  end
  op,scale=DiffPumas.Tomography.profiled_log_observation_operator(_->(p,zeros(length(p),1)),obs;weights=weights,robust_scale=config["robust_log_scale"],normalization=config["pinned_normalization"])
  r,_=op([0.]);data=.5dot(r,r);pv=DiffPumas.Tomography.prior_value(prior,wv)
  Dict("objective"=>data+pv,"data"=>data,"prior"=>pv,"normalization"=>scale(p)),p
 end
 base,p0=full(w);trial,p1=full(candidate)
 expected=Float64(best["objective"]);difference=trial["objective"]-base["objective"]
 @assert isapprox(base["objective"],100.59269073659428;atol=1e-10,rtol=0)
 @assert abs(trial["objective"]-expected)<=1e-10
 report=Dict("baseline"=>base,"trial"=>trial,"direction"=>string(best["direction"]),"step"=>Float64(best["step"]),"difference"=>difference,"expected_objective"=>expected,"absolute_trial_discrepancy"=>abs(trial["objective"]-expected),"full_paths_recomputed_per_field"=>length(paths),"supported_max_displacement"=>maximum(abs.(candidate-w)),"scope"=>"Independent full forward objective recomputation; no stationarity certificate.")
 mkpath(args.output_dir);open(io->TOML.print(io,report),joinpath(args.output_dir,"review.toml"),"w")
end
review3()
