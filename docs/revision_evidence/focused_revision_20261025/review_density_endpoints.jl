include(joinpath(@__DIR__,"..","..","..","examples","lvd_tomography.jl"))
using DelimitedFiles
function review_density_rows(path)
 a,h=readdlm(path,',',Any;header=true);[Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function review_density_state(ph,shallow,cfg,site,paths,energies,water,support,obs,W,prior,configuration)
 p,j=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,water,energies;n_cells=length(water),threaded=true)
 oper,profile=DiffPumas.Tomography.profiled_log_observation_operator(_->(p,j.*transpose(Float64.(support))),obs;weights=W,robust_scale=configuration["robust_log_scale"],normalization=configuration["pinned_normalization"])
 residual,J=oper(water);gradient=transpose(J)*residual+laplacian_gradient(prior,water);pg=norm((water.-clamp.(water.-gradient,0.,MAX_WATER_FRACTION)).*support,Inf)
 data=.5dot(residual,residual);pv=DiffPumas.Tomography.prior_value(prior,water)
 (predictions=p,objective=data+pv,data=data,prior=pv,normalization=profile(p),pg=pg)
end
function review_density_main()
 raw=copy(ARGS);length(raw)>=5||error("usage DUMP ARTIFACT BASELINE ARMS OUTPUT gridoptions")
 dump,artifact,baseline,arms,out=raw[1:5];empty!(ARGS);append!(ARGS,raw[6:end]);args=parse_commandline();ispath(out)&&error("fresh output")
 configuration=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"));fields=review_density_rows(joinpath(baseline,"lvd_fit_field.csv"));saved=review_density_rows(joinpath(baseline,"lvd_fit_predictions.csv"));wb=Float64[r["w"] for r in fields]
 ph=load_tomography_physics(dump);idx(n)=get_material_index(ph,n);porous=idx("PorousWetRock")
 makecfg(rho)=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,rho,Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,0.)
 set_adjoint_derivative_mode!(:forwarddiff);load_csda_corrections!(artifact);emap=LVDTopo.build_elevation_map();vol=create_sensitivity_volume(emap,makecfg(2710.),args);nc=num_cells(vol);@assert nc==600
 for c in 1:nc;@assert Tuple(Float64(fields[c][k]) for k in ("x_m","y_m","z_m"))==cell_centroid(vol,c);end
 shallow=Bool[is_shallow_cell(vol,c) for c in 1:nc];site=build_site(30000.);bins=load_measured_intensity(args.match_data);covered=Int[r["bin_idx"] for r in saved]
 zen=Float64[(b.zen_lo+b.zen_hi)/2 for b in bins];az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : (b.az_hi+b.az_lo)/2 for b in bins];paths=[trace_path(vol,zen[b],az[b],emap) for b in covered]
 obs=Float64[r["measured_flux"] for r in saved];W=Float64[r["fit_mask"] for r in saved]./(1 .+(0.15 .*max.(1 ./cosd.(zen[covered]).-1,0)).^2);energies=csda_energy_quadrature(32,1e-3,1e9);prior=SmoothnessPrior(cell_neighbors(vol),configuration["absolute_prior_weight"])
 crossed=falses(nc);fitcrossed=falses(nc)
 for k in eachindex(paths),segment in paths[k].segments
  segment.distance>0||continue;crossed[segment.cell_idx]=true;W[k]>0&&(fitcrossed[segment.cell_idx]=true)
 end
 mkpath(out)
 selected=review_density_rows(joinpath(@__DIR__,"current_grid_tail_selection.csv"))
 open(joinpath(out,"selected_tail_geometry_depth.csv"),"w") do io
  println(io,"case,band,bin,cell,height_above_detector_m,local_surface_height_above_detector_m,centroid_depth_below_local_surface_m,centroid_below_surface")
  for r in selected
   c=Int(r["cell"]);x,y,z=cell_centroid(vol,c);surface=local_surface_height(emap,x,y);depth=surface-z
   println(io,join((r["case"],r["band"],r["bin"],c,z,surface,depth,depth>=0),','))
  end
 end
 results=Dict{String,Any}[]
 for arm in ("baseline","rho2660","rho2760","cut400","cut800")
  p=joinpath(arms,arm,"results");report=TOML.parsefile(joinpath(p,"report.toml"));f=review_density_rows(joinpath(p,"field.csv"));pr=review_density_rows(joinpath(p,"predictions.csv"));rho=Float64(report["rock_density"]);cut=Float64(report["support_cutoff_m"]);support=Bool[string(r["support"])=="true" for r in f]
  @assert support==((Float64[r["z_m"] for r in fields].>cut).&crossed)
  @assert (count(support),count(support.&fitcrossed),count(support.&.!fitcrossed))==(cut==400 ? (205,118,87) : cut==800 ? (86,59,27) : (147,94,53))
  initial=Float64[r["initial_water"] for r in f];final=Float64[r["final_water"] for r in f];@assert initial==wb.*support
  @assert all(final[.!support].==0)&&all(0 .<=final.<=MAX_WATER_FRACTION)
  for (label,water,pcol) in (("initial",initial,"initial_flux"),("final",final,"final_flux"))
   actual=review_density_state(ph,shallow,makecfg(rho),site,paths,energies,water,support,obs,W,prior,configuration);expected=report[label]
   @assert isequal(actual.predictions,Float64[r[pcol] for r in pr])
   @assert isapprox(actual.objective,expected["objective"];rtol=0,atol=1e-10)
   @assert isapprox(actual.pg,expected["selected_branch_pg"];rtol=1e-8,atol=1e-10)
   @assert isapprox(actual.normalization,expected["normalization"];rtol=1e-12,atol=0)
   push!(results,Dict("arm"=>arm,"stage"=>label,"objective"=>actual.objective,"data"=>actual.data,"prior"=>actual.prior,"normalization"=>actual.normalization,"branch_pg"=>actual.pg,"all1003_predictions_bitwise_equal"=>true,"support_cells"=>count(support)))
  end
 end
 open(io->TOML.print(io,Dict("passed"=>true,"checks"=>results,"scope"=>"Independent full initial/final forward+Jacobian and objective/profile/branchPG check for five finite-budget sensitivity arms. NoMC/refitting, no stationarity or physical uncertainty certificate.")),joinpath(out,"independent_endpoints.toml"),"w")
end
review_density_main()
