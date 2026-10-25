include(joinpath(@__DIR__,"..","..","..","examples","lvd_tomography.jl"))
using DelimitedFiles
function rows3(path)
 a,h=readdlm(path,',',Any;header=true)
 [Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function review_final3()
 raw=copy(ARGS);finalfile=popfirst!(raw);baseline=popfirst!(raw);empty!(ARGS);append!(ARGS,raw)
 args=parse_commandline();ispath(args.output_dir)&&error("fresh result path required")
 config=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
 fld=rows3(joinpath(baseline,"lvd_fit_field.csv"));saved=rows3(joinpath(baseline,"lvd_fit_predictions.csv"))
 w=Float64[r["w"] for r in fld];support=Bool[string(r["aquifer_mask"])=="true" for r in fld]
 final=TOML.parsefile(finalfile);candidate=Float64.(final["field"])
 @assert length(candidate)==length(w) && all(0 .<=candidate .<=MAX_WATER_FRACTION)
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
 p,j=assemble_forward_and_jacobian(ph,shallow,cfg,site,paths,candidate,energies;n_cells=length(w),threaded=true)
 oper,_=DiffPumas.Tomography.profiled_log_observation_operator(_->(p,j.*transpose(Float64.(support))),obs;weights=weights,robust_scale=config["robust_log_scale"],normalization=config["pinned_normalization"])
 residual,jac=oper(candidate);g=transpose(jac)*residual+laplacian_gradient(prior,candidate)
 pg=(candidate.-clamp.(candidate.-g,0.,MAX_WATER_FRACTION)).*support
 expected=Float64(final["objective"]);difference=trial["objective"]-base["objective"]
 @assert isapprox(base["objective"],100.59269073659428;atol=1e-10,rtol=0)
 @assert abs(trial["objective"]-expected)<=1e-10
 @assert isapprox(norm(pg,Inf),final["selected_branch_projected_gradient"];rtol=1e-7,atol=1e-10)
 @assert maximum(abs.(p.-p1)./max.(abs.(p1),1e-300))<1e-11
 @assert maximum(abs.(p1.-Float64.(final["predictions"]))./max.(abs.(p1),1e-300))<1e-11
 report=Dict("baseline"=>base,"trial"=>trial,"difference"=>difference,"expected_objective"=>expected,"absolute_trial_discrepancy"=>abs(trial["objective"]-expected),"full_paths_recomputed_per_field"=>length(paths),"supported_max_displacement"=>maximum(abs.(candidate-w)),"selected_branch_pg"=>norm(pg,Inf),"scope"=>"Independent full forward objective and selected-branch gradient recomputation; no stationarity certificate.")
 harm=TOML.parsefile(joinpath(dirname(finalfile),"harmonic_reduction.toml"))
 seen=falses(length(w))
 for k in eachindex(paths)
  weights[k]>0 || continue
  for seg in paths[k].segments;seen[seg.cell_idx]=true;end
 end
 unseen=findall(support .& .!seen);reduced=findall(support .& seen)
 @assert unseen==harm["unseen_cells"] && reduced==harm["reduced_cells"]
 @assert length(unseen)==53 && length(reduced)==94
 harmonicfield=Float64.(harm["field"]);harmobjective,harmpred=full(harmonicfield)
 previousfile=joinpath(@__DIR__,"..","convergence_round3_20261025","polish_run","results","final.toml")
 previous=TOML.parsefile(previousfile);previousfield=Float64.(previous["field"]);previousobjective,previouspred=full(previousfield)
 @assert harmpred[weights.>0]==previouspred[weights.>0]
 @assert abs(harmobjective["objective"]-harm["objective_after"])<1e-10
 @assert harmobjective["objective"]<=previousobjective["objective"]+1e-10
 L=zeros(length(w),length(w))
 for c in eachindex(w)
  e=zeros(length(w));e[c]=1;L[:,c].=laplacian_gradient(prior,e)
 end
 @assert maximum(abs.(L-L'))<1e-12
 @assert norm((L*harmonicfield)[unseen],Inf)<1e-10
 T=Float64.(reduce(vcat,transpose.(harm["reduced_tangent"])))
 @assert size(T)==(600,94)
 @assert maximum(abs.((L*T)[unseen,:]))<1e-10
 @assert norm(g[unseen],Inf)<1e-10
 report["harmonic_unseen_count"]=length(unseen);report["reduced_count"]=length(reduced)
 report["harmonic_fit_flux_identical"]=true;report["harmonic_objective"]=harmobjective["objective"]
 report["final_unseen_gradient_inf"]=norm(g[unseen],Inf)
 report["harmonic_tangent_prior_residual_inf"]=maximum(abs.((L*T)[unseen,:]))
 mkpath(args.output_dir);open(io->TOML.print(io,report),joinpath(args.output_dir,"review.toml"),"w")
end
review_final3()
