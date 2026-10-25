# Same stored case/window deterministic audit. No MC, fitting, or reselection.
include(joinpath(@__DIR__,"..","..","..","dev","validate_surrogate_v2.jl"))
function isolated_contrast_case(s,ph,shallow,mat,site,es,nc,vol)
  w=s.w;wl=copy(w);wh=copy(w);wl[s.cell]=s.lo;wh[s.cell]=s.hi
  _,J=assemble_forward_and_jacobian(ph,shallow,mat,site,[s.path],w,es;n_cells=nc,threaded=false)
  ad=J[1,s.cell];@assert isapprox(ad,s.stored;rtol=1e-12,atol=1e-20)
  fl=compute_directional_flux_csda(ph,shallow,mat,site,s.path,wl,es);fh=compute_directional_flux_csda(ph,shallow,mat,site,s.path,wh,es);sec=(fh-fl)/(s.hi-s.lo)
  difference=sec-ad
  return (population=s.population,bin=s.bin,cell=s.cell,field=s.field,zenith=s.path.zenith,azimuth=s.path.azimuth,slant_m=DiffPumas.Tomography.path_rock_length(s.path),cell_z_m=vol.centroids[3,s.cell],local_fraction=w[s.cell],fraction_lo=s.lo,fraction_hi=s.hi,width=s.hi-s.lo,ad=ad,stored_ad=s.stored,stored_ad_bitwise_equal=isequal(ad,s.stored),midpoint_secant=sec,secant_minus_ad=difference,absolute_difference=abs(difference),relative_difference=abs(difference)/max(abs(ad),abs(sec),1e-30),flux_lo=fl,flux_hi=fh)
end
function contrast_audit()
 length(ARGS)==3 || error("usage: DUMP FROZEN_ARTIFACT FRESH_OUTPUT")
 dump,artifact,out=ARGS;ispath(out)&&error("fresh output required")
 root=normpath(joinpath(@__DIR__,"..","..",".."));ev=joinpath(root,"docs","revision_evidence")
 ph,mat,site,em,vol,shallow,es=v2state(dump);nc=num_cells(vol)
 set_adjoint_derivative_mode!(:forwarddiff);load_csda_corrections!(artifact)
 @assert get_adjoint_derivative_mode()==:forwarddiff
 @assert get_csda_material_correction().enabled
 oldpaths=filter(p->p.valid,precompute_paths(vol,em,[0.,15.,30.,45.,60.,75.],exclusive_degree_grid(360.,15.)))
 newpaths=filter(p->p.valid,precompute_paths(vol,em,[15.,45.,60.,75.],[37.5,157.5,277.5]))
 lowplan=TOML.parsefile(joinpath(ev,"lowfraction_round3_20261025","pilot","results","preregistered_plan.toml"))
 lowpaths=precompute_paths(vol,em,lowplan["test_zenith"],lowplan["test_azimuth"])
 fields=[fill(.35,nc),[.15+.4*sin(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc],[.15+.4*cos(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc],[vol.centroids[3,c]>500 && hypot(vol.centroids[1,c],vol.centroids[2,c])<1500 ? .6 : 0. for c in 1:nc]]
 lowfields=[fill(.04,nc),fill(.11,nc),[.015+.12*sin(vol.centroids[1,c]/1130+vol.centroids[3,c]/710+.37)^2 for c in 1:nc],[vol.centroids[3,c]>650 && hypot(vol.centroids[1,c]-210,vol.centroids[2,c]+170)<1370 ? .14 : .01 for c in 1:nc]]
 replay=readrows(joinpath(ev,"surrogate_v2","forwarddiff_replay","ad_fd_replay.csv"))
 adlookup=Dict((string(r["population"]),Int(r["bin"]),Int(r["cell"]),Float64(r["field"]))=>Float64(r["ad"]) for r in replay if Float64(r["step"])==1e-6)
 hist=readrows(joinpath(ev,"corrected_validation","mc_jacobian.csv"));new=readrows(joinpath(ev,"surrogate_v2","production","heldout_derivatives.csv"));low=readrows(joinpath(ev,"lowfraction_round3_20261025","pilot","results","heldout_derivatives.csv"));lowad=readrows(joinpath(ev,"lowfraction_round3_20261025","pilot","results","ad_fd_steps.csv"))
 lowlookup=Dict((Int(r["bin"]),Int(r["cell"]),Int(r["field"]))=>Float64(r["ad"]) for r in lowad if string(r["arm"])=="control" && Float64(r["step"])==1e-6)
 cases=Any[]
 for (name,rows,pool) in (("historical",hist,oldpaths),("new",new,newpaths),("lowfraction_control",filter(r->string(r["arm"])=="control",low),lowpaths))
  unique=Dict((Int(r["bin"]),Int(r["cell"]),name=="historical" ? Float64(r["fraction"]) : Float64(r["field"]))=>r for r in rows)
  for ((b,c,k),r) in sort!(collect(unique);by=first)
   w=name=="historical" ? fill(k,nc) : copy(name=="new" ? fields[Int(k)] : lowfields[Int(k)])
   lo=Float64(r[name=="lowfraction_control" ? "lo" : "fraction_lo"]);hi=Float64(r[name=="lowfraction_control" ? "hi" : "fraction_hi"])
   stored=name=="lowfraction_control" ? lowlookup[(b,c,Int(k))] : adlookup[(name,b,c,k)]
   push!(cases,(population=name,bin=b,cell=c,field=k,w=w,lo=lo,hi=hi,path=pool[b],stored=stored))
  end
 end
 @assert length(cases)==168
 rows=Vector{Any}(undef,length(cases))
 Threads.@threads for i in eachindex(cases)
  rows[i]=isolated_contrast_case(cases[i],ph,shallow,mat,site,es,nc,vol)
 end
 mkpath(out);csv_rows(joinpath(out,"finite_contrast_vs_local_ad.csv"),rows)
 open(io->TOML.print(io,Dict("cases"=>length(cases),"historical_cases"=>count(r->r.population=="historical",rows),"new_cases"=>count(r->r.population=="new",rows),"lowfraction_control_cases"=>count(r->r.population=="lowfraction_control",rows),"backend"=>string(get_adjoint_derivative_mode()),"all_stored_ad_bitwise_equal"=>all(r->r.stored_ad_bitwise_equal,rows),"quadrature_nodes"=>32,"artifact_sha256"=>bytes2hex(sha256(read(artifact))),"scope"=>"Same stored cases and finite windows, same current production correction, local AD versus deterministic32node finite contrast. No MC/refitting/reselection. Finite contrast differences do not bound stochastic derivatives.")),joinpath(out,"status.toml"),"w")
end
contrast_audit()
