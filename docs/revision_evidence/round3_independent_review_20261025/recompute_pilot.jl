include(joinpath(@__DIR__,"..","..","..","dev","validate_surrogate_v2.jl"))
function pilot_review3()
 length(ARGS)==3 || error("usage: RAW_FOLDER DUMP FRESH_OUTPUT")
 raw,dump,out=ARGS;ispath(out)&&error("fresh output required")
 plan=TOML.parsefile(joinpath(raw,"preregistered_plan.toml"));ph,mat,site,em,vol,shallow,es=v2state(dump);nc=num_cells(vol)
 install_dry(joinpath(@__DIR__,"..","corrected_validation","calibration.toml"))
 paths=precompute_paths(vol,em,plan["test_zenith"],plan["test_azimuth"])
 control=TOML.parsefile(joinpath(@__DIR__,"..","surrogate_v2","production","material_calibration.toml"))
 candidate=TOML.parsefile(joinpath(raw,"candidate_frozen_before_validation.toml"))["material"]
 flux=readrows(joinpath(raw,"heldout_flux.csv"));derivative=readrows(joinpath(raw,"heldout_derivatives.csv"));adrows=readrows(joinpath(raw,"ad_fd_steps.csv"));results=Dict{String,Any}[]
 for (b,k) in ((1,1),(12,4))
  path=paths[b]
  w=k==1 ? fill(.04,nc) : [vol.centroids[3,c]>650 && hypot(vol.centroids[1,c]-210,vol.centroids[2,c]+170)<1370 ? .14 : .01 for c in 1:nc]
  cells=sort(unique([s.cell_idx for s in path.segments]));c=cells[cld(length(cells),2)]
  wl=copy(w);wh=copy(w);wl[c]=max(0.,w[c]-.01);wh[c]=w[c]+.01
  seed=plan["test_seed"]+100000b+1000k+10;energies=sample_energy_set(plan["nmc"],1e-3,1e9,seed)
  reference_fluxrow=only(r for r in flux if r["bin"]==b && r["field"]==k && r["replicate"]==1 && r["arm"]=="control")
  reference_derow=only(r for r in derivative if r["bin"]==b && r["field"]==k && r["replicate"]==1 && r["arm"]=="control")
  replay=Dict{String,Float64}()
  for (name,field,saved) in (("dry",zeros(nc),reference_fluxrow["mc_dry"]),("wet",w,reference_fluxrow["mc"]),("lo",wl,reference_derow["mc_lo"]),("hi",wh,reference_derow["mc_hi"]))
   mats,dens=build_cell_properties_for_mc(shallow,field,mat)
   value=compute_directional_flux_mc(ph,mat,site,path,mats,dens,energies,seed+1;straggling=true,scattering=false,energy_threshold_low=100.)[1]
   @assert isequal(value,Float64(saved)) "MC serial replay mismatch bin=$b field=$k state=$name replay=$value saved=$saved"
   replay[name]=value
  end
  push!(results,Dict("bin"=>b,"field"=>k,"replicate"=>1,"seed"=>seed,"mc_replay"=>replay,"exact_match"=>true,"histories"=>4*plan["nmc"]))
  for (arm,coef) in (("control",control),("candidate",candidate))
   set_csda_material_correction!(linear=coef["linear"],quadratic=coef["quadratic"])
   row=only(r for r in flux if r["bin"]==b && r["field"]==k && r["replicate"]==1 && r["arm"]==arm)
   drow=only(r for r in derivative if r["bin"]==b && r["field"]==k && r["replicate"]==1 && r["arm"]==arm)
   f=compute_directional_flux_csda(ph,shallow,mat,site,path,w,es);dry=compute_directional_flux_csda(ph,shallow,mat,site,path,zeros(nc),es)
   _,J=assemble_forward_and_jacobian(ph,shallow,mat,site,[path],w,es;n_cells=nc,threaded=false)
   secant=(compute_directional_flux_csda(ph,shallow,mat,site,path,wh,energies)-compute_directional_flux_csda(ph,shallow,mat,site,path,wl,energies))/(wh[c]-wl[c])
   @assert isapprox(f,Float64(row["csda"]);rtol=1e-12,atol=0)
   @assert isapprox(dry,Float64(row["csda_dry"]);rtol=1e-12,atol=0)
   @assert isapprox(secant,Float64(drow["csda_secant"]);rtol=1e-10,atol=1e-20)
   fderrors=Float64[]
   for h in (1e-4,1e-5,1e-6)
    low=copy(w);high=copy(w);low[c]-=h;high[c]+=h
    fd=(compute_directional_flux_csda(ph,shallow,mat,site,path,high,es)-compute_directional_flux_csda(ph,shallow,mat,site,path,low,es))/(2h)
    ar=only(r for r in adrows if r["bin"]==b && r["field"]==k && r["arm"]==arm && r["step"]==h)
    @assert isapprox(J[1,c],Float64(ar["ad"]);rtol=1e-12,atol=1e-20)
    @assert isapprox(fd,Float64(ar["fd"]);rtol=1e-10,atol=1e-20)
    push!(fderrors,abs(fd-J[1,c])/max(abs(fd),abs(J[1,c]),1e-30))
   end
   push!(results,Dict("bin"=>b,"field"=>k,"arm"=>arm,"replicate"=>1,"cell"=>c,"seed"=>seed,"flux"=>f,"dry_flux"=>dry,"local_ad"=>J[1,c],"matched_energy_secant"=>secant,"ad_fd_errors"=>fderrors))
  end
 end
 mkpath(out);open(io->TOML.print(io,Dict("passed"=>true,"checks"=>results,"scope"=>"Preselected deterministic flux/Jacobian/secant and exact serial MC seed replay; verification only, no independent new replicate or refitting")),joinpath(out,"review.toml"),"w")
end
pilot_review3()
