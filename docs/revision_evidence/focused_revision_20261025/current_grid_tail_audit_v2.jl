include(joinpath(@__DIR__,"..","..","..","examples","lvd_tomography.jl"))
using DelimitedFiles
function tailrows(path)
 a,h=readdlm(path,',',Any;header=true);[Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function focused_fd_from_values(f0,fa,fb,h,stencil)
 stencil=="central" && return (fb-fa)/(2*h)
 stencil=="forward2" && return (-3*f0+4*fa-fb)/(2*h)
 stencil=="backward2" && return (3*f0-4*fa+fb)/(2*h)
 error("unknown stencil")
end
function isolated_tail_case(s,ph,shallow,cfg,site,paths,energies,w,nc,vol)
  b=Int(s["bin"]);c=Int(s["cell"]);path=paths[b]
  p,J=assemble_forward_and_jacobian(ph,shallow,cfg,site,[path],w,energies;n_cells=nc,threaded=false);f0=p[1];ad=J[1,c]
  repeat=compute_directional_flux_csda(ph,shallow,cfg,site,path,w,energies);floor=max(10abs(repeat-f0),100eps(abs(f0)))
  stored=Float64(s["stored_ad"]);storedok=isapprox(ad,stored;rtol=1e-10,atol=1e-25);localrows=NamedTuple[]
  for h in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-7)
   a=copy(w);d=copy(w)
   if w[c]-h>=0 && w[c]+h<=MAX_WATER_FRACTION
    a[c]-=h;d[c]+=h;fa=compute_directional_flux_csda(ph,shallow,cfg,site,path,a,energies);fd=compute_directional_flux_csda(ph,shallow,cfg,site,path,d,energies)
    stencil="central";value=focused_fd_from_values(f0,fa,fd,h,"central")
   elseif w[c]+2h<=MAX_WATER_FRACTION
    a[c]+=h;d[c]+=2h;fa=compute_directional_flux_csda(ph,shallow,cfg,site,path,a,energies);fd=compute_directional_flux_csda(ph,shallow,cfg,site,path,d,energies)
    stencil="forward2";value=focused_fd_from_values(f0,fa,fd,h,"forward2")
   else
    @assert w[c]-2h>=0;a[c]-=h;d[c]-=2h;fa=compute_directional_flux_csda(ph,shallow,cfg,site,path,a,energies);fd=compute_directional_flux_csda(ph,shallow,cfg,site,path,d,energies)
    stencil="backward2";value=focused_fd_from_values(f0,fa,fd,h,"backward2")
   end
   zero=string(s["band"])=="structural_zero";zerobits=isequal(fa,f0)&&isequal(fd,f0)&&ad==0
   resolved=max(abs(fa-f0),abs(fd-f0))>floor;absolute=abs(value-ad);relative=absolute/max(abs(value),abs(ad),1e-300)
   push!(localrows,(case=Int(s["case"]),band=string(s["band"]),bin=b,cell=c,zenith=path.zenith,slant_m=DiffPumas.Tomography.path_rock_length(path),cell_z_m=cell_centroid(vol,c)[3],local_fraction=w[c],step=h,stencil=stencil,flux=f0,repeat_flux=repeat,numerical_floor=floor,ad=ad,stored_ad=stored,stored_ad_consistent=storedok,fd=value,signed_error=value-ad,absolute_error=absolute,relative_error=relative,finite_change_resolved=resolved,relative_error_interpretable=resolved&&!zero,expected_structural_zero=zero,zero_perturbations_bitwise_equal=zerobits,flux_a=fa,flux_b=fd))
  end
  return localrows
end
function tail_audit()
 raw=copy(ARGS);length(raw)>=5||error("usage DUMP ARTIFACT BASELINE SELECTION OUTPUT [grid options]")
 dump,artifact,baseline,selection,out=raw[1:5];empty!(ARGS);append!(ARGS,raw[6:end]);args=parse_commandline();ispath(out)&&error("fresh output required")
 ph=load_tomography_physics(dump);idx(n)=get_material_index(ph,n);porous=idx("PorousWetRock")
 cfg=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,2710.,Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,0.)
 set_adjoint_derivative_mode!(:forwarddiff);load_csda_corrections!(artifact)
 em=LVDTopo.build_elevation_map();vol=create_sensitivity_volume(em,cfg,args);nc=num_cells(vol);@assert nc==600
 fields=tailrows(joinpath(baseline,"lvd_fit_field.csv"));w=Float64[r["w"] for r in fields]
 for c in 1:nc;@assert Tuple(Float64(fields[c][k]) for k in ("x_m","y_m","z_m"))==cell_centroid(vol,c);end
 shallow=Bool[is_shallow_cell(vol,c) for c in 1:nc];site=build_site(30000.)
 bins=load_measured_intensity(args.match_data);preds=tailrows(joinpath(baseline,"lvd_fit_predictions.csv"));covered=Int[r["bin_idx"] for r in preds]
 zen=Float64[(b.zen_lo+b.zen_hi)/2 for b in bins];az=Float64[b.az_hi-b.az_lo>=359.999 ? 0. : (b.az_hi+b.az_lo)/2 for b in bins]
 paths=Dict(b=>trace_path(vol,zen[b],az[b],em) for b in covered);energies=csda_energy_quadrature(32,1e-3,1e9)
 cases=tailrows(selection);@assert length(cases)==48
 # This geometric preflight occurs before any forward/Jacobian/FD evaluation.
 for s in cases
  if string(s["band"])=="structural_zero"
   @assert all(segment.cell_idx!=Int(s["cell"]) for segment in paths[Int(s["bin"])].segments) "selected zero not geometrically disjoint"
  end
 end
 mkpath(out);cp(selection,joinpath(out,"selection_before_evaluation.csv"))
 output=Vector{Any}(undef,48)
 Threads.@threads for i in eachindex(cases)
  output[i]=isolated_tail_case(cases[i],ph,shallow,cfg,site,paths,energies,w,nc,vol)
 end
 rows=vcat(output...)
 open(joinpath(out,"tail_fd_checks.csv"),"w") do io
  println(io,join(propertynames(first(rows)),','));for r in rows;println(io,join(values(r),','));end
 end
 allstored=all(r->r.stored_ad_consistent,rows);allzero=all(r->!r.expected_structural_zero||r.zero_perturbations_bitwise_equal,rows)
 open(io->TOML.print(io,Dict("cases"=>48,"nonzero_cases"=>36,"structural_zero_cases"=>12,"steps_per_case"=>6,"all_stored_ad_consistent"=>allstored,"all_zero_perturbations_bitwise_equal"=>allzero,"zero_geometry_disjoint_preflight"=>true,"backend"=>string(get_adjoint_derivative_mode()),"scope"=>"Prespecified current-grid geometric/sensitivity-tail deterministic checks; finite-step/noise-aware evidence, noMC or domain-wide derivative certificate.")),joinpath(out,"status.toml"),"w")
 allstored&&allzero||error("stored AD or exact structural-zero audit failed; complete rows retained")
end
if abspath(PROGRAM_FILE)==@__FILE__;tail_audit();end
