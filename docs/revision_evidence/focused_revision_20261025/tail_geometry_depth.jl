# Geometry-only metadata; frozen sensitivity selection and numerical outputs untouched.
include(joinpath(@__DIR__,"..","..","..","examples","lvd_tomography.jl"))
using DelimitedFiles
function georows(path)
 a,h=readdlm(path,',',Any;header=true);[Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function traced_midpoints(volume,theta,phi,emap)
 dir=direction_vector(theta,phi);surface,ok,_=ray_surface_exit_distance(emap,dir);@assert ok
 x=dir[1]*POINT_EPS;y=dir[2]*POINT_EPS;z=dir[3]*POINT_EPS;traveled=POINT_EPS;rows=NamedTuple[]
 while traveled<surface-POINT_EPS
  ijk=voxel_cell_indices(volume,x,y,z);ijk===nothing&&break;i,j,k=ijk;c=voxel_linear_index(volume,i,j,k)
  distance=distance_to_voxel_boundary(volume,x,y,z,dir[1],dir[2],dir[3]);(!isfinite(distance)||distance<=0)&&break
  remaining=surface-traveled;step=min(distance,remaining)
  if step>1e-8
   mx=x+.5*step*dir[1];my=y+.5*step*dir[2];mz=z+.5*step*dir[3];surf=local_surface_height(emap,mx,my)
   push!(rows,(cell=c,distance=step,midpoint_x=mx,midpoint_y=my,midpoint_z=mz,midpoint_surface_height=surf,midpoint_depth=surf-mz));traveled+=step
  end
  distance>=remaining-1e-6&&break;x+=dir[1]*(step+POINT_EPS);y+=dir[2]*(step+POINT_EPS);z+=dir[3]*(step+POINT_EPS)
 end
 original=trace_path(volume,theta,phi,emap);@assert [(r.cell,r.distance) for r in rows]==[(s.cell_idx,s.distance) for s in original.segments]
 rows
end
function geometry_main()
 raw=copy(ARGS);selection,fieldfile,coverage,out=raw[1:4];empty!(ARGS);append!(ARGS,raw[5:end]);args=parse_commandline();ispath(out)&&error("fresh output")
 fields=georows(fieldfile);cases=georows(selection);cov=Dict(Int(r["bin_idx"])=>r for r in georows(coverage));cfg=MaterialConfig(1,2,3,-1,2710.,1000.,0.,0.);emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,cfg,args)
 @assert num_cells(volume)==600
 for c in 1:600;@assert Tuple(Float64(fields[c][k]) for k in ("x_m","y_m","z_m"))==cell_centroid(volume,c);end
 mkpath(out)
 open(joinpath(out,"selected_geometry_depth.csv"),"w") do io
  println(io,"case,band,bin,cell,cell_height_above_detector_m,cell_centroid_depth_below_local_surface_m,selected_ray_has_segment,ray_segment_distance_m,ray_midpoint_height_above_detector_m,ray_midpoint_depth_below_local_surface_m")
  for s in cases
   b=Int(s["bin"]);c=Int(s["cell"]);theta=Float64(cov[b]["zenith_deg"]);phi=Float64(cov[b]["azimuth_deg"]);segments=traced_midpoints(volume,theta,phi,emap);chosen=filter(r->r.cell==c,segments);x,y,z=cell_centroid(volume,c);centroiddepth=local_surface_height(emap,x,y)-z
   if isempty(chosen)
    @assert string(s["band"])=="structural_zero";println(io,join((s["case"],s["band"],b,c,z,centroiddepth,false,"","",""),','))
   else
    @assert string(s["band"])!="structural_zero"&&length(chosen)==1;r=only(chosen);println(io,join((s["case"],s["band"],b,c,z,centroiddepth,true,r.distance,r.midpoint_z,r.midpoint_depth),','))
   end
  end
 end
 open(io->TOML.print(io,Dict("cases"=>48,"nonzero_cases"=>36,"structural_zero_cases"=>12,"source_tracer_segment_parity"=>true,"scope"=>"Frozen-selection geometry metadata only. Cellheight is above detector; signed centroiddepth can be negative for partial topographic voxels. Actual selected-ray segment midpointdepth is separately reported. NoMC/AD/selection change.")),joinpath(out,"status.toml"),"w")
end
geometry_main()
