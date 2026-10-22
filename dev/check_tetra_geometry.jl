#!/usr/bin/env julia
# Real-topography geometry smoke, independent of transport physics tables.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
using TOML
out=length(ARGS)==1 ? ARGS[1] : joinpath(@__DIR__,"..","examples","data","tetra_geometry_verification_20261023")
mkpath(out)
emap=LVDTopo.build_elevation_map()
mat=MaterialConfig(1,2,3,4,2710.0,1000.0,0.0,0.0)
x,y,z=build_surface_grid(emap,1.0,0.5)
measurement=@timed create_tetra_volume(emap,mat;half_width_km=1.0,surface_step_km=0.5,max_cell_volume_m3=5e8)
volume=measurement.value
@assert maximum(volume.mesh.tetrahedronlist)<=size(volume.mesh.pointlist,2)
@assert minimum(volume.mesh.tetrahedronlist)>=1
paths=[trace_path(volume,theta,phi,emap) for theta in (0.0,15.0,30.0), phi in (0.0,90.0,180.0,270.0)]
open(joinpath(out,"paths.csv"),"w") do io
    println(io,"zenith_deg,azimuth_deg,valid,segments,distance_in_volume_m,surface_distance_m,remaining_rock_m")
    for p in paths
        println(io,join((p.zenith,p.azimuth,p.valid,length(p.segments),p.distance_in_volume,p.surface_distance,p.remaining_rock_distance),','))
    end
end
@assert all(p->p.valid && !isempty(p.segments) && p.distance_in_volume>0 && p.surface_distance>0,paths)
_,_,peripheral_surface=build_surface_grid(emap,5.0,1.0)
@assert any(z->z<=MIN_MESH_OVERBURDEN_M,peripheral_surface)
autoargs=(geometry="auto",mesh_half_km=5.0,surface_step_km=1.0,grid_nz=2,tet_max_volume=5e8)
autovolume=create_sensitivity_volume(emap,mat,autoargs)
@assert autovolume isa VoxelVolume
open(joinpath(out,"verification.toml"),"w") do io
    TOML.print(io,Dict("julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),"half_width_km"=>1.0,
        "surface_step_km"=>0.5,"max_tetra_volume_m3"=>5e8,"points"=>size(volume.mesh.pointlist,2),
        "tetrahedra"=>num_cells(volume),"valid_test_paths"=>length(paths),"surface_min_m"=>minimum(z),
        "surface_floor_nodes"=>count(<=(MIN_MESH_OVERBURDEN_M),z),
        "auto_5km_floor_nodes"=>count(<=(MIN_MESH_OVERBURDEN_M),peripheral_surface),
        "auto_5km_geometry"=>geometry_name(autovolume),"geometry_seconds"=>measurement.time,
        "allocated_bytes"=>measurement.bytes,"process_peak_rss_bytes"=>Sys.maxrss(),
        "scope"=>"central1km real-topography tetrahedral geometry and12traced paths; no transport/inverse claim"))
end
println("Tetrahedral geometry smoke passed: $(num_cells(volume)) cells, $(length(paths)) valid paths")
