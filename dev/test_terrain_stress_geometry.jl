# Small geometry regression; no physics-table load or numerical reconstruction.
using Test
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
include(joinpath(@__DIR__,"terrain_stress_geometry.jl"))
using .TerrainStressGeometry
@testset "Localized assumed terrain stress" begin
    # Synthetic planar surface spanning the declared lens: detector at z=0.
    info=MapInfo(61,61,(-3.,3.),(-3.,3.),(0.,2000.))
    emap=map_create(info);emap.data .= 1000.
    volume=create_voxel_volume(emap,MaterialConfig(1,2,3,0,2710.,1000.,0.,0.);
        half_width_km=2.,surface_step_km=1.,nz=4)
    original_data=copy(emap.data);original_grid=copy(volume.centroids)
    zero=stressed_elevation_map(emap,0.,0.)
    plus=stressed_elevation_map(emap,.05,0.)
    minus=stressed_elevation_map(emap,-.05,0.)
    @test reinterpret(UInt64,vec(zero.data))==reinterpret(UInt64,vec(emap.data))
    @test plus.data.+minus.data ≈ 2 .*emap.data
    @test 0 < maximum(plus.data.-emap.data) <= 50.
    @test terrain_lens_weight(sind(157.5)*tand(47.5),cosd(157.5)*tand(47.5),1.) ≈ 1.
    for (theta,azimuth) in ((40.,157.5),(55.,157.5),(47.5,135.),(47.5,180.),(45.,300.))
        x=sind(azimuth)*tand(theta);y=cosd(azimuth)*tand(theta)
        @test terrain_lens_weight(x,y,1.) ≈ 0. atol=1e-28
    end
    # Example ray tracer subtracts its physical detector elevation.
    for map in (emap,zero,plus,minus);map.data .+= LVDTopo.DETECTOR_ELEVATION;end
    p=trace_path(volume,47.5,157.5,emap)
    p0=trace_path(volume,47.5,157.5,zero)
    pp=trace_path(volume,47.5,157.5,plus)
    pm=trace_path(volume,47.5,157.5,minus)
    @test isequal(path_snapshot(p),path_snapshot(p0))
    @test pp.surface_distance > p.surface_distance > pm.surface_distance
    for path in (p,p0,pp,pm)
        @test path.valid
        @test all(s->s.distance>0,path.segments)
        @test path.surface_exit_z ≈ path.surface_distance*cosd(path.zenith)
        @test path.surface_distance ≈ path.distance_in_volume+path.remaining_rock_distance atol=1e-9
    end
    @test volume.centroids==original_grid
    @test emap.data.-LVDTopo.DETECTOR_ELEVATION==original_data
    outside=trace_path(volume,47.5,300.,emap)
    @test isequal(path_snapshot(outside),path_snapshot(trace_path(volume,47.5,300.,plus)))
    @test_throws ArgumentError stressed_elevation_map(emap,.051,0.)
end
