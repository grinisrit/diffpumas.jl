# Standalone: julia --project=. test/test_revision_suite.jl
using Test
include(joinpath(@__DIR__, "..", "examples", "revision_suite.jl"))
@testset "TetGen Julia connectivity" begin
    mesh=RawTetGenIO{Float64}()
    mesh.pointlist=[0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    mesh.tetrahedronlist=reshape(Cint[1,2,3,4],4,1)
    @test tetra_centroids(mesh) ≈ fill(0.25,3,1)
    @test build_tetra_neighbors(mesh)==zeros(Int,1,4)
    volume=TetraVolume(mesh,zeros(Int,1,4),tetra_centroids(mesh),falses(1),1.0,1.0)
    face,distance,hit=nearest_tetra_face_intersection(volume,1,Point(0.1,0.1,0.1),(0.0,0.0,1.0))
    @test face>0
    @test distance≈0.7
    @test hit.z≈0.8
    triangle=Triangle(Point(0.0,0.0,1.0),Point(1.0,0.0,1.0),Point(0.0,1.0,1.0))
    boundary=tolerant_tetra_face_hit(Point(-1e-13,0.2,0.0),(0.0,0.0,1.0),triangle)
    @test boundary !== nothing && boundary[1]≈1.0
    @test tolerant_tetra_face_hit(Point(-0.01,0.2,0.0),(0.0,0.0,1.0),triangle) === nothing
    @test point_in_tetrahedron(Point(0.1,0.1,0.1),Int.(mesh.tetrahedronlist[:,1]),mesh.pointlist)
    mesh.tetrahedronlist[4,1]=5
    @test_throws ArgumentError tetra_centroids(mesh)
end
@testset "Adjoint finite difference controls" begin
    original=get_adjoint_fd_step()
    try
        @test set_adjoint_fd_step!(1e-5)==1e-5
        @test get_adjoint_fd_step()==1e-5
        for invalid in (0.0,-1.0,Inf,NaN,0.1)
            @test_throws ArgumentError set_adjoint_fd_step!(invalid)
        end
    finally
        set_adjoint_fd_step!(original)
    end
end
@testset "Revision option bounds" begin
    @test_throws ErrorException revision_options(["--samples","0"])
    @test_throws ErrorException revision_options(["--rays-step","0"])
    @test_throws ErrorException revision_options(["--unknown","1"])
end

@testset "Measured robustness controls" begin
    saved=copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS,["--papermatch-fit-only","--papermatch-prior-cutoff","350",
            "--papermatch-mask-ratio","Inf","--papermatch-calibration-water","0.3"])
        opts=parse_commandline()
        @test opts.papermatch && opts.papermatch_only && opts.papermatch_fit_only
        @test opts.papermatch_prior_cutoff==350
        @test isinf(opts.papermatch_mask_ratio)
        @test opts.papermatch_calibration_water==0.3
        @test opts.csda_quadrature=="midpoint"
        @test opts.n_samples==32
    finally
        empty!(ARGS);append!(ARGS,saved)
    end
end

@testset "Masked forward/Jacobian consistency" begin
    toy=w->([w[1]+2w[2]],sparse(reshape([1.0,2.0],1,2)))
    restricted=masked_material_operator(toy,Bool[true,false])
    w=[0.2,0.8]
    prediction,J=restricted(w)
    @test prediction≈[0.2]
    @test Matrix(J)≈[1.0 0.0]
    for c in eachindex(w)
        h=1e-6;delta=zeros(2);delta[c]=h
        fd=(first(restricted(w+delta))-first(restricted(w-delta)))/(2h)
        @test fd≈Vector(J[:,c]) atol=1e-10
    end
end

@testset "Validation angular configuration" begin
    @test validation_zeniths(revision_options(String[])) == [0,15,30,45,60,75]
    @test validation_zeniths(revision_options(["--validation-zeniths","75,0,60"])) == [0,60,75]
    for invalid in ("0,90","0,NaN","0,0")
        @test_throws ErrorException revision_options(["--validation-zeniths",invalid])
    end
end

@testset "High-zenith validation geometry" begin
    emap=LVDTopo.build_elevation_map()
    mat=MaterialConfig(1,2,3,4,2710.0,1000.0,0.0,0.0)
    vol=create_voxel_volume(emap,mat;half_width_km=3.0,surface_step_km=1.5,nz=8)
    paths=precompute_paths(vol,emap,[60.0,75.0],[0.0,90.0,180.0,270.0])
    @test length(paths)==8
    @test all(p->p.valid && !isempty(p.segments),paths)
    @test all(p->isfinite(p.surface_distance) && p.surface_distance>0,paths)
    @test all(p->p.distance_in_volume>0 && p.remaining_rock_distance>=0,paths)
end

@testset "Seeded selected/population zenith coverage" begin
    paths=[(zenith=t,surface_distance=1000.0+10b) for (b,t) in enumerate([0,15,30,45,60,75])]
    volume=(centroids=[zeros(1,8);zeros(1,8);reshape(1.0:8.0,1,8)],)
    J=sparse([10.0b+c for b in 1:6,c in 1:8])
    one=validation_pairs(paths,volume,J,12,MersenneTwister(42),[0,15,30,45,60,75])
    two=validation_pairs(paths,volume,J,12,MersenneTwister(42),[0,15,30,45,60,75])
    @test one==two
    @test length(unique(one))==12
    @test Set(paths[p.bin].zenith for p in one)==Set([0,15,30,45,60,75])
end
