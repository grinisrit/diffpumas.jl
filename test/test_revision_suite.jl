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
