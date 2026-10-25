using Test, LinearAlgebra
const MAX_WATER_FRACTION=0.6
include(joinpath(@__DIR__,"convergence_round3_sampled_direction.jl"))
@testset "Finite sampled-gradient convex hull and box normals" begin
    G=[1. -1.; 0.4 0.4]
    d,alpha,r,it,change=round3_sampled_direction(G,[.2,.2],[1,2])
    @test d≈[0.,-.4] atol=1e-10
    @test sum(alpha)≈1
    @test minimum(alpha)>=0
    @test r≈.4
    d,alpha,r,it,change=round3_sampled_direction(reshape([2.],1,1),[0.],[1])
    @test d==[0.] && r==0
    d,alpha,r,it,change=round3_sampled_direction(reshape([-2.],1,1),[0.],[1])
    @test d==[2.] && r==2
    d,alpha,r,it,change=round3_sampled_direction(reshape([-2.],1,1),[.6],[1])
    @test d==[0.] && r==0
    d,alpha,r,it,change=round3_sampled_direction(reshape([2.],1,1),[.6],[1])
    @test d==[-2.] && r==2
    d,alpha,r,it,change=round3_sampled_direction([3. -1.; 1. 1.],[.2,.2],[1,2])
    @test d≈[0.,-1.] atol=1e-8
    @test alpha≈[.25,.75] atol=1e-8
    @test round3_simplex_projection([-10.,4.,2.])≈[0.,1.,0.]
end
