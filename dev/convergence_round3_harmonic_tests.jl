using Test,LinearAlgebra
const MAX_WATER_FRACTION=.9
include(joinpath(@__DIR__,"convergence_round3_harmonic.jl"))
@testset "Harmonic reduction and explicit reduced derivative" begin
    L=[1. -1. 0.;-1. 2. -1.;0. -1. 1.]
    ext,U,C,records,T=round3_harmonic_map(L,trues(3),Bool[1,0,1],[.2,.8,.4])
    x=ext([.2,.8,.4])
    @test x≈[.2,.3,.4]
    @test T≈[1. 0.; .5 .5;0. 1.]
    f(x)=.5dot(x,L*x)+.5(x[1]-.1)^2+.5(x[3]-.8)^2
    g=L*x+[x[1]-.1,0.,x[3]-.8];reduced=transpose(T)*g
    for j in eachindex(C)
        step=zeros(3);step[C[j]]=1e-6
        fd=(f(ext(x.+step))-f(ext(x.-step)))/(2e-6)
        @test fd≈reduced[j] atol=1e-9
    end
    @test norm((L*x)[U],Inf)<1e-15
    L=[0. 0. 0.;0. 1. -1.;0. -1. 1.]
    ext,U,C,records,T=round3_harmonic_map(L,trues(3),Bool[1,0,0],[.1,.2,.4])
    @test ext([.1,.2,.4])≈[.1,.3,.3]
    @test T≈reshape([1.,0.,0.],3,1)
    @test !records[1]["anchored"]
end
