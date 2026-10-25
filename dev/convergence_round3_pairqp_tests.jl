using Test, LinearAlgebra
const MAX_WATER_FRACTION=0.9
include(joinpath(@__DIR__,"convergence_round3_pairqp.jl"))
@testset "Exact pair simplex transfers and dual gap" begin
    for (G,w,expected) in (([1. -1.; .4 .4],[.2,.2],[0.,-.4]),
                           ([3. -1.; 1. 1.],[.2,.2],[0.,-1.]),
                           ([2. 3.; -1. -1.],[0.,.2],[0.,1.]))
        d,a,r,s,gap,ok=round3_pairwise_direction(G,w,[1,2])
        @test d≈expected atol=1e-10
        @test ok && gap<=1e-12
        @test minimum(a)>=0 && sum(a)≈1
    end
    G=[1. -1. 1.; 1. 1. -1.]
    d,a,r,s,gap,ok=round3_pairwise_direction(G,[.2,.2],[1,2])
    @test norm(d)<1e-6
    @test gap<=1e-12 && ok
    for (g,w) in ((2.,0.),(-2.,.9))
        d,a,r,s,gap,ok=round3_pairwise_direction(reshape([g],1,1),[w],[1])
        @test d==[0.] && gap==0 && ok
    end
end
if !isempty(ARGS)
    using TOML, DelimitedFiles
    state=TOML.parsefile(ARGS[1]);rows,header=readdlm(ARGS[2],',',Any;header=true)
    col=findfirst(==("aquifer_mask"),vec(header));active=findall(i->string(rows[i,col])=="true",axes(rows,1))
    results=Dict{String,Any}[]
    @testset "Actual saved sampled-gradient convex subproblems" begin
        for sample in state["sample_records"]
            G=hcat(sample["gradients"]...);w=Float64.(sample["field_before"])
            d,a,r,s,gap,ok=round3_pairwise_direction(G,w,active)
            oldpsi=sample["sampled_residual_l2"]^2/2
            @test ok && gap<=1e-12
            @test r^2/2<=oldpsi+1e-15
            @test minimum(a)>=-1e-12 && abs(sum(a)-1)<1e-10
            @test norm(d[.!([i in active for i in eachindex(w)])])==0
            push!(results,Dict("radius"=>sample["radius"],"alpha"=>a,"direction"=>d,
                "residual_l2"=>r,"objective"=>r^2/2,"original_objective"=>oldpsi,
                "frank_wolfe_gap"=>gap,"converged"=>ok,"sweeps"=>s))
        end
    end
    open(ARGS[3],"w") do io;TOML.print(io,Dict("results"=>results));end
end
