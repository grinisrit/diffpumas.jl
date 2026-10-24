using Test
include(joinpath(@__DIR__,"prototype_stability_guarded_ray_cache.jl"))
using .GuardedStabilityRayCachePrototype
@testset "guarded exact ray cache" begin
    paths=[(segments=[(cell_idx=1,)],),(segments=[(cell_idx=2,)],),(segments=NamedTuple[],)]
    coefficient=Ref(1.);nodes=[1.,2.]
    context=()->(coefficient=coefficient[],nodes=nodes,paths=paths)
    cache=GuardedRayCache(paths,context)
    forward=(i,w)->i==1 ? copysign(coefficient[],w[1]) : i==2 ? w[2]*nodes[1] : 7.
    a,h=evaluate!(cache,[0.,2.],forward;threaded=true)
    @test h==0 && a==[1.,2.,7.]
    b,h=evaluate!(cache,[-0.,2.],forward;threaded=true)
    @test h==2 && b==[-1.,2.,7.]
    # Concurrent callers serialize; each receives independent correct output.
    tasks=[Threads.@spawn(evaluate!(cache,[Float64(i),Float64(i)],forward;threaded=true)) for i in 1:8]
    for (i,t) in enumerate(tasks)
        p,_=fetch(t);@test p==[1.,Float64(i),7.]
    end
    nodes[1]=2.
    @test_throws ArgumentError evaluate!(cache,[0.,2.],forward)
    nodes[1]=1.; coefficient[]=2.
    @test_throws ArgumentError evaluate!(cache,[0.,2.],forward)
    coefficient[]=1.;push!(paths[1].segments,(cell_idx=2,))
    @test_throws ArgumentError evaluate!(cache,[0.,2.],forward)
end
