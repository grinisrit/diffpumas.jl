using Test
isdefined(Main,:StabilityRayCachePrototype) || include(joinpath(@__DIR__,"prototype_stability_ray_cache.jl"))
using .StabilityRayCachePrototype
@testset "exact bounded row cache" begin
    paths=[(segments=[(cell_idx=1,),(cell_idx=1,)],),
           (segments=[(cell_idx=2,)],),(segments=NamedTuple[],)]
    cache=RayCache(paths,:first)
    calls=zeros(Int,3)
    forward=(i,w)->begin
        calls[i]+=1
        i==1 ? copysign(1.,w[1]) : i==2 ? w[2]^2 : 7.
    end
    w=[0.,2.]
    a,h=evaluate!(cache,w,forward)
    @test h==0 && calls==[1,1,1]
    b,h=evaluate!(cache,w,forward)
    @test h==3 && isequal(a,b)
    b[1]=99.; @test cache.values[1]==1. # caller cannot mutate cache
    w[1]=-0.; b,h=evaluate!(cache,w,forward)
    @test h==2 && b[1]==-1. && calls==[2,1,1]
    w[2]=3.; b,h=evaluate!(cache,w,forward)
    @test h==2 && b[2]==9. && calls==[2,2,1]
    b,h=evaluate!(cache,w,forward;context=:second)
    @test h==0 && calls==[3,3,2]
    @test sum(length,cache.snapshots)==2 # repeated path cells deduplicated
end
