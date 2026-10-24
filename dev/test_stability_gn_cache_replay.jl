using Test, DiffPumas
include(joinpath(@__DIR__,"prototype_stability_ray_cache.jl"))
include(joinpath(@__DIR__,"prototype_stability_gn_replay.jl"))
using .StabilityRayCachePrototype
@testset "accepted-step exact GN cache replay" begin
    paths=[(segments=[(cell_idx=1,)],),(segments=[(cell_idx=2,)],),(segments=NamedTuple[],)]
    raw=w->[exp(w[1]),exp(2w[2]),2.]
    model=w->(raw(w),[exp(w[1]) 0.;0. 2exp(2w[2]);0. 0.])
    cache=RayCache(paths,:fixed)
    cached=w->first(evaluate!(cache,w,(i,x)->i==1 ? exp(x[1]) : i==2 ? exp(2x[2]) : 2.))
    outputs=Any[];traces=Any[]
    for predict in (raw,cached)
        trace=Any[];diag=Any[]
        result=DiffPumas.Tomography.replay_cache_gauss_newton_reconstruct(model,[1.2,1.6,2.];
            w0=[0.,0.],n_iter=4,box=(0.,1.),solver_mode=:active_set,
            predict_only=predict,trial_trace=trace,diagnostics=diag)
        push!(outputs,(result,diag));push!(traces,trace)
    end
    @test isequal(outputs[1],outputs[2])
    @test isequal(traces[1],traces[2])
    @test length(outputs[1][1][2])>1 # accepted updates, unlike only a plateau fixture
    @test all(outputs[1][1][1].>0.)
end
