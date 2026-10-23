using Test, Random, DiffPumas
using DiffPumas.Physics: get_mixture_table, clear_mixture_cache!, _MIXTURE_TABLE_CACHE,
    _MIXTURE_TABLE_LOCK, _MIXTURE_CACHE_LIMIT

function ephemeral_mixture_owner(p)
    temporary=deepcopy(p)
    get_mixture_table(temporary,MaterialMixture([1,2],[.25,.75]))
    return WeakRef(temporary.tables),objectid(temporary.tables)
end

@testset "Mixture cache identity, mutation and concurrency" begin
    p=create_physics(MUON;n_energies=4,K_min=1.,K_max=8.,materials=[STANDARD_ROCK,WATER],
        settings=PhysicsSettings(dedx_path="/nonexistent"))
    # Constant stopping powers make the integrated range exactly K/S_mix.
    for (i,t) in enumerate(p.tables)
        t.csda_stopping_power .= (i==1 ? 2. : 6.)
        t.mixed_stopping_power .= (i==1 ? 1. : 3.)
    end
    mix=MaterialMixture([1,2],[.25,.75])
    original=get_mixture_table(p,mix)
    @test original.csda_range[end] ≈ 8/5
    @test original.mixed_range[end] ≈ 8/2.5
    @test get_mixture_table(p,MaterialMixture([1,2],[.25,.75])) === original

    # Later changes to both caller-owned vectors cannot corrupt stored keys.
    mix.materials .= [2,1]
    @test get_mixture_table(p,mix).csda_range[end] ≈ 8/3
    mix.fractions .= [.5,.5]
    @test get_mixture_table(p,mix).csda_range[end] ≈ 8/4
    @test get_mixture_table(p,MaterialMixture([1,2],[.25,.75])) === original

    q=deepcopy(p)
    copied=get_mixture_table(q,MaterialMixture([1,2],[.25,.75]))
    @test q.tables !== p.tables
    @test copied !== original
    @test copied.csda_range == original.csda_range

    # Stress concurrent misses, hits and two structurally equal physics owners.
    result=zeros(512)
    Threads.@threads for i in eachindex(result)
        owner=iseven(i) ? p : q
        fraction=(i % 251 + .5)/252
        material=MaterialMixture([1,2],[fraction,1-fraction])
        for _ in 1:16
            result[i]=get_mixture_table(owner,material).csda_range[end]
        end
    end
    expected=[8/(2*((i % 251+.5)/252)+6*(1-(i % 251+.5)/252)) for i in eachindex(result)]
    @test result ≈ expected rtol=1e-14

    # A weak owner mismatch cannot be accepted even if a numeric identity key
    # were reused after collection. Simulate the collision deterministically.
    lock(_MIXTURE_TABLE_LOCK) do
        _MIXTURE_TABLE_CACHE[objectid(p.tables)]=_MIXTURE_TABLE_CACHE[objectid(q.tables)]
    end
    repaired=get_mixture_table(p,MaterialMixture([1,2],[.25,.75]))
    @test repaired !== copied
    @test repaired.csda_range == copied.csda_range

    # Numeric-table mutations require explicit invalidation; other owners remain.
    for t in p.tables; t.csda_stopping_power .*= 2; end
    clear_mixture_cache!(p)
    refreshed=get_mixture_table(p,MaterialMixture([1,2],[.25,.75]))
    @test refreshed !== repaired
    @test refreshed.csda_range[end] ≈ 8/10
    @test get_mixture_table(q,MaterialMixture([1,2],[.25,.75])) === copied

    # Live owners have bounded storage; eviction cannot change a later result.
    for i in 1:_MIXTURE_CACHE_LIMIT+20
        fraction=i/(_MIXTURE_CACHE_LIMIT+21)
        get_mixture_table(p,MaterialMixture([1,2],[fraction,1-fraction]))
    end
    count=lock(_MIXTURE_TABLE_LOCK) do
        length(_MIXTURE_TABLE_CACHE[objectid(p.tables)].tables)
    end
    @test count<=_MIXTURE_CACHE_LIMIT
    @test get_mixture_table(p,MaterialMixture([1,2],[.25,.75])).csda_range[end] ≈ 8/10

    weak,identity=ephemeral_mixture_owner(p)
    GC.gc()
    @test weak.value === nothing
    get_mixture_table(p,MaterialMixture([1,2],[.25,.75]))
    @test lock(_MIXTURE_TABLE_LOCK) do
        !haskey(_MIXTURE_TABLE_CACHE,identity)
    end
end
