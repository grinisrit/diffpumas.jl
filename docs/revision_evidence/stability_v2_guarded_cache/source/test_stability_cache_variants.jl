# Run in a separate, bounded Julia process with --threads=2; no production edits.
module GuardedCacheTests
include(joinpath(@__DIR__,"test_stability_guarded_ray_cache.jl"))
end
module AcceptedGNTests
include(joinpath(@__DIR__,"test_stability_gn_cache_replay.jl"))
end
