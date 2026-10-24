# Isolated, unreferenced prototype. Cache lifetime belongs to one immutable
# physics/material/site/path/quadrature/correction context. No shared mutation
# during a call; each threaded row owns its separate snapshot and cached scalar.
module StabilityRayCachePrototype
export RayCache, evaluate!
mutable struct RayCache
    cells::Vector{Vector{Int}}
    snapshots::Vector{Vector{UInt64}}
    values::Vector{Float64}
    valid::BitVector
    context::Any
end
function RayCache(paths, context)
    cells=[unique(Int[s.cell_idx for s in p.segments]) for p in paths]
    RayCache(cells,[zeros(UInt64,length(c)) for c in cells],zeros(length(cells)),falses(length(cells)),context)
end
"""Exact per-row last-value reuse; context token must change with any physical context.
`forward(i,w)` must be deterministic and depend on w only through cells[i].
Calls are serial here to avoid BitVector write races and produce clean timings.
Concurrent evaluations of the same cache are forbidden. Returned vector is owned.
"""
function evaluate!(cache::RayCache, w::Vector{Float64}, forward; context=cache.context)
    isequal(context,cache.context) || (fill!(cache.valid,false);cache.context=context)
    out=similar(cache.values); hits=0
    for i in eachindex(cache.cells)
        cells=cache.cells[i]; bits=cache.snapshots[i]
        hit=cache.valid[i] && all(k->bits[k]==reinterpret(UInt64,w[cells[k]]),eachindex(cells))
        if hit
            hits+=1
        else
            cache.values[i]=forward(i,w)
            for k in eachindex(cells)
                bits[k]=reinterpret(UInt64,w[cells[k]])
            end
            cache.valid[i]=true
        end
        out[i]=cache.values[i]
    end
    out,hits
end
end
