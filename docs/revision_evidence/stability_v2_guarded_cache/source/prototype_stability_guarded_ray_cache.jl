# Unreferenced successor design; no production imports.
module GuardedStabilityRayCachePrototype
export GuardedRayCache, evaluate!
mutable struct GuardedRayCache{F}
    cells::Vector{Vector{Int}}
    snapshots::Vector{Vector{UInt64}}
    values::Vector{Float64}
    valid::Vector{Bool} # NOT packed: separate threaded row writes are independent
    guard::F
    context_snapshot::Any
    lock::ReentrantLock
end
"""Construct a bounded exact last-value cache.
`guard()` must return actual value snapshots of all mutable context (correction
coefficients, material/site configuration, shallow flags, energy nodes, paths).
It must not return only paths or hashes in place of actual values. Immutable
physics tables may be bound to an externally audited identity only when their
owner guarantees no table mutation for this closure's lifetime. The same
no-concurrent-external-mutation rule as the underlying forward model applies.
The guard snapshot is deep-copied, so later mutation cannot mutate both sides.
The whole call lock serializes competing calls; rows inside a call can parallelize.
"""
function GuardedRayCache(paths,guard)
    cells=[unique(Int[s.cell_idx for s in p.segments]) for p in paths]
    GuardedRayCache(cells,[zeros(UInt64,length(c)) for c in cells],zeros(length(cells)),
        fill(false,length(cells)),guard,deepcopy(guard()),ReentrantLock())
end
function evaluate!(cache::GuardedRayCache,w::Vector{Float64},forward;threaded=false)
    lock(cache.lock) do
        context=cache.guard()
        if !isequal(context,cache.context_snapshot)
            # Any context change, especially geometry incidence, requires a fresh
            # instance; failing closed avoids silently reusing a stale cell list.
            throw(ArgumentError("physical context changed; rebuild ray cache"))
        end
        out=similar(cache.values); hit=fill(false,length(out))
        function row!(i)
            cells=cache.cells[i];bits=cache.snapshots[i]
            reuse=cache.valid[i] && all(k->bits[k]==reinterpret(UInt64,w[cells[k]]),eachindex(cells))
            if !reuse
                cache.valid[i]=false
                value=forward(i,w)
                for k in eachindex(cells)
                    bits[k]=reinterpret(UInt64,w[cells[k]])
                end
                cache.values[i]=value
                cache.valid[i]=true
            end
            out[i]=cache.values[i];hit[i]=reuse
        end
        if threaded
            Threads.@threads for i in eachindex(out)
                row!(i)
            end
        else
            for i in eachindex(out)
                row!(i)
            end
        end
        out,count(hit)
    end
end
end
