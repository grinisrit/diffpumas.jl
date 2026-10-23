# Included in Physics. Keep PhysicsTables' serialized layout compatible with old dumps.
# Registry keys are identities, never structural hashes of mutable table vectors.
const _recipe_lock = ReentrantLock()
const _recipe_registry = Dict{UInt,Any}()
const PHYSICS_ENVELOPE_VERSION = 2

snapshot_recipe(m::BaseMaterial) = (
    name=m.name, density=m.density, I=m.I, ZoA=m.ZoA,
    elements=Tuple((name=e.name, Z=e.Z, A=e.A, I=e.I, fraction=f)
                   for (e,f) in zip(m.elements,m.fractions)))

# Canonical primitive encoding independent of Julia Serialization's wire format.
function _fingerprint_write(io,x)
    if x isa AbstractString
        print(io,"s",ncodeunits(x),":",x)
    elseif x isa AbstractFloat
        print(io,"f",bitstring(Float64(x)),";")
    elseif x isa Integer || x isa Enum
        print(io,"i",Int(x),";")
    elseif x isa AbstractDict
        print(io,"dict",length(x),":")
        for (k,v) in sort!(collect(x);by=first)
            _fingerprint_write(io,k);_fingerprint_write(io,v)
        end
    elseif x isa NamedTuple
        print(io,"named",length(x),":")
        for n in keys(x)
            _fingerprint_write(io,String(n));_fingerprint_write(io,getproperty(x,n))
        end
    elseif x isa Tuple || x isa AbstractArray
        print(io,"seq",length(x),":")
        for v in x;_fingerprint_write(io,v);end
    else
        print(io,"struct",nameof(typeof(x)),":")
        for n in fieldnames(typeof(x))
            _fingerprint_write(io,String(n));_fingerprint_write(io,getfield(x,n))
        end
    end
end
function _canonical_fingerprint(x)
    io=IOBuffer();_fingerprint_write(io,x)
    bytes2hex(sha256(take!(io)))
end
physics_fingerprint(p::PhysicsTables) = _canonical_fingerprint(p)
recipe_fingerprint(recipes) = _canonical_fingerprint(recipes)

function register_material_recipes!(p::PhysicsTables, recipes::Tuple;
                                    fingerprint=physics_fingerprint(p),
                                    table_version=Materials.RADIATIVE_DCS_TABLE_VERSION)
    length(recipes) == length(p.tables) || error("Material recipe/table count mismatch")
    for (r,t) in zip(recipes,p.tables)
        r.name == t.name && isapprox(r.I,t.I;rtol=1e-12) && isapprox(r.density,t.density;rtol=1e-12) ||
            error("Material recipe does not match table $(t.name)")
        all(isfinite,(r.density,r.I,r.ZoA)) && min(r.density,r.I,r.ZoA)>0 || error("Invalid material scalars")
        all(e -> all(isfinite,(e.Z,e.A,e.I,e.fraction)) && min(e.Z,e.A,e.I)>0 && e.fraction>=0,r.elements) || error("Invalid elemental recipe")
        isapprox(sum(e.fraction*e.Z/e.A for e in r.elements),r.ZoA;rtol=1e-12) || error("Elemental recipe ZoA mismatch")
        isempty(r.elements) && error("Empty elemental recipe for $(t.name)")
        isapprox(sum(e.fraction for e in r.elements),1;atol=1e-12) ||
            error("Invalid elemental fractions for $(t.name)")
    end
    metadata = (recipes=deepcopy(recipes),recipe_fingerprint=recipe_fingerprint(recipes),fingerprint=fingerprint,table_version=table_version)
    lock(_recipe_lock) do
        for k in collect(keys(_recipe_registry))
            _recipe_registry[k].owner.value === nothing && delete!(_recipe_registry,k)
        end
        _recipe_registry[objectid(p.tables)] = (owner=WeakRef(p.tables),metadata=metadata)
    end
    return metadata
end

"""Return validated immutable recipes, or nothing for a legacy/unregistered dump.

Validation hashes the complete tables once at context preparation/save time. Inner
transport loops must retain their prepared snapshot rather than call this function.
Mutating PhysicsTables arrays requires explicit regeneration, never re-registration.
"""
function material_metadata(p::PhysicsTables; required=false)
    entry = lock(_recipe_lock) do
        item = get(_recipe_registry,objectid(p.tables),nothing)
        item === nothing || item.owner.value !== p.tables ? nothing : item.metadata
    end
    if entry === nothing
        required && error("Stochastic mixed transport requires versioned elemental recipes. Regenerate physics from its original MDF and stopping tables with create_physics, then save_physics to a NEW dump path; legacy dumps support deterministic transport only.")
        return nothing
    end
    entry.table_version == Materials.RADIATIVE_DCS_TABLE_VERSION ||
        error("Derived physics tables are obsolete; regenerate from the original MDF and stopping tables into a NEW dump.")
    entry.fingerprint == physics_fingerprint(p) ||
        error("Physics tables changed after recipe validation; regenerate physics before stochastic transport or saving.")
    return entry
end
