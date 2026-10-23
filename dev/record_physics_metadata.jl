# Record inspectable provenance for an already generated dump without changing it.
using DiffPumas
using TOML
length(ARGS)==2 || error("Usage: record_physics_metadata.jl DUMP OUTPUT.toml")
ispath(ARGS[2]) && error("Refusing to replace existing metadata")
p=load_physics(ARGS[1]);p===nothing && error("Unreadable dump")
md=DiffPumas.Physics.material_metadata(p;required=true)
metadata=Dict("julia_version"=>string(VERSION),"dump_format_version"=>2,
    "radiative_table_version"=>md.table_version,"physics_fingerprint"=>md.fingerprint,
    "recipe_fingerprint"=>md.recipe_fingerprint,
    "particle"=>string(p.particle),"mass_GeV"=>p.mass,"ctau_m"=>p.ctau,
    "cutoff"=>p.settings.cutoff,"elastic_ratio"=>p.settings.elastic_ratio,
    "materials"=>[Dict("name"=>t.name,"nodes"=>length(t.energies),
        "energy_min_GeV"=>first(t.energies),"energy_max_GeV"=>last(t.energies),
        "density_kg_m3"=>t.density,"imported_header_ZoA"=>t.ZoA,
        "recipe_ZoA"=>r.ZoA,
        "elements"=>[Dict(string(k)=>getproperty(e,k) for k in keys(e)) for e in r.elements])
        for (t,r) in zip(p.tables,md.recipes)])
open(io->TOML.print(io,metadata),ARGS[2],"w")
