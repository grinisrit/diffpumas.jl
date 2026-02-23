"""
    Loader

Smart materials loader for PUMAS physics tables.
Implements caching via binary dumps for fast initialization.
Supports loading materials from XML Material Description Files (MDF).
"""
module Loader

using ..Types
using ..Physics
using ..Materials
using Serialization
using LightXML

export load_physics, save_physics, load_or_create_physics
export parse_mdf, print_physics_summary

"""
    load_physics(filepath)

Load physics tables from a binary dump file.

# Arguments
- `filepath`: Path to the dump file

# Returns
Physics tables or `nothing` if loading fails.
"""
function load_physics(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    
    try
        return open(filepath, "r") do io
            deserialize(io)
        end
    catch e
        @warn "Failed to load physics from $filepath: $e"
        return nothing
    end
end

"""
    save_physics(physics, filepath)

Save physics tables to a binary dump file.

# Arguments
- `physics`: Physics tables to save
- `filepath`: Path to the dump file

# Returns
`true` on success, `false` on failure.
"""
function save_physics(physics::PhysicsTables, filepath::String)
    try
        # Create directory if needed
        dir = dirname(filepath)
        if !isempty(dir) && !isdir(dir)
            mkpath(dir)
        end
        
        open(filepath, "w") do io
            serialize(io, physics)
        end
        return true
    catch e
        @warn "Failed to save physics to $filepath: $e"
        return false
    end
end

"""
    parse_mdf(filepath)

Parse a Material Description File (MDF) in XML format.
Returns a vector of BaseMaterial objects.

# Arguments
- `filepath`: Path to the MDF XML file

# Returns
Vector of BaseMaterial objects parsed from the file.

# Example MDF format
```xml
<pumas>
  <element name="H" Z="1" A="1.008700" I="19.2" />
  <material name="StandardRock" density="2.65">
    <component name="Rk" fraction="1" />
  </material>
</pumas>
```
"""
function parse_mdf(filepath::String)
    if !isfile(filepath)
        error("MDF file not found: $filepath")
    end
    
    @info "Parsing MDF file: $filepath"
    
    # Parse XML
    xdoc = parse_file(filepath)
    xroot = root(xdoc)
    
    if name(xroot) != "pumas"
        error("Invalid MDF file: root element must be 'pumas'")
    end
    
    # First pass: parse elements
    elements = Dict{String, AtomicElement}()
    
    for elem in child_elements(xroot)
        if name(elem) == "element"
            elem_name = attribute(elem, "name")
            Z = parse(Float64, attribute(elem, "Z"))
            A = parse(Float64, attribute(elem, "A"))
            I_ev = parse(Float64, attribute(elem, "I"))  # in eV
            I_gev = I_ev * 1e-9  # Convert to GeV
            
            elements[elem_name] = AtomicElement(elem_name, Z, A, I_gev)
        end
    end
    
    @info "  Parsed $(length(elements)) elements"
    
    # Second pass: parse materials
    materials = BaseMaterial[]
    
    for elem in child_elements(xroot)
        if name(elem) == "material"
            mat_name = attribute(elem, "name")
            density_str = attribute(elem, "density")
            density = parse(Float64, density_str) * 1000.0  # g/cm³ to kg/m³
            
            # Optional mean excitation energy override
            I_override = nothing
            I_attr = attribute(elem, "I"; required=false)
            if I_attr !== nothing
                I_override = parse(Float64, I_attr) * 1e-9  # eV to GeV
            end
            
            # Parse components
            mat_elements = AtomicElement[]
            fractions = Float64[]
            
            for comp in child_elements(elem)
                if name(comp) == "component"
                    comp_name = attribute(comp, "name")
                    fraction = parse(Float64, attribute(comp, "fraction"))
                    
                    if haskey(elements, comp_name)
                        push!(mat_elements, elements[comp_name])
                        push!(fractions, fraction)
                    else
                        @warn "Unknown element '$comp_name' in material '$mat_name'"
                    end
                end
            end
            
            if !isempty(mat_elements)
                # Calculate mean excitation energy if not overridden
                I_material = if I_override !== nothing
                    I_override
                else
                    # Bragg additivity with logarithmic weighting:
                    # ln(I) = Σ (w_i Z_i/A_i) ln(I_i) / Σ (w_i Z_i/A_i)
                    total_f = sum(fractions)
                    nf = fractions ./ total_f
                    zoa_sum = sum(e.Z / e.A * f for (e, f) in zip(mat_elements, nf))
                    ln_I = sum(e.Z / e.A * f * log(e.I) for (e, f) in zip(mat_elements, nf)) / zoa_sum
                    exp(ln_I)
                end
                
                mat = BaseMaterial(mat_name, density, I_material, mat_elements, fractions)
                push!(materials, mat)
                @info "  Parsed material: $mat_name (density=$(density) kg/m³)"
            end
        end
    end
    
    free(xdoc)
    
    @info "  Total materials: $(length(materials))"
    return materials
end

"""
    load_or_create_physics(particle; dump_path, mdf_path, kwargs...)

Load physics from dump if available, otherwise create from MDF and dump.
This is the smart loader implementation.

# Arguments
- `particle`: Particle type (MUON or TAU)

# Keyword Arguments
- `dump_path`: Path for binary dump (default: "materials.pumas")
- `mdf_path`: Path to MDF XML file (optional)
- `materials`: Vector of materials (used if mdf_path not provided)
- Other kwargs passed to `create_physics`

# Returns
Physics tables
"""
function load_or_create_physics(particle::Particle = MUON;
                                dump_path::String = "materials.pumas",
                                mdf_path::Union{String, Nothing} = nothing,
                                materials::Union{Vector{BaseMaterial}, Nothing} = nothing,
                                kwargs...)
    
    # Try to load from dump
    physics = load_physics(dump_path)
    if physics !== nothing
        @info "Loaded physics from $dump_path"
        return physics
    end
    
    # Determine materials source
    mats = if mdf_path !== nothing
        # Parse materials from MDF file
        parse_mdf(mdf_path)
    elseif materials !== nothing
        materials
    else
        # Use built-in defaults
        @info "Using built-in materials (StandardRock, Air)"
        [STANDARD_ROCK, AIR]
    end
    
    # Create physics tables
    @info "Creating physics tables..."
    physics = create_physics(particle; materials=mats, kwargs...)
    
    # Save to dump
    if save_physics(physics, dump_path)
        @info "Saved physics to $dump_path"
    end
    
    return physics
end

"""
    print_physics_summary(physics)

Print a human-readable summary of the physics tables.
"""
function print_physics_summary(physics::PhysicsTables)
    println("=" ^ 60)
    println("PUMAS Physics Summary")
    println("=" ^ 60)
    println()
    
    particle_name = physics.particle == MUON ? "Muon" : "Tau"
    println("Particle: $particle_name")
    println("Mass: $(physics.mass) GeV/c²")
    println("Decay length: $(physics.ctau) m")
    println()
    
    println("Settings:")
    println("  Cutoff: $(physics.settings.cutoff)")
    println("  Elastic ratio: $(physics.settings.elastic_ratio)")
    println("  Bremsstrahlung model: $(physics.settings.bremsstrahlung)")
    println("  Pair production model: $(physics.settings.pair_production)")
    println("  Photonuclear model: $(physics.settings.photonuclear)")
    println()
    
    println("Materials: $(length(physics.tables))")
    for (name, idx) in physics.materials
        table = physics.tables[idx]
        println("  [$idx] $name")
        println("      Density: $(table.density) kg/m³")
        println("      I: $(table.I * 1e9) eV")
        println("      Z/A: $(table.ZoA)")
        println("      Energy range: $(table.energies[1]) - $(table.energies[end]) GeV")
        println("      Table entries: $(length(table.energies))")
    end
    
    println()
    println("=" ^ 60)
end

end # module Loader
