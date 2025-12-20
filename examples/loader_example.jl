#!/usr/bin/env julia
"""
Loader Example - Smart Materials Loading from MDF

This example demonstrates:
1. Loading materials from an XML Material Description File (MDF)
2. Creating physics tables from the materials
3. Saving physics to a binary dump for fast subsequent loads

Equivalent to the C example: pumas/examples/pumas/loader.c

Usage:
    julia --project=. examples/loader_example.jl --mdf examples/data/materials.xml --dump examples/data/materials.pumas
    julia --project=. examples/loader_example.jl -m examples/data/materials.xml -d examples/data/materials.pumas
"""

using DiffPumas
using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        prog = "loader_example.jl",
        description = "Load materials from MDF and create physics dump"
    )
    
    @add_arg_table! s begin
        "--mdf", "-m"
            help = "Path to Material Description File (XML)"
            arg_type = String
            default = "examples/data/materials.xml"
        "--dump", "-d"
            help = "Path to output binary dump file"
            arg_type = String
            default = "examples/data/materials.pumas"
        "--particle", "-p"
            help = "Particle type: muon or tau"
            arg_type = String
            default = "muon"
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    mdf_path = args["mdf"]
    dump_path = args["dump"]
    particle_str = lowercase(args["particle"])
    
    particle = particle_str == "tau" ? TAU : MUON
    
    println("=" ^ 60)
    println(" DiffPumas - Smart Materials Loader")
    println("=" ^ 60)
    println()
    
    println("Configuration:")
    println("  MDF file:   $(mdf_path)")
    println("  Dump file:  $(dump_path)")
    println("  Particle:   $(particle_str)")
    println()
    
    # Check if MDF file exists
    if !isfile(mdf_path)
        println("ERROR: MDF file not found: $(mdf_path)")
        println()
        println("Please provide a valid MDF file path.")
        println("Example: julia --project=. examples/loader_example.jl --mdf examples/data/materials.xml")
        return 1
    end
    
    # Create the dump directory if it doesn't exist
    dump_dir = dirname(dump_path)
    if !isempty(dump_dir) && !isdir(dump_dir)
        println("Creating directory: $(dump_dir)")
        mkpath(dump_dir)
    end
    
    # Load materials from MDF and create physics (or load from existing dump)
    println("Loading/creating physics tables...")
    physics = load_or_create_physics(
        particle;
        dump_path = dump_path,
        mdf_path = mdf_path
    )
    
    # Print summary
    print_physics_summary(physics)
    
    # Example usage - demonstrate property lookups
    println()
    println("Example Property Lookups:")
    println("-" ^ 40)
    
    # Find available materials
    material_names = collect(keys(physics.materials))
    println("Available materials: $(join(material_names, ", "))")
    println()
    
    # Try to get StandardRock and Air if available
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx > 0
        println("StandardRock (index $rock_idx):")
        println("  CSDA Ranges:")
        for energy in [1.0, 10.0, 100.0, 1000.0]
            range_val = property_range(physics, ENERGY_LOSS_CSDA, rock_idx, energy)
            println("    E = $(energy) GeV: $(round(range_val, sigdigits=4)) kg/m²")
        end
        println()
    end
    
    if air_idx > 0
        println("Air (index $air_idx):")
        println("  CSDA Ranges:")
        for energy in [1.0, 10.0, 100.0, 1000.0]
            range_val = property_range(physics, ENERGY_LOSS_CSDA, air_idx, energy)
            println("    E = $(energy) GeV: $(round(range_val, sigdigits=4)) kg/m²")
        end
        println()
    end
    
    println("=" ^ 60)
    println("Physics dump saved to: $(dump_path)")
    println()
    println("To use this dump in geometry_example.jl:")
    println("  julia --project=. examples/geometry_example.jl --dump $(dump_path)")
    println("=" ^ 60)
    
    return 0
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
