#!/usr/bin/env julia
"""
Geometry Example - Backward Monte Carlo Flux Calculation

This example demonstrates:
1. Loading physics tables from a binary dump (created by loader_example.jl)
2. Running backward Monte Carlo to compute transmitted flux
3. Computing gradients of flux w.r.t. rock density using Zygote.jl

Equivalent to the C example: pumas/examples/pumas/geometry.c

Geometry:

                    PRIMARY_ALTITUDE (1000m)
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                    (exponential density)
                         |
    ──────────────────────────────────────── z = rock_thickness
                         |
                    ROCK LAYER
                    (uniform density)
                         |
                         |      ╱ incoming muon
                         |    ╱   at elevation angle
                         |  ╱
                         |╱ elevation
    ─────────────────────●─────────────────── z = 0 (Detector)
                      Detector
                     (observer)
                         
    - - - - - - - - - - - - - - - - - - - - z < 0 (outside)

Note: For this example to work, a PUMAS physics dump must have been generated
first by running the loader_example.jl script:

    julia --project=. examples/loader_example.jl --mdf examples/data/materials.xml

Usage:
    julia --project=. examples/geometry_example.jl [OPTIONS]

Options:
    --dump, -d PATH          Path to physics binary dump file (default: examples/data/materials.pumas)
    --thickness, -t FLOAT    Rock thickness in meters (default: 100.0)
    --elevation, -e FLOAT    Elevation angle in degrees (default: 45.0)
    --energy-min FLOAT       Minimum kinetic energy in GeV (default: 1.0)
    --energy-max FLOAT       Maximum kinetic energy in GeV (optional)
    --samples, -n INT        Number of Monte Carlo samples (default: 10000)
    --gradient, -g            Compute gradient ∂flux/∂density
    --no-straggling           Disable energy straggling
    --no-scattering           Disable scattering
    --threshold FLOAT         Energy threshold for mode switching in GeV (default: 100.0)

Examples:
    julia --project=. examples/geometry_example.jl --thickness 500 --elevation 0 --energy-min 1e-3 --energy-max 1e9
    julia --project=. examples/geometry_example.jl --thickness 100 --elevation 90 --samples 50000
    julia --project=. examples/geometry_example.jl --thickness 200 --elevation 45 --samples 20000 --gradient
    julia --project=. examples/geometry_example.jl --thickness 200 --elevation 45 --no-straggling --no-scattering
"""

using DiffPumas
using DiffPumas.Physics: get_material_index
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: compute_flux, compute_flux_gradient, compute_flux_differentiable
using ArgParse
using Zygote
using Printf

function parse_commandline()
    s = ArgParseSettings(
        prog = "geometry_example.jl",
        description = "Backward Monte Carlo flux calculation through rock"
    )
    
    @add_arg_table! s begin
        "--dump", "-d"
            help = "Path to physics binary dump file"
            arg_type = String
            default = "examples/data/materials.pumas"
        "--thickness", "-t"
            help = "Rock thickness in meters"
            arg_type = Float64
            default = 100.0
        "--elevation", "-e"
            help = "Elevation angle in degrees"
            arg_type = Float64
            default = 45.0
        "--energy-min"
            help = "Minimum kinetic energy in GeV"
            arg_type = Float64
            default = 1.0
        "--energy-max"
            help = "Maximum kinetic energy in GeV (omit for point estimate)"
            arg_type = Float64
            default = nothing
        "--samples", "-n"
            help = "Number of Monte Carlo samples"
            arg_type = Int
            default = 10000
        "--gradient", "-g"
            help = "Compute gradient ∂flux/∂density"
            action = :store_true
        "--no-straggling"
            help = "Disable energy straggling"
            action = :store_true
        "--no-scattering"
            help = "Disable scattering"
            action = :store_true
        "--threshold"
            help = "Energy threshold for mode switching in GeV (default: 100.0)"
            arg_type = Float64
            default = 100.0
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    dump_path = args["dump"]
    rock_thickness = args["thickness"]
    elevation = args["elevation"]
    energy_min = args["energy-min"]
    energy_max = args["energy-max"]
    n_samples = args["samples"]
    compute_grad = args["gradient"]
    straggling = !args["no-straggling"]
    scattering = !args["no-scattering"]
    energy_threshold_low = args["threshold"]
    
    println("=" ^ 60)
    println(" DiffPumas - Backward Monte Carlo Flux Calculation")
    println("=" ^ 60)
    println()
    
    println("Configuration:")
    println("  Dump file:      $(dump_path)")
    println("  Rock thickness: $(rock_thickness) m")
    println("  Elevation:      $(elevation)°")
    if energy_max !== nothing
        println("  Energy range:   $(energy_min) - $(energy_max) GeV")
    else
        println("  Energy:         $(energy_min) GeV (point estimate)")
    end
    println("  MC samples:     $(n_samples)")
    println("  Compute grad:   $(compute_grad)")
    println("  Straggling:     $(straggling)")
    println("  Scattering:     $(scattering)")
    println("  Energy threshold: $(energy_threshold_low) GeV")
    println()
    
    # Load PUMAS physics from binary dump (created by loader_example.jl)
    println("Loading physics from binary dump...")
    println("  Dump file: $(dump_path)")
    
    physics = load_physics(dump_path)
    
    if physics === nothing
        println()
        println("ERROR: Physics dump file not found: $(dump_path)")
        println()
        println("Please run loader_example.jl first to generate the physics dump:")
        println("  julia --project=. examples/loader_example.jl --mdf examples/data/materials.xml --dump $(dump_path)")
        println()
        return 1
    end
    
    println("✓ Physics loaded successfully")
    print_physics_summary(physics)
    println()
    
    # Map the PUMAS material indices (like geometry.c lines 205-208)
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials not found in physics tables")
        println("  StandardRock index: $(rock_idx)")
        println("  Air index: $(air_idx)")
        println()
        println("The dump file must contain 'StandardRock' and 'Air' materials.")
        return 1
    end
    
    println("Material indices:")
    println("  StandardRock: $(rock_idx)")
    println("  Air: $(air_idx)")
    println()
    
    # Run backward Monte Carlo using compute_flux (as in flux_comparison.jl)
    println("Running backward Monte Carlo simulation...")
    
    # Handle energy_max for point estimate (same as run_backward_mc)
    energy_max_actual = energy_max !== nothing ? energy_max : energy_min
    
    # Compute flux using compute_flux directly (as in flux_comparison.jl)
    flux, sigma = compute_flux(
        physics,
        2650.0,  # Standard rock density (kg/m³)
        rock_thickness,
        elevation,
        energy_min,
        energy_max_actual;
        n_samples = n_samples,
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low
    )
    
    # Compute gradient if requested
    grad_density = nothing
    if compute_grad
        energy_test = energy_max !== nothing ? sqrt(energy_min * energy_max) : energy_min
        _, grad_density = compute_flux_gradient(
            physics, 2650.0, rock_thickness, elevation, energy_test, 1.0
        )
    end
    
    # Print results
    println()
    println("Results:")
    println("-" ^ 40)
    unit = energy_max !== nothing ? "" : "GeV⁻¹ "
    @printf("  Flux: %.5e ± %.5e %sm⁻² s⁻¹ sr⁻¹\n", flux, sigma, unit)
    
    if grad_density !== nothing
        @printf("  ∂flux/∂ρ: %.5e\n", grad_density)
    end
    
    println()
    println("=" ^ 60)
    
    # Demonstrate gradient computation explicitly if requested
    if compute_grad
        println()
        println("Gradient Computation Demo:")
        println("-" ^ 40)
        
        rock_density = 2650.0
        energy_test = energy_max !== nothing ? sqrt(energy_min * energy_max) : energy_min
        
        println("Computing ∂flux/∂density for a single particle...")
        println("  Rock density: $(rock_density) kg/m³")
        println("  Test energy: $(energy_test) GeV")
        
        flux_single, grad = compute_flux_gradient(
            physics, rock_density, rock_thickness, elevation, energy_test, 1.0
        )
        
        @printf("  Flux contribution: %.5e\n", flux_single)
        if grad !== nothing
            @printf("  AD Gradient: %.5e\n", grad)
        else
            println("  AD Gradient: not available")
        end
        
        # Numerical gradient check
        println()
        println("Numerical gradient check (finite differences):")
        h = 1.0  # Step size
        f_plus = compute_flux_differentiable(physics, rock_density + h, rock_thickness, 
                                              elevation, energy_test, 1.0)
        f_minus = compute_flux_differentiable(physics, rock_density - h, rock_thickness,
                                               elevation, energy_test, 1.0)
        numerical_grad = (f_plus - f_minus) / (2h)
        @printf("  Numerical gradient: %.5e\n", numerical_grad)
    end
    
    println()
    println("Done!")
    return 0
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
