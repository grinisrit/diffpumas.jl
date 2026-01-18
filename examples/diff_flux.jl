#!/usr/bin/env julia
"""
Diff Flux - Backward Monte Carlo Integrated Flux Calculation

This example demonstrates:
1. Loading physics tables from a binary dump
2. Computing integrated muon flux over zenith angle and energy ranges
3. Computing gradient of flux w.r.t. rock density using Zygote.jl

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
                         |    ╱   at zenith angle θ
                         |  ╱
                         |╱ θ (zenith)
    ─────────────────────●─────────────────── z = 0 (Detector)
                      Detector
                     (observer)
                         
    - - - - - - - - - - - - - - - - - - - - z < 0 (outside)

Monte Carlo Integration:
- Zenith angles: n_angles points uniformly sampled in [zenith_min, zenith_max]
- For each angle, n_samples energies sampled log-uniformly from [E_min, E_max]
    kf = E_min * exp(log(E_max/E_min) * rand())
    weight = kf * log(E_max/E_min)
- Charge: sampled 50-50 (μ+ or μ-)

Total MC evaluations = n_angles × n_samples

Usage:
    julia --project=. examples/diff_flux.jl [OPTIONS]

Options:
    --dump, -d PATH           Path to physics binary dump file (default: examples/data/materials.pumas)
    --thickness, -t FLOAT     Rock thickness in meters (default: 100.0)
    --density FLOAT           Rock density in kg/m³ (default: 2650.0)
    --zenith-min FLOAT        Minimum zenith angle in degrees (default: 0.0)
    --zenith-max FLOAT        Maximum zenith angle in degrees (default: 60.0)
    --energy-min FLOAT        Minimum kinetic energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum kinetic energy in GeV (default: 1e9)
    --n-angles INT            Number of zenith angle points (default: 100)
    --samples, -n INT         Number of energy samples per angle (default: 100)
    --gradient, -g            Compute gradient ∂flux/∂density
    --straggling              Enable energy straggling (default: disabled for CSDA)
    --scattering              Enable scattering (default: disabled)
    --threshold FLOAT         Energy threshold for mode switching in GeV (default: 0.0)

Examples:
    # Basic CSDA flux computation
    julia --project=. examples/diff_flux.jl --thickness 100 --zenith-max 45

    # Higher statistics
    julia --project=. examples/diff_flux.jl --thickness 200 --n-angles 200 --samples 500

    # Compute density gradient (sensitivity analysis)
    julia --project=. examples/diff_flux.jl --thickness 100 --gradient

    # Enable straggling for more accurate physics
    julia --project=. examples/diff_flux.jl --thickness 100 --straggling --threshold 100
"""

using DiffPumas
using DiffPumas.Physics: get_material_index
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: compute_flux_single, compute_flux_differentiable, TwoLayerGeometry
using DiffPumas.GaisserFlux: flux_gaisser
using DiffPumas: zenith_to_elevation, sample_energy_loguniform
using DiffPumas.ExamplesCommon: load_or_create_physics
using Printf
using Random

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")

function parse_commandline()
    dump_path = DEFAULT_DUMP
    rock_thickness = 100.0
    rock_density = 2650.0
    zenith_min = 0.0
    zenith_max = 60.0
    energy_min = 1e-3
    energy_max = 1e9
    n_angles = 100
    n_samples = 100
    compute_grad = false
    straggling = false
    scattering = false
    energy_threshold_low = 0.0
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--dump" || arg == "-d"
            i + 1 <= length(ARGS) || error("--dump requires a path")
            dump_path = ARGS[i + 1]
            i += 2
        elseif arg == "--thickness" || arg == "-t"
            i + 1 <= length(ARGS) || error("--thickness requires a value")
            rock_thickness = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--density"
            i + 1 <= length(ARGS) || error("--density requires a value")
            rock_density = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--zenith-min"
            i + 1 <= length(ARGS) || error("--zenith-min requires a value")
            zenith_min = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--zenith-max"
            i + 1 <= length(ARGS) || error("--zenith-max requires a value")
            zenith_max = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--energy-min"
            i + 1 <= length(ARGS) || error("--energy-min requires a value")
            energy_min = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--energy-max"
            i + 1 <= length(ARGS) || error("--energy-max requires a value")
            energy_max = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--n-angles"
            i + 1 <= length(ARGS) || error("--n-angles requires a value")
            n_angles = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--samples" || arg == "-n"
            i + 1 <= length(ARGS) || error("--samples requires a value")
            n_samples = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--gradient" || arg == "-g"
            compute_grad = true
            i += 1
        elseif arg == "--straggling"
            straggling = true
            i += 1
        elseif arg == "--scattering"
            scattering = true
            i += 1
        elseif arg == "--threshold"
            i + 1 <= length(ARGS) || error("--threshold requires a value")
            energy_threshold_low = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--help" || arg == "-h"
            println(@doc diff_flux)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end
    
    return (
        dump_path = dump_path,
        rock_thickness = rock_thickness,
        rock_density = rock_density,
        zenith_min = zenith_min,
        zenith_max = zenith_max,
        energy_min = energy_min,
        energy_max = energy_max,
        n_angles = n_angles,
        n_samples = n_samples,
        compute_grad = compute_grad,
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low
    )
end

"""
    compute_integrated_flux_mc(physics, geometry, zenith_min, zenith_max,
                               energy_min, energy_max, n_angles, n_samples; kwargs...)

Compute integrated muon flux using Monte Carlo integration.

Integration structure:
- Outer loop: n_angles zenith points uniformly sampled in [zenith_min, zenith_max]
- Inner loop: for each angle, n_samples energies log-uniformly sampled in [E_min, E_max]

Returns (flux, sigma).
"""
function compute_integrated_flux_mc(physics, geometry::TwoLayerGeometry{Float64},
                                    zenith_min::Float64, zenith_max::Float64,
                                    energy_min::Float64, energy_max::Float64,
                                    n_angles::Int, n_samples::Int;
                                    straggling::Bool=false,
                                    scattering::Bool=false,
                                    energy_threshold_low::Float64=0.0,
                                    verbose::Bool=true)
    
    rng = MersenneTwister(42)
    
    d_zenith = zenith_max - zenith_min
    n_total = n_angles * n_samples
    
    w_sum = 0.0
    w2_sum = 0.0
    n_done = 0
    
    for i_angle in 1:n_angles
        # Sample zenith uniformly in the interval
        zenith = zenith_min + d_zenith * rand(rng)
        elevation = zenith_to_elevation(zenith)
        
        # For this angle, sample n_samples energies
        for i_sample in 1:n_samples
            n_done += 1
            
            # Sample energy log-uniformly (as in geometry.c)
            kf, w_energy = sample_energy_loguniform(energy_min, energy_max, rng)
            
            # Sample charge (50-50)
            charge = rand(rng) > 0.5 ? 1.0 : -1.0
            
            # Compute flux contribution
            flux_single = compute_flux_single(physics, geometry, kf, elevation, charge;
                                              rng=rng, straggling=straggling, scattering=scattering,
                                              energy_threshold_low=energy_threshold_low)
            
            # Weight: energy sampling * charge sampling (2 for 50-50)
            wi = 2.0 * w_energy * flux_single
            
            w_sum += wi
            w2_sum += wi * wi
            
            if verbose && (n_done % 1000 == 0 || n_done == 1)
                @info "[$n_done/$n_total] θ=$(round(zenith, digits=1))°, E=$(round(kf, sigdigits=3)) GeV"
            end
        end
    end
    
    # Average over samples
    n = Float64(n_total)
    flux = w_sum / n
    
    # Standard error
    variance = (w2_sum / n - flux^2) / max(1.0, n - 1)
    sigma = sqrt(max(0.0, variance))
    
    return flux, sigma
end

"""
    compute_integrated_flux_gradient_mc(physics, rock_density, rock_thickness,
                                        zenith_min, zenith_max, energy_min, energy_max,
                                        n_angles, n_samples; kwargs...)

Compute integrated flux and its gradient ∂flux/∂ρ using Zygote.

Pre-samples all random values to ensure deterministic gradient computation.
"""
function compute_integrated_flux_gradient_mc(physics, rock_density::Float64, rock_thickness::Float64,
                                             rock_idx::Int, air_idx::Int,
                                             zenith_min::Float64, zenith_max::Float64,
                                             energy_min::Float64, energy_max::Float64,
                                             n_angles::Int, n_samples::Int;
                                             straggling::Bool=false,
                                             energy_threshold_low::Float64=0.0,
                                             verbose::Bool=true)
    
    using Zygote
    
    rng = MersenneTwister(42)
    d_zenith = zenith_max - zenith_min
    
    # Pre-sample all random values
    samples = Vector{NTuple{4, Float64}}()
    for i_angle in 1:n_angles
        zenith = zenith_min + d_zenith * rand(rng)
        elevation = zenith_to_elevation(zenith)
        
        for i_sample in 1:n_samples
            kf, w_energy = sample_energy_loguniform(energy_min, energy_max, rng)
            charge = rand(rng) > 0.5 ? 1.0 : -1.0
            push!(samples, (elevation, kf, charge, w_energy))
        end
    end
    
    n_total = length(samples)
    
    # Differentiable flux function
    function flux_fn(ρ)
        w_sum = 0.0
        for (elevation, kf, charge, w_energy) in samples
            flux_single = compute_flux_differentiable(physics, ρ, rock_thickness,
                                                       elevation, kf, charge;
                                                       straggling=straggling, scattering=false,
                                                       energy_threshold_low=energy_threshold_low)
            w_sum += 2.0 * w_energy * flux_single
        end
        return w_sum / n_total
    end
    
    verbose && println("Computing integrated flux (deterministic for AD)...")
    flux = flux_fn(rock_density)
    
    verbose && println("Computing gradient ∂flux/∂ρ using Zygote...")
    grad_density = Zygote.gradient(flux_fn, rock_density)[1]
    
    # Compute variance using MC with RNG
    verbose && println("Computing statistical uncertainty...")
    geometry = TwoLayerGeometry{Float64}(rock_thickness, rock_density, rock_idx, air_idx)
    
    rng2 = MersenneTwister(42)
    w_sum = 0.0
    w2_sum = 0.0
    
    for (elevation, kf, charge, w_energy) in samples
        flux_single = compute_flux_single(physics, geometry, kf, elevation, charge;
                                          rng=rng2, straggling=straggling, scattering=false,
                                          energy_threshold_low=energy_threshold_low)
        wi = 2.0 * w_energy * flux_single
        w_sum += wi
        w2_sum += wi * wi
    end
    
    flux_mc = w_sum / n_total
    variance = (w2_sum / n_total - flux_mc^2) / max(1.0, n_total - 1)
    sigma = sqrt(max(0.0, variance))
    
    return flux, sigma, grad_density
end

function main()
    args = parse_commandline()
    
    println("=" ^ 60)
    println(" DiffPumas - Integrated Flux Calculation (CSDA)")
    println("=" ^ 60)
    println()
    
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  Rock thickness:  $(args.rock_thickness) m")
    println("  Rock density:    $(args.rock_density) kg/m³")
    println("  Zenith range:    $(args.zenith_min)° - $(args.zenith_max)°")
    println("  Energy range:    $(args.energy_min) - $(args.energy_max) GeV")
    println("  Angle points:    $(args.n_angles)")
    println("  Samples/angle:   $(args.n_samples)")
    println("  Total samples:   $(args.n_angles * args.n_samples)")
    println("  Compute grad:    $(args.compute_grad)")
    println("  Straggling:      $(args.straggling)")
    println("  Scattering:      $(args.scattering)")
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println()
    
    # Load physics
    physics = load_or_create_physics(args.dump_path)
    if physics === nothing
        println("ERROR: Failed to load physics from $(args.dump_path)")
        return 1
    end
    print_physics_summary(physics)
    println()
    
    # Get material indices
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials not found")
        println("  StandardRock: $(rock_idx)")
        println("  Air: $(air_idx)")
        return 1
    end
    
    println("Material indices:")
    println("  StandardRock: $(rock_idx)")
    println("  Air: $(air_idx)")
    println()
    
    # Create geometry
    geometry = TwoLayerGeometry{Float64}(
        args.rock_thickness, args.rock_density, rock_idx, air_idx
    )
    
    # Compute integrated flux
    println("Running Monte Carlo integration...")
    println("-" ^ 40)
    
    if args.compute_grad
        flux, sigma, grad_density = compute_integrated_flux_gradient_mc(
            physics, args.rock_density, args.rock_thickness, rock_idx, air_idx,
            args.zenith_min, args.zenith_max, args.energy_min, args.energy_max,
            args.n_angles, args.n_samples;
            straggling=args.straggling,
            energy_threshold_low=args.energy_threshold_low,
            verbose=true
        )
        
        println()
        println("Results:")
        println("=" ^ 40)
        @printf("  Integrated flux:    %.5e ± %.5e m⁻² s⁻¹ sr⁻¹\n", flux, sigma)
        @printf("  Relative error:     %.2f%%\n", 100 * sigma / max(flux, 1e-30))
        println()
        @printf("  ∂flux/∂ρ:           %.5e m⁻² s⁻¹ sr⁻¹ / (kg/m³)\n", grad_density)
        @printf("  Sensitivity:        %.2f%% per 1%% density change\n", 
                100 * grad_density * args.rock_density / max(flux, 1e-30))
        
        # Numerical gradient check
        println()
        println("Numerical gradient check (finite differences):")
        h = 1.0
        geometry_plus = TwoLayerGeometry{Float64}(args.rock_thickness, args.rock_density + h, rock_idx, air_idx)
        geometry_minus = TwoLayerGeometry{Float64}(args.rock_thickness, args.rock_density - h, rock_idx, air_idx)
        
        flux_plus, _ = compute_integrated_flux_mc(physics, geometry_plus,
            args.zenith_min, args.zenith_max, args.energy_min, args.energy_max,
            args.n_angles, args.n_samples;
            straggling=args.straggling, scattering=false,
            energy_threshold_low=args.energy_threshold_low, verbose=false)
        
        flux_minus, _ = compute_integrated_flux_mc(physics, geometry_minus,
            args.zenith_min, args.zenith_max, args.energy_min, args.energy_max,
            args.n_angles, args.n_samples;
            straggling=args.straggling, scattering=false,
            energy_threshold_low=args.energy_threshold_low, verbose=false)
        
        numerical_grad = (flux_plus - flux_minus) / (2 * h)
        @printf("  Numerical gradient: %.5e\n", numerical_grad)
        @printf("  AD gradient:        %.5e\n", grad_density)
        @printf("  Relative diff:      %.2f%%\n", 100 * abs(numerical_grad - grad_density) / max(abs(grad_density), 1e-30))
    else
        flux, sigma = compute_integrated_flux_mc(physics, geometry,
            args.zenith_min, args.zenith_max, args.energy_min, args.energy_max,
            args.n_angles, args.n_samples;
            straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low, verbose=true)
        
        println()
        println("Results:")
        println("=" ^ 40)
        @printf("  Integrated flux:    %.5e ± %.5e m⁻² s⁻¹ sr⁻¹\n", flux, sigma)
        @printf("  Relative error:     %.2f%%\n", 100 * sigma / max(flux, 1e-30))
    end
    
    # Compare with Gaisser analytical flux (no rock)
    println()
    println("Comparison with Gaisser (no rock):")
    println("-" ^ 40)
    
    # Compute Gaisser flux at center of ranges
    zenith_mid = (args.zenith_min + args.zenith_max) / 2
    energy_mid = sqrt(args.energy_min * args.energy_max)
    cos_theta = cosd(zenith_mid)
    gaisser_point = flux_gaisser(cos_theta, energy_mid, 0.0)
    
    @printf("  Gaisser flux at θ=%.1f°, E=%.1e GeV: %.5e GeV⁻¹ m⁻² s⁻¹ sr⁻¹\n",
            zenith_mid, energy_mid, gaisser_point)
    
    # Rough integrated Gaisser estimate
    rk = log(args.energy_max / args.energy_min)
    gaisser_integrated_estimate = gaisser_point * energy_mid * rk
    @printf("  Rough integrated Gaisser estimate: %.5e m⁻² s⁻¹ sr⁻¹\n", gaisser_integrated_estimate)
    @printf("  Attenuation factor (flux/Gaisser): %.3e\n", flux / max(gaisser_integrated_estimate, 1e-30))
    
    println()
    println("=" ^ 60)
    println("Done!")
    return 0
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
