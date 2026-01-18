#!/usr/bin/env julia
"""
flux_density.jl - Compute and visualize cumulative integrated muon flux

This script computes CUMULATIVE INTEGRATED FLUX: for each (zenith, E_level), the flux
is integrated from E_min to E_level. This shows:
- At low E_level: small cumulative flux (only lowest-energy contributions)
- At high E_level (= E_max): total integrated flux which properly DECAYS with zenith angle

Surfaces:
1. Gaisser reference (blue): analytical sea-level flux integrated from E_min to E_level
2. Thickness surfaces: MC-simulated flux through rock layers

Energy sampling:
- For each (zenith, E_level) point, n_samples energies are sampled log-uniformly in [E_min, E_level]
- For Gaisser: numerical integration over [E_min, E_level]

Zenith angle convention:
- θ = 0°  → vertical (muons from directly above) → maximum flux
- θ = 60° → inclined → reduced flux (longer atmospheric/rock path)

Usage:
    julia --project=. examples/flux_density.jl [OPTIONS]

Options:
    --dump PATH, -d PATH      Path to physics binary dump file (default: examples/data/materials.pumas)
    --samples N, -n N         Number of MC samples per (zenith, energy_level) point (default: 1000)
    --no-straggling           Disable energy straggling (default: enabled)
    --no-scattering           Disable scattering (default: enabled)
    --output PATH, -o PATH    Output path for plot HTML file (default: examples/data/flux_density.html)
    --thickness FLOATS        Rock thickness in meters (comma-separated, e.g. "0,100,200,400") (default: "0,100")
    --n-zeniths N             Number of zenith angle points (default: 10, range: 0-60°)
    --n-energies N            Number of energy level points for Y-axis (default: 8, range: E_min to E_max)
    --energy-min FLOAT        Minimum energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum energy in GeV (default: 1e9)
    --threshold FLOAT         Energy threshold for mode switching in GeV (default: 100.0)

Example:
    julia --project=. examples/flux_density.jl --thickness "0,100,200" --n-zeniths 12 --n-energies 10
"""

using DiffPumas
using DiffPumas.Physics: get_material_index
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: compute_flux
using DiffPumas.GaisserFlux: flux_gaisser
using DiffPumas: zenith_to_elevation
using DiffPumas.Pumas: load_or_create_physics
using PlotlyJS
using Printf

const PHYSICS_DUMP = joinpath(@__DIR__, "data", "materials.pumas")

"""
    compute_flux_grid_3d(physics, thicknesses, zeniths, energy_levels, energy_min; kwargs...)

Compute cumulative integrated flux for a 3D grid of (thickness, zenith, energy_level) parameters.
For each point, flux is integrated from energy_min to energy_level.

# Arguments
- `physics`: Physics tables
- `thicknesses`: Vector of rock thicknesses in meters
- `zeniths`: Vector of zenith angles in degrees
- `energy_levels`: Vector of upper energy bounds in GeV (for cumulative integration)
- `energy_min`: Lower energy bound in GeV (fixed)
- `n_samples`: Number of MC samples per point
- `straggling`: Enable energy straggling
- `scattering`: Enable scattering

# Returns
- 3D array: flux[thickness_idx, zenith_idx, energy_idx] (cumulative integrated flux)
- 3D array: sigma[thickness_idx, zenith_idx, energy_idx]
"""
function compute_flux_grid_3d(physics,
                               thicknesses::Vector{Float64},
                               zeniths::Vector{Float64},
                               energy_levels::Vector{Float64},
                               energy_min::Float64;
                               n_samples::Int = 1000,
                               straggling::Bool = true,
                               scattering::Bool = true,
                               energy_threshold_low::Float64 = 100.0)
    
    n_t = length(thicknesses)
    n_z = length(zeniths)
    n_e = length(energy_levels)
    
    flux_3d = zeros(n_t, n_z, n_e)
    sigma_3d = zeros(n_t, n_z, n_e)
    
    n_total = n_t * n_z * n_e
    n_done = 0
    
    for (t_idx, thickness) in enumerate(thicknesses)
        for (z_idx, zenith) in enumerate(zeniths)
            elevation = zenith_to_elevation(zenith)
            
            for (e_idx, energy_max) in enumerate(energy_levels)
                n_done += 1
                
                if n_done % 10 == 0 || n_done == 1
                    @info "[$n_done/$n_total] thickness=$(thickness)m, θ=$(zenith)°, E_max=$(round(energy_max, sigdigits=2)) GeV"
                end
                
                # Cumulative integrated flux from energy_min to energy_max
                flux, sigma = compute_flux(physics, 2650.0, thickness, elevation,
                                           energy_min, energy_max; 
                                           n_samples=n_samples,
                                           straggling=straggling,
                                           scattering=scattering,
                                           energy_threshold_low=energy_threshold_low)
                
                flux_3d[t_idx, z_idx, e_idx] = flux
                sigma_3d[t_idx, z_idx, e_idx] = sigma
            end
        end
    end
    
    return flux_3d, sigma_3d
end

"""
    compute_gaisser_cumulative(zeniths, energy_levels, energy_min; n_integration_points=100)

Compute cumulative Gaisser flux: for each (zenith, E_level), integrate from energy_min to E_level.

# Returns
- 2D array: cumulative_flux[zenith_idx, energy_idx] in m⁻² s⁻¹ sr⁻¹
"""
function compute_gaisser_cumulative(zeniths::Vector{Float64},
                                     energy_levels::Vector{Float64},
                                     energy_min::Float64;
                                     n_integration_points::Int = 100)
    n_z = length(zeniths)
    n_e = length(energy_levels)
    
    cumulative_flux = zeros(n_z, n_e)
    
    for (z_idx, zenith) in enumerate(zeniths)
        cos_theta = cosd(zenith)
        
        for (e_idx, energy_max) in enumerate(energy_levels)
            # Numerical integration from energy_min to energy_max
            log_e_min = log10(energy_min)
            log_e_max = log10(energy_max)
            
            # Scale number of points with energy range
            n_points = max(10, Int(ceil(n_integration_points * (log_e_max - log_e_min) / 12)))
            log_energies = range(log_e_min, log_e_max, length=n_points+1)
            
            integral = 0.0
            for i in 1:n_points
                e1 = 10.0^log_energies[i]
                e2 = 10.0^log_energies[i+1]
                e_mid = sqrt(e1 * e2)
                de = e2 - e1
                
                flux_mid = flux_gaisser(cos_theta, e_mid, 0.0)
                integral += flux_mid * de
            end
            
            cumulative_flux[z_idx, e_idx] = integral
        end
    end
    
    return cumulative_flux
end

"""
    create_combined_flux_plot(thicknesses, zeniths, energy_levels, flux_3d, gaisser_flux)

Create a combined 3D surface plot with cumulative integrated flux:
- X-axis: Zenith angle (°)
- Y-axis: Energy level E_max (GeV) - upper bound of integration
- Z-axis: Cumulative flux ∫ᴱᵐⁱⁿᴱᵐᵃˣ Φ(E',θ) dE'

At the highest energy, surfaces show total integrated flux which decays with zenith.
"""
function create_combined_flux_plot(thicknesses::Vector{Float64},
                                    zeniths::Vector{Float64},
                                    energy_levels::Vector{Float64},
                                    flux_3d::Array{Float64, 3},
                                    gaisser_flux::Matrix{Float64})
    
    n_z = length(zeniths)
    n_e = length(energy_levels)
    n_t = length(thicknesses)
    
    traces = PlotlyJS.GenericTrace[]
    
    # Color scales for different surfaces
    colorscales = ["Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Hot", "Cool", "Viridis"]
    
    # Create meshgrid
    zenith_grid = reshape(zeniths, (n_z, 1)) .* ones(1, n_e)
    energy_grid = reshape(energy_levels, (1, n_e)) .* ones(n_z, 1)
    
    # 1. Add Gaisser reference surface (no rock)
    gaisser_flux_log = [f > 0 ? log10(max(f, 1e-30)) : NaN for f in gaisser_flux]
    
    gaisser_trace = surface(
        x = zenith_grid,
        y = energy_grid,
        z = gaisser_flux_log,
        name = "Gaisser (no rock)",
        colorscale = "Blues",
        showscale = true,
        colorbar = attr(title = "log₁₀(Flux)", x = 1.02),
        opacity = 0.9
    )
    push!(traces, gaisser_trace)
    
    # 2. Add surfaces for each thickness
    for (t_idx, thickness) in enumerate(thicknesses)
        flux_grid = flux_3d[t_idx, :, :]
        flux_log = [f > 0 ? log10(max(f, 1e-30)) : NaN for f in flux_grid]
        
        colorscale = colorscales[(t_idx - 1) % length(colorscales) + 1]
        
        trace = surface(
            x = zenith_grid,
            y = energy_grid,
            z = flux_log,
            name = "$(Int(thickness))m rock",
            colorscale = colorscale,
            showscale = false,
            opacity = 0.7
        )
        push!(traces, trace)
    end
    
    layout = Layout(
        title = "Cumulative Muon Flux: ∫ᴱᵐⁱⁿᴱᵐᵃˣ Φ(E',θ) dE'",
        scene = attr(
            xaxis = attr(title = "Zenith Angle θ (°)"),
            yaxis = attr(title = "E_max (GeV)", type = "log"),
            zaxis = attr(title = "log₁₀(Flux) [m⁻² s⁻¹ sr⁻¹]"),
            camera = attr(eye = attr(x = 1.5, y = 1.5, z = 1.2))
        ),
        width = 1200,
        height = 900,
        legend = attr(x = 0.02, y = 0.98)
    )
    
    return Plot(traces, layout)
end

function parse_commandline()
    dump_path = PHYSICS_DUMP
    n_samples = 1000
    straggling = true
    scattering = true
    output_path = nothing
    thickness_str = "0,100"
    n_zeniths = 10
    n_energies = 8
    energy_min = 1e-3
    energy_max = 1e9
    energy_threshold_low = 100.0
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--dump" || arg == "-d"
            if i + 1 <= length(ARGS)
                dump_path = ARGS[i + 1]
                i += 2
            else
                error("--dump requires a path")
            end
        elseif arg == "--samples" || arg == "-n"
            if i + 1 <= length(ARGS)
                n_samples = parse(Int, ARGS[i + 1])
                i += 2
            else
                error("--samples requires a value")
            end
        elseif arg == "--no-straggling"
            straggling = false
            i += 1
        elseif arg == "--no-scattering"
            scattering = false
            i += 1
        elseif arg == "--output" || arg == "-o"
            if i + 1 <= length(ARGS)
                output_path = ARGS[i + 1]
                i += 2
            else
                error("--output requires a path")
            end
        elseif arg == "--thickness"
            if i + 1 <= length(ARGS)
                thickness_str = ARGS[i + 1]
                i += 2
            else
                error("--thickness requires a value")
            end
        elseif arg == "--n-zeniths"
            if i + 1 <= length(ARGS)
                n_zeniths = parse(Int, ARGS[i + 1])
                i += 2
            else
                error("--n-zeniths requires a value")
            end
        elseif arg == "--n-energies"
            if i + 1 <= length(ARGS)
                n_energies = parse(Int, ARGS[i + 1])
                i += 2
            else
                error("--n-energies requires a value")
            end
        elseif arg == "--energy-min"
            if i + 1 <= length(ARGS)
                energy_min = parse(Float64, ARGS[i + 1])
                i += 2
            else
                error("--energy-min requires a value")
            end
        elseif arg == "--energy-max"
            if i + 1 <= length(ARGS)
                energy_max = parse(Float64, ARGS[i + 1])
                i += 2
            else
                error("--energy-max requires a value")
            end
        elseif arg == "--threshold"
            if i + 1 <= length(ARGS)
                energy_threshold_low = parse(Float64, ARGS[i + 1])
                i += 2
            else
                error("--threshold requires a value")
            end
        elseif arg == "--help" || arg == "-h"
            println(@doc flux_density)
            exit(0)
        elseif startswith(arg, "--")
            error("Unknown argument: $arg (use --help for usage)")
        else
            error("Unknown argument: $arg (use --help for usage)")
        end
    end
    
    if output_path === nothing
        output_path = joinpath(@__DIR__, "data", "flux_density.html")
    end
    
    return (dump_path=dump_path, n_samples=n_samples, straggling=straggling, 
            scattering=scattering, output_path=output_path, thickness_str=thickness_str,
            n_zeniths=n_zeniths, n_energies=n_energies, energy_min=energy_min, 
            energy_max=energy_max, energy_threshold_low=energy_threshold_low)
end

function main()
    args = parse_commandline()
    
    # Parse thickness values
    thicknesses = [parse(Float64, strip(t)) for t in split(args.thickness_str, ",")]
    
    println("=" ^ 60)
    println(" DiffPumas - Cumulative Integrated Flux")
    println("=" ^ 60)
    println()
    
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  Thicknesses:     $(join([string(Int(t)) for t in thicknesses], ", ")) m")
    println("  N zeniths:       $(args.n_zeniths) (0° to 60°)")
    println("  N energy levels: $(args.n_energies) ($(args.energy_min) to $(args.energy_max) GeV)")
    println("  MC samples/pt:   $(args.n_samples)")
    println("  Straggling:      $(args.straggling)")
    println("  Scattering:      $(args.scattering)")
    println("  Energy threshold: $(args.energy_threshold_low) GeV")
    println("  Output:          $(args.output_path)")
    println()
    
    # Load or create physics tables
    physics = load_or_create_physics(args.dump_path)
    
    if physics === nothing
        error("Failed to load or create physics tables!")
    end
    
    print_physics_summary(physics)
    println()
    
    # Create zenith angle grid (0° to 60°)
    zeniths = collect(range(0.0, 60.0, length=args.n_zeniths))
    
    # Create energy level grid (log-spaced from energy_min to energy_max)
    log_e_min = log10(args.energy_min)
    log_e_max = log10(args.energy_max)
    energy_levels = [10.0^e for e in range(log_e_min, log_e_max, length=args.n_energies)]
    
    n_total_points = length(thicknesses) * length(zeniths) * length(energy_levels)
    
    println("Parameter grid:")
    println("  Thicknesses: $(length(thicknesses)) values")
    println("  Zenith angles: $(length(zeniths)) points from $(zeniths[1])° to $(zeniths[end])°")
    println("  Energy levels: $(length(energy_levels)) points from $(energy_levels[1]) to $(energy_levels[end]) GeV")
    println("  Total MC computations: $(n_total_points)")
    println("  Total MC samples: $(n_total_points * args.n_samples)")
    println()
    
    # Compute cumulative Gaisser flux (analytical, no rock)
    println("Computing cumulative Gaisser flux (analytical, no rock)...")
    gaisser_flux = compute_gaisser_cumulative(zeniths, energy_levels, args.energy_min)
    @printf("  Gaisser at θ=0°, E_max=%.0e GeV: %.4e m⁻² s⁻¹ sr⁻¹\n", energy_levels[end], gaisser_flux[1, end])
    @printf("  Gaisser at θ=60°, E_max=%.0e GeV: %.4e m⁻² s⁻¹ sr⁻¹\n", energy_levels[end], gaisser_flux[end, end])
    @printf("  Ratio (0°/60°): %.2f\n", gaisser_flux[1, end] / gaisser_flux[end, end])
    println("✓ Cumulative Gaisser flux computed")
    println()
    
    # Compute MC flux grid for all thicknesses
    println("Computing MC cumulative flux for rock thicknesses...")
    flux_3d, sigma_3d = compute_flux_grid_3d(physics, thicknesses, zeniths, energy_levels, args.energy_min;
                                              n_samples=args.n_samples,
                                              straggling=args.straggling,
                                              scattering=args.scattering,
                                              energy_threshold_low=args.energy_threshold_low)
    println()
    
    # Print summary for each thickness at max energy
    println("Results summary (at E_max = $(energy_levels[end]) GeV):")
    println("-" ^ 55)
    @printf("  %-15s  %12s  %12s  %8s\n", "Surface", "Flux(θ=0°)", "Flux(θ=60°)", "Ratio")
    @printf("  %-15s  %12s  %12s  %8s\n", "", "(m⁻²s⁻¹sr⁻¹)", "(m⁻²s⁻¹sr⁻¹)", "(0°/60°)")
    println("-" ^ 55)
    @printf("  %-15s  %12.4e  %12.4e  %8.2f\n", "Gaisser", gaisser_flux[1, end], gaisser_flux[end, end], 
            gaisser_flux[1, end] / gaisser_flux[end, end])
    for (t_idx, thickness) in enumerate(thicknesses)
        flux_0 = flux_3d[t_idx, 1, end]
        flux_60 = flux_3d[t_idx, end, end]
        ratio = flux_0 / max(flux_60, 1e-30)
        @printf("  %-15s  %12.4e  %12.4e  %8.2f\n", "$(Int(thickness))m rock", flux_0, flux_60, ratio)
    end
    println("-" ^ 55)
    println()
    
    # Create plot
    println("Creating 3D cumulative flux plot...")
    plot = create_combined_flux_plot(thicknesses, zeniths, energy_levels, flux_3d, gaisser_flux)
    
    # Save plot
    output_file = args.output_path
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        PlotlyJS.savefig(io, plot, format="html")
    end
    println("✓ Saved plot: $(output_file)")
    
    # Display plot if in interactive environment
    try
        display(plot)
    catch
        # Not in interactive environment
    end
    
    println()
    println("Done!")
    println("View plot: $(output_file)")
    println("  X-axis: Zenith angle θ (°)")
    println("  Y-axis: E_max - upper bound of energy integration")
    println("  Z-axis: Cumulative flux ∫ᴱᵐⁱⁿᴱᵐᵃˣ Φ(E',θ) dE'")
    println("  At Y=E_max edge: total integrated flux, decays with zenith")
    println()
    
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
