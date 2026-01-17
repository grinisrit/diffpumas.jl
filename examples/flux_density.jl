#!/usr/bin/env julia
"""
flux_density.jl - Compute and visualize muon flux density over 3D parameter space

This script:
1. Computes muon flux over a 3D grid of (thickness, zenith, energy) parameters
2. Creates an interactive 3D PlotlyJS visualization with:
   - Main volume/heatmap: flux(thickness, zenith, energy)
   - X-Z plane projection: integrated flux over energy
   - Y-Z plane projection: integrated flux over zenith angle

Usage:
    julia --project=. examples/flux_density.jl [OPTIONS]

Options:
    --dump PATH, -d PATH      Path to physics binary dump file (default: examples/data/materials.pumas)
    --samples N, -n N         Number of Monte Carlo samples per point (default: 10000)
    --no-straggling           Disable energy straggling (default: enabled)
    --no-scattering           Disable scattering (default: enabled)
    --output PATH, -o PATH    Output path for 3D plot HTML file
                              (default: examples/data/flux_density_3d.html)
    --n-thicknesses N         Number of thickness points (default: 9, range: 0-800m)
    --n-zeniths N             Number of zenith angle points (default: 7, range: 0-60°)
    --n-energies N            Number of energy points (default: 10, range: 1e-3 to 1e9 GeV)
    --threshold FLOAT         Energy threshold for mode switching in GeV (default: 100.0)
"""

using DiffPumas
using DiffPumas.Physics: get_material_index
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: compute_flux
using PlotlyJS
using Printf

const PHYSICS_DUMP = joinpath(@__DIR__, "data", "materials.pumas")

"""
    zenith_to_elevation(zenith)

Convert zenith angle to elevation angle.
elevation = 90° - zenith
"""
zenith_to_elevation(zenith::Float64) = 90.0 - zenith

"""
    load_or_create_physics(dump_path)

Load physics tables from dump file, or create new ones if not found.
"""
function load_or_create_physics(dump_path::String)
    if isfile(dump_path)
        println("Loading physics from binary dump...")
        println("  Dump file: $(dump_path)")
        physics = load_physics(dump_path)
        if physics !== nothing
            println("✓ Physics loaded successfully")
            return physics
        end
    end
    
    println("Physics dump not found, creating new physics tables...")
    physics = create_physics(MUON; n_energies=200, K_min=1e-3, K_max=1e9)
    mkpath(dirname(dump_path))
    save_physics(physics, dump_path)
    println("✓ Physics tables created and saved to $(dump_path)")
    return physics
end

"""
    compute_flux_grid_3d(physics, thicknesses, zeniths, energies; kwargs...)

Compute flux for a 3D grid of parameters.
For each (thickness, zenith, energy) point, compute the flux at that specific energy.

# Arguments
- `physics`: Physics tables
- `thicknesses`: Vector of rock thicknesses in meters
- `zeniths`: Vector of zenith angles in degrees
- `energies`: Vector of energy values in GeV (point estimates)
- `n_samples`: Number of MC samples per point
- `straggling`: Enable energy straggling
- `scattering`: Enable scattering

# Returns
- 3D array: flux[thickness_idx, zenith_idx, energy_idx]
- 3D array: sigma[thickness_idx, zenith_idx, energy_idx]
"""
function compute_flux_grid_3d(physics,
                               thicknesses::Vector{Float64},
                               zeniths::Vector{Float64},
                               energies::Vector{Float64};
                               n_samples::Int = 10000,
                               straggling::Bool = true,
                               scattering::Bool = true,
                               energy_threshold_low::Float64 = 100.0)
    
    n_t = length(thicknesses)
    n_z = length(zeniths)
    n_e = length(energies)
    
    flux_3d = zeros(n_t, n_z, n_e)
    sigma_3d = zeros(n_t, n_z, n_e)
    
    n_total = n_t * n_z * n_e
    n_done = 0
    
    # Compute flux for each energy point (point estimate: energy_min == energy_max)
    for (e_idx, energy) in enumerate(energies)
        for (t_idx, thickness) in enumerate(thicknesses)
            for (z_idx, zenith) in enumerate(zeniths)
                n_done += 1
                elevation = zenith_to_elevation(zenith)
                
                if n_done % 10 == 0 || n_done == 1
                    @info "[$n_done/$n_total] Computing: thickness=$(thickness)m, θ=$(zenith)°, E=$(energy)GeV"
                end
                
                # Point estimate: energy_min == energy_max
                flux, sigma = compute_flux(physics, 2650.0, thickness, elevation,
                                           energy, energy; 
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
    create_3d_flux_plot(thicknesses, zeniths, energies, flux_3d)

Create interactive 3D PlotlyJS plot with:
- Main volume: flux(thickness, zenith, energy) as heatmap
  Axes: x=zenith, y=energy, z=thickness, color=flux
- X-Z projection: integrated flux over energy (zenith-thickness plane at y=0)
- Y-Z projection: integrated flux over zenith (energy-thickness plane at x=0)
"""
function create_3d_flux_plot(thicknesses::Vector{Float64},
                             zeniths::Vector{Float64},
                             energies::Vector{Float64},
                             flux_3d::Array{Float64, 3})
    
    n_t = length(thicknesses)
    n_z = length(zeniths)
    n_e = length(energies)
    
    # Compute integrated fluxes for projections
    # X-Z plane: integrated over energy (for each thickness, zenith)
    flux_xz = zeros(n_z, n_t)
    for t_idx in 1:n_t, z_idx in 1:n_z
        # Integrate over energy bins
        flux_xz[z_idx, t_idx] = sum(flux_3d[t_idx, z_idx, :])
    end
    
    # Y-Z plane: integrated over zenith (for each thickness, energy)
    flux_yz = zeros(n_e, n_t)
    for t_idx in 1:n_t, e_idx in 1:n_e
        # Integrate over zenith angles
        if n_z > 1
            flux_yz[e_idx, t_idx] = sum(flux_3d[t_idx, :, e_idx])
        else
            flux_yz[e_idx, t_idx] = flux_3d[t_idx, 1, e_idx]
        end
    end
    
    traces = PlotlyJS.GenericTrace[]
    
    # Main 3D volume: create surfaces for different energy slices
    # For a cleaner visualization, show several energy slices
    n_slices = min(5, n_e)
    slice_indices = round.(Int, range(1, n_e, length=n_slices))
    
    for (slice_idx, e_idx) in enumerate(slice_indices)
        flux_slice = zeros(n_z, n_t)
        for t_idx in 1:n_t, z_idx in 1:n_z
            flux_slice[z_idx, t_idx] = flux_3d[t_idx, z_idx, e_idx]
        end
        
        # Convert to log scale for better visualization
        flux_slice_log = [f > 0 ? log10(max(f, 1e-20)) : NaN for f in flux_slice]
        
        # Create surface at this energy level
        # x=zenith, y=energy (constant), z=thickness, color=flux
        thickness_grid = reshape(thicknesses', (1, n_t)) .* ones(n_z, 1)
        energy_grid = fill(energies[e_idx], (n_z, n_t))
        
        surface_trace = surface(
            x = zeniths,           # X axis: zenith angle
            y = energy_grid,       # Y axis: energy (constant for this slice)
            z = thickness_grid,    # Z axis: thickness
            surfacecolor = flux_slice_log,
            name = "E=$(Printf.@sprintf("%.2e", energies[e_idx]))GeV",
            colorscale = "Viridis",
            showscale = (slice_idx == 1),
            colorbar = attr(title = "log₁₀(Flux) [m⁻² s⁻¹ sr⁻¹]"),
            opacity = 0.8
        )
        push!(traces, surface_trace)
    end
    
    # X-Z plane projection: integrated flux over energy (zenith-thickness)
    # This is projected onto y=0 plane
    flux_xz_log = [f > 0 ? log10(max(f, 1e-20)) : NaN for f in flux_xz]
    
    thickness_grid_xz = reshape(thicknesses', (1, n_t)) .* ones(n_z, 1)
    energy_grid_xz = zeros(n_z, n_t)  # Project onto y=0
    
    xz_projection = surface(
        x = zeniths,               # X axis: zenith angle
        y = energy_grid_xz,        # Y axis: energy (projected to 0)
        z = thickness_grid_xz,     # Z axis: thickness
        surfacecolor = flux_xz_log,
        name = "Integrated over energy",
        colorscale = "Blues",
        showscale = false,
        opacity = 0.7
    )
    push!(traces, xz_projection)
    
    # Y-Z plane projection: integrated flux over zenith (energy-thickness)
    # This is projected onto x=0 plane
    flux_yz_log = [f > 0 ? log10(max(f, 1e-20)) : NaN for f in flux_yz]
    
    energy_grid_yz = reshape(energies, (n_e, 1)) .* ones(1, n_t)
    thickness_grid_yz = reshape(thicknesses', (1, n_t)) .* ones(n_e, 1)
    zenith_grid_yz = zeros(n_e, n_t)  # Project onto x=0
    
    yz_projection = surface(
        x = zenith_grid_yz,        # X axis: zenith (projected to 0)
        y = energy_grid_yz,        # Y axis: energy
        z = thickness_grid_yz,     # Z axis: thickness
        surfacecolor = flux_yz_log,
        name = "Integrated over zenith",
        colorscale = "Reds",
        showscale = false,
        opacity = 0.7
    )
    push!(traces, yz_projection)
    
    layout = Layout(
        title = "Muon Flux Density: 3D Parameter Space (x=zenith, y=energy, z=thickness)",
        scene = attr(
            xaxis = attr(title = "Zenith Angle θ (°)"),
            yaxis = attr(title = "Energy (GeV)", type = "log"),
            zaxis = attr(title = "Rock Thickness (m)"),
            camera = attr(eye = attr(x = 1.5, y = 1.5, z = 1.2))
        ),
        width = 1200,
        height = 900
    )
    
    plot = Plot(traces, layout)
    return plot
end

function parse_commandline()
    dump_path = PHYSICS_DUMP
    n_samples = 10000
    straggling = true
    scattering = true
    output_path = nothing
    n_thicknesses = 9
    n_zeniths = 7
    n_energies = 10
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
        elseif arg == "--output" || arg == "-o"
            if i + 1 <= length(ARGS)
                output_path = ARGS[i + 1]
                i += 2
            else
                error("--output requires a path")
            end
        elseif arg == "--n-thicknesses"
            if i + 1 <= length(ARGS)
                n_thicknesses = parse(Int, ARGS[i + 1])
                i += 2
            else
                error("--n-thicknesses requires a value")
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
        elseif arg == "--threshold"
            if i + 1 <= length(ARGS)
                energy_threshold_low = parse(Float64, ARGS[i + 1])
                i += 2
            else
                error("--threshold requires a value")
            end
        elseif arg == "--no-straggling"
            straggling = false
            i += 1
        elseif arg == "--no-scattering"
            scattering = false
            i += 1
        elseif startswith(arg, "--samples=")
            n_samples = parse(Int, split(arg, "=")[2])
            i += 1
        elseif startswith(arg, "--dump=") || startswith(arg, "-d=")
            dump_path = split(arg, "=", limit=2)[2]
            i += 1
        elseif startswith(arg, "--output=") || startswith(arg, "-o=")
            output_path = split(arg, "=", limit=2)[2]
            i += 1
        elseif startswith(arg, "--n-thicknesses=")
            n_thicknesses = parse(Int, split(arg, "=")[2])
            i += 1
        elseif startswith(arg, "--n-zeniths=")
            n_zeniths = parse(Int, split(arg, "=")[2])
            i += 1
        elseif startswith(arg, "--n-energies=")
            n_energies = parse(Int, split(arg, "=")[2])
            i += 1
        elseif startswith(arg, "--threshold=")
            energy_threshold_low = parse(Float64, split(arg, "=")[2])
            i += 1
        else
            error("Unknown argument: $arg (use --help for usage)")
        end
    end
    
    if output_path === nothing
        output_path = joinpath(@__DIR__, "data", "flux_density_3d.html")
    end
    
    return dump_path, n_samples, straggling, scattering, output_path, n_thicknesses, n_zeniths, n_energies, energy_threshold_low
end

function main()
    dump_path, n_samples, straggling, scattering, output_path, n_thicknesses, n_zeniths, n_energies, energy_threshold_low = parse_commandline()
    
    println("=" ^ 60)
    println(" DiffPumas - 3D Flux Density Calculation")
    println("=" ^ 60)
    println()
    
    println("Configuration:")
    println("  Dump file:      $(dump_path)")
    println("  MC samples:     $(n_samples)")
    println("  Straggling:     $(straggling)")
    println("  Scattering:     $(scattering)")
    println("  N thicknesses:  $(n_thicknesses)")
    println("  N zeniths:      $(n_zeniths)")
    println("  N energies:     $(n_energies)")
    println("  Energy threshold: $(energy_threshold_low) GeV")
    println("  Output:         $(output_path)")
    println()
    
    # Load or create physics tables
    physics = load_or_create_physics(dump_path)
    
    if physics === nothing
        error("Failed to load or create physics tables!")
    end
    
    print_physics_summary(physics)
    println()
    
    # Verify required materials
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1 || air_idx == -1
        error("Required materials not found in physics tables")
    end
    
    println("Material indices:")
    println("  StandardRock: $(rock_idx)")
    println("  Air: $(air_idx)")
    println()
    
    # Define parameter grid
    # Thickness: 0 to 800m
    if n_thicknesses == 1
        thicknesses = [400.0]  # Single point at middle
    else
        thicknesses = collect(range(0.0, 800.0, length=n_thicknesses))
    end
    
    # Zenith: 0° to 60°
    if n_zeniths == 1
        zeniths = [30.0]  # Single point at middle
    else
        zeniths = collect(range(0.0, 60.0, length=n_zeniths))
    end
    
    # Energies: logarithmic spacing from 1e-3 to 1e9 GeV (point estimates)
    energy_log_min = log10(1e-3)
    energy_log_max = log10(1e9)
    energy_log_values = range(energy_log_min, energy_log_max, length=n_energies)
    energies = [10.0^e for e in energy_log_values]
    
    println("Parameter grid:")
    println("  Thicknesses: $(length(thicknesses)) points from $(thicknesses[1]) to $(thicknesses[end]) m")
    println("  Zenith angles: $(length(zeniths)) points from $(zeniths[1]) to $(zeniths[end])°")
    println("  Energies: $(length(energies)) points from $(energies[1]) to $(energies[end]) GeV")
    println("  Total computations: $(length(thicknesses) * length(zeniths) * length(energies))")
    println()
    
    # Compute flux grid
    println("Computing 3D flux grid...")
    flux_3d, sigma_3d = compute_flux_grid_3d(physics, thicknesses, zeniths, energies;
                                              n_samples=n_samples,
                                              straggling=straggling,
                                              scattering=scattering,
                                              energy_threshold_low=energy_threshold_low)
    println()
    
    # Create 3D plot
    println("Creating interactive 3D plot...")
    plot = create_3d_flux_plot(thicknesses, zeniths, energies, flux_3d)
    
    # Save plot
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        PlotlyJS.savefig(io, plot, format="html")
    end
    @info "Saved 3D flux plot to: $(output_path)"
    
    # Display plot if in interactive environment
    try
        display(plot)
    catch
        # Not in interactive environment
    end
    
    println()
    println("Done!")
    println("View 3D plot: $(output_path)")
    println()
    
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
