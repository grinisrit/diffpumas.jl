#!/usr/bin/env julia
"""
muography.jl - Muon flux muography simulation with aquifer detection

This script demonstrates muography techniques:
1. Baseline flux vs zenith angle for various rock depths (0-1000m, step 100m)
   - Zenith angles: 0° to 60° in 2° steps
   - One line per depth in 2D interactive plot
2. Aquifer detection using water material in a cubic rock volume (detector at 1000m)
   Materials: Standard Rock (2650 kg/m³), Water (1000 kg/m³), Air (1.205 kg/m³).
   Shallow air: all cells within 100m of the surface get 10% air mixed in.
   - 2.1a: Water fraction sweep from 0% (pure rock) to 90% (always ≥10% rock)
            at 500m depth, 50m × 50m × 50m aquifer
   - 2.1b: 90% water aquifer, enlarge from 50m to 500m at 500m depth
   - 2.2a: Moving 90% water aquifer (50m cube) vertically from detector to surface
   - 2.2b: Moving 90% water aquifer (50m cube) along y (zenith direction) at 100m depth
3. Rock-to-water sweep with material mixtures (max 90% water) for a 200m-thick
   aquifer at z=[400,600]m

Geometry:
    - Part 1: Variable rock thickness (0-1000m)
    - Part 2 & 3: Detector at 1000m depth (z=0), rock cube above
    - Air layer above rock to PRIMARY_ALTITUDE

Usage:
    julia --project=. examples/muography.jl --output-dir PATH [OPTIONS]

Options:
    --output-dir, -o PATH     Output directory for plots (REQUIRED)
    --dump, -d PATH           Path to physics binary dump file
    --samples, -n INT         Number of MC samples per point (default: 1000)
    --threshold FLOAT         Energy threshold for mode switching (default: 100.0)
    --energy-min FLOAT        Minimum energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum energy in GeV (default: 1e9)
    --no-straggling           Disable straggling
    --no-scattering           Disable scattering
    --part INT                Run only part 1, 2, 3 or all (default: all)

Example:
    julia --project=. examples/muography.jl --output-dir examples/muography --samples 2000
"""

using DiffPumas
using DiffPumas.Physics: get_material_index, property_range, property_kinetic_energy
using DiffPumas.Physics: property_stopping_power, ENERGY_LOSS_CSDA
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: PRIMARY_ALTITUDE, compute_air_grammage, compute_flux
using DiffPumas.Geometry: compute_decay_weight_from_path, locals_air_at_altitude
using DiffPumas.Geometry: transport_backward_step, transport_backward_step_full, transport_backward_step_mixed
using DiffPumas.GaisserFlux: flux_gccly
using DiffPumas.Types: State, Vec3, MaterialMixture, is_single_material, single_material
using DiffPumas.Materials: STANDARD_ROCK, AIR, WATER
using DiffPumas: zenith_to_elevation, sample_energy_loguniform
using DiffPumas.Pumas: load_or_create_physics
using PlotlyJS
using Printf
using Random

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")

# =============================================================================
# Part 1: Baseline flux vs angle for various depths
# =============================================================================

"""
    compute_flux_vs_angle(physics, depths, zeniths; kwargs...)

Compute integrated muon flux for each depth and zenith angle.
Similar to flux_comparison.jl but without the C code comparison.
"""
function compute_flux_vs_angle(physics,
                                depths::Vector{Float64},
                                zeniths::Vector{Float64};
                                n_samples::Int = 1000,
                                straggling::Bool = true,
                                scattering::Bool = true,
                                energy_threshold_low::Float64 = 100.0,
                                energy_min::Float64 = 1e-3,
                                energy_max::Float64 = 1e9,
                                verbose::Bool = true)
    
    n_depths = length(depths)
    n_zeniths = length(zeniths)
    
    flux_grid = zeros(n_depths, n_zeniths)
    sigma_grid = zeros(n_depths, n_zeniths)
    
    n_total = n_depths * n_zeniths
    n_done = 0
    
    for (d_idx, depth) in enumerate(depths)
        for (z_idx, zenith) in enumerate(zeniths)
            n_done += 1
            elevation = zenith_to_elevation(zenith)
            
            if verbose && (n_done % 10 == 0 || n_done == 1)
                @info "[$n_done/$n_total] depth=$(depth)m, θ=$(zenith)°"
            end
            
            flux, sigma = compute_flux(physics, 2650.0, depth, elevation,
                                       energy_min, energy_max;
                                       n_samples=n_samples,
                                       straggling=straggling,
                                       scattering=scattering,
                                       energy_threshold_low=energy_threshold_low)
            
            flux_grid[d_idx, z_idx] = flux
            sigma_grid[d_idx, z_idx] = sigma
        end
    end
    
    return flux_grid, sigma_grid
end

"""
    create_flux_vs_angle_plot(depths, zeniths, flux_grid, sigma_grid; output_path)

Create interactive 2D plot: flux vs zenith angle, one line per depth.
"""
function create_flux_vs_angle_plot(depths::Vector{Float64},
                                    zeniths::Vector{Float64},
                                    flux_grid::Matrix{Float64},
                                    sigma_grid::Matrix{Float64};
                                    output_path::String,
                                    title::String = "Muon Flux vs Zenith Angle")
    
    traces = GenericTrace[]
    
    # Color scale from light to dark
    n_depths = length(depths)
    colors = ["hsl($(round(Int, 240 - 240 * i / n_depths)), 70%, 50%)" for i in 1:n_depths]
    
    for (d_idx, depth) in enumerate(depths)
        flux_values = flux_grid[d_idx, :]
        sigma_values = sigma_grid[d_idx, :]
        
        # Main line
        trace = scatter(
            x = zeniths,
            y = flux_values,
            mode = "lines+markers",
            name = "$(Int(depth))m",
            line = attr(color = colors[d_idx], width = 2),
            marker = attr(size = 6),
            error_y = attr(
                type = "data",
                array = sigma_values,
                visible = true,
                color = colors[d_idx]
            )
        )
        push!(traces, trace)
    end
    
    layout = Layout(
        title = title,
        xaxis = attr(
            title = "Zenith Angle θ (°)",
            range = [minimum(zeniths) - 1, maximum(zeniths) + 1]
        ),
        yaxis = attr(
            title = "Flux (m⁻² s⁻¹ sr⁻¹)",
            type = "log"
        ),
        legend = attr(
            title = attr(text = "Rock Depth"),
            x = 1.02,
            y = 1.0
        ),
        width = 1000,
        height = 700,
        hovermode = "closest"
    )
    
    fig = Plot(traces, layout)
    
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    
    return fig
end

# =============================================================================
# Part 2: Cube geometry with density anomalies
# =============================================================================

"""
    MuographyCube

Represents a tessellated rock cube for muography simulations.
Fixed 10m x 10m x 10m cells.
"""
struct MuographyCube{T<:Real}
    # Cube dimensions
    x_min::T
    x_max::T
    y_min::T
    y_max::T
    z_min::T  # Detector level (0)
    z_max::T  # Rock thickness
    
    # Number of cells
    nx::Int
    ny::Int
    nz::Int
    
    # Cell size (10m)
    cell_size::T
    
    # Material indices
    rock_material::Int
    air_material::Int
end

function MuographyCube(rock_thickness::T, width::T, cell_size::T,
                        rock_material::Int, air_material::Int) where T<:Real
    x_min = -width / 2
    x_max = width / 2
    y_min = -width / 2
    y_max = width / 2
    z_min = T(0)
    z_max = rock_thickness
    
    nx = Int(ceil(width / cell_size))
    ny = Int(ceil(width / cell_size))
    nz = Int(ceil(rock_thickness / cell_size))
    
    return MuographyCube{T}(x_min, x_max, y_min, y_max, z_min, z_max,
                            nx, ny, nz, cell_size, rock_material, air_material)
end

"""Get cell indices from position."""
function get_cell_indices(cube::MuographyCube{T}, x::T, y::T, z::T) where T<:Real
    if x < cube.x_min || x > cube.x_max ||
       y < cube.y_min || y > cube.y_max ||
       z < cube.z_min || z > cube.z_max
        return nothing
    end
    
    i = clamp(floor(Int, (x - cube.x_min) / cube.cell_size) + 1, 1, cube.nx)
    j = clamp(floor(Int, (y - cube.y_min) / cube.cell_size) + 1, 1, cube.ny)
    k = clamp(floor(Int, (z - cube.z_min) / cube.cell_size) + 1, 1, cube.nz)
    
    return (i, j, k)
end

"""Linear index from (i, j, k)."""
@inline function cell_linear_index(cube::MuographyCube, i::Int, j::Int, k::Int)
    return i + (j - 1) * cube.nx + (k - 1) * cube.nx * cube.ny
end

"""Total number of cells."""
@inline function num_cells(cube::MuographyCube)
    return cube.nx * cube.ny * cube.nz
end

"""Get cell center coordinates."""
function get_cell_center(cube::MuographyCube{T}, i::Int, j::Int, k::Int) where T<:Real
    x = cube.x_min + (i - 0.5) * cube.cell_size
    y = cube.y_min + (j - 0.5) * cube.cell_size
    z = cube.z_min + (k - 0.5) * cube.cell_size
    return (x, y, z)
end

"""Compute step to next cell boundary."""
function compute_step_to_cell_boundary(cube::MuographyCube{T}, x::T, y::T, z::T,
                                        ux::T, uy::T, uz::T,
                                        i::Int, j::Int, k::Int) where T<:Real
    
    eps_val = T(1e-10)
    step_min = T(Inf)
    
    x_lo = cube.x_min + (i - 1) * cube.cell_size
    x_hi = cube.x_min + i * cube.cell_size
    y_lo = cube.y_min + (j - 1) * cube.cell_size
    y_hi = cube.y_min + j * cube.cell_size
    z_lo = cube.z_min + (k - 1) * cube.cell_size
    z_hi = cube.z_min + k * cube.cell_size
    
    # X boundaries
    if abs(ux) > eps_val
        t = ux > 0 ? (x_hi - x) / ux : (x_lo - x) / ux
        if t > eps_val
            step_min = min(step_min, t)
        end
    end
    
    # Y boundaries
    if abs(uy) > eps_val
        t = uy > 0 ? (y_hi - y) / uy : (y_lo - y) / uy
        if t > eps_val
            step_min = min(step_min, t)
        end
    end
    
    # Z boundaries
    if abs(uz) > eps_val
        t = uz > 0 ? (z_hi - z) / uz : (z_lo - z) / uz
        if t > eps_val
            step_min = min(step_min, t)
        end
    end
    
    return step_min == T(Inf) ? T(1000.0) : step_min
end

"""
    compute_flux_cube_single(physics, cube, cell_densities, elevation, energy_final, charge; kwargs...)

Compute single flux contribution through the cube.
Optionally accepts `cell_materials` to specify a `MaterialMixture` per cell.
When `cell_materials` is `nothing`, uses `cube.rock_material` for all cells.
"""
function compute_flux_cube_single(physics, cube::MuographyCube{T},
                                   cell_densities::Vector{T},
                                   elevation::T,
                                   energy_final::T,
                                   charge::T;
                                   rng::AbstractRNG,
                                   straggling::Bool=true,
                                   scattering::Bool=true,
                                   energy_threshold_low::T=T(100.0),
                                   cell_materials::Union{Vector{MaterialMixture}, Nothing}=nothing) where T<:Real
    
    theta = (T(90) - elevation) * T(π) / T(180)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    
    # Fix azimuth to 0 to keep trajectory in x=0 plane (along y-axis)
    # With azimuth = 0: dir_x = 0, dir_y = sin_theta, dir_z = cos_theta
    # This ensures the trajectory stays in the x=0 plane when there's no scattering
    azimuth = zero(T)
    
    dir_x = sin_theta * sin(azimuth)  # = 0 (stays in x=0 plane)
    dir_y = sin_theta * cos(azimuth)  # = sin_theta (along y-axis)
    dir_z = cos_theta
    
    state = State{T}(
        charge = charge,
        energy = energy_final,
        distance = zero(T),
        grammage = zero(T),
        time = zero(T),
        weight = one(T),
        position = Vec3{T}(zero(T), zero(T), zero(T)),
        direction = Vec3{T}(-dir_x, -dir_y, -dir_z),
        decayed = false
    )
    
    energy_threshold = T(1e12)
    STEP_EPSILON = T(1e-7)
    STEP_MIN = T(1e-6)
    
    max_outer_steps = 1000
    outer_step = 0
    
    while outer_step < max_outer_steps && state.energy < energy_threshold - eps(T)
        outer_step += 1
        
        x, y, z = state.position[1], state.position[2], state.position[3]
        
        if z < zero(T) || z >= PRIMARY_ALTITUDE - eps(T)
            break
        end
        
        if state.weight <= zero(T) || !isfinite(state.weight)
            break
        end
        
        if state.energy < energy_threshold_low - eps(T)
            use_straggled = straggling
            use_scattering = scattering
            current_energy_limit = energy_threshold_low
        else
            use_straggled = false
            use_scattering = false
            current_energy_limit = energy_threshold
        end
        
        inner_steps = 0
        max_inner_steps = 50000
        
        while inner_steps < max_inner_steps
            inner_steps += 1
            x, y, z = state.position[1], state.position[2], state.position[3]
            
            if z < zero(T) || z >= PRIMARY_ALTITUDE - eps(T)
                break
            end
            
            if state.energy >= current_energy_limit - eps(T)
                break
            end
            
            if state.weight <= zero(T) || !isfinite(state.weight)
                break
            end
            
            ux_motion = -state.direction[1]
            uy_motion = -state.direction[2]
            uz_motion = -state.direction[3]
            
            # Determine material (Int or MaterialMixture) and density for current position
            mix::MaterialMixture = MaterialMixture(cube.rock_material)
            density = cell_densities[1]
            step_to_boundary = T(1000.0)

            if z < cube.z_max
                cell_idx = get_cell_indices(cube, x, y, z)
                
                if cell_idx === nothing
                    mix = cell_materials !== nothing ? MaterialMixture(cube.rock_material) : MaterialMixture(cube.rock_material)
                    density = cell_densities[1]
                    
                    if uz_motion > eps(T)
                        step_to_boundary = min(step_to_boundary, (cube.z_max - z) / uz_motion)
                    elseif uz_motion < -eps(T)
                        step_to_boundary = min(step_to_boundary, -z / uz_motion)
                    end
                else
                    i, j, k = cell_idx
                    linear_idx = cell_linear_index(cube, i, j, k)
                    mix = cell_materials !== nothing ? cell_materials[linear_idx] : MaterialMixture(cube.rock_material)
                    density = cell_densities[linear_idx]
                    
                    step_to_boundary = compute_step_to_cell_boundary(cube, x, y, z,
                                                                      ux_motion, uy_motion, uz_motion,
                                                                      i, j, k)
                end
            else
                mix = MaterialMixture(cube.air_material)
                density = locals_air_at_altitude(z).density
                
                if uz_motion > eps(T)
                    step_to_boundary = (PRIMARY_ALTITUDE - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = (cube.z_max - z) / uz_motion
                end
            end
            
            Xi = property_range(physics, ENERGY_LOSS_CSDA, mix, state.energy)
            max_grammage_frac = T(0.01)
            max_grammage_step = max_grammage_frac * Xi
            max_geometric_step = max_grammage_step / max(density, T(1e-6))
            
            if z >= cube.z_max
                h = T(12e3)
                uz_abs = max(abs(uz_motion), T(0.05))
                altitude_step = T(0.05) * h / uz_abs
                max_geometric_step = min(max_geometric_step, altitude_step)
            end
            
            step_size = min(step_to_boundary + STEP_EPSILON, max_geometric_step)
            step_size = clamp(step_size, STEP_MIN, T(1000.0))
            
            if state.energy < energy_threshold_low - eps(T)
                if use_straggled && use_scattering
                    state, _ = transport_backward_step_full(physics, state, mix, density, step_size, rng;
                                                            mode=:straggled, scattering=true,
                                                            energy_limit=current_energy_limit)
                elseif use_straggled
                    state = transport_backward_step(physics, state, mix, density, step_size;
                                                    rng=rng, straggling=true)
                else
                    state = transport_backward_step(physics, state, mix, density, step_size;
                                                    rng=nothing, straggling=false)
                end
            else
                if straggling
                    state = transport_backward_step_mixed(physics, state, mix, density, step_size, rng;
                                                          energy_limit=current_energy_limit)
                else
                    state = transport_backward_step(physics, state, mix, density, step_size;
                                                    rng=nothing, straggling=false)
                end
            end
        end
    end
    
    if state.position[3] >= PRIMARY_ALTITUDE - T(1.0)
        cos_theta_final = -state.direction[3]
        cos_theta_final = clamp(cos_theta_final, zero(T), one(T))
        
        if cos_theta_final <= zero(T)
            return zero(T)
        end
        
        flux = flux_gccly(cos_theta_final, state.energy, charge)
        return state.weight * flux
    else
        return zero(T)
    end
end

"""
    compute_cube_flux_vs_angle(physics, cube, cell_densities, zeniths; kwargs...)

Compute flux vs zenith angle through the cube geometry.
"""
function compute_cube_flux_vs_angle(physics, cube::MuographyCube{T},
                                     cell_densities::Vector{T},
                                     zeniths::Vector{Float64};
                                     n_samples::Int = 1000,
                                     straggling::Bool = true,
                                     scattering::Bool = true,
                                     energy_threshold_low::Float64 = 100.0,
                                     energy_min::Float64 = 1e-3,
                                     energy_max::Float64 = 1e9,
                                     verbose::Bool = true,
                                     cell_materials::Union{Vector{MaterialMixture}, Nothing} = nothing) where T<:Real
    
    rng = MersenneTwister(42)
    n_zeniths = length(zeniths)
    
    flux_values = zeros(n_zeniths)
    sigma_values = zeros(n_zeniths)
    
    for (z_idx, zenith) in enumerate(zeniths)
        elevation = T(90) - zenith
        
        if verbose
            @info "Computing zenith=$(zenith)°..."
        end
        
        w_sum = zero(T)
        w2_sum = zero(T)
        
        for _ in 1:n_samples
            kf, w_energy = sample_energy_loguniform(Float64(energy_min), Float64(energy_max), rng)
            charge = rand(rng) > 0.5 ? T(1) : T(-1)
            
            flux_single = compute_flux_cube_single(physics, cube, cell_densities,
                                                    elevation, T(kf), charge;
                                                    rng=rng, straggling=straggling, scattering=scattering,
                                                    energy_threshold_low=T(energy_threshold_low),
                                                    cell_materials=cell_materials)
            
            wi = T(2) * T(w_energy) * flux_single
            w_sum += wi
            w2_sum += wi * wi
        end
        
        n = T(n_samples)
        flux_values[z_idx] = w_sum / n
        variance = (w2_sum / n - (w_sum / n)^2) / max(one(T), n - one(T))
        sigma_values[z_idx] = sqrt(max(zero(T), variance))
    end
    
    return flux_values, sigma_values
end

"""
    create_cell_config(cube, rock_idx, water_idx, air_idx,
                       aquifer_center, aquifer_half_size, water_fraction;
                       shallow_air_fraction=0.10, shallow_depth=100.0,
                       rock_density=2650.0, water_density=1000.0,
                       air_density_surface=1.205)

Create per-cell density and MaterialMixture arrays.

The aquifer region (box around `aquifer_center` with `aquifer_half_size`) has its
rock replaced by `water_fraction` water (by mass).  Everywhere within
`shallow_depth` of the surface (z > rock_thickness - shallow_depth) an additional
`shallow_air_fraction` of air is mixed in (both inside and outside the aquifer).

Effective density per cell: sum(f_i * rho_i).

# Returns
- `(densities::Vector{T}, materials::Vector{MaterialMixture})`
"""
function create_cell_config(cube::MuographyCube{T},
                            rock_idx::Int, water_idx::Int, air_idx::Int,
                            aquifer_center::Tuple{T,T,T},
                            aquifer_half_size::Tuple{T,T,T},
                            water_fraction::T;
                            shallow_air_fraction::T = T(0.10),
                            shallow_depth::T = T(100),
                            rock_density::T = T(2650),
                            water_density::T = T(1000),
                            air_density_surface::T = T(1.205)) where T<:Real

    n = num_cells(cube)
    densities = fill(rock_density, n)
    materials = fill(MaterialMixture(rock_idx), n)

    ax, ay, az = aquifer_center
    hx, hy, hz = aquifer_half_size
    shallow_z_threshold = cube.z_max - shallow_depth

    for k in 1:cube.nz
        for j in 1:cube.ny
            for i in 1:cube.nx
                cx, cy, cz = get_cell_center(cube, i, j, k)
                idx = cell_linear_index(cube, i, j, k)

                in_aquifer = abs(cx - ax) <= hx && abs(cy - ay) <= hy && abs(cz - az) <= hz
                in_shallow = shallow_depth > zero(T) && cz >= shallow_z_threshold

                # Start with pure rock
                rock_frac = one(T)
                water_frac = zero(T)
                air_frac = zero(T)

                # Replace rock with water inside aquifer (always keep ≥10% rock)
                if in_aquifer
                    wf_clamped = min(water_fraction, T(0.9))
                    water_frac = wf_clamped
                    rock_frac = one(T) - wf_clamped
                end

                # Add shallow air everywhere in the shallow zone
                if in_shallow
                    air_frac = shallow_air_fraction
                    remaining = one(T) - air_frac
                    rock_frac *= remaining
                    water_frac *= remaining
                end

                # Build the mixture and effective density
                if water_frac > zero(T) && air_frac > zero(T)
                    materials[idx] = MaterialMixture(
                        [rock_idx, water_idx, air_idx],
                        [rock_frac, water_frac, air_frac])
                    densities[idx] = rock_frac * rock_density +
                                     water_frac * water_density +
                                     air_frac * air_density_surface
                elseif water_frac > zero(T)
                    materials[idx] = MaterialMixture(
                        [rock_idx, water_idx],
                        [rock_frac, water_frac])
                    densities[idx] = rock_frac * rock_density +
                                     water_frac * water_density
                elseif air_frac > zero(T)
                    materials[idx] = MaterialMixture(
                        [rock_idx, air_idx],
                        [rock_frac, air_frac])
                    densities[idx] = rock_frac * rock_density +
                                     air_frac * air_density_surface
                end
                # else: stays pure rock at rock_density (the default)
            end
        end
    end

    return densities, materials
end

"""
Create multi-line plot for aquifer scenarios.
"""
function create_aquifer_plot(zeniths::Vector{Float64},
                              scenarios::Vector{Tuple{String, Vector{Float64}, Vector{Float64}}};
                              output_path::String,
                              title::String)
    
    traces = GenericTrace[]
    
    n_scenarios = length(scenarios)
    colors = ["hsl($(round(Int, 360 * i / n_scenarios)), 70%, 50%)" for i in 0:n_scenarios-1]
    
    for (idx, (name, flux, sigma)) in enumerate(scenarios)
        trace = scatter(
            x = zeniths,
            y = flux,
            mode = "lines+markers",
            name = name,
            line = attr(color = colors[idx], width = 2),
            marker = attr(size = 6),
            error_y = attr(
                type = "data",
                array = sigma,
                visible = true,
                color = colors[idx]
            )
        )
        push!(traces, trace)
    end
    
    layout = Layout(
        title = title,
        xaxis = attr(
            title = "Zenith Angle θ (°)",
            range = [minimum(zeniths) - 1, maximum(zeniths) + 1]
        ),
        yaxis = attr(
            title = "Flux (m⁻² s⁻¹ sr⁻¹)",
            type = "log"
        ),
        legend = attr(
            x = 1.02,
            y = 1.0
        ),
        width = 1100,
        height = 700,
        hovermode = "closest"
    )
    
    fig = Plot(traces, layout)
    
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    
    return fig
end

# =============================================================================
# Part 2.1: Water-fraction sweep and hole enlargement
# =============================================================================

"""
Run Part 2.1: Water-fraction sweep and hole enlargement.

Part 2.1a: Fixed 50m aquifer at 500m depth; sweep water fraction from 0% to 100%.
Part 2.1b: 90% water aquifer; enlarge from 50m to 500m.

Shallow air (10%) is mixed into all cells within 100m of the surface.
"""
function run_part2_1(physics, rock_idx::Int, air_idx::Int, water_idx::Int;
                      n_samples::Int = 1000,
                      straggling::Bool = true,
                      scattering::Bool = true,
                      energy_threshold_low::Float64 = 100.0,
                      energy_min::Float64 = 1e-3,
                      energy_max::Float64 = 1e9,
                      output_dir::String)
    
    println()
    println("=" ^ 60)
    println(" Part 2.1: Water-Fraction Sweep and Hole Enlargement")
    println("=" ^ 60)
    println()
    
    rock_thickness = 1000.0
    cube_width = 500.0
    cell_size = 10.0
    
    cube = MuographyCube(rock_thickness, cube_width, cell_size, rock_idx, air_idx)
    
    println("Cube geometry:")
    println("  Rock thickness: $(rock_thickness)m")
    println("  Cube width: $(cube_width)m")
    println("  Cell size: $(cell_size)m")
    println("  Grid: $(cube.nx) × $(cube.ny) × $(cube.nz)")
    println()
    
    zeniths = collect(0.0:2.0:60.0)
    
    # Aquifer at 500m depth: z=[450, 500], centered at z=475
    aquifer_depth = 500.0
    aquifer_hs = (25.0, 25.0, 25.0)  # 50m cube
    aquifer_center = (0.0, 0.0, aquifer_depth - aquifer_hs[3])  # center at (0,0,475)
    
    # -------------------------------------------------------------------------
    # Part 2.1a: Sweep water fraction from 0% (pure rock) to 90% (max water)
    # -------------------------------------------------------------------------
    println("Part 2.1a: Water-fraction sweep at 500m depth (50m × 50m × 50m)")
    println("-" ^ 50)
    
    scenarios_water = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    
    for water_pct in 0:10:90
        wf = Float64(water_pct) / 100.0
        
        densities, cell_mats = create_cell_config(
            cube, rock_idx, water_idx, air_idx,
            aquifer_center, aquifer_hs, wf)
        
        label = if water_pct == 0
            "Baseline (100% rock)"
        else
            "$(water_pct)% water"
        end
        
        @info "Computing $label..."
        flux, sigma = compute_cube_flux_vs_angle(
            physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
            cell_materials=cell_mats)
        
        push!(scenarios_water, (label, flux, sigma))
    end
    
    create_aquifer_plot(zeniths, scenarios_water;
                        output_path=joinpath(output_dir, "part2_1a_water_fraction.html"),
                        title="Part 2.1a: Flux vs Angle - Water Fraction at 500m<br>" *
                              "<sub>50m cube, 0-90% water (≥10% rock), shallow 10% air above 900m</sub>")
    
    # -------------------------------------------------------------------------
    # Part 2.1b: 90% water aquifer, enlarge from 50m to 500m
    # -------------------------------------------------------------------------
    println()
    println("Part 2.1b: Hole enlargement with 90% water aquifer")
    println("-" ^ 50)
    
    scenarios_hole = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    
    # Re-use baseline from sweep
    densities_bl, mats_bl = create_cell_config(
        cube, rock_idx, water_idx, air_idx,
        aquifer_center, aquifer_hs, 0.0)  # 0% water = baseline
    flux_bl, sigma_bl = compute_cube_flux_vs_angle(
        physics, cube, densities_bl, zeniths;
        n_samples=n_samples, straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
        cell_materials=mats_bl)
    push!(scenarios_hole, ("Baseline", flux_bl, sigma_bl))
    
    # Enlarge: half-size from 25m to 250m.  Aquifer grows downward from 500m.
    for half_size in 25.0:25.0:250.0
        size = 2 * half_size
        center_z = aquifer_depth - half_size         # top stays at 500m
        hs = (25.0, half_size, half_size)             # x stays 50m; y & z grow
        center = (0.0, 0.0, center_z)
        
        densities, cell_mats = create_cell_config(
            cube, rock_idx, water_idx, air_idx,
            center, hs, 0.9)  # 90% water (10% rock retained)
        
        @info "Computing hole=$(Int(size))m (90% water)..."
        flux, sigma = compute_cube_flux_vs_angle(
            physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
            cell_materials=cell_mats)
        
        push!(scenarios_hole, ("$(Int(size))m aquifer", flux, sigma))
    end
    
    create_aquifer_plot(zeniths, scenarios_hole;
                        output_path=joinpath(output_dir, "part2_1b_hole_enlargement.html"),
                        title="Part 2.1b: Flux vs Angle - 90% Water Aquifer Enlargement<br>" *
                              "<sub>Top at 500m, enlarging downward; shallow 10% air above 900m</sub>")
    
    println()
    println("Part 2.1 complete!")
end

# =============================================================================
# Part 2.2: Moving aquifer position
# =============================================================================

"""
Run Part 2.2: Move a 90% water aquifer vertically and horizontally.

Geometry note: In backward MC the particle starts at (0, 0, 0) and the motion
direction is (+dir_x, +dir_y, +dir_z) where dir_y = sin(theta) > 0 for
positive zenith angles.  So the muon trajectory crosses positive-y territory.
A muon at zenith angle theta and altitude z crosses the y-axis at
    y(z) = z * tan(theta)
Part 2.2b places the aquifer at 100m depth and sweeps it along y so the flux
bump scans across zenith angles.

Shallow air (10%) is mixed into all cells within 100m of the surface.
"""
function run_part2_2(physics, rock_idx::Int, air_idx::Int, water_idx::Int;
                      n_samples::Int = 1000,
                      straggling::Bool = true,
                      scattering::Bool = true,
                      energy_threshold_low::Float64 = 100.0,
                      energy_min::Float64 = 1e-3,
                      energy_max::Float64 = 1e9,
                      output_dir::String)
    
    println()
    println("=" ^ 60)
    println(" Part 2.2: Moving 90% Water Aquifer")
    println("=" ^ 60)
    println()
    
    rock_thickness = 1000.0
    cube_width = 1100.0  # Wide enough for horizontal displacement
    cell_size = 10.0
    
    cube = MuographyCube(rock_thickness, cube_width, cell_size, rock_idx, air_idx)
    
    println("Cube geometry:")
    println("  Rock thickness: $(rock_thickness)m")
    println("  Cube width: $(cube_width)m")
    println("  Cell size: $(cell_size)m")
    println("  Grid: $(cube.nx) × $(cube.ny) × $(cube.nz)")
    println()
    
    zeniths = collect(0.0:2.0:60.0)
    aquifer_hs = (25.0, 25.0, 25.0)  # 50m cube
    
    # ---- Baseline (no aquifer, with shallow air) ----------------------------
    densities_bl, mats_bl = create_cell_config(
        cube, rock_idx, water_idx, air_idx,
        (0.0, 0.0, -1000.0),  # put aquifer outside the cube (no effect)
        aquifer_hs, 0.0)
    flux_baseline, sigma_baseline = compute_cube_flux_vs_angle(
        physics, cube, densities_bl, zeniths;
        n_samples=n_samples, straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
        cell_materials=mats_bl)
    
    # -------------------------------------------------------------------------
    # Part 2.2a: Move 90% water aquifer vertically (y=0, x=0)
    # z from 25m to 975m (i.e. depth 50m to 1000m)
    # -------------------------------------------------------------------------
    println("Part 2.2a: Moving 90% water aquifer vertically (on-axis)")
    println("-" ^ 50)
    
    scenarios_vert = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    push!(scenarios_vert, ("Baseline", flux_baseline, sigma_baseline))
    
    for depth in 100.0:100.0:1000.0
        center_z = depth - aquifer_hs[3]  # top of aquifer at `depth`
        center = (0.0, 0.0, center_z)
        
        densities, cell_mats = create_cell_config(
            cube, rock_idx, water_idx, air_idx,
            center, aquifer_hs, 0.9)  # 90% water (10% rock retained)
        
        @info "Computing depth=$(Int(depth))m  (z=[$(Int(depth-50)),$(Int(depth))]m) ..."
        flux, sigma = compute_cube_flux_vs_angle(
            physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
            cell_materials=cell_mats)
        
        push!(scenarios_vert, ("depth=$(Int(depth))m", flux, sigma))
    end
    
    create_aquifer_plot(zeniths, scenarios_vert;
                        output_path=joinpath(output_dir, "part2_2a_vertical_movement.html"),
                        title="Part 2.2a: Flux vs Angle - 90% Water Aquifer Moving Vertically<br>" *
                              "<sub>50m cube, on-axis (x=y=0); shallow 10% air above 900m</sub>")
    
    # -------------------------------------------------------------------------
    # Part 2.2b: Aquifer at z=[50,100] (100m depth), move along +y
    #
    # A muon at zenith θ crosses altitude z=75 at y = 75 * tan(θ).
    # By placing the aquifer at that y we expect a flux bump near θ.
    # Sweep y_center from 0 to ~150m so the bump scans across angles.
    # -------------------------------------------------------------------------
    println()
    println("Part 2.2b: Moving 90% water aquifer along y (zenith direction) at 100m depth")
    println("-" ^ 50)
    
    scenarios_horiz = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    push!(scenarios_horiz, ("Baseline", flux_baseline, sigma_baseline))
    
    aquifer_z_center = 75.0  # z=[50,100]
    
    for y_center in 0.0:25.0:150.0
        center = (0.0, y_center, aquifer_z_center)
        
        # Which zenith angle does this correspond to?
        theta_expected = atand(y_center / aquifer_z_center)
        
        densities, cell_mats = create_cell_config(
            cube, rock_idx, water_idx, air_idx,
            center, aquifer_hs, 0.9)  # 90% water (10% rock retained)
        
        @info "Computing y_center=$(Int(y_center))m (θ≈$(@sprintf("%.0f",theta_expected))°) ..."
        flux, sigma = compute_cube_flux_vs_angle(
            physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
            cell_materials=cell_mats)
        
        push!(scenarios_horiz, ("y=$(Int(y_center))m (θ≈$(@sprintf("%.0f",theta_expected))°)", flux, sigma))
    end
    
    create_aquifer_plot(zeniths, scenarios_horiz;
                        output_path=joinpath(output_dir, "part2_2b_horizontal_movement.html"),
                        title="Part 2.2b: Flux vs Angle - 90% Water Aquifer Moving Along y<br>" *
                              "<sub>50m cube at z=[50,100]m (100m depth); shallow 10% air above 900m</sub>")
    
    println()
    println("Part 2.2 complete!")
end

# =============================================================================
# Part 3: Rock-to-Water Sweep with Shallow Air Mix (Material Mixtures)
# =============================================================================

"""
Run Part 3: Aquifer with rock-to-water material sweep.
Water fraction goes from 0% (100% rock) to 90% (max water, 10% rock retained).
200m-thick aquifer at z=[400,600]m.  Shallow 10% air above 900m (everywhere).
Uses `create_cell_config` (same as Part 2).
"""
function run_part3(physics, rock_idx::Int, air_idx::Int, water_idx::Int;
                    n_samples::Int = 1000,
                    straggling::Bool = true,
                    scattering::Bool = true,
                    energy_threshold_low::Float64 = 100.0,
                    energy_min::Float64 = 1e-3,
                    energy_max::Float64 = 1e9,
                    output_dir::String)

    println()
    println("=" ^ 60)
    println(" Part 3: Rock-to-Water Sweep with Material Mixtures")
    println("=" ^ 60)
    println()

    rock_thickness = 1000.0
    cube_width = 500.0
    cell_size = 10.0

    cube = MuographyCube(rock_thickness, cube_width, cell_size, rock_idx, air_idx)

    println("Cube geometry:")
    println("  Rock thickness: $(rock_thickness)m")
    println("  Cube width: $(cube_width)m")
    println("  Cell size: $(cell_size)m")
    println("  Grid: $(cube.nx) x $(cube.ny) x $(cube.nz)")
    println()

    zeniths = collect(0.0:2.0:60.0)

    # Aquifer occupies z = 400m to 600m (200m thick centred at 500m)
    # Represented as a box: center z=500, half-size z=100
    aquifer_center = (0.0, 0.0, 500.0)
    aquifer_hs = (250.0, 250.0, 100.0)  # covers full cube width, 200m in z

    # -------------------------------------------------------------------------
    # Sweep from 0% water (pure rock) to 90% water in 10% steps
    # -------------------------------------------------------------------------
    scenarios = Tuple{String, Vector{Float64}, Vector{Float64}}[]

    for water_pct in 0:10:90
        water_frac = Float64(water_pct) / 100.0

        densities, cell_mats = create_cell_config(
            cube, rock_idx, water_idx, air_idx,
            aquifer_center, aquifer_hs, water_frac)

        label = if water_pct == 0
            "100% rock (baseline)"
        else
            "$(100 - water_pct)% rock + $(water_pct)% water"
        end

        @info "Computing mixture: $label ..."
        flux, sigma = compute_cube_flux_vs_angle(
            physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min, energy_max=energy_max,
            cell_materials=cell_mats)

        push!(scenarios, (label, flux, sigma))
    end

    create_aquifer_plot(zeniths, scenarios;
                        output_path=joinpath(output_dir, "part3_rock_to_water_sweep.html"),
                        title="Part 3: Aquifer Rock-to-Water Sweep (max 90% water)<br>" *
                              "<sub>z=[400,600]m, shallow (z>900m) gets 10% air (everywhere)</sub>")

    println()
    println("Part 3 complete!")
end

# =============================================================================
# Main
# =============================================================================

function parse_commandline()
    dump_path = DEFAULT_DUMP
    n_samples = 1000
    energy_threshold_low = 100.0
    energy_min = 1e-3
    energy_max = 1e9
    straggling = true
    scattering = true
    output_dir = nothing  # Required, must be provided via CLI
    part = 0  # 0 = all
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--dump" || arg == "-d"
            i + 1 <= length(ARGS) || error("--dump requires a path")
            dump_path = ARGS[i + 1]
            i += 2
        elseif arg == "--samples" || arg == "-n"
            i + 1 <= length(ARGS) || error("--samples requires a value")
            n_samples = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--threshold"
            i + 1 <= length(ARGS) || error("--threshold requires a value")
            energy_threshold_low = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--energy-min"
            i + 1 <= length(ARGS) || error("--energy-min requires a value")
            energy_min = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--energy-max"
            i + 1 <= length(ARGS) || error("--energy-max requires a value")
            energy_max = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--no-straggling"
            straggling = false
            i += 1
        elseif arg == "--no-scattering"
            scattering = false
            i += 1
        elseif arg == "--output-dir" || arg == "-o"
            i + 1 <= length(ARGS) || error("--output-dir requires a path")
            output_dir = ARGS[i + 1]
            i += 2
        elseif arg == "--part"
            i + 1 <= length(ARGS) || error("--part requires a value")
            part = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--help" || arg == "-h"
            println(@doc muography)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end
    
    # Validate required arguments
    if output_dir === nothing
        error("--output-dir is required. Please specify the output directory for plots.")
    end
    
    return (
        dump_path = dump_path,
        n_samples = n_samples,
        energy_threshold_low = energy_threshold_low,
        energy_min = energy_min,
        energy_max = energy_max,
        straggling = straggling,
        scattering = scattering,
        output_dir = output_dir,
        part = part
    )
end

function main()
    args = parse_commandline()
    
    println("=" ^ 60)
    println(" DiffPumas - Muography Simulation")
    println("=" ^ 60)
    println()
    
    strag_str = args.straggling ? "enabled" : "disabled"
    scat_str = args.scattering ? "enabled" : "disabled"
    
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  MC samples:      $(args.n_samples)")
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println("  Energy range:    $(args.energy_min) - $(args.energy_max) GeV")
    println("  Straggling:      $strag_str")
    println("  Scattering:      $scat_str")
    println("  Output dir:      $(args.output_dir)")
    part_str = args.part == 0 ? "all" : string(args.part)
    println("  Running part:    $part_str")
    println()
    
    # Create output directory
    mkpath(args.output_dir)
    
    # Load physics
    physics = load_or_create_physics(args.dump_path)
    if physics === nothing
        println("ERROR: Failed to load physics")
        return 1
    end
    print_physics_summary(physics)
    println()
    
    # Get material indices
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    water_idx = get_material_index(physics, "Water")
    
    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials (StandardRock, Air) not found")
        return 1
    end
    if water_idx == -1
        println("WARNING: Water material not found; Parts 2 and 3 will be skipped")
    end
    
    println("Material indices: rock=$rock_idx, air=$air_idx, water=$water_idx")
    println()
    
    # =========================================================================
    # Part 1: Baseline flux vs angle for various depths
    # =========================================================================
    if args.part == 0 || args.part == 1
        println("=" ^ 60)
        println(" Part 1: Baseline Flux vs Zenith Angle")
        println("=" ^ 60)
        println()
        
        # Depths from 0m to 1000m in 100m steps
        depths = collect(0.0:100.0:1000.0)
        # Zenith angles from 0° to 45° in 5° steps
        zeniths = collect(0.0:2.0:60.0)
        
        println("Computing flux for $(length(depths)) depths × $(length(zeniths)) angles...")
        println("  Depths: $(Int.(depths))m")
        println("  Zeniths: $(zeniths)°")
        println()
        
        flux_grid, sigma_grid = compute_flux_vs_angle(
            physics, depths, zeniths;
            n_samples=args.n_samples,
            straggling=args.straggling,
            scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low,
            energy_min=args.energy_min,
            energy_max=args.energy_max
        )
        
        # Create plot
        create_flux_vs_angle_plot(depths, zeniths, flux_grid, sigma_grid;
                                   output_path=joinpath(args.output_dir, "part1_flux_vs_angle.html"),
                                   title="Part 1: Muon Flux vs Zenith Angle by Rock Depth")
        
        println()
        println("Part 1 complete!")
    end
    
    # =========================================================================
    # Part 2: Water aquifer detection (requires Water material)
    # =========================================================================
    if (args.part == 0 || args.part == 2) && water_idx != -1
        run_part2_1(physics, rock_idx, air_idx, water_idx;
                    n_samples=args.n_samples,
                    straggling=args.straggling,
                    scattering=args.scattering,
                    energy_threshold_low=args.energy_threshold_low,
                    energy_min=args.energy_min,
                    energy_max=args.energy_max,
                    output_dir=args.output_dir)
        
        run_part2_2(physics, rock_idx, air_idx, water_idx;
                    n_samples=args.n_samples,
                    straggling=args.straggling,
                    scattering=args.scattering,
                    energy_threshold_low=args.energy_threshold_low,
                    energy_min=args.energy_min,
                    energy_max=args.energy_max,
                    output_dir=args.output_dir)
    end
    
    # =========================================================================
    # Part 3: Rock-to-Water Sweep with Material Mixtures
    # =========================================================================
    if (args.part == 0 || args.part == 3) && water_idx != -1
        run_part3(physics, rock_idx, air_idx, water_idx;
                  n_samples=args.n_samples,
                  straggling=args.straggling,
                  scattering=args.scattering,
                  energy_threshold_low=args.energy_threshold_low,
                  energy_min=args.energy_min,
                  energy_max=args.energy_max,
                  output_dir=args.output_dir)
    end
    
    println()
    println("=" ^ 60)
    println(" All simulations complete!")
    println("=" ^ 60)
    println()
    println("Output files saved to: $(args.output_dir)")
    println()
    
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
