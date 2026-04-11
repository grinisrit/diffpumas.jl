#!/usr/bin/env julia
"""
Diff Flux Cube - Backward Monte Carlo with Tessellated Rock Volume

This example demonstrates:
1. Creating a tessellated cube representing the rock volume with cells
2. Each cell has a density parameter (all initially set to standard rock density)
3. Running detailed transport simulation through the cube + air layer
4. Running direct CSDA simulation and computing gradients w.r.t. cell densities
5. Visualizing the cube with transparency and coloring cells by gradient magnitude

Geometry:

                    PRIMARY_ALTITUDE (1000m)
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                    (exponential density)
                         |
    ──────────────────────────────────────── z = rock_thickness
                         |  ┌─────────────┐
                    ROCK │  │ Cell Grid   │
                    CUBE │  │ (nx×ny×nz)  │
                         │  └─────────────┘
                         |╱ θ (zenith)
    ─────────────────────●─────────────────── z = 0 (Detector)
                      Detector

The cube is tessellated into cells, each with its own density.
We compute ∂flux/∂ρ_i for cells adjacent to the air interface (top layer).

Usage:
    julia --project=. examples/diff_flux_cube.jl [OPTIONS]

Options:
    --dump, -d PATH           Path to physics binary dump file
    --thickness, -t FLOAT     Rock thickness in meters (default: 100.0)
    --density FLOAT           Default rock density in kg/m³ (default: 2650.0)
    --zenith-min FLOAT        Minimum zenith angle in degrees (default: 0.0)
    --zenith-max FLOAT        Maximum zenith angle in degrees (default: 60.0)
    --energy-min FLOAT        Minimum kinetic energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum kinetic energy in GeV (default: 1e9)
    --n-angles INT            Number of zenith angle points (default: 20)
    --samples, -n INT         Number of energy samples per angle (default: 20)
    --threshold FLOAT         Energy threshold for mode switching in GeV (default: 100.0)
    --threshold-scan-low FLOAT
                              Lower multiplicative factor for transport systematic
                              threshold scan (default: 0.5)
    --threshold-scan-high FLOAT
                              Upper multiplicative factor for transport systematic
                              threshold scan (default: 2.0)
    --no-straggling           Disable straggling in detailed transport (default: enabled)
    --no-scattering           Disable scattering in detailed transport (default: enabled)
    --nx INT                  Number of cells in x direction (default: 3)
    --ny INT                  Number of cells in y direction (default: 3)
    --nz INT                  Number of cells in z direction (default: 5)
    --output, -o PATH         Output HTML file for visualization
"""

using DiffPumas
using DiffPumas.Physics: get_material_index, property_range, property_kinetic_energy
using DiffPumas.Physics: property_stopping_power, ENERGY_LOSS_CSDA
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: PRIMARY_ALTITUDE, compute_air_grammage
using DiffPumas.Geometry: compute_decay_weight_from_path, compute_flux_single, TwoLayerGeometry
using DiffPumas.Geometry: locals_air_at_altitude
using DiffPumas.Geometry: transport_backward_step, transport_backward_step_full, transport_backward_step_mixed
using DiffPumas.GaisserFlux: flux_gccly
using DiffPumas.Types: State, Vec3
using DiffPumas: zenith_to_elevation, sample_energy_loguniform
using DiffPumas.Pumas: load_or_create_physics
using Printf
using Random
using Zygote
using LinearAlgebra
using PlotlyJS

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")

"""
    CubeGeometry

Represents a tessellated rock cube with cells of varying density.
"""
struct CubeGeometry{T<:Real}
    # Cube dimensions in world coordinates
    x_min::T
    x_max::T
    y_min::T
    y_max::T
    z_min::T  # Bottom of rock (detector level)
    z_max::T  # Top of rock (air interface)
    
    # Number of cells in each direction
    nx::Int
    ny::Int
    nz::Int
    
    # Cell sizes
    dx::T
    dy::T
    dz::T
    
    # Material index for rock
    rock_material::Int
    air_material::Int
end

function CubeGeometry(rock_thickness::T, nx::Int, ny::Int, nz::Int, 
                      rock_material::Int, air_material::Int;
                      width::T = T(200.0)) where T<:Real
    # Center the cube at origin in x,y
    x_min = -width / 2
    x_max = width / 2
    y_min = -width / 2
    y_max = width / 2
    z_min = T(0)  # Detector at z=0
    z_max = rock_thickness
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    
    return CubeGeometry{T}(x_min, x_max, y_min, y_max, z_min, z_max,
                           nx, ny, nz, dx, dy, dz,
                           rock_material, air_material)
end

"""
Get cell index (i, j, k) from position. Returns nothing if outside cube.
"""
function get_cell_indices(geo::CubeGeometry{T}, x::T, y::T, z::T) where T<:Real
    if x < geo.x_min || x > geo.x_max || 
       y < geo.y_min || y > geo.y_max ||
       z < geo.z_min || z > geo.z_max
        return nothing
    end
    
    i = clamp(floor(Int, (x - geo.x_min) / geo.dx) + 1, 1, geo.nx)
    j = clamp(floor(Int, (y - geo.y_min) / geo.dy) + 1, 1, geo.ny)
    k = clamp(floor(Int, (z - geo.z_min) / geo.dz) + 1, 1, geo.nz)
    
    return (i, j, k)
end

"""
Convert (i, j, k) cell index to linear index.
"""
@inline function cell_linear_index(geo::CubeGeometry, i::Int, j::Int, k::Int)
    return i + (j - 1) * geo.nx + (k - 1) * geo.nx * geo.ny
end

"""
Get total number of cells.
"""
@inline function num_cells(geo::CubeGeometry)
    return geo.nx * geo.ny * geo.nz
end

"""
Check if cell is in top layer (adjacent to air).
"""
@inline function is_top_layer(geo::CubeGeometry, k::Int)
    return k == geo.nz
end

"""
Get indices of all top-layer cells.
"""
function get_top_layer_indices(geo::CubeGeometry)
    indices = Int[]
    for j in 1:geo.ny
        for i in 1:geo.nx
            push!(indices, cell_linear_index(geo, i, j, geo.nz))
        end
    end
    return indices
end

"""
Get cell center coordinates.
"""
function get_cell_center(geo::CubeGeometry{T}, i::Int, j::Int, k::Int) where T<:Real
    x = geo.x_min + (i - 0.5) * geo.dx
    y = geo.y_min + (j - 0.5) * geo.dy
    z = geo.z_min + (k - 0.5) * geo.dz
    return (x, y, z)
end

"""
Compute distance to next cell boundary along ray direction.
Returns (distance, next_cell_indices) or (distance_to_exit, nothing) if exiting cube.
"""
function distance_to_boundary(geo::CubeGeometry{T}, x::T, y::T, z::T, 
                               dx::T, dy::T, dz::T) where T<:Real
    
    idx = get_cell_indices(geo, x, y, z)
    if idx === nothing
        return (zero(T), nothing)
    end
    i, j, k = idx
    
    # Cell boundaries
    x_lo = geo.x_min + (i - 1) * geo.dx
    x_hi = geo.x_min + i * geo.dx
    y_lo = geo.y_min + (j - 1) * geo.dy
    y_hi = geo.y_min + j * geo.dy
    z_lo = geo.z_min + (k - 1) * geo.dz
    z_hi = geo.z_min + k * geo.dz
    
    # Compute distance to each boundary
    t_min = T(Inf)
    next_i, next_j, next_k = i, j, k
    exiting = false
    
    eps = T(1e-10)
    
    # X boundaries
    if abs(dx) > eps
        if dx > 0
            t = (x_hi - x) / dx
            if t > eps && t < t_min
                t_min = t
                next_i = i + 1
                exiting = (next_i > geo.nx)
            end
        else
            t = (x_lo - x) / dx
            if t > eps && t < t_min
                t_min = t
                next_i = i - 1
                exiting = (next_i < 1)
            end
        end
    end
    
    # Y boundaries
    if abs(dy) > eps
        if dy > 0
            t = (y_hi - y) / dy
            if t > eps && t < t_min
                t_min = t
                next_i, next_j = i, j + 1
                exiting = (next_j > geo.ny)
            end
        else
            t = (y_lo - y) / dy
            if t > eps && t < t_min
                t_min = t
                next_i, next_j = i, j - 1
                exiting = (next_j < 1)
            end
        end
    end
    
    # Z boundaries
    if abs(dz) > eps
        if dz > 0
            t = (z_hi - z) / dz
            if t > eps && t < t_min
                t_min = t
                next_i, next_j, next_k = i, j, k + 1
                exiting = (next_k > geo.nz)  # Exit to air
            end
        else
            t = (z_lo - z) / dz
            if t > eps && t < t_min
                t_min = t
                next_i, next_j, next_k = i, j, k - 1
                exiting = (next_k < 1)  # Exit below detector
            end
        end
    end
    
    if exiting || t_min == T(Inf)
        return (t_min, nothing)
    else
        return (t_min, (next_i, next_j, next_k))
    end
end

"""
A segment through a cell with its geometric properties.
"""
struct CellSegment
    cell_idx::Int       # Linear index of the cell
    distance::Float64   # Path length through this cell
end

"""
    trace_cube_path(geo, elevation, azimuth, start_x, start_y)

Trace a path through the cube and return segments (cell index, distance pairs).
This is the NON-DIFFERENTIABLE part - pure geometry.

Returns (segments, cos_theta, z_exit) where z_exit is the z-coordinate where the path
exits the cube (either through top at z_max, or through sides at z < z_max).

# Arguments
- `elevation`: Elevation angle in degrees (90 = vertical)
- `azimuth`: Azimuth angle in degrees (0 = +y direction)
- `start_x`, `start_y`: Starting position at z=0
"""
function trace_cube_path(geo::CubeGeometry{T}, elevation::T, azimuth::T,
                          start_x::T, start_y::T) where T<:Real
    # Convert elevation to direction (backward mode: trace upward from detector)
    elevation_rad = deg2rad(elevation)
    azimuth_rad = deg2rad(azimuth)
    cos_theta = sin(elevation_rad)  # cos(zenith) = sin(elevation)
    sin_theta = cos(elevation_rad)
    
    # Direction pointing upward (backward tracing) with azimuthal rotation
    dir_x = sin_theta * sin(azimuth_rad)
    dir_y = sin_theta * cos(azimuth_rad)
    dir_z = cos_theta
    
    # Start at detector (z=0) with given x,y offset
    pos_x = start_x
    pos_y = start_y
    pos_z = T(0)
    z_exit = T(0)  # Track where we exit the cube
    
    segments = CellSegment[]
    
    max_steps = 1000
    step = 0
    
    while step < max_steps
        step += 1
        
        idx = get_cell_indices(geo, pos_x, pos_y, pos_z)
        if idx === nothing
            break  # Exited cube
        end
        i, j, k = idx
        
        cell_idx = cell_linear_index(geo, i, j, k)
        
        # Distance to next cell boundary
        dist, next_idx = distance_to_boundary(geo, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z)
        
        if dist <= T(1e-10) || !isfinite(dist)
            # Move a tiny bit forward
            pos_x += dir_x * T(0.01)
            pos_y += dir_y * T(0.01)
            pos_z += dir_z * T(0.01)
            continue
        end
        
        push!(segments, CellSegment(cell_idx, dist))
        
        # Update position
        pos_x += dir_x * dist
        pos_y += dir_y * dist
        pos_z += dir_z * dist
        z_exit = pos_z
        
        if next_idx === nothing
            break  # Exited cube
        end
    end
    
    return segments, cos_theta, z_exit
end

"""
    compute_flux_from_segments(physics, geo, segments, cos_theta, z_exit, cell_densities, energy_initial, charge)

Compute flux given pre-computed path segments. This IS differentiable w.r.t. cell_densities.

If z_exit < z_max, adds an additional rock segment with uniform density (first cell's density)
for the path from z_exit to z_max.
"""
function compute_flux_from_segments(physics, geo::CubeGeometry{T},
                                     segments::Vector{CellSegment},
                                     cos_theta::T,
                                     z_exit::T,
                                     cell_densities::AbstractVector{T},
                                     energy_initial::T,
                                     charge::T) where T<:Real
    
    energy = energy_initial
    weight = one(T)
    
    # Transport through each segment in the cube
    for seg in segments
        density = cell_densities[seg.cell_idx]
        dist = T(seg.distance)
        
        # Compute grammage for this segment
        grammage = dist * density
        
        # Energy lookup using CSDA tables
        X_initial = property_range(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy)
        X_final = X_initial + grammage
        energy_after = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, geo.rock_material, X_final)
        
        if energy_after >= T(1e12) || !isfinite(energy_after)
            return zero(T)  # Particle cannot reach this energy
        end
        
        # Weight factor (Jacobian)
        dedx_initial = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy)
        dedx_final = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy_after)
        weight *= dedx_final / dedx_initial
        
        # Decay weight
        E_avg = (energy + energy_after) / 2
        decay = compute_decay_weight_from_path(physics, dist, E_avg)
        weight *= decay
        
        energy = energy_after
    end
    
    # If exited cube through side (z_exit < z_max), add remaining rock segment
    if z_exit < geo.z_max - eps(T)
        cos_theta_safe = max(abs(cos_theta), T(0.01))
        remaining_rock_dist = (geo.z_max - z_exit) / cos_theta_safe
        default_density = cell_densities[1]  # Use first cell density
        
        grammage_remaining = remaining_rock_dist * default_density
        X_initial = property_range(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy)
        X_final = X_initial + grammage_remaining
        energy_after_rock = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, geo.rock_material, X_final)
        
        if energy_after_rock >= T(1e12) || !isfinite(energy_after_rock)
            return zero(T)
        end
        
        dedx_initial = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy)
        dedx_final = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.rock_material, energy_after_rock)
        weight *= dedx_final / dedx_initial
        
        E_avg = (energy + energy_after_rock) / 2
        decay = compute_decay_weight_from_path(physics, remaining_rock_dist, E_avg)
        weight *= decay
        
        energy = energy_after_rock
    end
    
    # Now transport through air layer
    X_air = compute_air_grammage(geo.z_max, PRIMARY_ALTITUDE, cos_theta)
    
    X_initial_air = property_range(physics, ENERGY_LOSS_CSDA, geo.air_material, energy)
    X_final_air = X_initial_air + X_air
    energy_final = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, geo.air_material, X_final_air)
    
    if energy_final >= T(1e12) || !isfinite(energy_final)
        return zero(T)
    end
    
    # Air weight factor
    dedx_initial_air = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.air_material, energy)
    dedx_final_air = property_stopping_power(physics, ENERGY_LOSS_CSDA, geo.air_material, energy_final)
    weight *= dedx_final_air / dedx_initial_air
    
    # Air decay
    cos_theta_safe = max(abs(cos_theta), T(0.01))
    path_air = (PRIMARY_ALTITUDE - geo.z_max) / cos_theta_safe
    E_avg_air = (energy + energy_final) / 2
    decay_air = compute_decay_weight_from_path(physics, path_air, E_avg_air)
    weight *= decay_air
    
    # Atmospheric flux
    atmospheric_flux = flux_gccly(cos_theta, energy_final, charge)
    
    return weight * atmospheric_flux
end

"""
Pre-computed trajectory data for a sample.
"""
struct TracedSample
    segments::Vector{CellSegment}
    cos_theta::Float64
    z_exit::Float64    # Z-coordinate where path exits cube
    energy::Float64
    charge::Float64
    weight::Float64  # Energy sampling weight
end

"""
Compute integrated flux through cube with gradients for top-layer cells.

This separates the computation into:
1. Geometric ray tracing (non-differentiable)
2. Physics computation (differentiable w.r.t. densities)
"""
function compute_flux_gradient_cube(physics, geo::CubeGeometry{T},
                                     cell_densities::Vector{T},
                                     zenith_min::T, zenith_max::T,
                                     energy_min::T, energy_max::T,
                                     n_angles::Int, n_samples::Int;
                                     verbose::Bool = true) where T<:Real
    
    rng = MersenneTwister(42)
    d_zenith = zenith_max - zenith_min
    
    verbose && println("Pre-computing trajectory geometry...")
    
    # Pre-compute all trajectories (non-differentiable geometry)
    # Each sample gets its own unique trajectory (like detailed transport)
    # RNG order MUST match compute_flux_detailed: zenith → (energy, charge, azimuth, start_x, start_y) per sample
    traced_samples = TracedSample[]
    for i_angle in 1:n_angles
        zenith = zenith_min + d_zenith * rand(rng)
        elevation = T(90) - zenith
        
        for i_sample in 1:n_samples
            # Sample energy and charge FIRST (to match detailed transport order)
            kf, w_energy = sample_energy_loguniform(Float64(energy_min), Float64(energy_max), rng)
            charge = rand(rng) > 0.5 ? T(1) : T(-1)
            
            # Then sample trajectory geometry (same order as compute_flux_cube_single)
            azimuth = T(360) * rand(rng)
            # Detector at center of lowest face
            start_x = T(0)
            start_y = T(0)
            
            # Trace the geometric path (non-differentiable)
            segments, cos_theta, z_exit = trace_cube_path(geo, elevation, azimuth, start_x, start_y)
            
            push!(traced_samples, TracedSample(segments, cos_theta, z_exit, T(kf), charge, T(w_energy)))
        end
    end
    
    n_total = length(traced_samples)
    verbose && println("  Traced $(n_total) unique trajectories")
    
    # Differentiable flux function (only depends on densities, not geometry)
    function flux_fn(densities)
        w_sum = zero(T)
        for sample in traced_samples
            flux_single = compute_flux_from_segments(
                physics, geo, sample.segments, T(sample.cos_theta), T(sample.z_exit),
                densities, T(sample.energy), T(sample.charge)
            )
            w_sum += T(2) * T(sample.weight) * flux_single
        end
        return w_sum / n_total
    end
    
    verbose && println("Computing integrated flux (direct CSDA through cube)...")
    flux = flux_fn(cell_densities)
    
    verbose && println("Computing gradient ∂flux/∂ρ_i using Zygote...")
    grad = Zygote.gradient(flux_fn, cell_densities)[1]
    
    return flux, grad
end

"""
    compute_flux_cube_single(physics, geo, cell_densities, elevation, energy_final, charge;
                             rng, straggling, scattering, energy_threshold_low)

Compute single flux contribution through the cube with cell-by-cell transport.
This respects cell boundaries and uses individual cell densities.
"""
function compute_flux_cube_single(physics, geo::CubeGeometry{T},
                                   cell_densities::Vector{T},
                                   elevation::T,
                                   energy_final::T,
                                   charge::T;
                                   rng::AbstractRNG,
                                   straggling::Bool=true,
                                   scattering::Bool=true,
                                   energy_threshold_low::T=T(100.0)) where T<:Real
    
    # Convert elevation to direction (backward tracing: upward from detector)
    theta = (T(90) - elevation) * T(π) / T(180)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    
    # Random azimuthal angle for variety in cell traversal
    azimuth = T(2π) * rand(rng)
    
    # Detector at center of lowest face
    start_x = T(0)
    start_y = T(0)
    
    # Direction pointing upward (backward mode)
    dir_x = sin_theta * sin(azimuth)
    dir_y = sin_theta * cos(azimuth)
    dir_z = cos_theta
    
    # Initial state at detector (z=0)
    state = State{T}(
        charge = charge,
        energy = energy_final,
        distance = zero(T),
        grammage = zero(T),
        time = zero(T),
        weight = one(T),
        position = Vec3{T}(start_x, start_y, zero(T)),
        direction = Vec3{T}(-dir_x, -dir_y, -dir_z),  # Points into rock (downward)
        decayed = false
    )
    
    energy_threshold = T(1e12)
    STEP_EPSILON = T(1e-7)
    STEP_MIN = T(1e-6)
    
    # Transport through cube cells
    max_outer_steps = 1000
    outer_step = 0
    
    while outer_step < max_outer_steps && state.energy < energy_threshold - eps(T)
        outer_step += 1
        
        x, y, z = state.position[1], state.position[2], state.position[3]
        
        # Check termination conditions
        if z < zero(T)
            break  # Below detector
        end
        
        if z >= PRIMARY_ALTITUDE - eps(T)
            break  # Reached primary altitude - success!
        end
        
        if state.weight <= zero(T) || !isfinite(state.weight)
            break
        end
        
        # Determine simulation mode based on energy
        if state.energy < energy_threshold_low - eps(T)
            use_straggled = straggling
            use_scattering = scattering
            current_energy_limit = energy_threshold_low
        else
            use_straggled = false
            use_mixed_mode = straggling
            use_scattering = false
            current_energy_limit = energy_threshold
        end
        
        # Inner transport loop
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
            
            # Motion direction (opposite to state direction in backward mode)
            ux_motion = -state.direction[1]
            uy_motion = -state.direction[2]
            uz_motion = -state.direction[3]
            
            # Determine which medium we're in and the step to boundary
            if z < geo.z_max
                # Check if inside the cube
                cell_idx = get_cell_indices(geo, x, y, z)
                
                if cell_idx === nothing
                    # Outside cube horizontally but below rock-air interface
                    # Treat as uniform rock with default density
                    material = geo.rock_material
                    density = cell_densities[1]  # Use first cell density as default
                    
                    # Step to rock-air interface
                    step_to_boundary = T(1000.0)
                    if uz_motion > eps(T)
                        step_to_boundary = min(step_to_boundary, (geo.z_max - z) / uz_motion)
                    elseif uz_motion < -eps(T)
                        step_to_boundary = min(step_to_boundary, -z / uz_motion)
                    end
                else
                    i, j, k = cell_idx
                    linear_idx = cell_linear_index(geo, i, j, k)
                    material = geo.rock_material
                    density = cell_densities[linear_idx]
                    
                    # Compute step to cell boundary
                    step_to_boundary = compute_step_to_cell_boundary(geo, x, y, z, 
                                                                      ux_motion, uy_motion, uz_motion,
                                                                      i, j, k)
                end
            else
                # In air layer
                material = geo.air_material
                density = locals_air_at_altitude(z).density
                
                # Step to primary altitude or rock-air interface
                if uz_motion > eps(T)
                    step_to_boundary = (PRIMARY_ALTITUDE - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = (geo.z_max - z) / uz_motion
                else
                    step_to_boundary = T(1000.0)
                end
            end
            
            # Adaptive step sizing (1% of range like PUMAS)
            Xi = property_range(physics, ENERGY_LOSS_CSDA, material, state.energy)
            max_grammage_frac = T(0.01)
            max_grammage_step = max_grammage_frac * Xi
            max_geometric_step = max_grammage_step / max(density, T(1e-6))
            
            # For non-uniform air, limit by altitude
            if z >= geo.z_max
                h = T(12e3)  # Scale height
                uz_abs = max(abs(uz_motion), T(0.05))
                altitude_step = T(0.05) * h / uz_abs
                max_geometric_step = min(max_geometric_step, altitude_step)
            end
            
            # Final step size
            step_size = min(step_to_boundary + STEP_EPSILON, max_geometric_step)
            step_size = clamp(step_size, STEP_MIN, T(1000.0))
            
            # Perform transport step
            if state.energy < energy_threshold_low - eps(T)
                if use_straggled && use_scattering
                    state, _ = transport_backward_step_full(physics, state, material, density, step_size, rng;
                                                            mode=:straggled, scattering=true,
                                                            energy_limit=current_energy_limit)
                elseif use_straggled
                    state = transport_backward_step(physics, state, material, density, step_size;
                                                    rng=rng, straggling=true)
                else
                    state = transport_backward_step(physics, state, material, density, step_size;
                                                    rng=nothing, straggling=false)
                end
            else
                if straggling
                    state = transport_backward_step_mixed(physics, state, material, density, step_size, rng;
                                                          energy_limit=current_energy_limit)
                else
                    state = transport_backward_step(physics, state, material, density, step_size;
                                                    rng=nothing, straggling=false)
                end
            end
        end
    end
    
    # Check if reached primary altitude
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
Compute step distance to the next cell boundary.
"""
function compute_step_to_cell_boundary(geo::CubeGeometry{T}, x::T, y::T, z::T,
                                        ux::T, uy::T, uz::T,
                                        i::Int, j::Int, k::Int) where T<:Real
    
    eps_val = T(1e-10)
    step_min = T(Inf)
    
    # Cell boundaries
    x_lo = geo.x_min + (i - 1) * geo.dx
    x_hi = geo.x_min + i * geo.dx
    y_lo = geo.y_min + (j - 1) * geo.dy
    y_hi = geo.y_min + j * geo.dy
    z_lo = geo.z_min + (k - 1) * geo.dz
    z_hi = geo.z_min + k * geo.dz
    
    # X boundaries
    if abs(ux) > eps_val
        if ux > 0
            t = (x_hi - x) / ux
            if t > eps_val
                step_min = min(step_min, t)
            end
        else
            t = (x_lo - x) / ux
            if t > eps_val
                step_min = min(step_min, t)
            end
        end
    end
    
    # Y boundaries
    if abs(uy) > eps_val
        if uy > 0
            t = (y_hi - y) / uy
            if t > eps_val
                step_min = min(step_min, t)
            end
        else
            t = (y_lo - y) / uy
            if t > eps_val
                step_min = min(step_min, t)
            end
        end
    end
    
    # Z boundaries
    if abs(uz) > eps_val
        if uz > 0
            t = (z_hi - z) / uz
            if t > eps_val
                step_min = min(step_min, t)
            end
        else
            t = (z_lo - z) / uz
            if t > eps_val
                step_min = min(step_min, t)
            end
        end
    end
    
    return step_min == T(Inf) ? T(1000.0) : step_min
end

"""
    compute_flux_detailed(physics, geo, cell_densities, zenith_min, zenith_max,
                          energy_min, energy_max, n_angles, n_samples; kwargs...)

Compute integrated flux using detailed transport through the cube.
Respects cell boundaries and uses individual cell densities.
"""
function compute_flux_detailed(physics, geo::CubeGeometry{T},
                               cell_densities::Vector{T},
                               zenith_min::T, zenith_max::T,
                               energy_min::T, energy_max::T,
                               n_angles::Int, n_samples::Int;
                               straggling::Bool=true,
                               scattering::Bool=true,
                               energy_threshold_low::T=T(100.0),
                               seed::Int=42,
                               verbose::Bool=true) where T<:Real
    
    rng = MersenneTwister(seed)
    
    d_zenith = zenith_max - zenith_min
    n_total = n_angles * n_samples
    
    w_sum = zero(T)
    w2_sum = zero(T)
    n_done = 0
    
    for i_angle in 1:n_angles
        zenith = zenith_min + d_zenith * rand(rng)
        elevation = T(90) - zenith
        
        for i_sample in 1:n_samples
            n_done += 1
            
            kf, w_energy = sample_energy_loguniform(Float64(energy_min), Float64(energy_max), rng)
            charge = rand(rng) > 0.5 ? T(1) : T(-1)
            
            flux_single = compute_flux_cube_single(physics, geo, cell_densities,
                                                    elevation, T(kf), charge;
                                                    rng=rng, straggling=straggling, scattering=scattering,
                                                    energy_threshold_low=energy_threshold_low)
            
            wi = T(2) * T(w_energy) * flux_single
            
            w_sum += wi
            w2_sum += wi * wi
        end
    end
    
    n = T(n_total)
    flux = w_sum / n
    
    variance = (w2_sum / n - flux^2) / max(one(T), n - one(T))
    sigma = sqrt(max(zero(T), variance))
    
    return flux, sigma
end

function compute_flux_detailed_budget(physics, geo::CubeGeometry{T},
                                      cell_densities::Vector{T},
                                      zenith_min::T, zenith_max::T,
                                      energy_min::T, energy_max::T,
                                      n_angles::Int, n_samples::Int;
                                      straggling::Bool=true,
                                      scattering::Bool=true,
                                      energy_threshold_low::T=T(100.0),
                                      threshold_factors::Tuple{Float64,Float64}=(0.5, 2.0),
                                      verbose::Bool=true) where T<:Real
    function evaluate(variation)
        return compute_flux_detailed(
            physics,
            geo,
            cell_densities,
            zenith_min,
            zenith_max,
            energy_min,
            energy_max,
            n_angles,
            n_samples;
            straggling = variation.straggling,
            scattering = variation.scattering,
            energy_threshold_low = variation.energy_threshold_low,
            seed = variation.seed,
            verbose = false,
        )
    end

    verbose && println("Running transport uncertainty budget...")
    return estimate_transport_uncertainty(
        evaluate;
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low,
        seed = 42,
        threshold_factors = threshold_factors,
    )
end

function summary_output_path(output_path::String)
    root, _ = splitext(output_path)
    return root * "_uncertainty.html"
end

function create_flux_uncertainty_summary_plot(detailed_budget,
                                              flux_csda::Float64,
                                              output_path::String)
    labels = ["Detailed transport", "Direct CSDA"]
    flux_values = [detailed_budget.value, flux_csda]
    total_sigma = [detailed_budget.sigma_total, 0.0]
    mc_sigma = [detailed_budget.sigma_mc, 0.0]
    hover_text = [
        @sprintf(
            "%s<br>Flux=%.5e<br>MC=%.2f%%<br>Transport syst=%.2f%%<br>Total=%.2f%%",
            labels[1],
            detailed_budget.value,
            100 * detailed_budget.sigma_mc / max(abs(detailed_budget.value), 1e-30),
            100 * detailed_budget.sigma_syst / max(abs(detailed_budget.value), 1e-30),
            100 * detailed_budget.sigma_total / max(abs(detailed_budget.value), 1e-30),
        ),
        @sprintf(
            "%s<br>Flux=%.5e<br>Relative difference vs detailed=%.2f%%",
            labels[2],
            flux_csda,
            100 * abs(flux_csda - detailed_budget.value) / max(abs(detailed_budget.value), 1e-30),
        ),
    ]

    traces = GenericTrace[
        scatter(
            x = labels,
            y = flux_values,
            mode = "markers",
            marker = attr(size = 13, color = "rgba(220,120,30,0.95)"),
            error_y = attr(type = "data", array = total_sigma, visible = true,
                           color = "rgba(220,120,30,0.95)", thickness = 1.6),
            text = hover_text,
            hovertemplate = "%{text}<extra></extra>",
            name = "Flux with total uncertainty",
        ),
        scatter(
            x = labels,
            y = flux_values,
            mode = "markers",
            marker = attr(size = 9, color = "rgba(40,110,220,0.95)"),
            error_y = attr(type = "data", array = mc_sigma, visible = true,
                           color = "rgba(40,110,220,0.95)", thickness = 1.2),
            hoverinfo = "skip",
            name = "MC uncertainty only",
        ),
    ]

    layout = Layout(
        title = "Cube Flux Uncertainty Summary",
        xaxis = attr(title = "Configuration"),
        yaxis = attr(title = "Flux (m⁻² s⁻¹ sr⁻¹)", type = "log"),
        width = 900,
        height = 600,
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Uncertainty summary saved to: $output_path")
    return fig
end

"""
Create 3D visualization of the cube with gradient coloring and density info.
"""
function create_cube_visualization(geo::CubeGeometry{T}, gradients::Vector{T},
                                    cell_densities::Vector{T},
                                    output_path::String;
                                    top_layer_only::Bool = false) where T<:Real
    
    traces = GenericTrace[]
    
    # Get gradient range for coloring (use all cells if showing all, else top only)
    if top_layer_only
        active_indices = get_top_layer_indices(geo)
    else
        active_indices = collect(1:length(gradients))
    end
    active_grads = [abs(gradients[i]) for i in active_indices]
    grad_min = minimum(active_grads)
    grad_max = maximum(active_grads)
    grad_range = max(grad_max - grad_min, 1e-30)
    
    # Get density range for display
    density_min = minimum(cell_densities)
    density_max = maximum(cell_densities)
    
    # Create mesh for each cell
    for k in 1:geo.nz
        for j in 1:geo.ny
            for i in 1:geo.nx
                cell_idx = cell_linear_index(geo, i, j, k)
                
                # Skip non-top cells if requested
                if top_layer_only && k != geo.nz
                    continue
                end
                
                # Cell boundaries
                x0 = geo.x_min + (i - 1) * geo.dx
                x1 = geo.x_min + i * geo.dx
                y0 = geo.y_min + (j - 1) * geo.dy
                y1 = geo.y_min + j * geo.dy
                z0 = geo.z_min + (k - 1) * geo.dz
                z1 = geo.z_min + k * geo.dz
                
                # Color based on gradient magnitude
                grad_val = abs(gradients[cell_idx])
                color_intensity = clamp((grad_val - grad_min) / grad_range, 0.0, 1.0)
                
                # Opacity: higher gradient = more opaque, lower layers slightly more transparent
                layer_factor = k / geo.nz  # 0 to 1 from bottom to top
                base_opacity = 0.2 + 0.3 * layer_factor  # 0.2 at bottom, 0.5 at top
                opacity = base_opacity + 0.4 * color_intensity
                
                # Color: blue (low gradient) to red (high gradient)
                r = round(Int, 255 * color_intensity)
                g = 50
                b = round(Int, 255 * (1 - color_intensity))
                color = "rgba($r, $g, $b, $opacity)"
                
                # Get cell density
                density = cell_densities[cell_idx]
                
                # Create cube mesh (8 vertices, 12 triangles)
                x = [x0, x1, x1, x0, x0, x1, x1, x0]
                y = [y0, y0, y1, y1, y0, y0, y1, y1]
                z = [z0, z0, z0, z0, z1, z1, z1, z1]
                
                # Triangle indices for mesh3d
                i_idx = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
                j_idx = [1, 2, 5, 6, 1, 4, 5, 2, 3, 4, 4, 7]
                k_idx = [2, 3, 6, 7, 5, 5, 6, 6, 7, 7, 7, 4]
                
                trace = mesh3d(
                    x=x, y=y, z=z,
                    i=i_idx, j=j_idx, k=k_idx,
                    color=color,
                    opacity=opacity,
                    flatshading=true,
                    showlegend=false,
                    hoverinfo="text",
                    hovertext="Cell ($i,$j,$k)<br>ρ = $(@sprintf("%.1f", density)) kg/m³<br>∂flux/∂ρ = $(@sprintf("%.2e", gradients[cell_idx]))"
                )
                push!(traces, trace)
            end
        end
    end
    
    # Add outline for the whole cube
    x_out = [geo.x_min, geo.x_max, geo.x_max, geo.x_min, geo.x_min,
             geo.x_min, geo.x_max, geo.x_max, geo.x_min, geo.x_min]
    y_out = [geo.y_min, geo.y_min, geo.y_max, geo.y_max, geo.y_min,
             geo.y_min, geo.y_min, geo.y_max, geo.y_max, geo.y_min]
    z_out = [geo.z_min, geo.z_min, geo.z_min, geo.z_min, geo.z_min,
             geo.z_max, geo.z_max, geo.z_max, geo.z_max, geo.z_max]
    
    outline = scatter3d(
        x=x_out, y=y_out, z=z_out,
        mode="lines",
        line=attr(color="black", width=2),
        showlegend=false
    )
    push!(traces, outline)
    
    # Add vertical edges
    for (x, y) in [(geo.x_min, geo.y_min), (geo.x_max, geo.y_min),
                   (geo.x_max, geo.y_max), (geo.x_min, geo.y_max)]
        edge = scatter3d(
            x=[x, x], y=[y, y], z=[geo.z_min, geo.z_max],
            mode="lines",
            line=attr(color="black", width=2),
            showlegend=false
        )
        push!(traces, edge)
    end
    
    title_str = top_layer_only ? "Muon Flux Sensitivity - Top Layer" : "Muon Flux Sensitivity - All Cells"
    layout = Layout(
        title="$title_str<br><sub>ρ ∈ [$(@sprintf("%.0f", density_min)), $(@sprintf("%.0f", density_max))] kg/m³, Color: |∂flux/∂ρ| (blue=low, red=high)</sub>",
        scene=attr(
            xaxis=attr(title="X (m)"),
            yaxis=attr(title="Y (m)"),
            zaxis=attr(title="Z (m)"),
            aspectmode="data"
        ),
        margin=attr(l=0, r=0, t=80, b=0)
    )
    
    fig = Plot(traces, layout)
    savefig(fig, output_path)
    println("Visualization saved to: $output_path")
    
    return fig
end

function parse_commandline()
    dump_path = DEFAULT_DUMP
    rock_thickness = 100.0
    rock_density = 2650.0
    zenith_min = 0.0
    zenith_max = 60.0
    energy_min = 1e-3
    energy_max = 1e9
    n_angles = 20
    n_samples = 20
    energy_threshold_low = 100.0
    threshold_scan_low = 0.5
    threshold_scan_high = 2.0
    straggling = true
    scattering = true
    nx = 3
    ny = 3
    nz = 5
    output_path = joinpath(@__DIR__, "data", "diff_flux_cube.html")
    
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
        elseif arg == "--threshold"
            i + 1 <= length(ARGS) || error("--threshold requires a value")
            energy_threshold_low = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--threshold-scan-low"
            i + 1 <= length(ARGS) || error("--threshold-scan-low requires a value")
            threshold_scan_low = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--threshold-scan-high"
            i + 1 <= length(ARGS) || error("--threshold-scan-high requires a value")
            threshold_scan_high = parse(Float64, ARGS[i + 1])
            i += 2
        elseif arg == "--no-straggling"
            straggling = false
            i += 1
        elseif arg == "--no-scattering"
            scattering = false
            i += 1
        elseif arg == "--nx"
            i + 1 <= length(ARGS) || error("--nx requires a value")
            nx = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--ny"
            i + 1 <= length(ARGS) || error("--ny requires a value")
            ny = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--nz"
            i + 1 <= length(ARGS) || error("--nz requires a value")
            nz = parse(Int, ARGS[i + 1])
            i += 2
        elseif arg == "--output" || arg == "-o"
            i + 1 <= length(ARGS) || error("--output requires a path")
            output_path = ARGS[i + 1]
            i += 2
        elseif arg == "--help" || arg == "-h"
            println(@doc diff_flux_cube)
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
        energy_threshold_low = energy_threshold_low,
        threshold_factors = (threshold_scan_low, threshold_scan_high),
        straggling = straggling,
        scattering = scattering,
        nx = nx,
        ny = ny,
        nz = nz,
        output_path = output_path
    )
end

function main()
    args = parse_commandline()
    
    println("=" ^ 60)
    println(" DiffPumas - Flux Cube with Cell Density Gradients")
    println("=" ^ 60)
    println()
    
    strag_str = args.straggling ? "enabled" : "disabled"
    scat_str = args.scattering ? "enabled" : "disabled"
    
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  Rock thickness:  $(args.rock_thickness) m")
    println("  Rock density:    $(args.rock_density) kg/m³")
    println("  Zenith range:    $(args.zenith_min)° - $(args.zenith_max)°")
    println("  Energy range:    $(args.energy_min) - $(args.energy_max) GeV")
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println("  Threshold scan:  ×$(args.threshold_factors[1]) / ×$(args.threshold_factors[2])")
    println("  Straggling:      $strag_str")
    println("  Scattering:      $scat_str")
    println("  Samples:         $(args.n_angles) angles × $(args.n_samples) energies = $(args.n_angles * args.n_samples)")
    println("  Grid:            $(args.nx) × $(args.ny) × $(args.nz) cells")
    println("  Output:          $(args.output_path)")
    println()
    
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
    
    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials not found")
        return 1
    end
    
    println("Material indices: rock=$rock_idx, air=$air_idx")
    println()
    
    # Compute cube width based on zenith angle range
    # Ensure trajectories at max zenith angle traverse through at least one cell width
    max_zenith_rad = deg2rad(args.zenith_max)
    horizontal_extent = args.rock_thickness * tan(max_zenith_rad) * 2
    cube_width = max(horizontal_extent * 1.5, 200.0)  # Add margin
    
    # Create cube geometry for cells
    geo = CubeGeometry(args.rock_thickness, args.nx, args.ny, args.nz,
                       rock_idx, air_idx; width=cube_width)
    
    n_cells = num_cells(geo)
    top_indices = get_top_layer_indices(geo)
    
    println("Cube geometry:")
    println("  X range: [$(geo.x_min), $(geo.x_max)] m")
    println("  Y range: [$(geo.y_min), $(geo.y_max)] m")
    println("  Z range: [$(geo.z_min), $(geo.z_max)] m")
    println("  Cell size: $(round(geo.dx, digits=2)) × $(round(geo.dy, digits=2)) × $(round(geo.dz, digits=2)) m")
    println("  Total cells: $n_cells")
    println("  Top layer cells: $(length(top_indices))")
    println()
    
    # Initialize cells with slightly randomized densities (±5% variation)
    rng_density = MersenneTwister(123)  # Fixed seed for reproducibility
    density_variation = 0.05  # 5% variation
    cell_densities = [args.rock_density * (1.0 + density_variation * (2*rand(rng_density) - 1)) 
                      for _ in 1:n_cells]
    
    println("Cell densities:")
    @printf("  Base density:  %.1f kg/m³\n", args.rock_density)
    @printf("  Variation:     ±%.1f%%\n", density_variation * 100)
    @printf("  Range:         [%.1f, %.1f] kg/m³\n", minimum(cell_densities), maximum(cell_densities))
    println()
    
    # =========================================================================
    # Part 1: Detailed Transport through cube (cell-by-cell)
    # =========================================================================
    println("=" ^ 60)
    println(" Part 1: Detailed Transport (straggling=$strag_str, scattering=$scat_str)")
    println("=" ^ 60)
    println()
    
    detailed_budget = compute_flux_detailed_budget(
        physics, geo, cell_densities,
        args.zenith_min, args.zenith_max,
        args.energy_min, args.energy_max,
        args.n_angles, args.n_samples;
        straggling=args.straggling,
        scattering=args.scattering,
        energy_threshold_low=args.energy_threshold_low,
        threshold_factors=args.threshold_factors,
        verbose=true
    )
    flux_detailed = detailed_budget.value
    sigma_detailed = detailed_budget.sigma_mc
    
    println()
    println("Detailed Transport Results:")
    println("-" ^ 40)
    @printf("  Integrated flux: %.5e m⁻² s⁻¹ sr⁻¹\n", flux_detailed)
    @printf("  MC error:        %.2e (%.2f%%)\n", sigma_detailed,
            100 * sigma_detailed / max(abs(flux_detailed), 1e-30))
    @printf("  Transport syst:  %.2e (%.2f%%)\n", detailed_budget.sigma_syst,
            100 * detailed_budget.sigma_syst / max(abs(flux_detailed), 1e-30))
    @printf("  Total:           %.2e (%.2f%%)\n", detailed_budget.sigma_total,
            100 * detailed_budget.sigma_total / max(abs(flux_detailed), 1e-30))
    
    # =========================================================================
    # Part 2: Direct CSDA with Gradient Computation
    # =========================================================================
    println()
    println("=" ^ 60)
    println(" Part 2: Direct CSDA with Gradient Computation")
    println("=" ^ 60)
    println()
    
    flux_csda, gradients = compute_flux_gradient_cube(
        physics, geo, cell_densities,
        args.zenith_min, args.zenith_max,
        args.energy_min, args.energy_max,
        args.n_angles, args.n_samples;
        verbose=true
    )
    
    println()
    println("Direct CSDA Results:")
    println("-" ^ 40)
    @printf("  Integrated flux: %.5e m⁻² s⁻¹ sr⁻¹\n", flux_csda)
    
    # Gradient statistics for ALL cells
    println()
    println("All cells gradient statistics:")
    println("-" ^ 40)
    @printf("  Min:  %.5e\n", minimum(gradients))
    @printf("  Max:  %.5e\n", maximum(gradients))
    @printf("  Mean: %.5e\n", sum(gradients) / length(gradients))
    
    # Also show top layer statistics
    top_grads = [gradients[i] for i in top_indices]
    println()
    println("Top layer gradient statistics:")
    println("-" ^ 40)
    @printf("  Min:  %.5e\n", minimum(top_grads))
    @printf("  Max:  %.5e\n", maximum(top_grads))
    @printf("  Mean: %.5e\n", sum(top_grads) / length(top_grads))
    
    # Print all cells sorted by gradient magnitude (show top 15)
    println()
    println("All cells by |gradient| (descending, top 15):")
    println("-" ^ 60)
    
    all_sorted = sortperm(gradients, by=abs, rev=true)
    for (rank, cell_idx) in enumerate(all_sorted[1:min(15, length(all_sorted))])
        # Convert to (i, j, k)
        k = div(cell_idx - 1, geo.nx * geo.ny) + 1
        remainder = (cell_idx - 1) % (geo.nx * geo.ny)
        j = div(remainder, geo.nx) + 1
        i = remainder % geo.nx + 1
        density = cell_densities[cell_idx]
        @printf("  %2d. Cell (%d,%d,%d): ρ=%.1f, ∂flux/∂ρ = %.5e\n", rank, i, j, k, density, gradients[cell_idx])
    end
    
    # =========================================================================
    # Part 3: Comparison
    # =========================================================================
    println()
    println("=" ^ 60)
    println(" Comparison Summary")
    println("=" ^ 60)
    println()
    
    rel_diff = abs(flux_csda - flux_detailed) / max(abs(flux_detailed), 1e-30) * 100
    @printf("  Detailed Transport:  %.5e m⁻² s⁻¹ sr⁻¹\n", flux_detailed)
    @printf("  Direct CSDA:         %.5e m⁻² s⁻¹ sr⁻¹\n", flux_csda)
    @printf("  Relative difference: %.2f%%\n", rel_diff)
    
    # =========================================================================
    # Create visualization
    # =========================================================================
    println()
    println("=" ^ 60)
    println(" Creating Visualization")
    println("=" ^ 60)
    println()
    
    mkpath(dirname(args.output_path))
    create_cube_visualization(geo, gradients, cell_densities, args.output_path; top_layer_only=false)
    create_flux_uncertainty_summary_plot(detailed_budget, flux_csda, summary_output_path(args.output_path))
    
    println()
    println("=" ^ 60)
    println("Done!")
    return 0
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
