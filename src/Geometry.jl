"""
    Geometry

Geometry module implementing the backward muon flux calculation example.
This module demonstrates differentiable transport w.r.t. material density.

The key differentiable quantity is the integrated flux w as a function of
rock density ρ. Using Zygote.jl, we can compute ∂w/∂ρ.
"""
module Geometry

using ..Constants
using ..Types
using ..Physics
using ..Materials
using ..Transport
using ..Context
using ..GaisserFlux: flux_gccly  # Use flux model from GaisserFlux module
using LinearAlgebra
using Zygote
using ChainRulesCore
using Random

export TwoLayerGeometry, create_geometry_context
export compute_flux, compute_flux_gradient, compute_flux_differentiable
export locals_rock, locals_air, medium_callback
export run_backward_mc

# Primary altitude for sampling (m)
const PRIMARY_ALTITUDE = 1e3

"""
    TwoLayerGeometry{T}

Simple two-layer geometry with rock and air layers.
The rock layer has configurable thickness and density.

# Fields
- `rock_thickness::T`: Rock layer thickness in m
- `rock_density::T`: Rock density in kg/m³
- `rock_material::Int`: Material index for rock
- `air_material::Int`: Material index for air
"""
struct TwoLayerGeometry{T<:Real}
    rock_thickness::T
    rock_density::T
    rock_material::Int
    air_material::Int
end

"""
    locals_rock(rock_density)

Create locals callback for uniform rock medium.
The density is explicitly parameterized for differentiation.
"""
function locals_rock(rock_density::T) where T<:Real
    return Locals{T}(rock_density, Vec3{T}(0, 0, 0))
end

"""
    locals_air(altitude)

Create locals for air with exponential density profile.
"""
function locals_air_at_altitude(altitude::T) where T<:Real
    # Geomagnetic field (simplified, assumed uniform)
    B = Vec3{T}(0, 2e-5, -4e-5)
    
    # Exponential atmosphere model
    ρ0 = T(1.205)  # kg/m³ at sea level
    h = T(12e3)    # Scale height
    
    density = ρ0 * exp(-altitude / h)
    
    return Locals{T}(density, B)
end

"""
    transport_backward_step(physics, state, material, density, step_distance)

Perform a single backward transport step.
In backward mode, particle moves OPPOSITE to its direction vector.
The energy INCREASES as we go backward (particle gains energy).
"""
function transport_backward_step(physics::PhysicsTables{T}, state::State{T},
                                  material::Int, density::T, 
                                  step_distance::T) where T<:Real
    
    ki = state.energy
    mass = physics.mass
    
    # Grammage for this step
    grammage_step = step_distance * density
    
    # In backward mode, we ADD to the range (particle gains energy going backward)
    # Get current range
    Xi = property_range(physics, ENERGY_LOSS_CSDA, material, ki)
    
    # Final range (larger since going backward = gaining energy)
    Xf = Xi + grammage_step
    
    # Get final (higher) energy
    kf = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, material, Xf)
    
    # Proper time update (in units of length, comparable to ctau)
    # proper_time = distance / (beta * gamma)
    gamma_i = one(T) + ki / mass
    gamma_f = one(T) + kf / mass
    gamma_avg = (gamma_i + gamma_f) / 2
    beta_avg = sqrt(one(T) - one(T) / gamma_avg^2)
    proper_time_step = step_distance / (beta_avg * gamma_avg)
    
    # Position update - BACKWARD mode means moving OPPOSITE to direction
    new_position = state.position - step_distance * state.direction
    
    # Weight update for backward MC
    # dE/dx ratio for importance sampling
    dedx_i = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, ki)
    dedx_f = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, kf)
    weight_factor = dedx_f / dedx_i
    
    # Decay weight
    decay_factor = exp(-proper_time_step / physics.ctau)
    
    new_weight = state.weight * weight_factor * decay_factor
    
    # Create new state
    new_state = State{T}(
        state.charge,
        kf,
        state.distance + step_distance,
        state.grammage + grammage_step,
        state.time + proper_time_step,
        new_weight,
        new_position,
        state.direction,
        state.decayed
    )
    
    return new_state
end

"""
    transport_backward_through_geometry(physics, geometry, initial_state, energy_threshold)

Transport particle backward through the two-layer geometry.
Particle starts at z=0 and travels upward (opposite to downward direction).
"""
function transport_backward_through_geometry(physics::PhysicsTables{T}, 
                                             geometry::TwoLayerGeometry{T},
                                             initial_state::State{T},
                                             energy_threshold::T) where T<:Real
    
    state = initial_state
    max_steps = 10000
    
    for step in 1:max_steps
        z = state.position[3]
        
        # Check if reached primary altitude (success)
        if z >= PRIMARY_ALTITUDE
            break
        end
        
        # Check if went below ground (failure)
        if z < zero(T)
            break
        end
        
        # Check energy threshold
        if state.energy >= energy_threshold
            break
        end
        
        # Check weight (numerical issues)
        if state.weight <= zero(T) || !isfinite(state.weight)
            break
        end
        
        # Determine current medium and compute step
        # In backward mode, uz is the vertical component of actual motion
        # (opposite to direction vector)
        uz_motion = -state.direction[3]  # Actual vertical motion direction
        
        if z < geometry.rock_thickness
            # In rock layer
            material = geometry.rock_material
            density = geometry.rock_density
            
            # Step to upper boundary of rock
            if uz_motion > eps(T)
                step_to_boundary = (geometry.rock_thickness - z) / uz_motion
            else
                # Moving down or horizontal - shouldn't happen in proper backward MC
                step_to_boundary = T(1.0)
            end
        else
            # In air layer
            material = geometry.air_material
            density = locals_air_at_altitude(z).density
            
            # Step to primary altitude
            if uz_motion > eps(T)
                step_to_boundary = (PRIMARY_ALTITUDE - z) / uz_motion
            else
                step_to_boundary = T(1.0)
            end
        end
        
        # Limit step size
        step_size = min(step_to_boundary + T(1e-6), T(100.0))
        step_size = max(step_size, T(STEP_MIN))
        
        # Perform backward transport step
        state = transport_backward_step(physics, state, material, density, step_size)
    end
    
    return state
end

"""
    compute_flux_single(physics, geometry, energy_final, elevation, charge)

Compute flux contribution from a single backward MC particle.
"""
function compute_flux_single(physics::PhysicsTables{T}, 
                             geometry::TwoLayerGeometry{T},
                             energy_final::T,
                             elevation::T,
                             charge::T) where T<:Real
    
    # Convert elevation to direction (pointing downward into rock)
    # elevation = 0 means horizontal, elevation = 90 means vertical (straight down)
    theta = (T(90) - elevation) * T(π) / T(180)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    
    # Initial state at z=0 (surface), direction pointing down
    # In backward MC, we trace backward (upward) from this state
    state = State{T}(
        charge = charge,
        energy = energy_final,
        distance = zero(T),
        grammage = zero(T),
        time = zero(T),
        weight = one(T),
        position = Vec3{T}(0, 0, 0),
        direction = Vec3{T}(-sin_theta, 0, -cos_theta),  # Pointing downward
        decayed = false
    )
    
    # Energy threshold (very high for backward MC)
    energy_threshold = T(1e12)
    
    # Transport backward through geometry
    final_state = transport_backward_through_geometry(physics, geometry, state, energy_threshold)
    
    # Check if reached primary altitude
    if final_state.position[3] >= PRIMARY_ALTITUDE - T(1.0)
        # Get the cos(theta) of the final direction (for flux evaluation)
        # In backward MC, the "incoming" direction at primary altitude
        # is opposite to what we traced
        cos_theta_final = -final_state.direction[3]
        
        # Compute flux contribution
        flux = flux_gccly(cos_theta_final, final_state.energy, charge)
        return final_state.weight * flux
    else
        return zero(T)
    end
end

"""
    compute_flux(physics, rock_density, rock_thickness, elevation, energy_min, energy_max; n_samples)

Compute integrated muon flux using backward Monte Carlo.
"""
function compute_flux(physics::PhysicsTables{T}, 
                      rock_density::T,
                      rock_thickness::T,
                      elevation::T,
                      energy_min::T,
                      energy_max::T;
                      n_samples::Int = 1000,
                      seed::Int = 42) where T<:Real
    
    rng = Random.MersenneTwister(seed)
    
    # Create geometry
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1
        rock_idx = 1
    end
    if air_idx == -1
        air_idx = length(physics.tables)
    end
    
    geometry = TwoLayerGeometry{T}(rock_thickness, rock_density, rock_idx, air_idx)
    
    # Integration bounds (log-uniform sampling)
    rk = log(energy_max / energy_min)
    
    # Accumulate flux
    w_sum = zero(T)
    w2_sum = zero(T)
    
    for i in 1:n_samples
        # Sample final energy (log-uniform)
        if rk > zero(T)
            kf = energy_min * exp(rk * rand(rng))
            wf = kf * rk  # Jacobian for log-uniform sampling
        else
            kf = energy_min
            wf = one(T)
        end
        
        # Sample charge (50-50)
        charge = rand(rng) > 0.5 ? one(T) : -one(T)
        wf *= T(2)  # Account for charge sampling
        
        # Compute contribution
        wi = wf * compute_flux_single(physics, geometry, kf, elevation, charge)
        
        w_sum += wi
        w2_sum += wi * wi
    end
    
    # Average flux
    flux_avg = w_sum / n_samples
    
    return flux_avg
end

"""
    compute_flux_differentiable(physics, rock_density, rock_thickness, elevation, energy_final, charge)

Compute flux for a single particle - fully differentiable version.
"""
function compute_flux_differentiable(physics::PhysicsTables{T},
                                     rock_density::T,
                                     rock_thickness::T,
                                     elevation::T,
                                     energy_final::T,
                                     charge::T) where T<:Real
    
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1
        rock_idx = 1
    end
    if air_idx == -1
        air_idx = length(physics.tables)
    end
    
    geometry = TwoLayerGeometry{T}(rock_thickness, rock_density, rock_idx, air_idx)
    
    return compute_flux_single(physics, geometry, energy_final, elevation, charge)
end

"""
    compute_flux_gradient(physics, rock_density, rock_thickness, elevation, energy_final, charge)

Compute gradient of flux w.r.t. rock density using Zygote.

# Returns
- `flux`: The flux value
- `grad_density`: ∂flux/∂rock_density
"""
function compute_flux_gradient(physics::PhysicsTables{T},
                               rock_density::T,
                               rock_thickness::T,
                               elevation::T,
                               energy_final::T,
                               charge::T) where T<:Real
    
    f = ρ -> compute_flux_differentiable(physics, ρ, rock_thickness, 
                                         elevation, energy_final, charge)
    
    flux = f(rock_density)
    grad = Zygote.gradient(f, rock_density)[1]
    
    return flux, grad
end

"""
    run_backward_mc(physics; kwargs...)

Run the backward Monte Carlo flux calculation example.
Equivalent to the geometry.c example.
"""
function run_backward_mc(physics::PhysicsTables{T};
                         rock_thickness::T = T(100.0),
                         rock_density::T = T(2650.0),
                         elevation::T = T(45.0),
                         energy_min::T = T(1.0),
                         energy_max::Union{T, Nothing} = nothing,
                         n_samples::Int = 10000,
                         compute_gradient::Bool = false,
                         seed::Int = 42) where T<:Real
    
    if energy_max === nothing
        energy_max = energy_min
    end
    
    @info "Running backward Monte Carlo flux calculation..."
    @info "  Rock thickness: $(rock_thickness) m"
    @info "  Rock density: $(rock_density) kg/m³"
    @info "  Elevation: $(elevation)°"
    @info "  Energy range: $(energy_min) - $(energy_max) GeV"
    @info "  Samples: $(n_samples)"
    
    # Compute flux
    flux = compute_flux(physics, rock_density, rock_thickness, elevation,
                        energy_min, energy_max; n_samples=n_samples, seed=seed)
    
    # Estimate error
    sigma = rock_thickness > 0 ? flux / sqrt(T(n_samples)) : zero(T)
    
    # Compute gradient if requested
    grad_density = nothing
    if compute_gradient
        @info "Computing gradient ∂flux/∂density..."
        energy_mid = sqrt(energy_min * energy_max)
        _, grad_density = compute_flux_gradient(
            physics, rock_density, rock_thickness, elevation, energy_mid, one(T)
        )
        @info "  ∂flux/∂density ≈ $(grad_density)"
    end
    
    unit = energy_max > energy_min ? "" : "GeV⁻¹ "
    @info "Flux: $(flux) ± $(sigma) $(unit)m⁻² s⁻¹ sr⁻¹"
    
    return (flux = flux, sigma = sigma, gradient = grad_density)
end

# Keep old functions for compatibility
locals_air(state::State{T}) where T = locals_air_at_altitude(state.position[3])
medium_callback(ctx, state, geo) = nothing
create_geometry_context(physics, geometry; kwargs...) = nothing

end # module Geometry
