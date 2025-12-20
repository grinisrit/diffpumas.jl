"""
    Transport

Monte Carlo transport algorithms for particle propagation.
Designed to be differentiable with Zygote.jl.
"""
module Transport

using ..Constants
using ..Types
using ..Physics
using LinearAlgebra
using ChainRulesCore
using Random

export transport_csda_uniform, transport_csda_magnetic
export transport_single_medium, transport_with_density
export compute_grammage_step, compute_energy_loss

"""
    transport_csda_uniform(physics, state, material, locals, distance_max)

Transport particle using CSDA in uniform medium.
Returns updated state - fully differentiable.

# Arguments
- `physics`: Physics tables
- `state`: Initial particle state
- `material`: Material index
- `locals`: Local medium properties (includes density)
- `distance_max`: Maximum distance to travel

# Returns
- `new_state`: Updated particle state
- `event`: Transport event flag
"""
function transport_csda_uniform(physics::PhysicsTables{T}, state::State{T}, 
                                material::Int, locals::Locals{T},
                                distance_max::T) where T<:Real
    
    density = locals.density
    ki = state.energy
    mass = physics.mass
    
    # Get mode (CSDA)
    mode = ENERGY_LOSS_CSDA
    
    # Compute grammage for this distance
    grammage_step = distance_max * density
    
    # Get initial range
    Xi = property_range(physics, mode, material, ki)
    
    # Get range at end of step
    Xf = Xi - grammage_step
    
    # Check if particle stops
    event = EVENT_NONE
    if Xf <= zero(T)
        # Particle stops
        Xf = zero(T)
        grammage_step = Xi
        distance = grammage_step / density
        kf = zero(T)
        event = EVENT_LIMIT_ENERGY
    else
        # Get final energy
        kf = property_kinetic_energy(physics, mode, material, Xf)
        distance = distance_max
    end
    
    # Proper time update (simplified)
    gamma_i = one(T) + ki / mass
    gamma_f = one(T) + kf / mass
    beta_avg = sqrt(one(T) - 2 / (gamma_i + gamma_f))
    proper_time_step = distance / beta_avg
    
    # Update position (straight line in CSDA without magnetic field)
    sgn = one(T)  # Forward mode
    new_position = state.position + sgn * distance * state.direction
    
    # Update weight for backward mode (identity for forward)
    new_weight = state.weight
    
    # Decay weight (if weighted decay mode)
    decay_factor = exp(-proper_time_step / physics.ctau)
    new_weight = new_weight * decay_factor
    
    # Create new state
    new_state = State{T}(
        state.charge,
        kf,
        state.distance + distance,
        state.grammage + grammage_step,
        state.time + proper_time_step,
        new_weight,
        new_position,
        state.direction,
        state.decayed
    )
    
    return new_state, event
end

"""
    transport_csda_magnetic(physics, state, material, locals, distance_max)

Transport with CSDA in magnetic field.
Tracks helical trajectory - differentiable.
"""
function transport_csda_magnetic(physics::PhysicsTables{T}, state::State{T},
                                 material::Int, locals::Locals{T},
                                 distance_max::T) where T<:Real
    
    density = locals.density
    B = locals.magnet
    B_norm = norm(B)
    
    if B_norm < T(1e-12)
        # No magnetic field, use straight transport
        return transport_csda_uniform(physics, state, material, locals, distance_max)
    end
    
    ki = state.energy
    mass = physics.mass
    mode = ENERGY_LOSS_CSDA
    
    # Momentum and velocity
    gamma_i = one(T) + ki / mass
    beta_i = sqrt(one(T) - one(T) / gamma_i^2)
    p_i = sqrt(ki * (ki + 2mass))
    
    # Larmor radius
    q = state.charge
    r_L = p_i / (q * LARMOR_FACTOR * B_norm)
    
    # Transverse and parallel components of velocity
    u = state.direction
    B_hat = B / B_norm
    u_par = dot(u, B_hat) * B_hat
    u_perp = u - u_par
    u_perp_norm = norm(u_perp)
    
    # Step in grammage
    grammage_step = distance_max * density
    Xi = property_range(physics, mode, material, ki)
    Xf = Xi - grammage_step
    
    event = EVENT_NONE
    if Xf <= zero(T)
        Xf = zero(T)
        grammage_step = Xi
        kf = zero(T)
        event = EVENT_LIMIT_ENERGY
    else
        kf = property_kinetic_energy(physics, mode, material, Xf)
    end
    
    distance = grammage_step / density
    
    # Final momentum and Larmor radius
    p_f = sqrt(max(zero(T), kf * (kf + 2mass)))
    gamma_f = one(T) + kf / mass
    
    # Average Larmor radius for the step
    r_L_avg = (p_i + p_f) / (2 * abs(q) * LARMOR_FACTOR * B_norm)
    
    # Rotation angle
    arc_length = distance * u_perp_norm
    θ = arc_length / r_L_avg
    
    # New direction (rotation in plane perpendicular to B)
    if u_perp_norm > T(1e-12)
        perp_hat = u_perp / u_perp_norm
        # Vector perpendicular to both B and u_perp
        side_hat = cross(B_hat, perp_hat)
        
        new_u_perp = u_perp_norm * (cos(θ) * perp_hat + sin(θ) * side_hat)
        new_u = u_par + new_u_perp
        new_u = new_u / norm(new_u)  # Renormalize
    else
        new_u = u
    end
    
    # Position update (simplified - assumes small deflection)
    new_position = state.position + distance * (u + new_u) / 2
    
    # Proper time
    beta_avg = sqrt(one(T) - 2 / (gamma_i + gamma_f))
    proper_time_step = distance / beta_avg
    
    # Weight update
    new_weight = state.weight * exp(-proper_time_step / physics.ctau)
    
    new_state = State{T}(
        state.charge,
        kf,
        state.distance + distance,
        state.grammage + grammage_step,
        state.time + proper_time_step,
        new_weight,
        new_position,
        new_u,
        state.decayed
    )
    
    return new_state, event
end

"""
    compute_grammage_step(physics, state, material, locals, distance)

Compute grammage for a given distance step.
This function is explicitly differentiable w.r.t. density.
"""
function compute_grammage_step(physics::PhysicsTables{T}, state::State{T},
                               material::Int, locals::Locals{T},
                               distance::T) where T<:Real
    return distance * locals.density
end

# Explicit rrule for grammage computation
function ChainRulesCore.rrule(::typeof(compute_grammage_step), 
                              physics::PhysicsTables{T}, state::State{T},
                              material::Int, locals::Locals{T}, 
                              distance::T) where T
    grammage = compute_grammage_step(physics, state, material, locals, distance)
    
    function grammage_pullback(Δ)
        # ∂grammage/∂density = distance
        Δ_density = Δ * distance
        Δ_locals = Tangent{Locals{T}}(; density = Δ_density, magnet = ZeroTangent())
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_locals, Δ * locals.density)
    end
    
    return grammage, grammage_pullback
end

"""
    compute_energy_loss(physics, state, material, grammage_step)

Compute energy loss for a given grammage step.
"""
function compute_energy_loss(physics::PhysicsTables{T}, state::State{T},
                             material::Int, grammage_step::T;
                             mode::EnergyLossMode = ENERGY_LOSS_CSDA) where T<:Real
    
    ki = state.energy
    Xi = property_range(physics, mode, material, ki)
    Xf = max(zero(T), Xi - grammage_step)
    kf = property_kinetic_energy(physics, mode, material, Xf)
    
    return ki - kf
end

"""
    transport_single_medium(physics, state, medium, distance_max; kwargs...)

Transport particle through a single uniform medium.
Main differentiable transport function.

# Arguments
- `physics`: Physics tables
- `state`: Initial particle state
- `medium`: Propagation medium
- `distance_max`: Maximum distance to travel

# Keyword Arguments
- `mode`: Energy loss mode (default: CSDA)

# Returns
- `new_state`: Final particle state
- `event`: Transport event
"""
function transport_single_medium(physics::PhysicsTables{T}, state::State{T},
                                 medium::Medium{T}, distance_max::T;
                                 mode::EnergyLossMode = ENERGY_LOSS_CSDA) where T<:Real
    
    # Get local properties
    if medium.locals_cb !== nothing
        locals, step_hint = medium.locals_cb(medium, state)
        # Could use step_hint for adaptive stepping
    else
        # Use default density
        table = physics.tables[medium.material]
        locals = Locals{T}(table.density)
    end
    
    # Check for valid density
    if locals.density <= zero(T)
        # Use default
        table = physics.tables[medium.material]
        locals = Locals{T}(table.density)
    end
    
    # Choose transport method based on magnetic field
    B_norm = norm(locals.magnet)
    
    if B_norm < T(1e-12)
        return transport_csda_uniform(physics, state, medium.material, locals, distance_max)
    else
        return transport_csda_magnetic(physics, state, medium.material, locals, distance_max)
    end
end

"""
    transport_with_density(physics, state, material, density, distance_max)

Transport with explicit density parameter for differentiation.
This is the key function for computing ∂flux/∂density.
"""
function transport_with_density(physics::PhysicsTables{T}, state::State{T},
                                material::Int, density::T, distance_max::T) where T<:Real
    
    locals = Locals{T}(density)
    
    ki = state.energy
    mass = physics.mass
    mode = ENERGY_LOSS_CSDA
    
    # Grammage step - key differentiable quantity
    grammage_step = distance_max * density
    
    # Range calculation
    Xi = property_range(physics, mode, material, ki)
    Xf = Xi - grammage_step
    
    event = EVENT_NONE
    if Xf <= zero(T)
        Xf = zero(T)
        grammage_step = Xi
        distance = grammage_step / density
        kf = zero(T)
        event = EVENT_LIMIT_ENERGY
    else
        kf = property_kinetic_energy(physics, mode, material, Xf)
        distance = distance_max
    end
    
    # Proper time
    gamma_i = one(T) + ki / mass
    gamma_f = one(T) + kf / mass
    beta_avg = sqrt(one(T) - 2 / (gamma_i + gamma_f))
    proper_time_step = distance / beta_avg
    
    # Position update
    new_position = state.position + distance * state.direction
    
    # Weight update with decay
    new_weight = state.weight * exp(-proper_time_step / physics.ctau)
    
    new_state = State{T}(
        state.charge,
        kf,
        state.distance + distance,
        state.grammage + grammage_step,
        state.time + proper_time_step,
        new_weight,
        new_position,
        state.direction,
        state.decayed
    )
    
    return new_state, event
end

# ChainRulesCore rrule for transport_with_density
function ChainRulesCore.rrule(::typeof(transport_with_density),
                              physics::PhysicsTables{T}, state::State{T},
                              material::Int, density::T, distance_max::T) where T
    
    new_state, event = transport_with_density(physics, state, material, density, distance_max)
    
    function transport_pullback(Δ)
        Δ_state, Δ_event = Δ
        
        # Compute sensitivities
        ki = state.energy
        mass = physics.mass
        mode = ENERGY_LOSS_CSDA
        
        # ∂grammage/∂density = distance_max (main contribution)
        Δ_density = zero(T)
        
        if Δ_state !== nothing && Δ_state !== ZeroTangent()
            # Sensitivity of grammage to density
            if hasproperty(Δ_state, :grammage) && Δ_state.grammage !== nothing
                Δ_density += Δ_state.grammage * distance_max
            end
            
            # Sensitivity of final energy to density (through grammage)
            if hasproperty(Δ_state, :energy) && Δ_state.energy !== nothing
                # ∂kf/∂density ≈ -dedx * distance_max
                dedx = property_stopping_power(physics, mode, material, ki)
                Δ_density += Δ_state.energy * (-dedx * distance_max)
            end
            
            # Sensitivity of weight to density (through proper time and energy)
            if hasproperty(Δ_state, :weight) && Δ_state.weight !== nothing
                # Complex dependency through proper time and decay
                # Simplified: ignore higher-order terms
            end
        end
        
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), Δ_density, NoTangent())
    end
    
    return (new_state, event), transport_pullback
end

end # module Transport

