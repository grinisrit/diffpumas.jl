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
using ..Straggling
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
    transport_backward_step_full(physics, state, material, density, step_distance, rng;
                                 mode=:straggled, scattering=true)

Perform a single backward transport step with full physics:
- Landau/Vavilov energy straggling
- Discrete energy loss (DEL) events
- Elastic hard scattering (EHS)
- Soft multiple scattering

This matches the PUMAS implementation for accurate muon transport.

# Arguments
- `physics`: Physics tables
- `state`: Current particle state
- `material`: Material index
- `density`: Material density in kg/m³
- `step_distance`: Step distance in m
- `rng`: Random number generator
- `mode`: Energy loss mode (:csda, :mixed, :straggled)
- `scattering`: Enable scattering (default: true)

# Returns
- `new_state`: Updated particle state
- `event`: Transport event that occurred (if any)
"""
@inline function transport_backward_step_full(physics::PhysicsTables{T}, state::State{T},
                                       material::Int, density::T, 
                                       step_distance::T,
                                       rng::AbstractRNG;
                                       mode::Symbol = :straggled,
                                       scattering::Bool = true) where T<:Real
    
    ki = state.energy
    mass = physics.mass
    
    # Grammage for this step
    dX = step_distance * density
    
    # Get initial CSDA total grammage (range)
    mode_enum = ENERGY_LOSS_CSDA
    Xi = property_range(physics, mode_enum, material, ki)
    
    # Backward: Xtot increases
    Xtot = Xi + dX
    
    # Apply energy fluctuation based on mode
    kf::T = ki
    ratio::T = one(T)
    event = EVENT_NONE
    
    if mode == :straggled && ki > T(0.001)
        # Use Landau/Vavilov fluctuation from Straggling module
        kf, ratio = fluctuate_energy_loss(physics, material, ki, dX, Xi, rng; backward=true)
        kf = max(kf, ki)  # Energy must increase in backward mode
    elseif mode == :mixed && ki > T(0.001)
        # Mixed mode: 80% CSDA + fluctuation
        kf_csda = property_kinetic_energy(physics, mode_enum, material, Xtot)
        dk_csda = kf_csda - ki
        
        # Small fluctuation around CSDA
        Omega = property_straggling(physics, material, ki)
        sigma = sqrt(max(Omega * dX * T(0.2), zero(T)))
        
        if sigma > zero(T) && dk_csda > zero(T)
            u = box_muller_randn(rng)
            while abs(u) > T(3.0)
                u = box_muller_randn(rng)
            end
            dk = dk_csda + sigma * u
            kf = ki + max(dk, zero(T))
            ratio = dk / dk_csda
        else
            kf = kf_csda
        end
    else
        # Pure CSDA
        kf = property_kinetic_energy(physics, mode_enum, material, Xtot)
    end
    
    # Check for discrete events (DEL) - these matter at ALL energies in backward mode
    # DEL events give large energy boosts, especially important for high zenith angles
    if mode == :straggled
        del_occurred, X_del, k_del, process = sample_del_event(
            physics, material, ki, kf, dX, Xi, ratio, rng; backward=true
        )
        
        if del_occurred && k_del > ki
            # DEL event occurred - update final energy and grammage
            kf = k_del
            dX = X_del
            # Set event based on process (1=brems, 2=pair, 3=photo)
            if process == 1
                event = EVENT_VERTEX_BREMSSTRAHLUNG
            elseif process == 2
                event = EVENT_VERTEX_PAIR_CREATION
            else
                event = EVENT_VERTEX_PHOTONUCLEAR
            end
        end
    end
    
    # Check for elastic hard scattering (EHS) - important at all energies
    new_direction = state.direction
    
    if scattering && mode == :straggled
        ehs_occurred, X_ehs, k_ehs = sample_ehs_event(
            physics, material, ki, kf, dX, Xi, ratio, rng; backward=true
        )
        
        if ehs_occurred && k_ehs > ki
            # EHS event - sample scattering angle and rotate direction
            mu = sample_scattering_angle(physics, material, k_ehs, rng)
            new_direction = rotate_direction(state.direction, mu, rng)
            
            # Update energy at scattering point (if EHS came before DEL)
            if X_ehs < dX && event != EVENT_VERTEX_BREMSSTRAHLUNG && 
               event != EVENT_VERTEX_PAIR_CREATION && event != EVENT_VERTEX_PHOTONUCLEAR
                kf = k_ehs
                dX = X_ehs
                event = EVENT_VERTEX_COULOMB
            end
        end
    end
    
    # Apply soft multiple scattering (Highland formula) when no hard event
    if scattering && event != EVENT_VERTEX_COULOMB
        mu_soft = sample_soft_scattering(physics, material, ki, dX, rng)
        if mu_soft > zero(T)
            new_direction = rotate_direction(state.direction, mu_soft, rng)
        end
    end
    
    # Recompute step distance from actual grammage
    actual_step = dX / density
    
    # Proper time update
    gamma_i = one(T) + ki / mass
    gamma_f = one(T) + kf / mass
    gamma_avg = (gamma_i + gamma_f) / 2
    beta_avg = sqrt(max(one(T) - one(T) / gamma_avg^2, T(1e-12)))
    proper_time_step = actual_step / (beta_avg * gamma_avg)
    
    # Position update - BACKWARD mode means moving OPPOSITE to direction
    new_position = state.position - actual_step * state.direction
    
    # Weight update for backward MC
    dedx_i = property_stopping_power(physics, mode_enum, material, ki)
    dedx_f = property_stopping_power(physics, mode_enum, material, kf)
    weight_factor = dedx_f / dedx_i
    
    # Decay weight
    decay_factor = exp(-proper_time_step / physics.ctau)
    
    new_weight = state.weight * weight_factor * decay_factor
    
    # Create new state
    new_state = State{T}(
        state.charge,
        kf,
        state.distance + actual_step,
        state.grammage + dX,
        state.time + proper_time_step,
        new_weight,
        new_position,
        new_direction,
        state.decayed
    )
    
    return new_state, event
end

"""
    transport_backward_step(physics, state, material, density, step_distance; rng=nothing, straggling=false)

Perform a single backward transport step.
In backward mode, particle moves OPPOSITE to its direction vector.
The energy INCREASES as we go backward (particle gains energy).

Uses pure CSDA for high energies (>10 GeV) where fluctuations are small relative to energy.
Uses mixed/straggled mode for lower energies where stochastic effects matter.

IMPORTANT: The Monte Carlo weight is ALWAYS computed from CSDA tables, even when
straggling is applied. This matches PUMAS's handling of the weight Jacobian.
"""
function transport_backward_step(physics::PhysicsTables{T}, state::State{T},
                                  material::Int, density::T, 
                                  step_distance::T;
                                  rng::Union{AbstractRNG, Nothing} = nothing,
                                  straggling::Bool = false) where T<:Real
    
    ki = state.energy
    mass = physics.mass
    grammage_step = step_distance * density
    
    # Get initial CSDA range
    Xi = property_range(physics, ENERGY_LOSS_CSDA, material, ki)
    Xf_csda = Xi + grammage_step
    
    # CSDA energy for this step (used for weight computation)
    kf_csda = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, material, Xf_csda)
    
    # Actual final energy (may differ due to straggling)
    kf = kf_csda
    
    if rng !== nothing && straggling && ki < T(100.0)
        # Apply straggling for lower energies using Landau-like fluctuation
        Omega_i = property_straggling(physics, material, ki)
        Omega_f = property_straggling(physics, material, kf_csda)
        
        # Variance for this step
        dk_var = T(0.5) * grammage_step * (Omega_i + Omega_f)
        
        if dk_var > zero(T)
            dk_sigma = sqrt(dk_var)
            dk_expected = kf_csda - ki
            
            # Landau-like: for dk >> sigma, use truncated Gaussian
            # for dk ~ sigma, use mixed model
            if dk_expected > T(3) * dk_sigma
                # Truncated Gaussian regime
                u = box_muller_randn(rng)
                while abs(u) > T(3)
                    u = box_muller_randn(rng)
                end
                # Scale by 1/1.015387 to match PUMAS normalization
                u = u / T(1.015387)
                kf = kf_csda + u * dk_sigma
            elseif dk_expected > T(1.7320508) * dk_sigma
                # Uniform regime
                u = T(1.7320508) * (one(T) - T(2) * rand(rng))
                kf = kf_csda + u * dk_sigma
            else
                # Small step: use modified uniform
                dk32 = T(3) * dk_var
                dk02 = dk_expected * dk_expected
                a = one(T) - (dk32 - dk02) / (dk32 + T(3) * dk02)
                
                if rand(rng) <= a
                    b = T(0.5) * (dk32 + T(3) * dk02) / max(dk_expected, T(1e-12))
                    kf = ki + b * rand(rng)
                else
                    kf = ki
                end
            end
            
            # Ensure energy increases in backward mode
            kf = max(kf, ki)
        end
    end
    
    # Kinematic calculations using CSDA energies for proper averaging
    gamma_i = one(T) + ki / mass
    gamma_csda = one(T) + kf_csda / mass  # Use CSDA energy for kinematics
    gamma_avg = (gamma_i + gamma_csda) / 2
    beta_avg = sqrt(max(one(T) - one(T) / gamma_avg^2, T(1e-12)))
    
    # Proper time step (in rest frame of particle)
    proper_time_step = step_distance / (beta_avg * gamma_avg)
    
    # Position update - BACKWARD mode means moving OPPOSITE to direction
    new_position = state.position - step_distance * state.direction
    
    # Weight update for backward MC
    # CRITICAL: Use CSDA energies for the Jacobian, NOT fluctuated energies
    # This matches PUMAS's handling in step_fluctuate
    dedx_i = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, ki)
    dedx_csda = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, kf_csda)
    weight_factor = dedx_csda / dedx_i
    
    # Decay weight (using CSDA proper time)
    decay_factor = exp(-proper_time_step / physics.ctau)
    
    new_weight = state.weight * weight_factor * decay_factor
    
    return State{T}(
        state.charge, kf, state.distance + step_distance,
        state.grammage + grammage_step, state.time + proper_time_step,
        new_weight, new_position, state.direction, state.decayed
    )
end

"""
    transport_backward_through_geometry(physics, geometry, initial_state, energy_threshold; rng=nothing, straggling=false, scattering=false, energy_threshold_low=100.0)

Transport particle backward through the two-layer geometry.
Particle starts at z=0 and travels upward (opposite to downward direction).

**CRITICAL**: This implements PUMAS's dual-mode transport:
- Below energy_threshold_low: STRAGGLED energy loss + MIXED scattering (detailed Geant4-like)
- Above energy_threshold_low: MIXED energy loss + DISABLED scattering (fast MUM-like)

This mode switching is essential for accurate flux calculation!

# Arguments
- `physics`: Physics tables
- `geometry`: Two-layer geometry definition
- `initial_state`: Starting particle state at detector
- `energy_threshold`: Maximum energy threshold
- `rng`: Random number generator for straggling
- `straggling`: Enable energy straggling (default: false)
- `scattering`: Enable full scattering (default: false)
- `energy_threshold_low`: Energy threshold for mode switching (default: 100.0 GeV)
"""
function transport_backward_through_geometry(physics::PhysicsTables{T}, 
                                             geometry::TwoLayerGeometry{T},
                                             initial_state::State{T},
                                             energy_threshold::T;
                                             rng::Union{AbstractRNG, Nothing} = nothing,
                                             straggling::Bool = false,
                                             scattering::Bool = false,
                                             energy_threshold_low::T = T(100.0)) where T<:Real
    
    state = initial_state
    
    # PUMAS-style dual mode transport with energy limit triggers
    ENERGY_THRESHOLD_LOW = energy_threshold_low  # Mode switch threshold (default 100 GeV)
    STEP_EPSILON = T(1e-7)  # Small offset to cross boundaries
    
    # Outer loop: runs until we exit simulation area or reach energy_threshold
    while state.energy < energy_threshold - eps(T)
        z = state.position[3]
        
        # Check termination conditions
        if z < zero(T)
            # Below ground - particle absorbed
            break
        end
        
        if z >= PRIMARY_ALTITUDE - eps(T)
            # Reached primary altitude - success!
            break
        end
        
        if state.weight <= zero(T) || !isfinite(state.weight)
            break
        end
        
        # Determine simulation mode based on energy (PUMAS dual-mode)
        if state.energy < ENERGY_THRESHOLD_LOW - eps(T)
            # Below 100 GeV: detailed simulation (Geant4-like)
            use_straggled = straggling
            use_scattering = scattering
            current_energy_limit = ENERGY_THRESHOLD_LOW
        else
            # Above 100 GeV: fast simulation (MUM-like)
            # Use MIXED mode (partial fluctuations) but no scattering
            use_straggled = straggling  # Use mixed mode behavior
            use_scattering = false  # Disable scattering above 100 GeV
            current_energy_limit = energy_threshold
        end
        
        # Inner transport loop until hitting energy limit or boundary
        inner_steps = 0
        max_inner_steps = 50000
        
        while inner_steps < max_inner_steps
            inner_steps += 1
            z = state.position[3]
            
            # Check boundary conditions
            if z < zero(T) || z >= PRIMARY_ALTITUDE - eps(T)
                break
            end
            
            # Check energy limit for mode switching
            if state.energy >= current_energy_limit - eps(T)
                break
            end
            
            if state.weight <= zero(T) || !isfinite(state.weight)
                break
            end
            
            # Compute motion direction (opposite to direction in backward mode)
            uz_motion = -state.direction[3]
            
            # Determine medium and step size
            if z < geometry.rock_thickness
                material = geometry.rock_material
                density = geometry.rock_density
                
                # Step to rock-air interface
                if uz_motion > eps(T)
                    step_to_boundary = (geometry.rock_thickness - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = -z / uz_motion
                else
                    step_to_boundary = T(1000.0)
                end
            else
                material = geometry.air_material
                density = locals_air_at_altitude(z).density
                
                # Step to primary altitude or rock-air interface
                if uz_motion > eps(T)
                    step_to_boundary = (PRIMARY_ALTITUDE - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = (geometry.rock_thickness - z) / uz_motion
                else
                    step_to_boundary = T(1000.0)
                end
            end
            
            # Adaptive step sizing for accuracy
            Xi = property_range(physics, ENERGY_LOSS_CSDA, material, state.energy)
            
            # Step fraction depends on mode and material
            if z < geometry.rock_thickness
                # Rock: use 5% of range for accuracy
                max_grammage_frac = T(0.05)
            else
                # Air: can use larger steps (lower density, less interaction)
                max_grammage_frac = T(0.1)
            end
            
            max_grammage_step = max_grammage_frac * Xi
            max_geometric_step = max_grammage_step / max(density, T(1e-6))
            
            # For non-uniform air, also limit by density change
            if z >= geometry.rock_thickness
                h = T(12e3)  # Scale height
                uz_abs = max(abs(uz_motion), T(0.05))
                altitude_step = T(0.05) * h / uz_abs
                max_geometric_step = min(max_geometric_step, altitude_step)
            end
            
            # Final step size (add small epsilon to cross boundaries)
            step_size = min(step_to_boundary + STEP_EPSILON, max_geometric_step)
            step_size = clamp(step_size, T(STEP_MIN), T(1000.0))
            
            # Perform transport step with appropriate mode
            if use_scattering && rng !== nothing && state.energy < ENERGY_THRESHOLD_LOW
                # Full physics below 100 GeV
                state, _ = transport_backward_step_full(physics, state, material, density, step_size, rng;
                                                        mode=use_straggled ? :straggled : :csda,
                                                        scattering=true)
            elseif use_straggled && rng !== nothing && state.energy >= ENERGY_THRESHOLD_LOW
                # Above 100 GeV: MIXED mode (simplified straggling, no scattering)
                state = transport_backward_step_mixed(physics, state, material, density, step_size, rng)
            elseif use_straggled && rng !== nothing
                # Below 100 GeV without scattering
                state = transport_backward_step(physics, state, material, density, step_size;
                                                rng=rng, straggling=true)
            else
                # Pure CSDA
                state = transport_backward_step(physics, state, material, density, step_size;
                                                rng=nothing, straggling=false)
            end
        end
    end
    
    return state
end

"""
    transport_backward_step_mixed(physics, state, material, density, step_distance, rng)

Perform a backward transport step in MIXED mode (for energies > 100 GeV).
Uses MIXED energy loss (partial straggling) with DISABLED scattering.
This is the fast "MUM-like" simulation mode.
"""
function transport_backward_step_mixed(physics::PhysicsTables{T}, state::State{T},
                                       material::Int, density::T, 
                                       step_distance::T,
                                       rng::AbstractRNG) where T<:Real
    
    ki = state.energy
    mass = physics.mass
    grammage_step = step_distance * density
    
    # Get CSDA properties
    Xi = property_range(physics, ENERGY_LOSS_CSDA, material, ki)
    Xf = Xi + grammage_step
    kf_csda = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, material, Xf)
    
    # MIXED mode: use CEL (Continuous Energy Loss) with radiative fluctuations
    # This accounts for hard radiative processes probabilistically
    kf = kf_csda
    
    # Add fluctuation from radiative processes
    Omega = property_straggling(physics, material, ki)
    dk_expected = kf_csda - ki
    
    if Omega > zero(T) && dk_expected > zero(T)
        # Variance scaled down for MIXED mode (80% CSDA + 20% stochastic)
        sigma = sqrt(Omega * grammage_step * T(0.2))
        
        # Gaussian fluctuation (truncated)
        u = box_muller_randn(rng)
        u = clamp(u, T(-3), T(3))
        
        kf = kf_csda + sigma * u
        kf = max(kf, ki)  # Energy must increase in backward mode
    end
    
    # Kinematics (using average of actual energies)
    gamma_i = one(T) + ki / mass
    gamma_f = one(T) + kf / mass
    gamma_avg = (gamma_i + gamma_f) / 2
    beta_avg = sqrt(max(one(T) - one(T) / gamma_avg^2, T(1e-12)))
    
    proper_time_step = step_distance / (beta_avg * gamma_avg)
    
    # Position update (backward = opposite to direction)
    new_position = state.position - step_distance * state.direction
    
    # Weight update using CSDA Jacobian
    dedx_i = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, ki)
    dedx_f = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, kf_csda)
    weight_factor = dedx_f / dedx_i
    
    # Decay factor
    decay_factor = exp(-proper_time_step / physics.ctau)
    
    new_weight = state.weight * weight_factor * decay_factor
    
    return State{T}(
        state.charge, kf, state.distance + step_distance,
        state.grammage + grammage_step, state.time + proper_time_step,
        new_weight, new_position, state.direction, state.decayed
    )
end

"""
    compute_flux_single(physics, geometry, energy_final, elevation, charge; rng=nothing, straggling=false, scattering=false, energy_threshold_low=100.0)

Compute flux contribution from a single backward MC particle.
"""
function compute_flux_single(physics::PhysicsTables{T}, 
                             geometry::TwoLayerGeometry{T},
                             energy_final::T,
                             elevation::T,
                             charge::T;
                             rng::Union{AbstractRNG, Nothing} = nothing,
                             straggling::Bool = false,
                             scattering::Bool = false,
                             energy_threshold_low::T = T(100.0)) where T<:Real
    
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
    
    # Transport backward through geometry with optional straggling and scattering
    final_state = transport_backward_through_geometry(physics, geometry, state, energy_threshold;
                                                      rng=rng, straggling=straggling, scattering=scattering,
                                                      energy_threshold_low=energy_threshold_low)
    
    # Check if reached primary altitude
    if final_state.position[3] >= PRIMARY_ALTITUDE - T(1.0)
        # Get the cos(theta) of the final direction (for flux evaluation)
        # In backward MC, the "incoming" direction at primary altitude
        # is opposite to what we traced
        cos_theta_final = -final_state.direction[3]
        
        # Clamp to valid range - particles that scattered too much
        # to be downward-going contribute zero flux
        cos_theta_final = clamp(cos_theta_final, zero(T), one(T))
        
        if cos_theta_final <= zero(T)
            return zero(T)
        end
        
        # Compute flux contribution
        flux = flux_gccly(cos_theta_final, final_state.energy, charge)
        return final_state.weight * flux
    else
        return zero(T)
    end
end

"""
    compute_flux(physics, rock_density, rock_thickness, elevation, energy_min, energy_max; n_samples, seed, straggling, scattering, energy_threshold_low)

Compute integrated muon flux using backward Monte Carlo.

# Arguments
- `physics`: Physics tables
- `rock_density`: Rock density in kg/m³
- `rock_thickness`: Rock thickness in m
- `elevation`: Elevation angle in degrees (90 = vertical, 0 = horizontal)
- `energy_min`: Minimum energy in GeV
- `energy_max`: Maximum energy in GeV
- `n_samples`: Number of MC samples (default: 1000)
- `seed`: Random seed (default: 42)
- `straggling`: Enable energy straggling (default: true)
- `scattering`: Enable full scattering model (default: true)
- `energy_threshold_low`: Energy threshold for mode switching in GeV (default: 100.0)
"""
function compute_flux(physics::PhysicsTables{T}, 
                      rock_density::T,
                      rock_thickness::T,
                      elevation::T,
                      energy_min::T,
                      energy_max::T;
                      n_samples::Int = 1000,
                      seed::Int = 42,
                      straggling::Bool = true,
                      scattering::Bool = true,
                      energy_threshold_low::T = T(100.0)) where T<:Real
    
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
        
        # Compute contribution with optional straggling and scattering
        wi = wf * compute_flux_single(physics, geometry, kf, elevation, charge;
                                      rng=rng, straggling=straggling, scattering=scattering,
                                      energy_threshold_low=energy_threshold_low)
        
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
                         seed::Int = 42,
                         straggling::Bool = true,
                         scattering::Bool = true,
                         energy_threshold_low::T = T(100.0)) where T<:Real
    
    if energy_max === nothing
        energy_max = energy_min
    end
    
    @info "Running backward Monte Carlo flux calculation..."
    @info "  Rock thickness: $(rock_thickness) m"
    @info "  Rock density: $(rock_density) kg/m³"
    @info "  Elevation: $(elevation)°"
    @info "  Energy range: $(energy_min) - $(energy_max) GeV"
    @info "  Samples: $(n_samples)"
    @info "  Straggling: $(straggling), Scattering: $(scattering)"
    @info "  Energy threshold: $(energy_threshold_low) GeV"
    
    # Compute flux
    flux = compute_flux(physics, rock_density, rock_thickness, elevation,
                        energy_min, energy_max; n_samples=n_samples, seed=seed,
                        straggling=straggling, scattering=scattering,
                        energy_threshold_low=energy_threshold_low)
    
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
