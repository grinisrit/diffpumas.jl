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
using ..MixedTransport
using LinearAlgebra
using Zygote
using ChainRulesCore
using Random

export TwoLayerGeometry, create_geometry_context
export compute_flux, compute_flux_gradient, compute_flux_differentiable
export compute_flux_csda_direct, compute_flux_differentiable_csda, compute_flux_gradient_csda
export locals_rock, locals_air, locals_air_at_altitude, medium_callback
export run_backward_mc
export PRIMARY_ALTITUDE, compute_air_grammage, compute_rock_grammage
export compute_decay_weight_from_path
export transport_backward_step, transport_backward_step_full, transport_backward_step_mixed
export compute_flux_single_with_state

# Primary altitude for sampling (m)
const PRIMARY_ALTITUDE = 3e4

"""
    TwoLayerGeometry{T}

Layered geometry with rock, air, and an optional shallow porous layer.

When `porous_material > 0`, the top `porous_thickness` metres of the rock
column (i.e. z ∈ [rock_thickness − porous_thickness, rock_thickness)) use
`porous_material` / `porous_density` instead of the bulk rock.

# Fields
- `rock_thickness::T`: Total rock column thickness in m
- `rock_density::T`: Bulk rock density in kg/m³
- `rock_material::Int`: Material index for rock
- `air_material::Int`: Material index for air
- `porous_material::Int`: Material index for shallow porous layer (≤0 to disable)
- `porous_density::T`: Density of the porous layer in kg/m³
- `porous_thickness::T`: Thickness of the porous layer in m
"""
struct TwoLayerGeometry{T<:Real}
    rock_thickness::T
    rock_density::T
    rock_material::Int
    air_material::Int
    porous_material::Int
    porous_density::T
    porous_thickness::T
    primary_altitude::T
end

function TwoLayerGeometry{T}(rock_thickness, rock_density, rock_material, air_material,
                              porous_material, porous_density, porous_thickness) where T<:Real
    TwoLayerGeometry{T}(rock_thickness, rock_density, rock_material, air_material,
                        porous_material, porous_density, porous_thickness, T(PRIMARY_ALTITUDE))
end

function TwoLayerGeometry{T}(rock_thickness, rock_density, rock_material, air_material) where T<:Real
    TwoLayerGeometry{T}(rock_thickness, rock_density, rock_material, air_material,
                        -1, zero(T), zero(T), T(PRIMARY_ALTITUDE))
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

Accepts either a single material index (Int) or a MaterialMixture.

# Arguments
- `physics`: Physics tables
- `state`: Current particle state
- `material`: Material index (Int) or MaterialMixture
- `density`: Material density in kg/m³
- `step_distance`: Step distance in m
- `rng`: Random number generator
- `mode`: Energy loss mode (:csda, :mixed, :straggled)
- `scattering`: Enable scattering (default: true)

# Returns
- `new_state`: Updated particle state
- `event`: Transport event that occurred (if any)
"""
# Backward-compatible overload: Int → MaterialMixture
@inline function transport_backward_step_full(physics::PhysicsTables{T}, state::State{T},
                                       material::Int, density::T, 
                                       step_distance::T,
                                       rng::AbstractRNG;
                                       mode::Symbol = :straggled,
                                       scattering::Bool = true,
                                       energy_limit::T = T(-1),
                                       kernel_context=nothing) where T<:Real
    return transport_backward_step_full(physics, state, MaterialMixture(material), density,
                                        step_distance, rng; mode=mode, scattering=scattering,
                                        energy_limit=energy_limit,kernel_context=kernel_context)
end

@inline function transport_backward_step_full(physics::PhysicsTables{T}, state::State{T},
                                       mix::MaterialMixture, density::T, 
                                       step_distance::T,
                                       rng::AbstractRNG;
                                       mode::Symbol = :straggled,
                                       scattering::Bool = true,
                                       energy_limit::T = T(-1),
                                       kernel_context=nothing) where T<:Real

    mode in (:csda,:mixed,:straggled) || throw(ArgumentError("Unknown transport mode $mode"))
    if mode==:mixed
        scattering && throw(ArgumentError("Prepared MIXED transport currently requires scattering=false"))
        return transport_backward_step_mixed(physics,state,mix,density,step_distance,rng;
            energy_limit=energy_limit,kernel_context=kernel_context),EVENT_NONE
    end
    kernel=mode==:straggled ? (kernel_context===nothing ? prepare_mixed_kernel(physics,mix) : MixedTransport.check_kernel_owner(kernel_context,physics,mix)) : nothing
    ki = state.energy
    mass = physics.mass
    
    # Grammage for this step
    dX = step_distance * density
    
    # PUMAS uses MIXED mode for range lookups in STRAGGLED mode
    # Official CEL Jacobian clamps STRAGGLED to restricted MIXED tables.
    range_mode = mode==:csda ? ENERGY_LOSS_CSDA : ENERGY_LOSS_MIXED
    weight_mode = mode==:csda ? ENERGY_LOSS_CSDA : ENERGY_LOSS_MIXED
    Xi = property_range(physics, range_mode, mix, ki)
    
    # Backward: Xtot increases
    Xtot = Xi + dX
    
    # Apply energy fluctuation based on mode
    kf::T = ki
    ratio::T = one(T)
    event = EVENT_NONE
    hit_energy_limit = false
    
    if mode == :straggled && ki > T(0.001)
        kf, ratio = fluctuate_energy_loss(physics, mix, ki, dX, Xi, rng; backward=true)
        kf = max(kf, ki)
    else
        kf = property_kinetic_energy(physics, range_mode, mix, Xtot)
    end
    
    # Check for energy limit
    if energy_limit > zero(T) && kf >= energy_limit
        dk = abs(ki - kf)
        if dk > zero(T)
            dk_to_limit = abs(energy_limit - ki)
            frac = dk_to_limit / dk
            dX = dX * frac
            kf = energy_limit
            hit_energy_limit = true
            event = EVENT_LIMIT_ENERGY
        end
    end
    
    # Sample both DEL and EHS; the event with shorter interaction length wins
    # This matches PUMAS C transport_limit where DEL and EHS compete
    del_process = 0
    del_occurred_flag = false
    ehs_occurred_flag = false
    ehs_kinetic = zero(T)
    
    del_occurred = false
    X_del = T(Inf)
    k_del = zero(T)
    del_proc = 0
    
    ehs_occurred = false
    X_ehs = T(Inf)
    k_ehs = zero(T)
    
    if mode == :straggled
        # Official step-local endpoint majorant and one-candidate thinning.
        rate=max(MixedTransport.lookup(ki,kernel.energies,kernel.sigma),
                 MixedTransport.lookup(kf,kernel.energies,kernel.sigma))
        if rate>0
            candidate=randexp(rng)/rate
            if candidate<dX
                kh=property_kinetic_energy(physics,ENERGY_LOSS_MIXED,mix,Xi+candidate)
                pre=ki+ratio*abs(kh-ki)
                if pre>0 && rand(rng)<=MixedTransport.lookup(pre,kernel.energies,kernel.sigma)/rate
                    del_occurred=true;X_del=candidate;k_del=pre
                end
            end
        end
        
        if scattering
            ehs_occurred, X_ehs_s, k_ehs_s = sample_ehs_event(
                physics, mix, ki, kf, dX, Xi, ratio, rng; backward=true
            )
            if ehs_occurred && k_ehs_s > zero(T)
                X_ehs = X_ehs_s
                k_ehs = k_ehs_s
            else
                ehs_occurred = false
            end
        end
    end
    
    # Whichever event has shorter interaction length wins
    if del_occurred && ehs_occurred
        if X_del <= X_ehs
            ehs_occurred = false
        else
            del_occurred = false
        end
    end
    
    discrete_weight=one(T)
    target_element=nothing
    continuous_kf=kf
    if del_occurred
        continuous_kf=k_del
        kf,discrete_weight,del_proc,target_element=reverse_del_event(kernel,Float64(k_del),rng;details=true)
        dX = X_del
        del_process = del_proc
        del_occurred_flag = true
        if del_proc == 1
            event = EVENT_VERTEX_BREMSSTRAHLUNG
        elseif del_proc == 2
            event = EVENT_VERTEX_PAIR_CREATION
        elseif del_proc==3
            event = EVENT_VERTEX_PHOTONUCLEAR
        elseif del_proc==4
            event = EVENT_VERTEX_DELTA_RAY
        else
            event = EVENT_NONE
        end
    end
    
    if ehs_occurred
        kf=k_ehs;continuous_kf=k_ehs;dX=X_ehs
        ehs_occurred_flag = true
        ehs_kinetic = k_ehs
        event = EVENT_VERTEX_COULOMB
    end
    
    # Recompute step distance from actual grammage (needed for MSC and position update)
    actual_step = dX / density
    
    # Scattering: DEL angular deflection + EHS deflection + soft MCS
    # Matches PUMAS C: direction updates require scattering == PUMAS_MODE_MIXED
    new_direction = state.direction
    if scattering
        # DEL angular deflection (applied first, matching PUMAS C transport_do_del)
        if del_occurred_flag && del_process > 0
            e=target_element
            mu_del = if del_process==1
                Straggling.polar_bremsstrahlung(mass,kf,continuous_kf,T(e.Z),T(e.A),rng)
            elseif del_process==2
                Straggling.polar_pair_production(mass,kf,continuous_kf,T(e.Z),T(e.A),rng)
            elseif del_process==3
                Straggling.polar_photonuclear(mass,kf,continuous_kf,T(e.Z),T(e.A),rng)
            else
                Straggling.polar_ionisation(mass,kf,continuous_kf)
            end
            if mu_del > zero(T)
                new_direction = rotate_direction(new_direction, mu_del, rng)
            end
        end
        
        # EHS angular deflection (large-angle Coulomb scatter)
        if ehs_occurred_flag
            mu_ehs = sample_scattering_angle(physics, mix, ehs_kinetic, rng)
            if mu_ehs > zero(T)
                new_direction = rotate_direction(new_direction, mu_ehs, rng)
            end
        end
        
        # Soft multiple scattering (MSC)
        # PUMAS C: ilb1 = 0.25 * step * (invlb1_start + invlb1_end)
        mu_soft = sample_soft_scattering(physics, mix, ki, continuous_kf, actual_step, density, rng)
        if mu_soft > zero(T)
            new_direction = rotate_direction(new_direction, mu_soft, rng)
        end
    end
    
    # Proper time update
    proper_time_step = actual_step * mass / 2 * (inv(sqrt(ki*(ki+2mass))) + inv(sqrt(continuous_kf*(continuous_kf+2mass))))
    
    # Position update - BACKWARD mode means moving OPPOSITE to direction
    new_position = state.position - actual_step * state.direction
    
    # Continuous Jacobian ends before the instantaneous reverse DEL jump.
    dedx_i = property_stopping_power(physics, weight_mode, mix, ki)
    dedx_f = property_stopping_power(physics, weight_mode, mix, continuous_kf)
    weight_factor = dedx_f / dedx_i
    
    # Decay weight
    decay_factor = exp(-proper_time_step / physics.ctau)
    
    new_weight = state.weight * weight_factor * decay_factor * discrete_weight
    
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

Perform a single backward transport step (CSDA or with simple straggling).
Accepts either a single material index (Int) or a MaterialMixture.
"""
# Backward-compatible overload
function transport_backward_step(physics::PhysicsTables{T}, state::State{T},
                                  material::Int, density::T, 
                                  step_distance::T;
                                  rng::Union{AbstractRNG, Nothing} = nothing,
                                  straggling::Bool = false) where T<:Real
    return transport_backward_step(physics, state, MaterialMixture(material), density,
                                   step_distance; rng=rng, straggling=straggling)
end

# Keep rejection sampling out of the deterministic step's reverse-mode trace.
# Random draws and all three straggling regimes retain their original physics.
function _sample_backward_straggling_energy(physics::PhysicsTables{T}, mix::MaterialMixture,
                                             ki::T, kf_csda::T, grammage_step::T,
                                             rng::AbstractRNG) where T<:Real
    kf = kf_csda
    Omega_i = property_straggling(physics, mix, ki)
    Omega_f = property_straggling(physics, mix, kf_csda)
    dk_var = T(0.5) * grammage_step * (Omega_i + Omega_f)

    if dk_var > zero(T)
        dk_sigma = sqrt(dk_var)
        dk_expected = kf_csda - ki

        if dk_expected > T(3) * dk_sigma
            u = box_muller_randn(rng)
            while abs(u) > T(3)
                u = box_muller_randn(rng)
            end
            u = u / T(1.015387)
            kf = kf_csda + u * dk_sigma
        elseif dk_expected > T(1.7320508) * dk_sigma
            u = T(1.7320508) * (one(T) - T(2) * rand(rng))
            kf = kf_csda + u * dk_sigma
        else
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
        kf = max(kf, ki)
    end
    return kf
end

# Dispatch on the RNG so a deterministic call has no stochastic branch on its
# reverse tape (including short-circuit conditions involving `nothing`).
_backward_step_energy(physics, mix, ki, kf, grammage, ::Nothing, straggling) = kf
function _backward_step_energy(physics, mix, ki::T, kf, grammage,
                               rng::AbstractRNG, straggling) where T
    return straggling && ki < T(100) ?
        _sample_backward_straggling_energy(physics, mix, ki, kf, grammage, rng) : kf
end

function transport_backward_step(physics::PhysicsTables{T}, state::State{T},
                                  mix::MaterialMixture, density::T, 
                                  step_distance::T;
                                  rng::Union{AbstractRNG, Nothing} = nothing,
                                  straggling::Bool = false) where T<:Real
    
    ki = state.energy
    mass = physics.mass
    grammage_step = step_distance * density
    
    Xi = property_range(physics, ENERGY_LOSS_CSDA, mix, ki)
    Xf_csda = Xi + grammage_step
    kf_csda = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, mix, Xf_csda)
    kf = _backward_step_energy(physics, mix, ki, kf_csda, grammage_step, rng, straggling)

    gamma_i = one(T) + ki / mass
    gamma_csda = one(T) + kf_csda / mass
    gamma_avg = (gamma_i + gamma_csda) / 2
    beta_avg = sqrt(max(one(T) - one(T) / gamma_avg^2, T(1e-12)))
    proper_time_step = step_distance / (beta_avg * gamma_avg)
    new_position = state.position - step_distance * state.direction
    
    dedx_i = property_stopping_power(physics, ENERGY_LOSS_CSDA, mix, ki)
    dedx_csda = property_stopping_power(physics, ENERGY_LOSS_CSDA, mix, kf_csda)
    weight_factor = dedx_csda / dedx_i
    decay_factor = exp(-proper_time_step / physics.ctau)
    new_weight = state.weight * weight_factor * decay_factor
    
    return State{T}(
        state.charge, kf, state.distance + step_distance,
        state.grammage + grammage_step, state.time + proper_time_step,
        new_weight, new_position, state.direction, state.decayed
    )
end

# Compute each adaptive limit in its own branch before combining them. This
# avoids loop-carried branch-local values in Zygote's reverse control flow.
function _backward_geometric_step_limit(physics::PhysicsTables{T}, state::State{T},
                                         material::Int, density::T, uz_motion::T,
                                         in_air::Bool, use_scattering::Bool) where T<:Real
    accuracy = T(0.01)
    Xi = property_range(physics, ENERGY_LOSS_CSDA, material, state.energy)
    range_limit = accuracy * Xi / max(density, T(1e-6))
    scattering_limit = if use_scattering
        lb1 = property_transport_path(physics, material, state.energy)
        lb1 > zero(T) && density > zero(T) ? accuracy * lb1 / density : T(Inf)
    else
        T(Inf)
    end
    # Bound density changes in the exponential atmosphere.
    air_limit = in_air ? T(0.05) * T(12e3) / max(abs(uz_motion), T(0.05)) : T(Inf)
    return min(range_limit, scattering_limit, air_limit)
end

# Explicit preparation is batch-scoped. Contexts copy all stochastic kernel
# data. Physics must remain unmodified during a batch: low-energy fluctuation
# and scattering helpers also read the original tables.
_prepare_mixed_contexts(physics,::Nothing,enabled) = nothing
function _prepare_mixed_contexts(physics,rng::AbstractRNG,enabled)
    enabled || return nothing
    return prepare_mixed_kernels(physics,eachindex(physics.tables))
end

# Keep the stochastic mode-selection branches outside the adaptive loop's
# reverse tape. The no-RNG path has always used deterministic CSDA.
function _backward_geometry_step(physics, state, material, density, step_size,
                                  ::Nothing, use_straggled, use_mixed_mode,
                                  use_scattering, energy_limit, mixed_contexts)
    return transport_backward_step(physics, state, material, density, step_size)
end

function _backward_geometry_step(physics, state, material, density, step_size,
                                  rng::AbstractRNG, use_straggled, use_mixed_mode,
                                  use_scattering, energy_limit, mixed_contexts)
    if use_straggled
        result, _ = transport_backward_step_full(physics, state, material, density,
            step_size, rng; mode=:straggled, scattering=use_scattering, energy_limit=energy_limit,
            kernel_context=mixed_contexts[material])
        return result
    elseif use_mixed_mode
        return transport_backward_step_mixed(physics, state, material, density,
                                              step_size, rng; energy_limit=energy_limit,
                                              kernel_context=mixed_contexts[material])
    else
        return transport_backward_step(physics, state, material, density, step_size)
    end
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
                                             energy_threshold_low::T = T(100.0),
                                             mixed_contexts=nothing) where T<:Real
    
    contexts = mixed_contexts === nothing ? _prepare_mixed_contexts(physics,rng,straggling) : mixed_contexts
    state = initial_state
    primary_alt = geometry.primary_altitude
    
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
        
        if z >= primary_alt - eps(T)
            # Reached primary altitude - success!
            break
        end
        
        if state.weight <= zero(T) || !isfinite(state.weight)
            break
        end
        
        # Determine simulation mode based on energy (PUMAS dual-mode)
        # This must EXACTLY match C geometry.c behavior:
        # - Below threshold: PUMAS_MODE_STRAGGLED + PUMAS_MODE_MIXED scattering
        # - Above threshold: PUMAS_MODE_MIXED energy loss + PUMAS_MODE_DISABLED scattering
        if state.energy < ENERGY_THRESHOLD_LOW - eps(T)
            # Below threshold: detailed simulation (Geant4-like)
            # PUMAS_MODE_STRAGGLED = full stochastic energy loss
            use_straggled = straggling
            use_mixed_mode = false  # Full straggling, not mixed
            use_scattering = scattering
            current_energy_limit = min(ENERGY_THRESHOLD_LOW,energy_threshold)
        else
            # Above threshold: fast simulation (MUM-like)
            # PUMAS_MODE_MIXED = deterministic CEL with stochastic DEL
            use_straggled = false  # No straggling above threshold
            use_mixed_mode = straggling  # Use MIXED mode instead
            use_scattering = false  # Disable scattering above threshold
            current_energy_limit = energy_threshold
        end
        
        # Inner transport loop until hitting energy limit or boundary
        inner_steps = 0
        max_inner_steps = 50000
        
        while inner_steps < max_inner_steps
            inner_steps += 1
            z = state.position[3]
            
            # Check boundary conditions
            if z < zero(T) || z >= primary_alt - eps(T)
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
            has_porous = geometry.porous_material > 0 && geometry.porous_thickness > zero(T)
            porous_z = has_porous ? geometry.rock_thickness - geometry.porous_thickness : geometry.rock_thickness

            if z < geometry.rock_thickness
                if has_porous && z >= porous_z
                    material = geometry.porous_material
                    density = geometry.porous_density
                else
                    material = geometry.rock_material
                    density = geometry.rock_density
                end

                if uz_motion > eps(T)
                    next_boundary = (has_porous && z < porous_z) ? porous_z : geometry.rock_thickness
                    step_to_boundary = (next_boundary - z) / uz_motion
                elseif uz_motion < -eps(T)
                    prev_boundary = (has_porous && z >= porous_z) ? porous_z : zero(T)
                    step_to_boundary = (prev_boundary - z) / uz_motion
                else
                    step_to_boundary = T(1000.0)
                end
            else
                material = geometry.air_material
                density = locals_air_at_altitude(z).density
                
                if uz_motion > eps(T)
                    step_to_boundary = (primary_alt - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = (geometry.rock_thickness - z) / uz_motion
                else
                    step_to_boundary = T(1000.0)
                end
            end
            
            max_geometric_step = _backward_geometric_step_limit(
                physics, state, material, density, uz_motion,
                z >= geometry.rock_thickness, use_scattering)

            # Final step size (add small epsilon to cross boundaries)
            step_size = min(step_to_boundary + STEP_EPSILON, max_geometric_step)
            step_size = clamp(step_size, T(STEP_MIN), T(1000.0))
            
            # Perform transport step with appropriate mode
            # Match PUMAS C geometry.c exactly:
            # - Below threshold: STRAGGLED + MIXED scattering (full physics)
            # - Above threshold: MIXED energy loss (no full straggling, no scattering)
            # NOTE: Mode flags are set at outer loop, use them consistently (no redundant energy check)
            state = _backward_geometry_step(physics, state, material, density, step_size,
                                            rng, use_straggled, use_mixed_mode,
                                            use_scattering, current_energy_limit, contexts)

        end
    end
    
    return state
end

"""
    transport_backward_step_mixed(physics, state, material, density, step_distance, rng; energy_limit=-1)

Perform a backward transport step in MIXED mode (for energies > 100 GeV).
Accepts either a single material index (Int) or a MaterialMixture.
"""
# Backward-compatible overload
function transport_backward_step_mixed(physics::PhysicsTables{T}, state::State{T},
                                       material, density::T, step_distance::T,
                                       rng::AbstractRNG;energy_limit::T=T(-1),
                                       kernel_context=nothing) where T<:Real
    context=kernel_context===nothing ? prepare_mixed_kernel(physics,material) : MixedTransport.check_kernel_owner(kernel_context,physics,material)
    return transport_mixed_step(context,state,density,step_distance,rng;energy_limit=energy_limit)
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
                             energy_threshold_low::T = T(100.0),
                                             mixed_contexts=nothing,
                                             source_energy_max::Union{Nothing,T}=nothing) where T<:Real
    
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
    supported_max=minimum(last(t.energies) for t in physics.tables)
    energy_threshold=source_energy_max===nothing ? min(T(1e9),supported_max) : source_energy_max
    zero(T)<energy_threshold<=supported_max || throw(ArgumentError("source_energy_max must be positive and within the physics energy grid"))
    
    # Transport backward through geometry with optional straggling and scattering
    final_state = transport_backward_through_geometry(physics, geometry, state, energy_threshold;
                                                      rng=rng, straggling=straggling, scattering=scattering,
                                                      energy_threshold_low=energy_threshold_low,mixed_contexts=mixed_contexts)
    
    # Check if reached primary altitude
    if final_state.position[3] >= geometry.primary_altitude - T(1.0) && final_state.energy < energy_threshold
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
    compute_flux_single_with_state(physics, geometry, energy_final, elevation, charge; ...)

Like `compute_flux_single` but also returns the final zenith angle (degrees)
at primary altitude. Returns `(flux_weight, zenith_final)` where `zenith_final`
is `NaN` if the particle did not reach the primary altitude.
"""
function compute_flux_single_with_state(physics::PhysicsTables{T},
                                        geometry::TwoLayerGeometry{T},
                                        energy_final::T,
                                        elevation::T,
                                        charge::T;
                                        rng::Union{AbstractRNG, Nothing} = nothing,
                                        straggling::Bool = false,
                                        scattering::Bool = false,
                                        energy_threshold_low::T = T(100.0),
                                             mixed_contexts=nothing,
                                             source_energy_max::Union{Nothing,T}=nothing) where T<:Real

    theta = (T(90) - elevation) * T(π) / T(180)
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    state = State{T}(
        charge = charge,
        energy = energy_final,
        distance = zero(T),
        grammage = zero(T),
        time = zero(T),
        weight = one(T),
        position = Vec3{T}(0, 0, 0),
        direction = Vec3{T}(-sin_theta, 0, -cos_theta),
        decayed = false
    )

    supported_max=minimum(last(t.energies) for t in physics.tables)
    energy_threshold=source_energy_max===nothing ? min(T(1e9),supported_max) : source_energy_max
    zero(T)<energy_threshold<=supported_max || throw(ArgumentError("source_energy_max must be positive and within the physics energy grid"))

    final_state = transport_backward_through_geometry(physics, geometry, state, energy_threshold;
                                                      rng=rng, straggling=straggling, scattering=scattering,
                                                      energy_threshold_low=energy_threshold_low,mixed_contexts=mixed_contexts)

    if final_state.position[3] >= geometry.primary_altitude - T(1.0) && final_state.energy < energy_threshold
        cos_theta_final = -final_state.direction[3]
        cos_theta_final = clamp(cos_theta_final, zero(T), one(T))

        if cos_theta_final <= zero(T)
            return zero(T), T(NaN)
        end

        zenith_final = acos(cos_theta_final) * T(180) / T(π)
        flux = flux_gccly(cos_theta_final, final_state.energy, charge)
        return final_state.weight * flux, zenith_final
    else
        return zero(T), T(NaN)
    end
end

function _compute_flux_estimate(physics::PhysicsTables{T},
                      rock_density::T,
                      rock_thickness::T,
                      elevation::T,
                      energy_min::T,
                      energy_max::T;
                      n_samples::Int = 1000,
                      seed::Int = 42,
                      straggling::Bool = true,
                      scattering::Bool = true,
                      energy_threshold_low::T = T(100.0),
                      porous_material::Int = -1,
                      porous_density::T = zero(T),
                      porous_thickness::T = zero(T),
                      primary_altitude::T = T(PRIMARY_ALTITUDE),
                      record_samples::Bool = false,
                      source_energy_max::Union{Nothing,T}=nothing) where T<:Real
    
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
    
    geometry = TwoLayerGeometry{T}(rock_thickness, rock_density, rock_idx, air_idx,
                                   porous_material, porous_density, porous_thickness,
                                   primary_altitude)

    mixed_contexts = _prepare_mixed_contexts(physics,rng,straggling)
    
    # Integration bounds (log-uniform sampling)
    rk = log(energy_max / energy_min)
    
    # Accumulate flux
    samples = record_samples ? NTuple{3,T}[] : nothing
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
        record_samples && push!(samples, (kf, charge, wf))
        
        # Compute contribution with optional straggling and scattering
        wi = wf * compute_flux_single(physics, geometry, kf, elevation, charge;
                                      rng=rng, straggling=straggling, scattering=scattering,
                                      energy_threshold_low=energy_threshold_low,mixed_contexts=mixed_contexts,source_energy_max=source_energy_max)
        
        w_sum += wi
        w2_sum += wi * wi
    end
    
    # Average flux
    flux_avg = w_sum / n_samples
    
    # Proper standard error: σ = sqrt(Var(X)/n) where Var(X) = E[X²] - E[X]²
    # This matches PUMAS C implementation
    variance = (w2_sum / n_samples) - flux_avg * flux_avg
    sigma = rock_thickness > zero(T) ? sqrt(max(variance, zero(T)) / n_samples) : zero(T)
    
    return flux_avg, sigma, samples
end

"""
    compute_flux(physics, rock_density, rock_thickness, elevation, energy_min, energy_max; n_samples, seed, straggling, scattering, energy_threshold_low, porous_material, porous_density, porous_thickness)

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
- `porous_material`: Material index for shallow porous layer (≤0 to disable, default: -1)
- `porous_density`: Density of the porous layer in kg/m³ (default: 0)
- `porous_thickness`: Thickness of the porous layer in m (default: 0)
- `primary_altitude`: Altitude at which the primary flux is sampled in m (default: PRIMARY_ALTITUDE)

# Returns
- `flux`: Average flux in m⁻² s⁻¹ sr⁻¹
- `sigma`: Standard error of the mean (proper variance calculation matching PUMAS C)
"""
function compute_flux(physics::PhysicsTables{T}, rock_density::T, rock_thickness::T,
                      elevation::T, energy_min::T, energy_max::T; kwargs...) where T<:Real
    flux, sigma, _ = _compute_flux_estimate(physics, rock_density, rock_thickness,
                                          elevation, energy_min, energy_max; kwargs...)
    return flux, sigma
end

# Differentiate the deterministic CSDA surrogate at the exact energy and charge
# samples used by the MC estimate, including their quadrature weights.
function _sampled_csda_flux_gradient(physics, density, thickness, elevation, samples;source_energy_max=nothing)
    flux = zero(density)
    gradient = zero(density)
    # Linearity lets us release each particle's reverse tape immediately, rather
    # than retaining one geometry tape for every MC sample at once.
    for (energy, charge, weight) in samples
        particle_flux, particle_gradient = compute_flux_gradient(
            physics, density, thickness, elevation, energy, charge;source_energy_max=source_energy_max)
        flux += weight * particle_flux
        gradient += weight * particle_gradient
    end
    return flux / length(samples), gradient / length(samples)
end


"""
    compute_flux_differentiable(physics, rock_density, rock_thickness, elevation, energy_final, charge; straggling=false, scattering=false, energy_threshold_low=100.0)

Compute flux for a single particle - fully differentiable version.

For Zygote compatibility:
- Uses deterministic transport (no RNG)
- straggling=true with no RNG uses deterministic CSDA approximation
- scattering=false is recommended for gradient computation
"""
function compute_flux_differentiable(physics::PhysicsTables{T},
                                     rock_density::T,
                                     rock_thickness::T,
                                     elevation::T,
                                     energy_final::T,
                                     charge::T;
                                     straggling::Bool = false,
                                     scattering::Bool = false,
                                     energy_threshold_low::T = T(100.0),
                         source_energy_max::Union{Nothing,T}=nothing) where T<:Real
    
    rock_idx = get_material_index(physics, "StandardRock")
    air_idx = get_material_index(physics, "Air")
    
    if rock_idx == -1
        rock_idx = 1
    end
    if air_idx == -1
        air_idx = length(physics.tables)
    end
    
    geometry = TwoLayerGeometry{T}(rock_thickness, rock_density, rock_idx, air_idx)
    
    # Note: rng=nothing for Zygote compatibility (deterministic transport)
    # With straggling=true but no rng, transport uses deterministic CSDA
    return compute_flux_single(physics, geometry, energy_final, elevation, charge;
                               rng=nothing, straggling=straggling, scattering=scattering,
                               energy_threshold_low=energy_threshold_low,source_energy_max=source_energy_max)
end

"""
    compute_flux_gradient(physics, rock_density, rock_thickness, elevation, energy_final, charge; straggling=false, scattering=false, energy_threshold_low=100.0)

Compute gradient of flux w.r.t. rock density using Zygote.

For Zygote compatibility:
- Uses deterministic transport (no RNG)
- straggling=true with no RNG uses deterministic CSDA approximation
- scattering=false is recommended for gradient computation

# Arguments
- `straggling`: Use straggling model (deterministic approximation for AD)
- `scattering`: Enable scattering (should be false for Zygote)
- `energy_threshold_low`: Energy threshold for mode switching in GeV

# Returns
- `flux`: The flux value
- `grad_density`: ∂flux/∂rock_density
"""
function compute_flux_gradient(physics::PhysicsTables{T},
                               rock_density::T,
                               rock_thickness::T,
                               elevation::T,
                               energy_final::T,
                               charge::T;
                               straggling::Bool = false,
                               scattering::Bool = false,
                               energy_threshold_low::T = T(100.0),
                               source_energy_max::Union{Nothing,T}=nothing) where T<:Real
    
    f = ρ -> compute_flux_differentiable(physics, ρ, rock_thickness, 
                                         elevation, energy_final, charge;
                                         straggling=straggling, scattering=scattering,
                                         energy_threshold_low=energy_threshold_low,source_energy_max=source_energy_max)
    
    flux = f(rock_density)
    grad = Zygote.gradient(f, rock_density)[1]
    
    # Zygote uses `nothing` when the density is unused (e.g. zero rock).
    return flux, something(grad, zero(T))
end

# =============================================================================
# Optimized Direct CSDA Transport (Zygote-friendly)
# =============================================================================

"""
    compute_air_grammage(z_start, z_end, cos_theta)

Compute integrated air grammage (kg/m²) along a straight path from z_start to z_end.
Uses exponential atmosphere model: ρ(z) = ρ₀ * exp(-z/h)

For a path at angle θ to vertical: dL = dz / cos(θ)
Grammage = ∫ ρ(z) dL = (ρ₀ * h / cos_θ) * [exp(-z_start/h) - exp(-z_end/h)]
"""
function compute_air_grammage(z_start::T, z_end::T, cos_theta::T) where T<:Real
    ρ0 = T(1.205)   # kg/m³ at sea level
    h = T(12e3)     # Scale height in meters
    
    # Safeguard for numerical stability
    cos_theta_safe = max(abs(cos_theta), T(0.01))
    
    # Integrated grammage through exponential atmosphere
    # Note: for upward path, z_end > z_start, and exp(-z_end/h) < exp(-z_start/h)
    grammage = (ρ0 * h / cos_theta_safe) * (exp(-z_start / h) - exp(-z_end / h))
    
    return max(grammage, zero(T))
end

"""
    compute_rock_grammage(rock_thickness, rock_density, cos_theta)

Compute grammage (kg/m²) through uniform rock layer.
"""
@inline function compute_rock_grammage(rock_thickness::T, rock_density::T, cos_theta::T) where T<:Real
    cos_theta_safe = max(abs(cos_theta), T(0.01))
    path_length = rock_thickness / cos_theta_safe
    return path_length * rock_density
end

"""
    compute_decay_weight_from_path(physics, path_length, E_avg)

Compute decay weight factor for CSDA transport from path length.

Uses the correct decay formula:
P(survive) = exp(-t_proper / τ) = exp(-L / (βγ * c * τ)) = exp(-L / (βγ * ctau))

where L is the lab-frame path length and ctau is the muon decay length.

# Arguments
- `physics`: Physics tables (for ctau)
- `path_length`: Lab-frame path length in meters
- `E_avg`: Average kinetic energy for computing βγ
"""
function compute_decay_weight_from_path(physics::PhysicsTables{T}, 
                                        path_length::T, 
                                        E_avg::T) where T<:Real
    mass = physics.mass
    gamma_avg = one(T) + E_avg / mass
    beta_avg = sqrt(max(one(T) - one(T) / gamma_avg^2, T(1e-12)))
    
    # Decay weight: exp(-L / (βγ * ctau))
    decay_length = beta_avg * gamma_avg * physics.ctau
    return exp(-path_length / decay_length)
end

"""
    compute_flux_csda_direct(physics, rock_material, air_material, rock_density, 
                              rock_thickness, elevation, energy_initial, charge)

Compute backward MC flux for a single particle using direct CSDA computation.
This is optimized for differentiation - no iterative stepping required.

For pure CSDA (no scattering, no straggling), the path is deterministic:
1. Transport through rock layer using rock material tables
2. Transport through air layer using air material tables
3. Compute weight from stopping power ratio and decay

The flux contribution is:
  w = (dE/dX)_final / (dE/dX)_initial * exp(-Δτ/ctau) * flux_gccly(E_final, cos_theta)

Returns flux contribution (m⁻² s⁻¹ sr⁻¹ GeV⁻¹).
"""
function compute_flux_csda_direct(physics::PhysicsTables{T},
                                  rock_material::Int,
                                  air_material::Int,
                                  rock_density::T,
                                  rock_thickness::T,
                                  elevation::T,
                                  energy_initial::T,
                                  charge::T;
                                  primary_altitude::T = T(PRIMARY_ALTITUDE)) where T<:Real
    
    # Convert elevation (degrees from horizontal) to cos(zenith)
    elevation_rad = deg2rad(elevation)
    cos_theta = cos(T(π/2) - elevation_rad)  # cos(zenith) = sin(elevation)
    
    # =========================================================================
    # Step 1: Transport through ROCK layer
    # =========================================================================
    X_rock = compute_rock_grammage(rock_thickness, rock_density, cos_theta)
    
    # Initial CSDA range at detector (in rock material)
    X_initial_rock = property_range(physics, ENERGY_LOSS_CSDA, rock_material, energy_initial)
    
    # Range after traversing rock (backward = range increases)
    X_after_rock = X_initial_rock + X_rock
    
    # Energy after rock layer
    energy_after_rock = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, rock_material, X_after_rock)
    
    # Check energy validity
    if energy_after_rock >= T(1e12) || !isfinite(energy_after_rock)
        return zero(T)
    end
    
    # Weight factor for rock: dedx_final / dedx_initial
    dedx_initial_rock = property_stopping_power(physics, ENERGY_LOSS_CSDA, rock_material, energy_initial)
    dedx_after_rock = property_stopping_power(physics, ENERGY_LOSS_CSDA, rock_material, energy_after_rock)
    weight_rock = dedx_after_rock / dedx_initial_rock
    
    # Decay weight through rock (using path length and average energy)
    cos_theta_safe = max(abs(cos_theta), T(0.01))
    path_rock = rock_thickness / cos_theta_safe
    E_avg_rock = (energy_initial + energy_after_rock) / 2
    decay_rock = compute_decay_weight_from_path(physics, path_rock, E_avg_rock)
    
    # =========================================================================
    # Step 2: Transport through AIR layer
    # =========================================================================
    X_air = compute_air_grammage(rock_thickness, primary_altitude, cos_theta)
    
    # Range at start of air (convert from rock energy to air range)
    X_initial_air = property_range(physics, ENERGY_LOSS_CSDA, air_material, energy_after_rock)
    
    # Range after traversing air
    X_final_air = X_initial_air + X_air
    
    # Final energy at top of atmosphere
    energy_final = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, air_material, X_final_air)
    
    # Check energy validity
    if energy_final >= T(1e12) || !isfinite(energy_final)
        return zero(T)
    end
    
    # Weight factor for air: dedx_final / dedx_initial
    dedx_initial_air = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_after_rock)
    dedx_final_air = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_final)
    weight_air = dedx_final_air / dedx_initial_air
    
    # Decay weight through air (path length from rock top to primary altitude)
    path_air = (primary_altitude - rock_thickness) / cos_theta_safe
    E_avg_air = (energy_after_rock + energy_final) / 2
    decay_air = compute_decay_weight_from_path(physics, path_air, E_avg_air)
    
    # =========================================================================
    # Step 3: Combine weights and compute flux
    # =========================================================================
    weight = weight_rock * decay_rock * weight_air * decay_air
    
    # Flux at top of atmosphere (using GCCLY model)
    # flux_gccly expects (cos_theta, energy, charge)
    atmospheric_flux = flux_gccly(cos_theta, energy_final, charge)
    
    return weight * atmospheric_flux
end

"""
    compute_flux_differentiable_csda(physics, rock_density, rock_thickness, elevation, 
                                     energy_final, charge)

Compute flux for a single particle using optimized direct CSDA.
This version is specifically optimized for Zygote autodiff.

No iteration, no RNG - pure functional computation suitable for AD.
"""
function compute_flux_differentiable_csda(physics::PhysicsTables{T},
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
    
    return compute_flux_csda_direct(physics, rock_idx, air_idx, 
                                    rock_density, rock_thickness, 
                                    elevation, energy_final, charge)
end

"""
    compute_flux_gradient_csda(physics, rock_density, rock_thickness, elevation, 
                               energy_final, charge)

Compute gradient of flux w.r.t. rock density using optimized direct CSDA.
This is significantly faster than the iterative version for Zygote.

# Returns
- `flux`: The flux value
- `grad_density`: ∂flux/∂rock_density
"""
function compute_flux_gradient_csda(physics::PhysicsTables{T},
                                    rock_density::T,
                                    rock_thickness::T,
                                    elevation::T,
                                    energy_final::T,
                                    charge::T) where T<:Real
    
    f = ρ -> compute_flux_differentiable_csda(physics, ρ, rock_thickness, 
                                              elevation, energy_final, charge)
    
    flux = f(rock_density)
    grad = Zygote.gradient(f, rock_density)[1]
    
    return flux, grad
end

# Export the new optimized functions
export compute_flux_csda_direct, compute_flux_differentiable_csda, compute_flux_gradient_csda

"""
    run_backward_mc(physics; kwargs...)

Run the backward Monte Carlo flux calculation example.
Equivalent to the geometry.c example.

With `compute_gradient=true`, `gradient` differentiates a deterministic CSDA
surrogate averaged over the same sampled energies, charges, and integration
weights as the MC estimate. `gradient_flux` is that surrogate's flux and
`gradient_method` is `:deterministic_csda_sample_average`. The gradient is not a
derivative of stochastic straggling or scattering, and holds the sampled inputs
fixed as density changes. With gradients disabled, both metadata fields are
`nothing`.
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
                         energy_threshold_low::T = T(100.0),
                         source_energy_max::Union{Nothing,T}=nothing) where T<:Real
    
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
    
    # Compute flux (now returns flux and proper sigma)
    flux, sigma, samples = _compute_flux_estimate(physics, rock_density, rock_thickness, elevation,
                        energy_min, energy_max; n_samples=n_samples, seed=seed,
                        straggling=straggling, scattering=scattering,
                        energy_threshold_low=energy_threshold_low,
                        record_samples=compute_gradient,source_energy_max=source_energy_max)
    
    # Compute gradient if requested
    grad_density = nothing
    gradient_flux = nothing
    gradient_method = nothing
    if compute_gradient
        @info "Computing density gradient of sampled deterministic CSDA flux..."
        gradient_flux, grad_density = _sampled_csda_flux_gradient(
            physics, rock_density, rock_thickness, elevation, samples;source_energy_max=source_energy_max)
        gradient_method = :deterministic_csda_sample_average
        @info "  ∂CSDA flux/∂density = $(grad_density)"
    end
    
    unit = energy_max > energy_min ? "" : "GeV⁻¹ "
    @info "Flux: $(flux) ± $(sigma) $(unit)m⁻² s⁻¹ sr⁻¹"
    
    return (flux = flux, sigma = sigma, gradient = grad_density,
            gradient_flux = gradient_flux, gradient_method = gradient_method,
            source_energy_max = source_energy_max===nothing ? min(T(1e9),minimum(last(t.energies) for t in physics.tables)) : source_energy_max)
end

# Keep old functions for compatibility
locals_air(state::State{T}) where T = locals_air_at_altitude(state.position[3])
medium_callback(ctx, state, geo) = nothing
create_geometry_context(physics, geometry; kwargs...) = nothing

end # module Geometry
