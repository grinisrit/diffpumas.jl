"""
    Straggling

Energy straggling and discrete energy loss (DEL) implementation.
Based on PUMAS implementation for accurate muon transport.

This module implements:
1. Landau/Vavilov energy fluctuation model (step_fluctuate)
2. Discrete energy loss (DEL) events (bremsstrahlung, pair production, photonuclear)
3. Elastic hard scattering (EHS) - Coulomb scattering
4. Soft multiple scattering (MCS)
"""
module Straggling

using ..Constants
using ..Types
using ..Physics
using ..Materials
using Random
using LinearAlgebra

export fluctuate_energy_loss, sample_del_event, sample_ehs_event
export compute_del_cross_section, compute_ehs_mean_free_path
export rotate_direction, box_muller_randn, sample_scattering_angle
export sample_soft_scattering, sample_msc_angle
export polar_del_angle, sample_target_element

# Constants from PUMAS
const BMC_ALPHA = 2.0         # Exponent for backward MC DEL sampling
const EHS_PATH_MAX = 1e9      # Maximum path for EHS
const X_THRESHOLD = 0.05      # Threshold for straggling correction
const STEP_EPSILON = 1e-7     # Small offset for step boundaries
const MSC_ACCURACY = 0.01     # Default accuracy for MSC angular distribution
const POLAR_MAX_TRIALS = 100  # Maximum rejection sampling iterations for polar angle

# Nuclear RMS radii in metres (from PUMAS data_nuclear_radius, Z=1..120)
const NUCLEAR_RMS_RADII = [
    2.098, 0.858, 1.680, 2.400, 2.518, 2.405, 2.470, 2.548, 2.734,
    2.900, 2.993, 2.940, 3.043, 3.035, 3.098, 3.187, 3.245, 3.360,
    3.413, 3.408, 3.477, 3.443, 3.595, 3.600, 3.644, 3.681, 3.748,
    3.843, 3.776, 3.943, 3.942, 4.032, 4.065, 4.078, 4.123, 4.135,
    4.188, 4.209, 4.237, 4.249, 4.306, 4.318, 4.363, 4.388, 4.432,
    4.435, 4.479, 4.520, 4.612, 4.646, 4.640, 4.630, 4.714, 4.706,
    4.756, 4.774, 4.823, 4.848, 4.878, 4.897, 4.927, 4.948, 5.054,
    5.093, 5.133, 5.177, 5.197, 5.210, 5.264, 5.301, 5.390, 5.371,
    5.429, 5.479, 5.423, 5.400, 5.409, 5.411, 5.387, 5.318, 5.404,
    5.471, 5.498, 5.520, 5.520, 5.529, 5.629, 5.637, 5.661, 5.669,
    5.710, 5.701, 5.784, 5.825, 5.824, 5.817, 5.843, 5.843, 5.868,
    5.875, 5.906, 5.912, 5.918, 5.937, 5.967, 5.973, 5.979, 5.985,
    5.980, 6.033, 6.051, 6.056, 6.074, 6.079, 6.097, 6.097, 6.119,
    6.125, 6.125, 2.958
] .* 1e-15  # Convert fm → m

"""
    box_muller_randn(rng)

Generate a standard normal random variate using Box-Muller transform.
"""
@inline function box_muller_randn(rng::AbstractRNG)
    r = sqrt(-2.0 * log(rand(rng)))
    phi = 2.0 * π * rand(rng)
    return r * sin(phi)
end

"""
    truncated_randn(rng, limit=3.0)

Generate a truncated normal random variate |u| <= limit.
The value is scaled by 1.015387 to match PUMAS normalization.
"""
@inline function truncated_randn(rng::AbstractRNG, limit::Float64 = 3.0)
    u = box_muller_randn(rng)
    # Rejection sampling (rarely loops more than once for limit=3)
    @inbounds while abs(u) > limit
        u = box_muller_randn(rng)
    end
    return u * 0.9848416472  # Pre-computed 1/1.015387
end

"""
    fluctuate_energy_loss(physics, material, ki, dX, Xtot, rng; backward=false)

Apply stochastic energy loss fluctuation following PUMAS step_fluctuate.

# Arguments
- `physics`: Physics tables
- `material`: Material index
- `ki`: Initial kinetic energy in GeV
- `dX`: Step grammage in kg/m²
- `Xtot`: Total initial CSDA grammage in kg/m²
- `rng`: Random number generator
- `backward`: Whether in backward transport mode

# Returns
- `kf`: Final kinetic energy
- `ratio`: Ratio of actual energy loss to expected (for backward MC weight)

This implements the Landau/Vavilov distribution approximation used by PUMAS:
- For dk0 >= 3*dk1: Truncated Gaussian
- For 3*dk1 > dk0 >= sqrt(3)*dk1: Uniform distribution
- For dk0 < sqrt(3)*dk1: Mixed model
"""
@inline function fluctuate_energy_loss(physics::PhysicsTables{T}, material::Int,
                               ki::T, dX::T, Xtot::T, rng::AbstractRNG;
                               backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    # PUMAS uses MIXED mode tables for STRAGGLED mode (see pumas.c line 4268, 4277, 4301)
    # STRAGGLED > MIXED, so it gets clamped to MIXED
    mode = ENERGY_LOSS_MIXED
    
    # Get expected final energy from MIXED tables (matching PUMAS C)
    k1 = property_kinetic_energy(physics, mode, material, Xtot - sgn * dX)
    dk0 = abs(ki - k1)
    
    ratio = one(T)
    
    if k1 > zero(T) && dk0 > zero(T)
        # Get straggling variance at initial and final energies
        Omega0 = property_straggling(physics, material, ki)
        Omega1 = property_straggling(physics, material, k1)
        
        # Compute variance for the step
        dk12 = T(0.5) * dX * (Omega0 + Omega1)
        
        # Apply correction for large steps
        tmp = dX / Xtot
        if tmp > T(X_THRESHOLD)
            de1 = property_stopping_power(physics, ENERGY_LOSS_MIXED, material, k1)
            de0 = property_stopping_power(physics, ENERGY_LOSS_MIXED, material, ki)
            dk12 *= one(T) + sgn * dX / dk0 * (de1 - de0)
        end
        
        dk1 = sqrt(max(dk12, zero(T)))
        
        if dk0 >= T(3) * dk1 && dk1 > zero(T)
            # Truncated Gaussian regime (Landau-like)
            u = truncated_randn(rng)
            k1 += u * dk1
        elseif dk0 >= T(1.7320508) * dk1 && dk1 > zero(T)
            # Uniform distribution regime
            u = T(1.7320508) * (one(T) - T(2) * rand(rng))
            k1 += u * dk1
        elseif dk1 > zero(T)
            # Mixed model for small steps
            dk32 = T(3) * dk12
            dk02 = dk0 * dk0
            a = one(T) - (dk32 - dk02) / (dk32 + T(3) * dk02)
            
            if rand(rng) <= a
                b = T(0.5) * (dk32 + T(3) * dk02) / dk0
                u = rand(rng)
                k1 = ki - sgn * b * u
                if k1 < zero(T)
                    k1 = zero(T)
                end
            else
                k1 = ki
            end
        end
        
        ratio = abs(ki - k1) / dk0
    end
    
    if ratio == zero(T)
        ratio = one(T)
    end
    
    # Ensure k1 is valid
    k1 = max(k1, zero(T))
    
    return k1, ratio
end

"""
    compute_del_cross_section(physics, material, kinetic)

Compute the total cross-section for discrete energy loss events.
Based on PUMAS del_cross_section() - uses tabulated values from radiative processes.

The cross-section represents the probability per unit grammage of a hard
(discrete) radiative event: bremsstrahlung, pair production, or photonuclear.

Returns cross-section in m²/kg.
"""
@inline function compute_del_cross_section(physics::PhysicsTables{T}, material::Int, 
                                           kinetic::T) where T<:Real
    if kinetic < T(1e-3)
        return zero(T)
    end
    
    @inbounds table = physics.tables[material]
    
    # Use tabulated cross-section (always try first - it's pre-computed)
    xs = interpolate_table(kinetic, table.energies, table.cross_section)
    
    # Cross-section is always positive if tabulated correctly
    return max(xs, zero(T))
end

"""
    sample_del_event(physics, material, ki, kf, dX, Xtot, ratio, rng; backward=false)

Sample whether a discrete energy loss event (DEL) occurs during the step.
Based on PUMAS's DEL sampling in step_fluctuate.

In backward MC, DEL events give large energy BOOSTS (the particle gains energy 
when traced backward through a hard radiative event). This is crucial for 
high zenith angles where hard events matter more.

# Returns
- `occurred`: Whether a DEL event occurred
- `X_del`: Grammage at which event occurred (if any)
- `k_del`: Energy at which event occurred (if any)
- `process`: Which process (0=none, 1=brems, 2=pair, 3=photo)
"""
@inline function sample_del_event(physics::PhysicsTables{T}, material::Int,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    # PUMAS uses MIXED mode tables for STRAGGLED mode
    mode = ENERGY_LOSS_MIXED
    
    # Get maximum cross-section over the step
    xs_del_i = compute_del_cross_section(physics, material, ki)
    xs_del_f = compute_del_cross_section(physics, material, kf)
    xs_del = max(xs_del_i, xs_del_f)
    
    if xs_del <= zero(T)
        return false, zero(T), zero(T), 0
    end
    
    # Sample interaction length (exponential distribution)
    zeta = rand(rng)
    if zeta <= T(1e-12) || zeta >= one(T) - T(1e-12)
        return false, zero(T), zero(T), 0
    end
    
    X_del = -log(zeta) / xs_del
    
    if X_del >= dX
        return false, zero(T), zero(T), 0
    end
    
    # Get energy at the interaction point (using MIXED tables like PUMAS C)
    k_h = property_kinetic_energy(physics, mode, material, Xtot - sgn * X_del)
    
    # Energy at interaction using fluctuation ratio
    k_at_del = ki - sgn * ratio * abs(ki - k_h)
    
    if k_at_del <= zero(T)
        return false, zero(T), zero(T), 0
    end
    
    # Acceptance/rejection based on cross-section at actual energy
    r = compute_del_cross_section(physics, material, k_at_del) / xs_del
    
    if rand(rng) > r
        return false, zero(T), zero(T), 0
    end
    
    # DEL occurred! Now sample the energy transfer
    # In backward mode: particle GAINS energy from the DEL
    # Energy transfer fraction ν is sampled from dσ/dν ∝ 1/ν^α with α = BMC_ALPHA = 2
    cutoff = physics.settings.cutoff
    
    # Sample ν from power law: P(ν) ∝ ν^(-α) for ν ∈ [cutoff, 1]
    # CDF: F(ν) = (ν^(1-α) - cutoff^(1-α)) / (1 - cutoff^(1-α))
    # Inverse: ν = [u * (1 - cutoff^(1-α)) + cutoff^(1-α)]^(1/(1-α))
    alpha = T(BMC_ALPHA)
    u = rand(rng)
    
    if abs(alpha - one(T)) < T(1e-6)
        # α ≈ 1: use log distribution
        nu = cutoff * exp(u * log(one(T) / cutoff))
    else
        # General power law
        cutoff_term = cutoff^(one(T) - alpha)
        nu = (u * (one(T) - cutoff_term) + cutoff_term)^(one(T) / (one(T) - alpha))
    end
    
    nu = clamp(nu, cutoff, one(T))
    
    # In backward mode, the energy AFTER the DEL is higher
    # k_del = k_at_del / (1 - ν) for forward; k_del = k_at_del * (1 + ν/(1-ν)) for backward
    if backward
        # Particle gains energy: k_del > k_at_del
        # The fractional energy loss ν in forward mode means energy boost in backward
        k_del = k_at_del / (one(T) - nu)
    else
        # Particle loses energy: k_del < k_at_del
        k_del = k_at_del * (one(T) - nu)
    end
    
    # Determine which process based on relative cross-sections
    table = physics.tables[material]
    brems = interpolate_table(k_at_del, table.energies, table.bremsstrahlung)
    pair = interpolate_table(k_at_del, table.energies, table.pair_production)
    photo = interpolate_table(k_at_del, table.energies, table.photonuclear)
    total = brems + pair + photo
    
    process = 1  # Default to bremsstrahlung
    if total > zero(T)
        r_proc = rand(rng) * total
        if r_proc < brems
            process = 1
        elseif r_proc < brems + pair
            process = 2
        else
            process = 3
        end
    end
    
    return true, X_del, k_del, process
end

"""
    compute_ehs_mean_free_path(physics, material, kinetic)

Compute the elastic hard scattering mean free path.
Based on PUMAS coulomb_ehs_length.

In PUMAS, the EHS MFP is computed from the elastic scattering tables.
The tabulated elastic_path already accounts for the momentum dependence.

Returns mean free path in kg/m².
"""
@inline function compute_ehs_mean_free_path(physics::PhysicsTables{T}, material::Int,
                                            kinetic::T) where T<:Real
    if kinetic < T(1e-3)
        return T(EHS_PATH_MAX)
    end
    
    @inbounds table = physics.tables[material]
    
    # Elastic path is stored as Lb = lb_h * p² (PUMAS convention)
    Lb = interpolate_table(kinetic, table.energies, table.elastic_path)
    p2 = kinetic * (kinetic + T(2) * physics.mass)
    
    if Lb > zero(T) && Lb < T(EHS_PATH_MAX) * p2
        return Lb / p2
    end
    
    # Fallback: compute from first principles
    # EHS MFP ≈ λ_1 / μ_0 where λ_1 is first transport MFP and μ_0 is cutoff angle
    # The elastic ratio determines what fraction of scatters are "hard"
    elastic_ratio = physics.settings.elastic_ratio
    transport_path = interpolate_table(kinetic, table.energies, table.transport_path)
    
    if transport_path > zero(T)
        # EHS path = transport_path / (hard fraction)
        # For small elastic_ratio, hard events are rare -> long MFP
        ehs_path = transport_path / elastic_ratio
        return min(ehs_path, T(EHS_PATH_MAX))
    end
    
    return T(EHS_PATH_MAX)
end

"""
    sample_ehs_event(physics, material, ki, kf, dX, Xtot, ratio, rng; backward=false)

Sample whether an elastic hard scattering event occurs during the step.
Based on PUMAS's EHS sampling.

# Returns
- `occurred`: Whether an EHS event occurred
- `X_ehs`: Grammage at which event occurred (if any)
- `k_ehs`: Energy at which event occurred (if any)
"""
@inline function sample_ehs_event(physics::PhysicsTables{T}, material::Int,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    # PUMAS uses MIXED mode tables for STRAGGLED mode
    mode = ENERGY_LOSS_MIXED
    
    # Get minimum/maximum energies
    kmin = min(ki, kf)
    kmax = max(ki, kf)
    
    # For very low energies, use kmax instead
    if kmin < T(0.001)
        kmin, kmax = kmax, kmin
    end
    
    # Get EHS mean free path at lower energy
    lb_ehs = compute_ehs_mean_free_path(physics, material, kmin)
    
    if lb_ehs <= zero(T) || lb_ehs >= T(EHS_PATH_MAX)
        return false, zero(T), zero(T)
    end
    
    # Sample interaction length
    zeta = rand(rng)
    if zeta <= zero(T) || zeta >= one(T)
        return false, zero(T), zero(T)
    end
    
    X_ehs = -log(zeta) * lb_ehs
    
    if X_ehs >= dX
        return false, zero(T), zero(T)
    end
    
    # Get energy at the interaction point
    k_h = property_kinetic_energy(physics, mode, material, Xtot - sgn * X_ehs)
    k_ehs = ki - sgn * ratio * abs(ki - k_h)
    
    if k_ehs <= zero(T)
        return false, zero(T), zero(T)
    end
    
    # Acceptance probability based on energy dependence
    lb_ehs_k = compute_ehs_mean_free_path(physics, material, k_ehs)
    
    if lb_ehs_k <= zero(T) || lb_ehs_k >= T(EHS_PATH_MAX)
        return false, zero(T), zero(T)
    end
    
    r = lb_ehs / lb_ehs_k
    
    if r <= zero(T) || rand(rng) > one(T) / r
        return false, zero(T), zero(T)
    end
    
    return true, X_ehs, k_ehs
end

"""
    sample_scattering_angle(physics, material, kinetic, rng; mu0=nothing)

Sample the polar scattering angle mu = 0.5*(1 - cos(theta)) for an EHS event.
Matches PUMAS C `transport_do_ehs`:
1. Sample mu1 in CM frame from Wentzel envelope with spin rejection
2. Transform from CM to Lab frame

Returns mu in lab frame (0 = forward, 1 = backward).
"""
@inline function sample_scattering_angle(physics::PhysicsTables{T}, material::Int,
                                  kinetic::T, rng::AbstractRNG;
                                  mu0::Union{T, Nothing} = nothing) where T<:Real
    
    mass = physics.mass
    gamma_lab = one(T) + kinetic / mass
    p2 = kinetic * (kinetic + T(2) * mass)
    
    table = physics.tables[material]
    if mu0 === nothing
        mu0 = interpolate_table(kinetic, table.energies, table.screening_parameter)
        mu0 = max(mu0, T(1e-12))
    end
    
    # Effective target mass (use mean A for composite material)
    A_target = table.ZoA > zero(T) ? one(T) / table.ZoA : T(22.0)
    M_target = A_target * T(NEUTRON_MASS)
    
    # CM frame parameters (PUMAS coulomb_frame_parameters)
    # kinetic0 = kinetic energy in CM frame
    s = mass^2 + M_target^2 + T(2) * (kinetic + mass) * M_target
    kinetic0 = (s - (mass + M_target)^2) / (T(2) * sqrt(s))
    
    # gamma_CM and tau for CM→Lab transformation
    E_cm = kinetic0 + mass
    p_cm = sqrt(max(kinetic0 * (kinetic0 + T(2) * mass), zero(T)))
    E_lab = kinetic + mass
    p_lab = sqrt(max(p2, zero(T)))
    
    gamma_CM = (p_lab > zero(T) && p_cm > zero(T)) ? (E_lab * E_cm + p_lab * p_cm) / (mass * sqrt(s)) : one(T)
    tau = (p_lab > zero(T) && p_cm > zero(T)) ? (E_lab * p_cm - E_cm * p_lab) / (p_lab * p_cm) : zero(T)
    
    # Wentzel screening parameter
    A = mu0 / T(4)
    
    fspin = interpolate_table(kinetic, table.energies, table.spin_factor)
    
    A_plus_mu0 = A + mu0
    A_plus_1 = A + one(T)
    one_minus_mu0 = one(T) - mu0
    
    # Sample mu1 in CM frame (Wentzel + spin rejection)
    mu1 = mu0
    @inbounds for _ in 1:100
        zeta = rand(rng)
        tmp = A_plus_1 - zeta * one_minus_mu0
        
        if tmp <= zero(T)
            continue
        end
        
        mu1_trial = A_plus_mu0 * A_plus_1 / tmp - A
        mu1_trial = clamp(mu1_trial, mu0, one(T))
        
        if rand(rng) <= one(T) - fspin * mu1_trial
            mu1 = mu1_trial
            break
        end
    end
    
    # CM → Lab frame transformation (PUMAS transport_do_ehs lines 5874-5884)
    if mu1 > T(1e-6)
        a = gamma_CM * (tau + one(T) - T(2) * mu1)
        ct_h = a / sqrt(T(4) * mu1 * (one(T) - mu1) + a * a)
        mu = T(0.5) * (one(T) - ct_h)
    else
        d = gamma_CM * (one(T) + tau)
        mu = mu1 / (d * d)
    end
    
    return clamp(mu, zero(T), one(T))
end

"""
    sample_msc_angle(physics, material, kinetic_start, kinetic_end, step_distance, density, rng)

Sample the multiple scattering angle for soft scattering.
Matches PUMAS C implementation exactly (pumas.c lines 6913-6922):

    ilb1 = 0.25 * step * (invlb1_start + invlb1_end)
    if ilb1 > 1: ilb1 = 1
    do { mu = -ilb1 * log(random); } while (mu > 1);

Where invlb1 = density / transport_path(energy).

# Arguments
- `kinetic_start`: Energy at start of step (before step)
- `kinetic_end`: Energy at end of step (after step)
- `step_distance`: Step distance in metres
- `density`: Material density in kg/m³

Returns mu = 0.5*(1 - cos(theta)).
"""
@inline function sample_msc_angle(physics::PhysicsTables{T}, material::Int,
                          kinetic_start::T, kinetic_end::T,
                          step_distance::T, density::T,
                          rng::AbstractRNG) where T<:Real
    
    @inbounds begin
        if step_distance <= zero(T) || density <= zero(T)
            return zero(T)
        end
        
        table = physics.tables[material]
        
        lb1_start = interpolate_table(kinetic_start, table.energies, table.transport_path)
        invlb1_start = (lb1_start > zero(T)) ? density / lb1_start : zero(T)
        
        if kinetic_end > zero(T)
            lb1_end = interpolate_table(kinetic_end, table.energies, table.transport_path)
            invlb1_end = (lb1_end > zero(T)) ? density / lb1_end : zero(T)
        else
            invlb1_end = invlb1_start
        end
        
        if invlb1_start <= zero(T) && invlb1_end <= zero(T)
            return zero(T)
        end
        
        # PUMAS C: ilb1 = 0.25 * step * (invlb1 + context_->step_invlb1)
        ilb1 = T(0.25) * step_distance * (invlb1_start + invlb1_end)
        
        if ilb1 < T(1e-12)
            return zero(T)
        end
        
        if ilb1 > one(T)
            ilb1 = one(T)
        end
        
        # PUMAS C rejection sampling: do { mu = -ilb1*log(rand()); } while (mu > 1);
        for _ in 1:1000
            u = rand(rng)
            if u < T(1e-300)
                continue
            end
            mu = -ilb1 * log(u)
            if mu <= one(T)
                return max(mu, zero(T))
            end
        end
        
        return ilb1
    end
end

"""
    sample_msc_angle(physics, material, kinetic, grammage, rng)

Legacy 3-argument form. Approximates PUMAS by using same energy for start/end
and converting grammage to step_distance with the material's reference density.
Prefer the 6-argument form when density and step_distance are available.
"""
@inline function sample_msc_angle(physics::PhysicsTables{T}, material::Int,
                          kinetic::T, grammage::T, rng::AbstractRNG) where T<:Real
    @inbounds begin
        if grammage <= zero(T) || kinetic <= zero(T)
            return zero(T)
        end
        
        table = physics.tables[material]
        density = table.density
        step_distance = (density > zero(T)) ? grammage / density : zero(T)
        
        return sample_msc_angle(physics, material, kinetic, kinetic,
                                step_distance, density, rng)
    end
end

"""
    sample_soft_scattering(physics, material, kinetic_start, kinetic_end, step_distance, density, rng)

Sample soft multiple scattering angle using the full PUMAS trapezoidal formula.
"""
@inline function sample_soft_scattering(physics::PhysicsTables{T}, material::Int,
                               kinetic_start::T, kinetic_end::T,
                               step_distance::T, density::T,
                               rng::AbstractRNG) where T<:Real
    return sample_msc_angle(physics, material, kinetic_start, kinetic_end,
                            step_distance, density, rng)
end

"""
    sample_soft_scattering(physics, material, kinetic, grammage, rng)

Legacy form for backward compatibility.
"""
@inline sample_soft_scattering(physics::PhysicsTables{T}, material::Int,
                               kinetic::T, grammage::T, rng::AbstractRNG) where T<:Real = 
    sample_msc_angle(physics, material, kinetic, grammage, rng)

"""
    rotate_direction(direction, mu, rng)

Rotate the direction vector randomly with polar angle given by mu = 0.5*(1 - cos(theta)).
Returns the new direction vector (normalized).

Based on PUMAS step_rotate_direction implementation.
"""
@inline function rotate_direction(direction::Vec3{T}, mu::T, rng::AbstractRNG) where T<:Real
    # Cosine and sine of scattering angle
    cos_theta = one(T) - T(2) * mu
    sin_theta_sq = T(4) * mu * (one(T) - mu)
    
    if sin_theta_sq <= zero(T)
        return direction
    end
    
    sin_theta = sqrt(sin_theta_sq)
    
    # Build orthonormal basis - following PUMAS exactly
    d = direction
    ax, ay, az = abs(d[1]), abs(d[2]), abs(d[3])
    
    # Select co-vectors for local basis
    u0x, u0y, u0z = zero(T), zero(T), zero(T)
    
    if ax > ay
        if ax > az
            # d[0] is largest
            nrm = one(T) / sqrt(d[1]^2 + d[3]^2)
            u0x, u0z = -d[3] * nrm, d[1] * nrm
        else
            # d[2] is largest
            nrm = one(T) / sqrt(d[2]^2 + d[3]^2)
            u0y, u0z = d[3] * nrm, -d[2] * nrm
        end
    else
        if ay > az
            # d[1] is largest
            nrm = one(T) / sqrt(d[1]^2 + d[2]^2)
            u0x, u0y = d[2] * nrm, -d[1] * nrm
        else
            # d[2] is largest
            nrm = one(T) / sqrt(d[2]^2 + d[3]^2)
            u0y, u0z = d[3] * nrm, -d[2] * nrm
        end
    end
    
    # Cross product: u1 = u0 × d (PUMAS convention)
    # Note: u0 × d = -d × u0, so signs are flipped from d × u0
    u1x = u0y * d[3] - u0z * d[2]
    u1y = u0z * d[1] - u0x * d[3]
    u1z = u0x * d[2] - u0y * d[1]
    
    # Random azimuthal angle (following PUMAS: phi = π*(1 - 2*random))
    phi = T(π) * (one(T) - T(2) * rand(rng))
    cp = cos(phi)
    sp = sin(phi)
    
    # Apply rotation
    new_dx = cos_theta * d[1] + sin_theta * (cp * u0x + sp * u1x)
    new_dy = cos_theta * d[2] + sin_theta * (cp * u0y + sp * u1y)
    new_dz = cos_theta * d[3] + sin_theta * (cp * u0z + sp * u1z)
    
    # Normalize (should be close to 1 already)
    nrm = sqrt(new_dx^2 + new_dy^2 + new_dz^2)
    if nrm > zero(T)
        new_dx /= nrm
        new_dy /= nrm
        new_dz /= nrm
    end
    
    return Vec3{T}(new_dx, new_dy, new_dz)
end

# =============================================================================
# DEL angular scattering (polar angle after discrete events)
# =============================================================================
# Implements PUMAS's polar angle sampling for each DEL process.
# All functions return mu = 0.5*(1 - cos(theta)).
# Enabled only when scattering (MCS) is on, matching PUMAS C behaviour
# where DEL direction update requires context->mode.scattering == PUMAS_MODE_MIXED.

"""
    nuclear_radius(Z)

Look up nuclear RMS radius in metres for atomic number Z.
Matches PUMAS `data_nuclear_radius`.
"""
@inline function nuclear_radius(Z::Real)
    iZ = clamp(round(Int, Z), 1, length(NUCLEAR_RMS_RADII))
    return NUCLEAR_RMS_RADII[iZ]
end

"""
    polar_bremsstrahlung(mass, ki, kf, Z, A, rng)

Sample polar angle mu = 0.5*(1-cos θ) for a bremsstrahlung or pair-production event.
Uses Tsai's DDCS with nuclear screening and rejection sampling.

# Arguments
- `mass`: Projectile mass (GeV/c²)
- `ki`: Kinetic energy before DEL (higher energy)
- `kf`: Kinetic energy after DEL (lower energy)
- `Z`, `A`: Atomic number / mass of target element
- `rng`: Random number generator

Reference: Y. Tsai, Rev. Mod. Phys. (1974); PUMAS polar_bremsstrahlung().
"""
function polar_bremsstrahlung(mass::T, ki::T, kf::T, Z::T, A::T,
                               rng::AbstractRNG) where T<:Real
    m = mass
    E = ki + m
    nu = ki - kf
    if nu <= zero(T) || E <= m
        return zero(T)
    end
    y = nu / E

    tmp = T(0.5) * m * nu / (E * (E - nu))
    mu0 = tmp * tmp
    c1 = (T(2) * (one(T) - y) + y * y) * mu0
    c2 = T(4) * (one(T) - y) * mu0 * mu0

    RN = T(nuclear_radius(Z))
    tmp2 = m * RN / T(HBAR_C)
    muni = tmp2 * tmp2 / T(6)
    muc = sqrt(mu0) - mu0
    if muc < zero(T)
        return zero(T)
    end

    # Precompute screening factor x0 at mu=0 (envelope normalisation)
    q0 = muni
    r0 = mu0
    qr0 = q0 * r0
    denom0 = r0 + qr0
    if denom0 <= zero(T) || (one(T) + qr0) <= zero(T)
        return zero(T)
    end
    x0 = (one(T) + T(2) * qr0) * log((one(T) + qr0) / denom0) -
         (one(T) - r0) * (one(T) + T(2) * q0) / (one(T) + q0)
    if x0 <= zero(T)
        return zero(T)
    end

    # Rejection sampling
    for _ in 1:POLAR_MAX_TRIALS
        r0_rng = rand(rng)
        r1 = rand(rng)

        # Sample mu from envelope: mu = r0*mu0 / (mu0 + 1 - r0)
        mu = r0_rng * mu0 / (mu0 + one(T) - r0_rng)

        mus = mu0 + mu
        mu2 = mus * mus
        d1 = one(T) / mu2
        d2 = d1 * d1
        score = r1 * c1 * d1

        # Unscreened PDF check
        pdf = c1 * d1 - mu * c2 * d2
        if score > pdf
            continue
        end

        # Nuclear screening check
        l2 = mu2 / (mu0 * mu0)
        q = muni * l2
        r = mu0 * l2
        qr = q * r
        if qr > T(1e5)
            continue  # numerically unstable region
        end
        denom = r + qr
        if denom <= zero(T) || (one(T) + qr) <= zero(T)
            continue
        end
        x = (one(T) + T(2) * qr) * log((one(T) + qr) / denom) -
            (one(T) - r) * (one(T) + T(2) * q) / (one(T) + q)

        pdf *= x / x0
        if score <= pdf
            return mu
        end
    end

    return zero(T)  # Fallback: no deflection
end

"""
    polar_pair_production(mass, ki, kf, Z, A, rng)

Sample polar angle for a pair-production event.
PUMAS uses the same formula as bremsstrahlung (virtual bremsstrahlung assumption).
"""
@inline polar_pair_production(mass::T, ki::T, kf::T, Z::T, A::T,
                              rng::AbstractRNG) where T<:Real =
    polar_bremsstrahlung(mass, ki, kf, Z, A, rng)

"""
    polar_photonuclear(mass, ki, kf, Z, A, rng)

Sample polar angle for a photonuclear event.
Simplified version: samples Q² uniformly in log-space from the DDCS envelope,
then converts to mu via energy-momentum conservation.

Reference: PUMAS polar_photonuclear().
"""
function polar_photonuclear(mass::T, ki::T, kf::T, Z::T, A::T,
                             rng::AbstractRNG) where T<:Real
    M = T(0.5) * (T(PROTON_MASS) + T(NEUTRON_MASS))
    q = ki - kf
    ml = mass
    E = ki + ml
    ml2 = ml * ml

    Q2min = ml2 * (q * q - T(0.5) * ml2) / (E * (E - q))
    Q2max = T(2) * M * (q - T(PION_MASS)) - T(PION_MASS)^2

    if Q2max < Q2min || Q2min < zero(T)
        return zero(T)
    end

    # Simplified sampling: uniform in log(Q²) then accept
    # For a simplified version we sample Q² uniformly in log-space
    lnQ2min = log(Q2min)
    lnQ2max = log(Q2max)
    rQ2 = lnQ2max - lnQ2min

    # Sample Q² (single draw; the full PUMAS version does rejection but that
    # requires the full photonuclear DDCS which we don't have tabulated.
    # Using the midpoint of the log range gives a representative angle.)
    u = rand(rng)
    Q2 = Q2min * exp(rQ2 * u)

    # Convert Q² → mu via kinematics
    p = sqrt(ki * (ki + T(2) * ml))
    E1 = E - q
    eps_ratio = ml / E1
    p1 = E1 * sqrt(max((one(T) + eps_ratio) * (one(T) - eps_ratio), zero(T)))
    p2 = p * p1
    if p2 <= zero(T)
        return zero(T)
    end

    tmp_val = p2 + ml2 - E * E1
    if abs(tmp_val) <= T(3) * eps(T(1.0)) * p2
        tmp_val = zero(T)
    end
    a_mu = T(0.5) * tmp_val / p2
    b_mu = T(0.25) / p2
    mu = a_mu + b_mu * Q2

    return clamp(mu, zero(T), one(T))
end

"""
    polar_ionisation(mass, ki, kf)

Compute polar angle for an ionisation (delta-ray) event.
Exact formula from energy-momentum conservation (electron at rest).

Reference: Salvat (2013), NIMB 316; PUMAS polar_ionisation().
"""
@inline function polar_ionisation(mass::T, ki::T, kf::T) where T<:Real
    nu = ki - kf
    p2 = ki * (ki + T(2) * mass)
    E = ki + mass

    denom = sqrt(p2 * (p2 + nu * nu - T(2) * nu * E))
    if denom <= zero(T)
        return zero(T)
    end

    c = (p2 - nu * (E + T(ELECTRON_MASS))) / denom
    return T(0.5) * (one(T) - c)
end

"""
    sample_target_element(material::BaseMaterial, rng)

Randomly sample a target atomic element from a material,
weighted by mass fraction. Returns (Z, A) of the selected element.
"""
function sample_target_element(material::BaseMaterial, rng::AbstractRNG)
    u = rand(rng)
    cumul = 0.0
    for (i, f) in enumerate(material.fractions)
        cumul += f
        if u <= cumul
            el = material.elements[i]
            return el.Z, el.A
        end
    end
    # Fallback: last element
    el = material.elements[end]
    return el.Z, el.A
end

"""
    polar_del_angle(physics, material_name, process, ki, kf, rng)

Sample the polar angle mu = 0.5*(1-cos θ) for a DEL event.

# Arguments
- `physics`: Physics tables (for particle mass)
- `material_name`: Name of the material (to look up elements)
- `process`: DEL process (1=bremsstrahlung, 2=pair production, 3=photonuclear)
- `ki`: Higher kinetic energy (before DEL in forward sense)
- `kf`: Lower kinetic energy (after DEL in forward sense)
- `rng`: Random number generator

Returns mu ∈ [0, 1]. The caller must ensure ki > kf.
"""
function polar_del_angle(physics::PhysicsTables{T}, material_name::String,
                          process::Int, ki::T, kf::T,
                          rng::AbstractRNG) where T<:Real
    mass = physics.mass

    # Look up material to get element-level data
    mat = get(MATERIALS, material_name, nothing)
    if mat === nothing
        return zero(T)  # Cannot compute angle without element data
    end

    # Sample target element
    Z, A = sample_target_element(mat, rng)
    Z_T = T(Z)
    A_T = T(A)

    if process == 1
        # Bremsstrahlung
        return polar_bremsstrahlung(mass, ki, kf, Z_T, A_T, rng)
    elseif process == 2
        # Pair production (same formula as bremsstrahlung)
        return polar_pair_production(mass, ki, kf, Z_T, A_T, rng)
    elseif process == 3
        # Photonuclear
        return polar_photonuclear(mass, ki, kf, Z_T, A_T, rng)
    else
        # Ionisation (delta ray) – process 4 if ever used
        return polar_ionisation(mass, ki, kf)
    end
end

"""
    polar_del_angle(physics, material_idx, process, ki, kf, rng)

Convenience overload accepting material index instead of name.
"""
function polar_del_angle(physics::PhysicsTables{T}, material_idx::Int,
                          process::Int, ki::T, kf::T,
                          rng::AbstractRNG) where T<:Real
    @inbounds table = physics.tables[material_idx]
    return polar_del_angle(physics, table.name, process, ki, kf, rng)
end

# =============================================================================
# Mixture-aware stochastic functions
# =============================================================================
# For mixtures, continuous quantities (straggling, stopping power) are weighted
# sums. For discrete events (DEL, EHS), we first compute mixture cross-sections
# then sample which material the interaction occurs in.

"""
    compute_del_cross_section(physics, mix::MaterialMixture, kinetic)

Compute mixture DEL cross-section: weighted sum of per-material cross-sections.
"""
@inline function compute_del_cross_section(physics::PhysicsTables{T}, mix::MaterialMixture,
                                           kinetic::T) where T<:Real
    if is_single_material(mix)
        return compute_del_cross_section(physics, single_material(mix), kinetic)
    end
    xs = zero(T)
    @inbounds for j in eachindex(mix.materials)
        xs += T(mix.fractions[j]) * compute_del_cross_section(physics, mix.materials[j], kinetic)
    end
    return xs
end

"""
    compute_ehs_mean_free_path(physics, mix::MaterialMixture, kinetic)

Compute mixture EHS mean free path: harmonic mean weighted by fractions.
    1/λ_mix = sum(f_i / λ_i)
"""
@inline function compute_ehs_mean_free_path(physics::PhysicsTables{T}, mix::MaterialMixture,
                                            kinetic::T) where T<:Real
    if is_single_material(mix)
        return compute_ehs_mean_free_path(physics, single_material(mix), kinetic)
    end
    inv_path = zero(T)
    @inbounds for j in eachindex(mix.materials)
        lp = compute_ehs_mean_free_path(physics, mix.materials[j], kinetic)
        if lp > zero(T) && lp < T(EHS_PATH_MAX)
            inv_path += T(mix.fractions[j]) / lp
        end
    end
    return inv_path > zero(T) ? one(T) / inv_path : T(EHS_PATH_MAX)
end

"""
    fluctuate_energy_loss(physics, mix::MaterialMixture, ki, dX, Xtot, rng; backward=false)

Apply energy fluctuation for a material mixture. Uses mixture straggling variance
and mixture stopping power for the Landau/Vavilov model.
"""
@inline function fluctuate_energy_loss(physics::PhysicsTables{T}, mix::MaterialMixture,
                               ki::T, dX::T, Xtot::T, rng::AbstractRNG;
                               backward::Bool = false) where T<:Real
    if is_single_material(mix)
        return fluctuate_energy_loss(physics, single_material(mix), ki, dX, Xtot, rng; backward=backward)
    end

    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_MIXED

    # Get expected final energy using mixture range tables
    k1 = property_kinetic_energy(physics, mode, mix, Xtot - sgn * dX)
    dk0 = abs(ki - k1)

    ratio = one(T)

    if k1 > zero(T) && dk0 > zero(T)
        Omega0 = property_straggling(physics, mix, ki)
        Omega1 = property_straggling(physics, mix, k1)
        dk12 = T(0.5) * dX * (Omega0 + Omega1)

        tmp = dX / Xtot
        if tmp > T(X_THRESHOLD)
            de1 = property_stopping_power(physics, ENERGY_LOSS_MIXED, mix, k1)
            de0 = property_stopping_power(physics, ENERGY_LOSS_MIXED, mix, ki)
            dk12 *= one(T) + sgn * dX / dk0 * (de1 - de0)
        end

        dk1 = sqrt(max(dk12, zero(T)))

        if dk0 >= T(3) * dk1 && dk1 > zero(T)
            u = truncated_randn(rng)
            k1 += u * dk1
        elseif dk0 >= T(1.7320508) * dk1 && dk1 > zero(T)
            u = T(1.7320508) * (one(T) - T(2) * rand(rng))
            k1 += u * dk1
        elseif dk1 > zero(T)
            dk32 = T(3) * dk12
            dk02 = dk0 * dk0
            a = one(T) - (dk32 - dk02) / (dk32 + T(3) * dk02)
            if rand(rng) <= a
                b = T(0.5) * (dk32 + T(3) * dk02) / dk0
                u = rand(rng)
                k1 = ki - sgn * b * u
                if k1 < zero(T)
                    k1 = zero(T)
                end
            else
                k1 = ki
            end
        end
        ratio = abs(ki - k1) / dk0
    end

    if ratio == zero(T)
        ratio = one(T)
    end
    k1 = max(k1, zero(T))
    return k1, ratio
end

"""
    sample_del_event(physics, mix::MaterialMixture, ki, kf, dX, Xtot, ratio, rng; backward=false)

Sample DEL event for a material mixture. The total cross-section is the mixture sum.
If a DEL occurs, one material is sampled according to fraction-weighted cross-sections
and that material's tables are used for the energy transfer and process identification.
"""
@inline function sample_del_event(physics::PhysicsTables{T}, mix::MaterialMixture,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    if is_single_material(mix)
        return sample_del_event(physics, single_material(mix), ki, kf, dX, Xtot, ratio, rng; backward=backward)
    end

    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_MIXED

    # Mixture cross-sections
    xs_del_i = compute_del_cross_section(physics, mix, ki)
    xs_del_f = compute_del_cross_section(physics, mix, kf)
    xs_del = max(xs_del_i, xs_del_f)

    if xs_del <= zero(T)
        return false, zero(T), zero(T), 0
    end

    zeta = rand(rng)
    if zeta <= T(1e-12) || zeta >= one(T) - T(1e-12)
        return false, zero(T), zero(T), 0
    end

    X_del = -log(zeta) / xs_del
    if X_del >= dX
        return false, zero(T), zero(T), 0
    end

    k_h = property_kinetic_energy(physics, mode, mix, Xtot - sgn * X_del)
    k_at_del = ki - sgn * ratio * abs(ki - k_h)
    if k_at_del <= zero(T)
        return false, zero(T), zero(T), 0
    end

    r = compute_del_cross_section(physics, mix, k_at_del) / xs_del
    if rand(rng) > r
        return false, zero(T), zero(T), 0
    end

    # DEL occurred – sample which material
    mat_idx = sample_mixture_material(physics, mix, k_at_del, rng)

    # Sample energy transfer
    cutoff = physics.settings.cutoff
    alpha = T(BMC_ALPHA)
    u = rand(rng)
    if abs(alpha - one(T)) < T(1e-6)
        nu = cutoff * exp(u * log(one(T) / cutoff))
    else
        cutoff_term = cutoff^(one(T) - alpha)
        nu = (u * (one(T) - cutoff_term) + cutoff_term)^(one(T) / (one(T) - alpha))
    end
    nu = clamp(nu, cutoff, one(T))

    if backward
        k_del = k_at_del / (one(T) - nu)
    else
        k_del = k_at_del * (one(T) - nu)
    end

    # Process identification from sampled material
    table = physics.tables[mat_idx]
    brems = interpolate_table(k_at_del, table.energies, table.bremsstrahlung)
    pair = interpolate_table(k_at_del, table.energies, table.pair_production)
    photo = interpolate_table(k_at_del, table.energies, table.photonuclear)
    total = brems + pair + photo

    process = 1
    if total > zero(T)
        r_proc = rand(rng) * total
        if r_proc < brems
            process = 1
        elseif r_proc < brems + pair
            process = 2
        else
            process = 3
        end
    end

    return true, X_del, k_del, process
end

"""
    sample_soft_scattering(physics, mix::MaterialMixture, kinetic_start, kinetic_end, step_distance, density, rng)

Sample soft multiple scattering angle for a material mixture.
Full PUMAS trapezoidal formula.
"""
@inline function sample_soft_scattering(physics::PhysicsTables{T}, mix::MaterialMixture,
                               kinetic_start::T, kinetic_end::T,
                               step_distance::T, density::T,
                               rng::AbstractRNG) where T<:Real
    if is_single_material(mix)
        return sample_soft_scattering(physics, single_material(mix),
                                      kinetic_start, kinetic_end,
                                      step_distance, density, rng)
    end

    if step_distance <= zero(T) || density <= zero(T)
        return zero(T)
    end

    lb1_start = property_transport_path(physics, mix, kinetic_start)
    invlb1_start = (lb1_start > zero(T)) ? density / lb1_start : zero(T)

    if kinetic_end > zero(T)
        lb1_end = property_transport_path(physics, mix, kinetic_end)
        invlb1_end = (lb1_end > zero(T)) ? density / lb1_end : zero(T)
    else
        invlb1_end = invlb1_start
    end

    if invlb1_start <= zero(T) && invlb1_end <= zero(T)
        return zero(T)
    end

    ilb1 = T(0.25) * step_distance * (invlb1_start + invlb1_end)

    if ilb1 < T(1e-12)
        return zero(T)
    end
    if ilb1 > one(T)
        ilb1 = one(T)
    end

    for _ in 1:1000
        u = rand(rng)
        if u < T(1e-300)
            continue
        end
        mu = -ilb1 * log(u)
        if mu <= one(T)
            return max(mu, zero(T))
        end
    end

    return ilb1
end

"""
    sample_soft_scattering(physics, mix::MaterialMixture, kinetic, grammage, rng)

Legacy form for backward compatibility.
"""
@inline function sample_soft_scattering(physics::PhysicsTables{T}, mix::MaterialMixture,
                               kinetic::T, grammage::T, rng::AbstractRNG) where T<:Real
    if is_single_material(mix)
        return sample_soft_scattering(physics, single_material(mix), kinetic, grammage, rng)
    end

    if grammage <= zero(T) || kinetic <= zero(T)
        return zero(T)
    end

    lb1 = property_transport_path(physics, mix, kinetic)
    density = (lb1 > zero(T)) ? grammage / (grammage / lb1 * lb1 + T(1e-30)) : zero(T)
    step_distance = (density > zero(T)) ? grammage / density : zero(T)

    return sample_soft_scattering(physics, mix, kinetic, kinetic,
                                  step_distance, density, rng)
end

"""
    sample_ehs_event(physics, mix::MaterialMixture, ki, kf, dX, Xtot, ratio, rng; backward=false)

Sample elastic hard scattering for a material mixture.
Uses mixture EHS mean free path.
"""
@inline function sample_ehs_event(physics::PhysicsTables{T}, mix::MaterialMixture,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    if is_single_material(mix)
        return sample_ehs_event(physics, single_material(mix), ki, kf, dX, Xtot, ratio, rng; backward=backward)
    end

    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_MIXED

    kmin = min(ki, kf)
    kmax = max(ki, kf)
    if kmin < T(0.001)
        kmin, kmax = kmax, kmin
    end

    lb_ehs = compute_ehs_mean_free_path(physics, mix, kmin)
    if lb_ehs <= zero(T) || lb_ehs >= T(EHS_PATH_MAX)
        return false, zero(T), zero(T)
    end

    zeta = rand(rng)
    if zeta <= zero(T) || zeta >= one(T)
        return false, zero(T), zero(T)
    end

    X_ehs = -log(zeta) * lb_ehs
    if X_ehs >= dX
        return false, zero(T), zero(T)
    end

    k_h = property_kinetic_energy(physics, mode, mix, Xtot - sgn * X_ehs)
    k_ehs = ki - sgn * ratio * abs(ki - k_h)
    if k_ehs <= zero(T)
        return false, zero(T), zero(T)
    end

    lb_ehs_k = compute_ehs_mean_free_path(physics, mix, k_ehs)
    if lb_ehs_k <= zero(T) || lb_ehs_k >= T(EHS_PATH_MAX)
        return false, zero(T), zero(T)
    end

    r = lb_ehs / lb_ehs_k
    if r <= zero(T) || rand(rng) > one(T) / r
        return false, zero(T), zero(T)
    end

    return true, X_ehs, k_ehs
end

"""
    sample_scattering_angle(physics, mix::MaterialMixture, kinetic, rng; mu0=nothing)

Sample EHS polar scattering angle for a material mixture using
fraction-weighted screening parameter, spin factor, and Z/A.
This matches how a precomputed composite table stores these quantities
(computed from the flattened atomic composition).
"""
@inline function sample_scattering_angle(physics::PhysicsTables{T}, mix::MaterialMixture,
                                  kinetic::T, rng::AbstractRNG;
                                  mu0::Union{T, Nothing} = nothing) where T<:Real
    if is_single_material(mix)
        return sample_scattering_angle(physics, single_material(mix), kinetic, rng; mu0=mu0)
    end

    mass = physics.mass
    p2 = kinetic * (kinetic + T(2) * mass)

    if mu0 === nothing
        mu0 = property_screening(physics, mix, kinetic)
    end

    ZoA_mix = mixture_ZoA(physics, mix)
    A_target = ZoA_mix > zero(T) ? one(T) / ZoA_mix : T(22.0)
    M_target = A_target * T(NEUTRON_MASS)

    s = mass^2 + M_target^2 + T(2) * (kinetic + mass) * M_target
    kinetic0 = (s - (mass + M_target)^2) / (T(2) * sqrt(s))

    E_cm = kinetic0 + mass
    p_cm = sqrt(max(kinetic0 * (kinetic0 + T(2) * mass), zero(T)))
    E_lab = kinetic + mass
    p_lab = sqrt(max(p2, zero(T)))

    gamma_CM = (p_lab > zero(T) && p_cm > zero(T)) ? (E_lab * E_cm + p_lab * p_cm) / (mass * sqrt(s)) : one(T)
    tau = (p_lab > zero(T) && p_cm > zero(T)) ? (E_lab * p_cm - E_cm * p_lab) / (p_lab * p_cm) : zero(T)

    A = mu0 / T(4)
    fspin = property_spin_factor(physics, mix, kinetic)

    A_plus_mu0 = A + mu0
    A_plus_1 = A + one(T)
    one_minus_mu0 = one(T) - mu0

    mu1 = mu0
    @inbounds for _ in 1:100
        zeta = rand(rng)
        tmp = A_plus_1 - zeta * one_minus_mu0
        if tmp <= zero(T)
            continue
        end
        mu1_trial = A_plus_mu0 * A_plus_1 / tmp - A
        mu1_trial = clamp(mu1_trial, mu0, one(T))
        if rand(rng) <= one(T) - fspin * mu1_trial
            mu1 = mu1_trial
            break
        end
    end

    if mu1 > T(1e-6)
        a = gamma_CM * (tau + one(T) - T(2) * mu1)
        ct_h = a / sqrt(T(4) * mu1 * (one(T) - mu1) + a * a)
        mu = T(0.5) * (one(T) - ct_h)
    else
        d = gamma_CM * (one(T) + tau)
        mu = mu1 / (d * d)
    end

    return clamp(mu, zero(T), one(T))
end

end # module Straggling
