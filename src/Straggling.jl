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

# Constants from PUMAS
const BMC_ALPHA = 2.0         # Exponent for backward MC DEL sampling
const EHS_PATH_MAX = 1e9      # Maximum path for EHS
const X_THRESHOLD = 0.05      # Threshold for straggling correction
const STEP_EPSILON = 1e-7     # Small offset for step boundaries
const MSC_ACCURACY = 0.01     # Default accuracy for MSC angular distribution

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
    
    # Get tabulated elastic path (primary source)
    elastic_path = interpolate_table(kinetic, table.energies, table.elastic_path)
    
    if elastic_path > zero(T) && elastic_path < T(EHS_PATH_MAX)
        return elastic_path
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
Uses Coulomb scattering with screening based on PUMAS implementation.

Returns mu (0 = forward, 1 = backward).
"""
@inline function sample_scattering_angle(physics::PhysicsTables{T}, material::Int,
                                  kinetic::T, rng::AbstractRNG;
                                  mu0::Union{T, Nothing} = nothing) where T<:Real
    
    mass = physics.mass
    gamma = one(T) + kinetic / mass
    beta2 = one(T) - one(T) / gamma^2
    p2 = kinetic * (kinetic + T(2) * mass)
    
    # Get cutoff angle (screening parameter) from table or compute
    table = physics.tables[material]
    if mu0 === nothing
        # Use tabulated or computed screening parameter
        mu0 = interpolate_table(kinetic, table.energies, table.screening_parameter)
        mu0 = max(mu0, T(1e-12))
    end
    
    # Screening parameter for nuclear form factor
    # A ≈ mu0 / 4 following Molière
    A = mu0 / T(4)
    
    # Precompute spin factor (outside loop for efficiency)
    fspin = interpolate_table(kinetic, table.energies, table.spin_factor)
    
    # Precompute constants for sampling
    A_plus_mu0 = A + mu0
    A_plus_1 = A + one(T)
    one_minus_mu0 = one(T) - mu0
    
    # Sample from Wentzel-like distribution: dσ/dμ ∝ 1/(A + μ)²
    # Using inverse CDF: μ = A * (A + 1) / (A + 1 - ξ * (1 - μ0)) - A
    @inbounds for _ in 1:100
        zeta = rand(rng)
        tmp = A_plus_1 - zeta * one_minus_mu0
        
        if tmp <= zero(T)
            continue
        end
        
        mu1 = A_plus_mu0 * A_plus_1 / tmp - A
        mu1 = clamp(mu1, mu0, one(T))
        
        # Spin correction factor rejection
        if rand(rng) <= one(T) - fspin * mu1
            return mu1
        end
    end
    
    # Fallback: return small angle
    return mu0
end

"""
    sample_msc_angle(physics, material, kinetic, grammage, rng)

Sample the multiple scattering angle for soft scattering.
Uses PUMAS-style exponential sampling matching C implementation.

The formula: mu = -invlb1 * grammage * 0.25 * log(rand()) with rejection for mu > 1

Returns mu = 0.5*(1 - cos(theta)).
"""
@inline function sample_msc_angle(physics::PhysicsTables{T}, material::Int,
                          kinetic::T, grammage::T, rng::AbstractRNG) where T<:Real
    
    @inbounds begin
        if grammage <= zero(T) || kinetic <= zero(T)
            return zero(T)
        end
        
        # Get transport path from table
        table = physics.tables[material]
        lb1 = interpolate_table(kinetic, table.energies, table.transport_path)
        
        if lb1 <= zero(T)
            return zero(T)
        end
        
        # PUMAS formula: ilb1 = 0.5 * step * (invlb1_start + invlb1_end)
        # For similar start/end values: ilb1 ≈ step / lb1
        # Use 0.5 as estimate (C averages start/end, we only have start)
        ilb1 = T(0.5) * grammage / lb1
        
        if ilb1 <= zero(T)
            return zero(T)
        end
        
        # Cap ilb1 at 1 as in C
        if ilb1 > one(T)
            ilb1 = one(T)
        end
        
        # Efficient sampling from truncated exponential distribution
        # mu ~ Exp(rate = 1/ilb1) truncated to [0, 1]
        # Using inverse CDF method (no rejection needed):
        # F(x) = (1 - exp(-x/ilb1)) / (1 - exp(-1/ilb1)) for x in [0, 1]
        # Inverse: x = -ilb1 * log(1 - u * (1 - exp(-1/ilb1)))
        u = rand(rng)
        inv_ilb1 = one(T) / ilb1
        exp_factor = one(T) - exp(-inv_ilb1)
        mu = -ilb1 * log(one(T) - u * exp_factor)
        
        # Ensure mu is in valid range (numerical safety)
        return min(max(mu, zero(T)), one(T))
    end
end

"""
    sample_soft_scattering(physics, material, kinetic, grammage, rng)

Sample soft multiple scattering angle for a given grammage step.
Alias for sample_msc_angle for backward compatibility.
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

end # module Straggling
