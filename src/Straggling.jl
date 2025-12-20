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
function box_muller_randn(rng::AbstractRNG)
    r = sqrt(-2.0 * log(rand(rng)))
    phi = 2.0 * π * rand(rng)
    return r * sin(phi)
end

"""
    truncated_randn(rng, limit=3.0)

Generate a truncated normal random variate |u| <= limit.
The value is scaled by 1.015387 to match PUMAS normalization.
"""
function truncated_randn(rng::AbstractRNG, limit::Float64 = 3.0)
    u = box_muller_randn(rng)
    while abs(u) > limit
        u = box_muller_randn(rng)
    end
    return u / 1.015387
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
function fluctuate_energy_loss(physics::PhysicsTables{T}, material::Int,
                               ki::T, dX::T, Xtot::T, rng::AbstractRNG;
                               backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_CSDA
    
    # Get expected final energy from CSDA
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
            de1 = property_stopping_power(physics, mode, material, k1)
            de0 = property_stopping_power(physics, mode, material, ki)
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
Uses the radiative stopping power to estimate the hard event cross-section.

Returns cross-section in m²/kg.
"""
function compute_del_cross_section(physics::PhysicsTables{T}, material::Int, 
                                   kinetic::T) where T<:Real
    if kinetic < T(1e-3)
        return zero(T)
    end
    
    # Get the tabulated cross-section (interpolated)
    # This is computed from the radiative components
    table = physics.tables[material]
    
    # Get radiative stopping powers
    brems = interpolate_table(kinetic, table.energies, table.bremsstrahlung)
    pair = interpolate_table(kinetic, table.energies, table.pair_production)
    photo = interpolate_table(kinetic, table.energies, table.photonuclear)
    
    radiative = brems + pair + photo
    
    # Cross-section from σ ≈ (dE/dx)_rad / (ε * K) where ε is cutoff
    # Using cutoff ~0.05 as in PUMAS
    cutoff = physics.settings.cutoff
    
    # The cross-section scales as radiative/(cutoff * K)
    # but with a lower limit to avoid numerical issues
    if radiative > zero(T) && kinetic > T(0.001)
        sigma = radiative / (cutoff * kinetic)
        return min(sigma, T(1e-3))  # Cap to avoid unrealistic values
    else
        return zero(T)
    end
end

"""
    sample_del_event(physics, material, ki, kf, dX, Xtot, ratio, rng; backward=false)

Sample whether a discrete energy loss event occurs during the step.
Based on PUMAS's DEL sampling in step_fluctuate.

# Returns
- `occurred`: Whether a DEL event occurred
- `X_del`: Grammage at which event occurred (if any)
- `k_del`: Energy at which event occurred (if any)
"""
function sample_del_event(physics::PhysicsTables{T}, material::Int,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_CSDA
    
    # Get maximum cross-section over the step
    xs_del = compute_del_cross_section(physics, material, ki)
    xs_del_f = compute_del_cross_section(physics, material, kf)
    xs_del = max(xs_del, xs_del_f)
    
    if xs_del <= zero(T)
        return false, zero(T), zero(T)
    end
    
    # Sample interaction length (exponential distribution)
    zeta = rand(rng)
    if zeta <= zero(T) || zeta >= one(T)
        return false, zero(T), zero(T)
    end
    
    X_del = -log(zeta) / xs_del
    
    if X_del >= dX
        return false, zero(T), zero(T)
    end
    
    # Get energy at the interaction point
    k_h = property_kinetic_energy(physics, mode, material, Xtot - sgn * X_del)
    k_del = ki - sgn * ratio * abs(ki - k_h)
    
    if k_del <= zero(T)
        return false, zero(T), zero(T)
    end
    
    # Acceptance probability
    r = compute_del_cross_section(physics, material, k_del) / xs_del
    
    if rand(rng) > r
        return false, zero(T), zero(T)
    end
    
    return true, X_del, k_del
end

"""
    compute_ehs_mean_free_path(physics, material, kinetic)

Compute the elastic hard scattering mean free path.
Based on PUMAS coulomb_ehs_length.

Returns mean free path in kg/m².
"""
function compute_ehs_mean_free_path(physics::PhysicsTables{T}, material::Int,
                                     kinetic::T) where T<:Real
    if kinetic < T(1e-3)
        return T(EHS_PATH_MAX)
    end
    
    mass = physics.mass
    table = physics.tables[material]
    
    # Get elastic scattering parameters
    p2 = kinetic * (kinetic + T(2) * mass)
    
    # The EHS path is Lb / p² where Lb is tabulated
    # We approximate Lb from the elastic_path table
    elastic_path = interpolate_table(kinetic, table.energies, table.elastic_path)
    
    # PUMAS divides by p² for the actual path
    # lb_ehs = Lb / p² where Lb includes the p² factor in tabulation
    # So the actual EHS mfp scales inversely with p²
    ehs_path = elastic_path * p2
    
    return min(ehs_path, T(EHS_PATH_MAX))
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
function sample_ehs_event(physics::PhysicsTables{T}, material::Int,
                          ki::T, kf::T, dX::T, Xtot::T, ratio::T,
                          rng::AbstractRNG;
                          backward::Bool = false) where T<:Real
    
    sgn = backward ? T(-1) : T(1)
    mode = ENERGY_LOSS_CSDA
    
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
function sample_scattering_angle(physics::PhysicsTables{T}, material::Int,
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
    
    # Sample from Wentzel-like distribution: dσ/dμ ∝ 1/(A + μ)²
    # Using inverse CDF: μ = A * (A + 1) / (A + 1 - ξ * (1 - μ0)) - A
    max_iter = 1000
    for _ in 1:max_iter
        zeta = rand(rng)
        tmp = one(T) + A - zeta * (one(T) - mu0)
        
        if tmp <= zero(T)
            continue
        end
        
        mu1 = (A + mu0) * (A + one(T)) / tmp - A
        mu1 = clamp(mu1, mu0, one(T))
        
        # Spin correction factor rejection
        fspin = interpolate_table(kinetic, table.energies, table.spin_factor)
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
Uses Gaussian approximation with Highland formula width.

Returns mu = 0.5*(1 - cos(theta)).
"""
function sample_msc_angle(physics::PhysicsTables{T}, material::Int,
                          kinetic::T, grammage::T, rng::AbstractRNG) where T<:Real
    
    if grammage <= zero(T) || kinetic <= zero(T)
        return zero(T)
    end
    
    mass = physics.mass
    p = sqrt(kinetic * (kinetic + T(2) * mass))
    beta = p / (kinetic + mass)
    
    # Get transport path (related to radiation length)
    table = physics.tables[material]
    lb1 = interpolate_table(kinetic, table.energies, table.transport_path)
    
    if lb1 <= zero(T)
        return zero(T)
    end
    
    # First transport coefficient: μ₁ ≈ x / λ₁
    mu1 = grammage / lb1
    
    # Highland formula RMS angle
    # θ_rms² ≈ 2 * μ₁ for Gaussian approximation
    theta_rms_sq = T(2) * mu1
    
    if theta_rms_sq <= zero(T)
        return zero(T)
    end
    
    theta_rms = sqrt(theta_rms_sq)
    
    # Sample from 2D Gaussian and convert to mu
    # Rayleigh distribution for |θ|
    u1 = box_muller_randn(rng) * theta_rms
    u2 = box_muller_randn(rng) * theta_rms
    theta = sqrt(u1^2 + u2^2)
    
    # Convert to mu = 0.5*(1 - cos(theta))
    if theta < T(0.1)
        # Small angle approximation
        mu = theta^2 / T(4)
    else
        mu = T(0.5) * (one(T) - cos(theta))
    end
    
    # Cap at 90 degrees (mu = 0.5)
    return min(mu, T(0.5))
end

"""
    sample_soft_scattering(physics, material, kinetic, grammage, rng)

Sample soft multiple scattering angle for a given grammage step.
Alias for sample_msc_angle for backward compatibility.
"""
sample_soft_scattering = sample_msc_angle

"""
    rotate_direction(direction, mu, rng)

Rotate the direction vector randomly with polar angle given by mu = 0.5*(1 - cos(theta)).
Returns the new direction vector (normalized).

Based on PUMAS step_rotate_direction implementation.
"""
function rotate_direction(direction::Vec3{T}, mu::T, rng::AbstractRNG) where T<:Real
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
    
    # Cross product: u1 = d × u0
    u1x = d[2] * u0z - d[3] * u0y
    u1y = d[3] * u0x - d[1] * u0z
    u1z = d[1] * u0y - d[2] * u0x
    
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
