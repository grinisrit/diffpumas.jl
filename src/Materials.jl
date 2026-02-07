"""
    Materials

Material properties and Differential Cross-Section (DCS) calculations.
"""
module Materials

using ..Constants
using ..Types
using SpecialFunctions
using ChainRulesCore

export AtomicElement, BaseMaterial, CompositeMaterial
export ELEMENTS, MATERIALS, STANDARD_ROCK, AIR, WATER
export electronic_stopping_power, electronic_density_effect
export elastic_dcs, elastic_path, electronic_dcs
export dcs_bremsstrahlung_ssr, dcs_pair_production_ssr, dcs_photonuclear_drss

"""
    AtomicElement

Properties of an atomic element.

# Fields
- `name::String`: Element name
- `Z::Float64`: Atomic number
- `A::Float64`: Mass number in g/mol
- `I::Float64`: Mean excitation energy in GeV
"""
struct AtomicElement
    name::String
    Z::Float64
    A::Float64
    I::Float64  # Mean excitation energy in GeV
end

# Standard atomic elements data
const ELEMENTS = Dict{String, AtomicElement}(
    "H" => AtomicElement("H", 1.0, 1.00794, 19.2e-9),
    "C" => AtomicElement("C", 6.0, 12.0107, 78.0e-9),
    "N" => AtomicElement("N", 7.0, 14.0067, 82.0e-9),
    "O" => AtomicElement("O", 8.0, 15.9994, 95.0e-9),
    "Na" => AtomicElement("Na", 11.0, 22.9898, 149.0e-9),
    "Mg" => AtomicElement("Mg", 12.0, 24.305, 156.0e-9),
    "Al" => AtomicElement("Al", 13.0, 26.9815, 166.0e-9),
    "Si" => AtomicElement("Si", 14.0, 28.0855, 173.0e-9),
    "K" => AtomicElement("K", 19.0, 39.0983, 190.0e-9),
    "Ca" => AtomicElement("Ca", 20.0, 40.078, 191.0e-9),
    "Fe" => AtomicElement("Fe", 26.0, 55.845, 286.0e-9),
    "Ar" => AtomicElement("Ar", 18.0, 39.948, 188.0e-9),
)

"""
    BaseMaterial

Properties of a base material (homogeneous composition).

# Fields
- `name::String`: Material name
- `density::Float64`: Reference density in kg/m³
- `I::Float64`: Mean excitation energy in GeV
- `elements::Vector{AtomicElement}`: Constituent elements
- `fractions::Vector{Float64}`: Mass fractions of elements
"""
struct BaseMaterial
    name::String
    density::Float64
    I::Float64
    elements::Vector{AtomicElement}
    fractions::Vector{Float64}
    
    # Precomputed properties
    ZoA::Float64  # Effective Z/A ratio
    
    function BaseMaterial(name, density, I, elements, fractions)
        # Normalize fractions
        total = sum(fractions)
        norm_fractions = fractions ./ total
        
        # Compute effective Z/A
        ZoA = sum(e.Z / e.A * f for (e, f) in zip(elements, norm_fractions))
        
        new(name, density, I, elements, norm_fractions, ZoA)
    end
end

"""
    CompositeMaterial

A composite material made of base materials.
"""
struct CompositeMaterial
    name::String
    components::Vector{BaseMaterial}
    fractions::Vector{Float64}
    
    function CompositeMaterial(name, components, fractions)
        total = sum(fractions)
        new(name, components, fractions ./ total)
    end
end

# Standard materials
const STANDARD_ROCK = BaseMaterial(
    "StandardRock",
    2650.0,  # kg/m³
    136.4e-9,  # GeV
    [ELEMENTS["O"], ELEMENTS["Si"], ELEMENTS["Al"], ELEMENTS["Fe"], 
     ELEMENTS["Ca"], ELEMENTS["Na"], ELEMENTS["Mg"], ELEMENTS["K"]],
    [0.466, 0.277, 0.081, 0.050, 0.036, 0.028, 0.021, 0.026]
)

const AIR = BaseMaterial(
    "Air",
    1.205,  # kg/m³ at STP
    85.7e-9,  # GeV
    [ELEMENTS["N"], ELEMENTS["O"], ELEMENTS["Ar"]],
    [0.755, 0.232, 0.013]
)

const WATER = BaseMaterial(
    "Water",
    1000.0,  # kg/m³
    75.0e-9,  # GeV (mean excitation energy for water)
    [ELEMENTS["H"], ELEMENTS["O"]],
    [0.1119, 0.8881]  # Mass fractions for H₂O
)

const MATERIALS = Dict{String, BaseMaterial}(
    "StandardRock" => STANDARD_ROCK,
    "Air" => AIR,
    "Water" => WATER
)

"""
    electronic_density_effect(material, gamma)

Compute the electronic density effect δ for a material.
Uses the Fano model with atomic binding energies.

# Arguments
- `material::BaseMaterial`: The material
- `gamma::Real`: Relativistic gamma factor

# Returns
The density effect δ
"""
function electronic_density_effect(material::BaseMaterial, gamma::T) where T<:Real
    # Plasma frequency squared (in GeV²)
    ωp² = 4π * ALPHA_EM * HBAR_C^3 * AVOGADRO_NUMBER * 
          material.density * 1e-3 * material.ZoA / ELECTRON_MASS
    
    β² = one(T) - one(T) / (gamma^2)
    β²γ² = β² * gamma^2
    
    # Sternheimer parametrization
    x = log10(sqrt(β²γ²))
    x0 = 0.2  # Typical values for condensed matter
    x1 = 3.0
    C = -2 * log(material.I / sqrt(ωp²))
    a = (C - 4.6052 * x0) / (x1 - x0)^3
    k = 3.0
    
    δ = if x < x0
        zero(T)
    elseif x < x1
        4.6052 * x + C + a * (x1 - x)^k
    else
        4.6052 * x + C
    end
    
    return δ
end

"""
    electronic_stopping_power(material, mass, energy)

Compute the electronic stopping power for a particle.
Based on the Bethe formula with density effect correction.

# Arguments
- `material::BaseMaterial`: Target material
- `mass::Real`: Projectile mass in GeV/c²
- `energy::Real`: Kinetic energy in GeV

# Returns
Stopping power in GeV/(kg/m²)
"""
function electronic_stopping_power(material::BaseMaterial, mass::T, energy::T) where T<:Real
    # Kinematic quantities
    gamma = one(T) + energy / mass
    β² = one(T) - one(T) / (gamma^2)
    β = sqrt(β²)
    
    # Maximum energy transfer to electron
    Tmax = 2 * ELECTRON_MASS * β² * gamma^2 / 
           (1 + 2 * gamma * ELECTRON_MASS / mass + (ELECTRON_MASS / mass)^2)
    
    # Density effect
    δ = electronic_density_effect(material, gamma)
    
    # Bethe formula prefactor
    K = 4π * AVOGADRO_NUMBER * ELECTRON_RADIUS^2 * ELECTRON_MASS
    
    # Stopping power (GeV m²/kg)
    dEdX = K * material.ZoA / β² * (
        0.5 * log(2 * ELECTRON_MASS * β² * gamma^2 * Tmax / material.I^2) - 
        β² - δ / 2
    )
    
    return dEdX
end

"""
    elastic_dcs(Z, A, m, K, θ)

Compute the elastic (Coulomb) differential cross-section.
Based on Salvat (2013) with nuclear form factors.

# Arguments
- `Z::Real`: Target atomic number
- `A::Real`: Target mass number
- `m::Real`: Projectile mass in GeV/c²
- `K::Real`: Kinetic energy in GeV
- `θ::Real`: Scattering angle in rad

# Returns
Atomic DCS in m²/rad
"""
function elastic_dcs(Z::Real, A::Real, m::Real, K::T, θ::T) where T<:Real
    # Kinematic quantities
    gamma = one(T) + K / m
    β² = one(T) - one(T) / (gamma^2)
    p = sqrt(K * (K + 2m))  # momentum in GeV/c
    
    # Effective mass for recoil
    M_target = A * 0.931494  # GeV/c² (approximate nucleon mass)
    m_eff = m * M_target / (m + M_target)
    
    # Momentum transfer
    sin_half = sin(θ / 2)
    q = 2 * p * sin_half  # GeV/c
    
    # Atomic screening (Thomas-Fermi)
    a_TF = BOHR_RADIUS * 0.8853 / Z^(1/3)  # Thomas-Fermi radius
    λ_a = HBAR_C / (a_TF * p)  # dimensionless screening parameter
    
    # Nuclear form factor (exponential approximation)
    R_n = 1.2e-15 * A^(1/3)  # nuclear radius in m
    λ_n = R_n * p / HBAR_C
    
    # Rutherford cross-section
    σ_R = (Z * ALPHA_EM * HBAR_C / (2 * p * β² * sin_half^2))^2
    
    # Screening factors
    F_atom = 1 / (1 + (sin_half / λ_a)^2)^2
    F_nucl = exp(-2 * (sin_half / λ_n)^2)
    
    # Spin correction (for spin-1/2 particles)
    F_spin = 1 - β² * sin_half^2
    
    # Total DCS (m²/rad)
    dσdθ = σ_R * F_atom^2 * F_nucl * F_spin * sin(θ)
    
    return dσdθ
end

"""
    elastic_path(order, Z, A, m, K)

Compute the (transport) mean free path for elastic collisions.

# Arguments
- `order::Int`: 0 for single collision m.f.p., 1 for transport m.f.p.
- `Z, A, m, K`: See `elastic_dcs`

# Returns
Path per unit mass in kg/m²
"""
function elastic_path(order::Int, Z::Real, A::Real, m::Real, K::T) where T<:Real
    if order != 0 && order != 1
        return -one(T)
    end
    
    # Numerical integration (simplified)
    N = 100
    dθ = π / N
    integral = zero(T)
    
    for i in 1:N
        θ = (i - 0.5) * dθ
        dσ = elastic_dcs(Z, A, m, K, θ)
        weight = order == 0 ? one(T) : (1 - cos(θ))
        integral += dσ * weight * dθ
    end
    
    # Convert to path length
    n = AVOGADRO_NUMBER / (A * 1e-3)  # atoms per kg
    λ = 1 / (n * integral)
    
    return λ
end

"""
    electronic_dcs(Z, I, m, K, q)

Electronic DCS restricted to close collisions (delta rays).

# Arguments
- `Z::Real`: Target atomic number
- `I::Real`: Mean excitation energy in GeV
- `m::Real`: Projectile mass in GeV/c²
- `K::Real`: Kinetic energy in GeV
- `q::Real`: Energy transfer in GeV

# Returns
Atomic DCS in m²/GeV
"""
function electronic_dcs(Z::Real, I::Real, m::Real, K::T, q::T) where T<:Real
    gamma = one(T) + K / m
    β² = one(T) - one(T) / (gamma^2)
    
    # Maximum energy transfer
    Tmax = 2 * ELECTRON_MASS * β² * gamma^2 / 
           (1 + 2 * gamma * ELECTRON_MASS / m + (ELECTRON_MASS / m)^2)
    
    if q <= zero(T) || q >= Tmax
        return zero(T)
    end
    
    # Prefactor
    K_factor = 2π * ELECTRON_RADIUS^2 * ELECTRON_MASS * Z / β²
    
    # DCS (Møller formula approximation)
    dσdq = K_factor / q^2 * (1 - β² * q / Tmax + q^2 / (2 * (K + m)^2))
    
    return dσdq
end

"""
    dcs_bremsstrahlung_ssr(Z, A, m, K, q)

Bremsstrahlung DCS using the SSR model (Sandrock, Soedingrekso, Rhode 2019).

# Arguments
- `Z, A, m, K, q`: Standard DCS arguments

# Returns
Atomic DCS in m²/GeV
"""
function dcs_bremsstrahlung_ssr(Z::Real, A::Real, m::Real, K::T, q::T) where T<:Real
    if q <= zero(T) || q >= K
        return zero(T)
    end
    
    v = q / (K + m)  # fractional energy loss
    if v >= one(T)
        return zero(T)
    end
    
    # Total energy
    E = K + m
    
    # Screening functions (simplified SSR)
    L_rad = log(184.15 * Z^(-1/3))
    L_rad_prime = log(1194.0 * Z^(-2/3))
    
    # Prefactor
    α = ALPHA_EM
    r_e = ELECTRON_RADIUS
    prefactor = 4 * α * r_e^2 * Z^2 / q
    
    # Nuclear contribution
    F_n = (4/3 - 4/3 * v + v^2) * (L_rad - log(v / (1 - v)) + 1/8)
    
    # Electron contribution (smaller, simplified)
    F_e = (4/3 - 4/3 * v + v^2) * L_rad_prime / Z * 0.1
    
    return prefactor * (F_n + F_e)
end

"""
    dcs_pair_production_ssr(Z, A, m, K, q)

e⁺e⁻ pair production DCS using the SSR model.
"""
function dcs_pair_production_ssr(Z::Real, A::Real, m::Real, K::T, q::T) where T<:Real
    if q <= 4 * ELECTRON_MASS || q >= K
        return zero(T)
    end
    
    v = q / (K + m)
    E = K + m
    
    # Simplified pair production formula
    α = ALPHA_EM
    r_e = ELECTRON_RADIUS
    me = ELECTRON_MASS
    
    prefactor = 4 * α * r_e^2 * Z^2 / q
    
    # Approximate differential cross section
    L = log(183 * Z^(-1/3))
    F = (1 - v + 3/4 * v^2) * L
    
    return prefactor * F / 3
end

"""
    dcs_photonuclear_drss(Z, A, m, K, q)

Photonuclear DCS using the DRSS model (Dutta, Reno, Sarcevic, Seckel 2001).
"""
function dcs_photonuclear_drss(Z::Real, A::Real, m::Real, K::T, q::T) where T<:Real
    if q <= zero(T) || q >= K
        return zero(T)
    end
    
    # Photonuclear interaction threshold
    ν_min = PION_MASS + PION_MASS^2 / (2 * PROTON_MASS)
    if q < ν_min
        return zero(T)
    end
    
    E = K + m
    v = q / E
    
    # Photon-nucleon cross-section (simplified)
    σ_γN = 114.3 * 1e-34  # m² (approximate at high energy)
    
    # Nuclear shadowing
    A_eff = A^0.91  # shadowing exponent
    
    # Prefactor
    α = ALPHA_EM
    prefactor = α * σ_γN * A_eff / (2π * q)
    
    # Virtual photon spectrum (Weizsäcker-Williams)
    F_virtual = (1 - v + v^2 / 2) * log(1 + (E / m)^2 * (1 - v) / v^2)
    
    return prefactor * F_virtual * 1e-3  # Scale factor
end

end # module Materials

