"""
    Materials

Material properties and Differential Cross-Section (DCS) calculations.
Exact port of PUMAS C DCS implementations (SSR bremsstrahlung, SSR pair production,
DRSS photonuclear with ALLM97 structure function, Salvat ionisation).
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
export gauss_quad_coefficients, compute_dcs_integral, dcs_ionisation_integrate
export AtomicShell, atomic_shell_create, atomic_shell_normalise
export radiation_logarithm, math_dilog

struct AtomicElement
    name::String
    Z::Float64
    A::Float64
    I::Float64
end

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

struct BaseMaterial
    name::String
    density::Float64
    I::Float64
    elements::Vector{AtomicElement}
    fractions::Vector{Float64}
    ZoA::Float64

    function BaseMaterial(name, density, I, elements, fractions)
        total = sum(fractions)
        norm_fractions = fractions ./ total
        ZoA = sum(e.Z / e.A * f for (e, f) in zip(elements, norm_fractions))
        new(name, density, I, elements, norm_fractions, ZoA)
    end
end

struct CompositeMaterial
    name::String
    components::Vector{BaseMaterial}
    fractions::Vector{Float64}

    function CompositeMaterial(name, components, fractions)
        total = sum(fractions)
        new(name, components, fractions ./ total)
    end
end

const STANDARD_ROCK = BaseMaterial(
    "StandardRock", 2650.0, 136.4e-9,
    [ELEMENTS["O"], ELEMENTS["Si"], ELEMENTS["Al"], ELEMENTS["Fe"],
     ELEMENTS["Ca"], ELEMENTS["Na"], ELEMENTS["Mg"], ELEMENTS["K"]],
    [0.466, 0.277, 0.081, 0.050, 0.036, 0.028, 0.021, 0.026]
)

const AIR = BaseMaterial(
    "Air", 1.205, 85.7e-9,
    [ELEMENTS["N"], ELEMENTS["O"], ELEMENTS["Ar"]],
    [0.755, 0.232, 0.013]
)

const WATER = BaseMaterial(
    "Water", 1000.0, 75.0e-9,
    [ELEMENTS["H"], ELEMENTS["O"]],
    [0.1119, 0.8881]
)

const MATERIALS = Dict{String, BaseMaterial}(
    "StandardRock" => STANDARD_ROCK,
    "Air" => AIR,
    "Water" => WATER
)

# ============================================================================
# Gauss-Legendre quadrature (exact PUMAS coefficients up to N=12)
# ============================================================================

const _GQ_X = (
    # N=1
    (0.5000000000000000,),
    # N=2
    (0.2113248654051871, 0.7886751345948129),
    # N=3
    (0.1127016653792583, 0.5000000000000000, 0.8872983346207417),
    # N=4
    (0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970262),
    # N=5
    (0.0469100770306680, 0.2307653449471584, 0.5000000000000000,
     0.7692346550528415, 0.9530899229693319),
    # N=6
    (0.0337652428984240, 0.1693953067668678, 0.3806904069584016,
     0.6193095930415985, 0.8306046932331322, 0.9662347571015760),
    # N=7
    (0.0254460438286208, 0.1292344072003028, 0.2970774243113014,
     0.5000000000000000, 0.7029225756886985, 0.8707655927996972,
     0.9745539561713792),
    # N=8
    (0.0198550717512319, 0.1016667612931866, 0.2372337950418355,
     0.4082826787521751, 0.5917173212478249, 0.7627662049581645,
     0.8983332387068134, 0.9801449282487682),
    # N=9
    (0.0159198802461870, 0.0819844463366821, 0.1933142836497048,
     0.3378732882980955, 0.5000000000000000, 0.6621267117019045,
     0.8066857163502952, 0.9180155536633179, 0.9840801197538130),
    # N=10
    (0.0130467357414141, 0.0674683166555077, 0.1602952158504878,
     0.2833023029353764, 0.4255628305091844, 0.5744371694908156,
     0.7166976970646236, 0.8397047841495122, 0.9325316833444923,
     0.9869532642585859),
    # N=11
    (0.0108856709269715, 0.0564687001159523, 0.1349239972129753,
     0.2404519353965941, 0.3652284220238275, 0.5000000000000000,
     0.6347715779761725, 0.7595480646034058, 0.8650760027870247,
     0.9435312998840477, 0.9891143290730284),
    # N=12
    (0.0092196828766404, 0.0479413718147625, 0.1150486629028477,
     0.2063410228566913, 0.3160842505009099, 0.4373832957442655,
     0.5626167042557344, 0.6839157494990901, 0.7936589771433087,
     0.8849513370971523, 0.9520586281852375, 0.9907803171233596),
)

const _GQ_W = (
    # N=1
    (1.0000000000000000,),
    # N=2
    (0.5000000000000000, 0.5000000000000000),
    # N=3
    (0.2777777777777778, 0.4444444444444444, 0.2777777777777778),
    # N=4
    (0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269),
    # N=5
    (0.1184634425280946, 0.2393143352496833, 0.2844444444444444,
     0.2393143352496833, 0.1184634425280946),
    # N=6
    (0.0856622461895852, 0.1803807865240693, 0.2339569672863455,
     0.2339569672863455, 0.1803807865240693, 0.0856622461895852),
    # N=7
    (0.0647424830844349, 0.1398526957446383, 0.1909150252525595,
     0.2089795918367347, 0.1909150252525595, 0.1398526957446383,
     0.0647424830844349),
    # N=8
    (0.0506142681451881, 0.1111905172266872, 0.1568533229389436,
     0.1813418916891810, 0.1813418916891810, 0.1568533229389436,
     0.1111905172266872, 0.0506142681451881),
    # N=9
    (0.0406371941807872, 0.0903240803474287, 0.1303053482014677,
     0.1561735385200015, 0.1651196775006299, 0.1561735385200015,
     0.1303053482014677, 0.0903240803474287, 0.0406371941807872),
    # N=10
    (0.0333356721543440, 0.0747256745752903, 0.1095431812579910,
     0.1346333596549981, 0.1477621123573765, 0.1477621123573765,
     0.1346333596549981, 0.1095431812579910, 0.0747256745752903,
     0.0333356721543440),
    # N=11
    (0.0278342835580868, 0.0627901847324523, 0.0931451054638671,
     0.1165968822959952, 0.1314022722551233, 0.1364625433889503,
     0.1314022722551233, 0.1165968822959952, 0.0931451054638671,
     0.0627901847324523, 0.0278342835580868),
    # N=12
    (0.0235876681932559, 0.0534696629976592, 0.0800391642716731,
     0.1015837133615330, 0.1167462682691774, 0.1245735229067014,
     0.1245735229067014, 0.1167462682691774, 0.1015837133615330,
     0.0800391642716731, 0.0534696629976592, 0.0235876681932559),
)

@inline function gauss_quad_coefficients(::Val{N}) where N
    return _GQ_X[N], _GQ_W[N]
end

@inline function gauss_quad_coefficients(n::Int)
    return _GQ_X[n], _GQ_W[n]
end

"""
Prepare Gauss quadrature nodes/weights for log-scale integration on [xmin, xmax].
Writes into pre-allocated buffers to avoid allocation in hot loops.
"""
function gauss_quad_logscale!(nodes::Vector{Float64}, weights::Vector{Float64},
                               n::Int, xmin::Float64, xmax::Float64)
    xref, wref = gauss_quad_coefficients(n)
    lmin = log(xmin)
    lmax = log(xmax)
    dx = lmax - lmin
    @inbounds for i in 1:n
        xi = exp(lmin + dx * xref[i])
        nodes[i] = xi
        weights[i] = wref[i] * dx * xi
    end
end

function gauss_quad_logscale(n::Int, xmin::Float64, xmax::Float64)
    nodes = Vector{Float64}(undef, n)
    weights = Vector{Float64}(undef, n)
    gauss_quad_logscale!(nodes, weights, n, xmin, xmax)
    return nodes, weights
end

function gauss_quad_linear(n::Int, xmin::Float64, xmax::Float64)
    xref, wref = gauss_quad_coefficients(n)
    dx = xmax - xmin
    nodes = Vector{Float64}(undef, n)
    weights = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        nodes[i] = xmin + dx * xref[i]
        weights[i] = wref[i] * dx
    end
    return nodes, weights
end

# ============================================================================
# Spence's dilogarithm (exact PUMAS implementation via Chebyshev expansion)
# ============================================================================

const _DILOG_C = (
     0.42996693560813697,  0.40975987533077105, -0.01858843665014592,
     0.00145751084062268, -0.00014304184442340,  0.00001588415541880,
    -0.00000190784959387,  0.00000024195180854, -0.00000003193341274,
     0.00000000434545063, -0.00000000060578480,  0.00000000008612098,
    -0.00000000001244332,  0.00000000000182256, -0.00000000000027007,
     0.00000000000004042, -0.00000000000000610,  0.00000000000000093,
    -0.00000000000000014,  0.00000000000000002)

@inline function math_dilog(x::Float64)
    PI6 = π^2 / 6.0
    PI12 = π^2 / 12.0
    PI3 = π^2 / 3.0

    if x == 1.0
        return PI6
    elseif x == -1.0 || (1.0 - x) == 0.0
        return -PI12
    end

    T = -x
    local Y::Float64, S::Float64, A_val::Float64

    if T <= -2.0
        Y = -1.0 / (1.0 + T)
        S = 1.0
        A_val = -PI3 + 0.5 * (log(-T)^2 - log(1.0 + 1.0/T)^2)
    elseif T < -1.0
        Y = -1.0 - T
        S = -1.0
        la = log(-T)
        A_val = -PI6 + la * (la + log(1.0 + 1.0/T))
    elseif T <= -0.5
        Y = -(1.0 + T) / T
        S = 1.0
        la = log(-T)
        A_val = -PI6 + la * (-0.5*la + log(1.0 + T))
    elseif T < 0.0
        Y = -T / (1.0 + T)
        S = -1.0
        A_val = 0.5 * log(1.0 + T)^2
    elseif T <= 1.0
        Y = T
        S = 1.0
        A_val = 0.0
    else
        Y = 1.0 / T
        S = -1.0
        A_val = PI6 + 0.5 * log(T)^2
    end

    H = Y + Y - 1.0
    ALFA = H + H
    B1 = 0.0
    B2 = 0.0
    @inbounds for i in 20:-1:1
        B0 = _DILOG_C[i] + ALFA * B1 - B2
        B2 = B1
        B1 = B0
    end

    return -(S * (B1 - H * B2) + A_val)
end

# ============================================================================
# Atomic shell data (from Geant4 10.7, ported from PUMAS C)
# ============================================================================

const _ATOMIC_SHELL_INDEX = UInt16[
       0,    1,    2,    4,    6,    9,   12,   16,   20,   23,   27,   32,
      37,   43,   49,   55,   61,   67,   74,   82,   90,   99,  108,  117,
     126,  135,  144,  153,  163,  173,  183,  194,  205,  216,  227,  238,
     250,  263,  276,  290,  304,  318,  332,  346,  360,  374,  389,  404,
     419,  435,  451,  467,  483,  499,  516,  534,  552,  571,  590,  609,
     628,  647,  666,  685,  705,  724,  743,  762,  781,  800,  820,  841,
     862,  883,  904,  925,  946,  967,  988, 1010, 1032, 1055, 1078, 1101,
    1124, 1148, 1172, 1197, 1222, 1248, 1274, 1301, 1328, 1355, 1381, 1407,
    1434, 1461, 1487, 1513, 1539
]

include("AtomicShellData.jl")

struct AtomicShell
    f::Float64   # oscillator strength
    E::Float64   # binding energy (eV, then scaled)
end

function _atomic_shell_getn1(Z::Int)
    iZ = clamp(Z, 1, 100)
    i0 = Int(_ATOMIC_SHELL_INDEX[iZ]) + 1   # 1-based
    i1 = Int(_ATOMIC_SHELL_INDEX[iZ + 1])    # 1-based exclusive end
    return i1 - i0 + 1, i0
end

function _atomic_shell_getn(Z::Float64, A::Float64)
    if Z == 11.0 && A == 22.0
        n1, _ = _atomic_shell_getn1(6)
        n2, _ = _atomic_shell_getn1(8)
        n3, _ = _atomic_shell_getn1(20)
        return n1 + n2 + n3
    else
        n, _ = _atomic_shell_getn1(round(Int, Z))
        return n
    end
end

function _atomic_shell_copyweight1!(shells::Vector{AtomicShell}, offset::Int, Z::Int, w::Float64)
    n, i0 = _atomic_shell_getn1(Z)
    @inbounds for i in 1:n
        is = i0 + i - 1
        shells[offset + i] = AtomicShell(w * Float64(_ATOMIC_SHELL_OCCUPANCY[is]),
                                          Float64(_ATOMIC_SHELL_ENERGY[is]))
    end
    return n
end

function _atomic_shell_copyweight!(shells::Vector{AtomicShell}, offset::Int,
                                    Z::Float64, A::Float64, w::Float64)
    if Z == 11.0 && A == 22.0
        w_scaled = w * 11.0 / (6.0 + 3*8.0 + 20.0)
        n = _atomic_shell_copyweight1!(shells, offset, 6, w_scaled)
        n += _atomic_shell_copyweight1!(shells, offset + n, 8, 3*w_scaled)
        n += _atomic_shell_copyweight1!(shells, offset + n, 20, w_scaled)
        return n
    else
        return _atomic_shell_copyweight1!(shells, offset, round(Int, Z), w)
    end
end

function atomic_shell_normalise!(shells::Vector{AtomicShell}, ZoA::Float64,
                                  I::Float64, density::Float64)
    ftot = 0.0
    lnI = 0.0
    @inbounds for s in shells
        ftot += s.f
        lnI += s.f * log(s.E)
    end
    inv_ftot = 1.0 / ftot
    lnI *= inv_ftot

    wp = 28.816 * sqrt(ZoA * density * 1e-3)
    aS = I * 1e9 / exp(lnI)
    r = aS / wp

    @inbounds for i in eachindex(shells)
        shells[i] = AtomicShell(shells[i].f * inv_ftot, shells[i].E * r)
    end
    return aS
end

"""
Build atomic shells for a material, matching PUMAS `atomic_shell_create`.
Returns (shells, aS).
"""
function atomic_shell_create(material::BaseMaterial)
    n_shells = 0
    for (e, _) in zip(material.elements, material.fractions)
        n_shells += _atomic_shell_getn(e.Z, e.A)
    end

    shells = Vector{AtomicShell}(undef, n_shells)
    is = 0
    for (e, f) in zip(material.elements, material.fractions)
        wi = f / e.A
        is += _atomic_shell_copyweight!(shells, is, e.Z, e.A, wi)
    end

    aS = atomic_shell_normalise!(shells, material.ZoA, material.I, material.density)
    return shells, aS
end

# ============================================================================
# Fano density effect (exact PUMAS bisection algorithm)
# ============================================================================

function electronic_density_effect(shells::Vector{AtomicShell}, gamma::Float64)
    y = 1.0 / (gamma * gamma)
    ymax = 0.0
    @inbounds for s in shells
        ymax += s.f / (s.E * s.E)
    end
    if ymax <= y
        return 0.0
    end

    l2min = 0.0
    l2max = 1.0 / y
    l2 = 0.0
    for _ in 1:200
        l2 = 0.5 * (l2min + l2max)
        yi = 0.0
        @inbounds for s in shells
            yi += s.f / (s.E * s.E + l2)
        end
        if abs(yi - y) <= eps(Float64) || (l2max - l2min) <= eps(Float64)
            break
        elseif yi > y
            l2min = l2
        else
            l2max = l2
        end
    end

    delta = -l2 * y
    @inbounds for s in shells
        delta += s.f * log(1.0 + l2 / (s.E * s.E))
    end
    return delta
end

function electronic_density_effect(material::BaseMaterial, gamma::T) where T<:Real
    shells = _get_shells(material)
    return T(electronic_density_effect(shells, Float64(gamma)))
end

# ============================================================================
# Electronic energy loss (modified Bethe-Bloch, exact PUMAS)
# ============================================================================

const _SHELL_CACHE = Dict{String, Vector{AtomicShell}}()

function _get_shells(material::BaseMaterial)
    get!(_SHELL_CACHE, material.name) do
        shells, _ = atomic_shell_create(material)
        shells
    end
end

@inline function electronic_stopping_power(material::BaseMaterial, mass::T, energy::T) where T<:Real
    shells = _get_shells(material)
    return T(_electronic_energy_loss(material.ZoA, material.I, shells, Float64(mass), Float64(energy)))
end

@fastmath function _electronic_energy_loss(ZoA::Float64, I::Float64, shells::Vector{AtomicShell},
                                  mass::Float64, kinetic::Float64)
    E = kinetic + mass
    P2 = kinetic * (kinetic + 2.0 * mass)
    beta2 = P2 / (E * E)
    gamma = E / mass

    r = ELECTRON_MASS / mass
    Qmax = 2.0 * r * P2 / (mass * (1.0 + r*r) + 2.0 * r * E)
    lQ = log(1.0 + 2.0 * Qmax / ELECTRON_MASS)
    Delta = ALPHA_EM / (2π) * (log(2.0 * gamma) - lQ / 3.0) * lQ * lQ

    delta = electronic_density_effect(shells, gamma)

    return 2π * ELECTRON_RADIUS^2 * ELECTRON_MASS *
        AVOGADRO_NUMBER * ZoA / (beta2 * 1e-3) * (
        log(2.0 * ELECTRON_MASS * beta2 * gamma * gamma * Qmax / (I * I)) -
        2.0 * beta2 - delta + 0.25 * Qmax * Qmax / (E * E) + Delta)
end

# ============================================================================
# Electronic DCS (exact Salvat NIMB316 (2013), matching PUMAS pumas_electronic_dcs)
# ============================================================================

function electronic_dcs(Z::Real, I::Real, m::Real, K::T, q::T) where T<:Real
    P2 = K * (K + 2.0 * m)
    E = K + m
    Wmax = 2.0 * ELECTRON_MASS * P2 / (m*m + ELECTRON_MASS * (ELECTRON_MASS + 2.0 * E))
    q > Wmax && return zero(T)
    Wmin = (13.6 / 19.2) * I
    q <= Wmin && return zero(T)

    E2 = E * E
    beta2 = P2 / E2
    a0 = 0.5 / E2
    a1 = beta2 / Wmax
    cs = 2π * ELECTRON_MASS * ELECTRON_RADIUS^2 * Z * (a0 + 1.0/q * (1.0/q - a1)) / beta2
    return cs
end

"""
Analytical integration of ionisation DCS (exact PUMAS `dcs_ionisation_integrate`).
mode=0: cross-section, mode=1: energy loss, mode=2: straggling.
Returns result in m²/kg (mode 0), GeV⋅m²/kg (mode 1), or GeV²⋅m²/kg (mode 2).
"""
function dcs_ionisation_integrate(Z::Float64, A::Float64, I::Float64,
                                   mass::Float64, K::Float64,
                                   xlow::Float64, xhigh::Float64, mode::Int)
    P2 = K * (K + 2.0 * mass)
    E = K + mass
    Wr = 2.0 * ELECTRON_MASS * P2 / (mass*mass + ELECTRON_MASS * (ELECTRON_MASS + 2.0 * E))
    qlow_abs = K * xlow
    qhigh_abs = K * xhigh
    Wmax = Wr
    Wmax < qlow_abs && return 0.0
    Wmax > qhigh_abs && (Wmax = qhigh_abs)
    Wmin = (13.6 / 19.2) * I
    qlow_abs >= Wmin && (Wmin = qlow_abs)

    Wmax <= Wmin && return 0.0

    E2 = E * E
    beta2 = P2 / E2
    a0 = 0.5 / E2
    a1 = beta2 / Wr

    local Ival::Float64
    if mode == 0
        Ival = a0 * (Wmax - Wmin) - a1 * log(Wmax / Wmin) + 1.0/Wmin - 1.0/Wmax
    elseif mode == 1
        Ival = 0.5 * a0 * (Wmax^2 - Wmin^2) - a1 * (Wmax - Wmin) + log(Wmax / Wmin)
    else
        Ival = a0 * (Wmax^3 - Wmin^3) / 3.0 - 0.5 * a1 * (Wmax^2 - Wmin^2) + Wmax - Wmin
    end

    return 2π * ELECTRON_MASS * ELECTRON_RADIUS^2 * Z * AVOGADRO_NUMBER / (A * 1e-3 * beta2) * Ival
end

# ============================================================================
# Radiation logarithm (exact PUMAS table for Z=1..92)
# ============================================================================

const _RADIATION_LOG = (
    202.4, 151.9, 159.9, 172.3, 177.9,
    178.3, 176.6, 173.4, 170.0, 165.8,
    165.8, 167.1, 169.1, 170.8, 172.2,
    173.4, 174.3, 174.8, 175.1, 175.6,
    176.2, 176.8,   0.0,   0.0,   0.0,
    175.8,   0.0,   0.0, 173.1,   0.0,
      0.0, 173.0,   0.0,   0.0, 173.5,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0, 175.9,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0, 177.4,
      0.0,   0.0, 178.6,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0, 177.6,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0, 178.0,   0.0,   0.0,   0.0,
      0.0,   0.0,   0.0,   0.0,   0.0,
      0.0, 179.8)

@inline function radiation_logarithm(Z::Float64)
    i = round(Int, Z)
    if 1 <= i <= 92
        l = _RADIATION_LOG[i]
        l > 0.0 && return l
    end
    return 182.7
end

# ============================================================================
# Bremsstrahlung SSR (exact PUMAS dcs_bremsstrahlung_SSR)
# ============================================================================

const _SQRTE = 1.648721270700128
const _ME_MEV = 0.5109989461       # MeV
const _RE_CM = 2.8179403227e-13    # cm
const _MMU_MEV = 105.6583745       # MeV

@fastmath function dcs_bremsstrahlung_ssr(Z::Real, A::Real, m::Real, K::T, q::T) where T<:Real
    (Z <= 0 || A <= 0 || m <= 0 || K <= 0 || q <= 0) && return zero(T)

    z13 = Z^(1.0/3.0)
    q >= K + m * (1.0 - 0.75 * _SQRTE * z13) && return zero(T)

    a_coeff = (-0.00349, 148.84, -987.531)
    b_coeff = (0.1642, 132.573, -585.361, 1407.77)
    c_coeff = (-2.8922, -19.0156, 57.698, -63.418, 14.1166, 1.84206)
    d_coeff = (2134.19, 581.823, -2708.85, 4767.05, 1.52918, 0.361933)

    energy = (K + m) * 1e3
    v = q * 1e3 / energy
    m_mev = m * 1e3

    Z13 = 1.0 / z13
    rad_log = radiation_logarithm(Float64(Z))
    rad_log_inel = Z == 1.0 ? 446.0 : 1429.0
    Dn = 1.54 * A^0.27

    mu_qc = m_mev / (_MMU_MEV * exp(1.0) / Dn)
    rho = sqrt(1.0 + 4.0 * mu_qc^2)

    log_rho = log((rho + 1.0) / (rho - 1.0))
    delta1 = log(mu_qc) + 0.5 * rho * log_rho
    delta2 = log(mu_qc) + 0.25 * (3.0*rho - rho^3) * log_rho + 2.0 * mu_qc^2

    delta = m_mev^2 * v / (2.0 * energy * (1.0 - v))

    phi1 = log(rad_log * Z13 * (m_mev / _ME_MEV) /
        (1.0 + rad_log * Z13 * exp(0.5) * delta / _ME_MEV))
    phi2 = log(rad_log * Z13 * exp(-1.0/6.0) * (m_mev / _ME_MEV) /
        (1.0 + rad_log * Z13 * exp(1.0/3.0) * delta / _ME_MEV))
    phi1 -= delta1 * (1.0 - 1.0/Z)
    phi2 -= delta2 * (1.0 - 1.0/Z)

    s_atomic_1 = log(m_mev / delta / (m_mev * delta / (_ME_MEV^2) + _SQRTE))
    s_atomic_2 = log(1.0 + _ME_MEV / (delta * rad_log_inel * Z13^2 * _SQRTE))
    s_atomic = (4.0/3.0*(1.0 - v) + v*v) * (s_atomic_1 - s_atomic_2)

    local s_rad::T
    if v < 0.0 || v > 1.0
        s_rad = zero(T)
    elseif v < 0.02
        s_rad = a_coeff[1] + a_coeff[2]*v + a_coeff[3]*v*v
    elseif v < 0.1
        s_rad = b_coeff[1] + b_coeff[2]*v + b_coeff[3]*v*v + b_coeff[4]*v^3
    elseif v < 0.9
        s_rad = c_coeff[1] + c_coeff[2]*v + c_coeff[3]*v*v
        tmp = log(1.0 - v)
        s_rad += c_coeff[4]*v*log(v) + c_coeff[5]*tmp + c_coeff[6]*tmp*tmp
    else
        s_rad = d_coeff[1] + d_coeff[2]*v + d_coeff[3]*v*v
        tmp = log(1.0 - v)
        s_rad += d_coeff[4]*v*log(v) + d_coeff[5]*tmp + d_coeff[6]*tmp*tmp
    end

    result = ((2.0 - 2.0*v + v*v) * phi1 - 2.0/3.0*(1.0 - v)*phi2) +
             1.0/Z * s_atomic + 0.25 * ALPHA_EM * phi1 * s_rad

    result <= 0.0 && return zero(T)

    aux = 2.0 * (_ME_MEV / m_mev) * _RE_CM * Z
    return T(aux * aux * (ALPHA_EM / q) * result * 1e-4)
end

# ============================================================================
# Pair production SSR (exact PUMAS, with full DDCS + 12-point Gauss quad)
# ============================================================================

@fastmath function _dcs_pair_production_d2_ssr(Z::Float64, A::Float64, mass::Float64,
                                      K::Float64, q::Float64, rho_in::Float64)
    energy = (K + mass) * 1e3
    v = q / (K + mass)
    m = mass * 1e3
    rad_log = radiation_logarithm(Z)

    const_pf = 4.0 / (3.0 * π) * Z * (ALPHA_EM * _RE_CM)^2
    Z13 = Z^(-1.0/3.0)
    d_n = 1.54 * A^0.27

    rho = 1.0 - rho_in
    rho2 = rho * rho

    g1, g2 = Z == 1.0 ? (4.4e-5, 4.8e-5) : (1.95e-5, 5.3e-5)

    zeta1 = 0.073 * log(energy / m / (1.0 + g1 * Z^(2.0/3.0) * energy / m)) - 0.26
    zeta2 = 0.058 * log(energy / m / (1.0 + g2 / Z13 * energy / m)) - 0.14
    zeta = (zeta1 > 0.0 && zeta2 > 0.0) ? zeta1 / zeta2 : 0.0

    beta = v*v / (2.0 * (1.0 - v))
    xi = (m * v / (2.0 * _ME_MEV))^2 * (1.0 - rho2) / (1.0 - v)

    Be = ((2.0 + rho2) * (1.0 + beta) + xi * (3.0 + rho2)) * log(1.0 + 1.0/xi) +
         (1.0 - rho2 - beta) / (1.0 + xi) - (3.0 + rho2)

    Ce2 = ((1.0 - rho2) * (1.0 + beta) + xi * (3.0 - rho2)) * log(1.0 + 1.0/xi) +
          2.0*(1.0 - beta - rho2)/(1.0 + xi) - (3.0 - rho2)
    Ce1 = Be - Ce2

    De = ((2.0 + rho2)*(1.0 + beta) + xi*(3.0 + rho2)) * math_dilog(1.0/(1.0 + xi)) -
         (2.0 + rho2)*xi*log(1.0 + 1.0/xi) -
         (xi + rho2 + beta)/(1.0 + xi)

    local Le1::Float64, Le2::Float64
    if De / Be > 0.0
        tmp = De / Be
        Xe = tmp < 100.0 ? exp(-tmp) : 0.0
        Le1 = log(rad_log * Z13 * sqrt(1.0 + xi) /
            (Xe + 2.0*_ME_MEV*exp(0.5)*rad_log*Z13*(1.0 + xi) /
            (energy*v*(1.0 - rho2)))) - tmp -
            0.5*log(Xe + (_ME_MEV/m*d_n)^2 * (1.0 + xi))
        Le2 = log(rad_log * Z13 * exp(-1.0/6.0) * sqrt(1.0 + xi) /
            (Xe + 2.0*_ME_MEV*exp(1.0/3.0)*rad_log*Z13*(1.0 + xi) /
            (energy*v*(1.0 - rho2)))) - tmp -
            0.5*log(Xe + (_ME_MEV/m*d_n)^2 * exp(-1.0/3.0) * (1.0 + xi))
    else
        tmp = De / Be
        Xe_inv = tmp > -100.0 ? exp(tmp) : 0.0
        Le1 = log(rad_log * Z13 * sqrt(1.0 + xi) /
            (1.0 + Xe_inv*2.0*_ME_MEV*exp(0.5)*rad_log*Z13*(1.0 + xi) /
            (energy*v*(1.0 - rho2)))) - 0.5*tmp -
            0.5*log(1.0 + Xe_inv*(_ME_MEV/m*d_n)^2 * (1.0 + xi))
        Le2 = log(rad_log * Z13 * exp(-1.0/6.0) * sqrt(1.0 + xi) /
            (1.0 + Xe_inv*2.0*_ME_MEV*exp(1.0/3.0)*rad_log*Z13*(1.0 + xi) /
            (energy*v*(1.0 - rho2)))) - 0.5*tmp -
            0.5*log(1.0 + Xe_inv*(_ME_MEV/m*d_n)^2 * exp(-1.0/3.0) * (1.0 + xi))
    end

    diagram_e = const_pf * (Z + zeta) * (1.0 - v) / v * (Ce1*Le1 + Ce2*Le2)
    diagram_e < 0.0 && (diagram_e = 0.0)

    Bm = ((1.0 + rho2)*(1.0 + 1.5*beta) - 1.0/xi*(1.0 + 2.0*beta)*(1.0 - rho2)) * log(1.0 + xi) +
         xi*(1.0 - rho2 - beta)/(1.0 + xi) + (1.0 + 2.0*beta)*(1.0 - rho2)

    Cm2 = ((1.0 - beta)*(1.0 - rho2) - xi*(1.0 + rho2)) * log(1.0 + xi)/xi -
          2.0*(1.0 - beta - rho2)/(1.0 + xi) + 1.0 - beta - (1.0 + beta)*rho2
    Cm1 = Bm - Cm2

    Dm = ((1.0 + rho2)*(1.0 + 1.5*beta) - 1.0/xi*(1.0 + 2.0*beta)*(1.0 - rho2)) *
         math_dilog(xi/(1.0 + xi)) +
         (1.0 + 1.5*beta)*(1.0 - rho2)/xi*log(1.0 + xi) +
         (1.0 - rho2 - 0.5*beta*(1.0 + rho2) + (1.0 - rho2)/(2.0*xi)*beta) * xi/(1.0 + xi)

    local Lm1::Float64, Lm2::Float64
    if Dm / Bm > 0.0
        tmp = Dm / Bm
        Xm = tmp < 100.0 ? exp(-tmp) : 0.0
        Lm1 = log(m/_ME_MEV * rad_log*Z13/d_n /
            (Xm + 2.0*_ME_MEV*exp(0.5)*rad_log*Z13*(1.0+xi) /
            (energy*v*(1.0 - rho2)))) - tmp
        Lm2 = log(m/_ME_MEV * rad_log*Z13/d_n /
            (Xm + 2.0*_ME_MEV*exp(1.0/3.0)*rad_log*Z13*(1.0+xi) /
            (energy*v*(1.0 - rho2)))) - tmp
    else
        tmp = Dm / Bm
        Xmv = tmp > -100.0 ? exp(tmp) : 0.0
        Lm1 = log(m/_ME_MEV * rad_log*Z13/d_n /
            (1.0 + 2.0*_ME_MEV*exp(0.5)*rad_log*Z13*(1.0+xi) /
            (energy*v*(1.0 - rho2)) * Xmv))
        Lm2 = log(m/_ME_MEV * rad_log*Z13/d_n /
            (1.0 + 2.0*_ME_MEV*exp(1.0/3.0)*rad_log*Z13*(1.0+xi) /
            (energy*v*(1.0 - rho2)) * Xmv))
    end

    diagram_mu = const_pf * (Z + zeta) * (1.0 - v) / v * (_ME_MEV/m)^2 * (Cm1*Lm1 + Cm2*Lm2)
    diagram_mu < 0.0 && (diagram_mu = 0.0)

    return (diagram_e + diagram_mu) * 0.1 / energy
end

function dcs_pair_production_ssr(Z::Real, A::Real, mass::Real, K::T, q::T) where T<:Real
    (Z <= 0 || A <= 0 || mass <= 0 || K <= 0 || q <= 0) && return zero(T)
    q <= 4.0 * ELECTRON_MASS && return zero(T)
    sqrte = 1.6487212707
    Z13 = Z^(1.0/3.0)
    q >= K + mass * (1.0 - 0.75 * sqrte * Z13) && return zero(T)

    gamma = 1.0 + K / mass
    x0 = 1.0 - 4.0 * ELECTRON_MASS / q
    x1 = 1.0 - 6.0 / (gamma * (gamma - q / mass))

    if x0 > 0.0 && x1 > 0.0
        rmax = sqrt(x0) * x1
    else
        return zero(T)
    end

    ri = 1.0 - rmax
    (ri <= 0.0 || ri >= 1.0) && return zero(T)
    tmin = log(ri)

    xGQ, wGQ = gauss_quad_coefficients(12)
    Ival = 0.0
    @inbounds for i in 1:12
        rho_val = exp(xGQ[i] * tmin)
        Ival -= _dcs_pair_production_d2_ssr(Float64(Z), Float64(A), Float64(mass),
                    Float64(K), Float64(q), rho_val) * rho_val * wGQ[i] * tmin
    end

    return T(Ival < 0.0 ? 0.0 : Ival)
end

# ============================================================================
# Photonuclear DRSS (exact PUMAS with ALLM97 + DRSS shadowing + 9pt GQ)
# ============================================================================

@fastmath function _dcs_photonuclear_f2p_ALLM97(x::Float64, Q2::Float64)
    m02 = 0.31985; mP2 = 49.457; mR2 = 0.15052; Q02 = 0.52544; Lambda2 = 0.06527
    cP1 = 0.28067; cP2 = 0.22291; cP3 = 2.1979
    aP1 = -0.0808; aP2 = -0.44812; aP3 = 1.1709
    bP1 = 0.36292; bP2 = 1.8917; bP3 = 1.8439
    cR1 = 0.80107; cR2 = 0.97307; cR3 = 3.4942
    aR1 = 0.58400; aR2 = 0.37888; aR3 = 2.6063
    bR1 = 0.01147; bR2 = 3.7582; bR3 = 0.49338

    M = 0.5 * (PROTON_MASS + NEUTRON_MASS)
    M2 = M * M
    W2 = M2 + Q2 * (1.0/x - 1.0)
    t = log(log((Q2 + Q02)/Lambda2) / log(Q02/Lambda2))
    xP = (Q2 + mP2) / (Q2 + mP2 + W2 - M2)
    xR = (Q2 + mR2) / (Q2 + mR2 + W2 - M2)
    cP = cP1 + (cP1 - cP2) * (1.0/(1.0 + t^cP3) - 1.0)
    aP = aP1 + (aP1 - aP2) * (1.0/(1.0 + t^aP3) - 1.0)
    bP = bP1 + bP2 * t^bP3
    cR = cR1 + cR2 * t^cR3
    aR = aR1 + aR2 * t^aR3
    bR = bR1 + bR2 * t^bR3

    F2P = cP * exp(aP * log(xP) + bP * log(1.0 - x))
    F2R = cR * exp(aR * log(xR) + bR * log(1.0 - x))
    return Q2 / (Q2 + m02) * (F2P + F2R)
end

function _dcs_photonuclear_shadowing_DRSS(Z::Float64, A::Float64, x::Float64)
    Z == 1.0 && return 1.0
    if x < 0.0014
        return exp(-0.1 * log(A))
    elseif x < 0.04
        return exp((0.069 * log10(x) + 0.097) * log(A))
    else
        return 1.0
    end
end

function _dcs_photonuclear_f2a_DRSS(Z::Float64, A::Float64, F2p::Float64,
                                      shadowing::Float64, x::Float64)
    return F2p * shadowing * (Z + (A - Z) * (1.0 + x*(-1.85 + x*(2.45 + x*(-2.35 + x)))))
end

@fastmath function _dcs_photonuclear_d2_DRSS(Z::Float64, A::Float64, ml::Float64,
                                     K::Float64, q::Float64, Q2::Float64)
    cf = 4π * ALPHA_EM^2 * HBAR_C^2
    M = 0.5 * (PROTON_MASS + NEUTRON_MASS)
    E = K + ml
    y = q / E
    x = 0.5 * Q2 / (M * q)
    F2p = _dcs_photonuclear_f2p_ALLM97(x, Q2)
    shadowing = _dcs_photonuclear_shadowing_DRSS(Z, A, x)
    F2A = _dcs_photonuclear_f2a_DRSS(Z, A, F2p, shadowing, x)

    dds = (1.0 - y + 0.5*(1.0 - 2.0*ml*ml/Q2) * (y*y + Q2/(E*E))) / (Q2*Q2) -
          0.25 / (E*E*Q2)
    return cf * F2A * dds / q
end

const _PHOTO_NODES = Vector{Float64}(undef, 9)
const _PHOTO_WEIGHTS = Vector{Float64}(undef, 9)

function dcs_photonuclear_drss(Z::Real, A::Real, m::Real, K::T, q::T) where T<:Real
    (Z <= 0 || A <= 0 || m <= 0 || K <= 0 || q <= 0) && return zero(T)

    M = 0.5 * (NEUTRON_MASS + PROTON_MASS)
    mpi = PION_MASS
    qmin = mpi + 0.5 * mpi^2 / M
    qmax = K + m - 0.5*(M + m*m/M)
    (q <= qmin || q >= qmax) && return zero(T)

    E = K + m
    ml2 = m * m
    Q2min = ml2 * (q*q - 0.5*ml2) / (E * (E - q))
    Q2max = 2.0 * M * (q - mpi) - mpi^2
    (Q2max < Q2min || Q2min < 0.0) && return zero(T)

    gauss_quad_logscale!(_PHOTO_NODES, _PHOTO_WEIGHTS, 9, Float64(Q2min), Float64(Q2max))
    ds = 0.0
    @inbounds for i in 1:9
        Q2 = _PHOTO_NODES[i]
        val = _dcs_photonuclear_d2_DRSS(Float64(Z), Float64(A), Float64(m),
                                         Float64(K), Float64(q), Q2)
        val > 0.0 && (ds += val * _PHOTO_WEIGHTS[i])
    end

    return T(ds < 0.0 ? 0.0 : ds)
end

# ============================================================================
# Unified DCS integration (exact PUMAS compute_dcs_integral)
#
# mode=0: cross-section ∫ dσ/dq dq  (m²/kg)
# mode=1: energy loss   ∫ q dσ/dq dq (GeV⋅m²/kg)
# mode=2: straggling    ∫ q² dσ/dq dq (GeV²⋅m²/kg)
#
# Integration is done per element (Z, A), with proper kinematic bounds
# and Gauss quadrature in log-space.
# ============================================================================

@fastmath function compute_dcs_integral(dcs_func::Symbol, Z::Float64, A::Float64, I::Float64,
                               mass::Float64, kinetic::Float64,
                               xlow::Float64, xhigh::Float64, mode::Int;
                               npoints::Int = 180)
    xlow <= 0.0 && (xlow = 1e-6)

    if dcs_func == :ionisation
        return dcs_ionisation_integrate(Z, A, I, mass, kinetic, xlow, xhigh, mode)
    end

    M = 0.5 * (NEUTRON_MASS + PROTON_MASS)
    mpi = PION_MASS
    local qmin_abs::Float64, qmax_abs::Float64

    if dcs_func == :photonuclear
        qmin_abs = mpi + 0.5 * mpi^2 / M
        qmax_abs = kinetic + mass - 0.5*(M + mass^2/M)
    else
        Z13 = Z^(1.0/3.0)
        sqrte_val = 1.648721271
        qmax_abs = kinetic + mass * (1.0 - 0.75 * sqrte_val * Z13)
        qmin_abs = dcs_func == :pair_production ? 4.0 * ELECTRON_MASS : 0.0
    end

    qlow = max(xlow * kinetic, qmin_abs)
    qhigh = min(xhigh * kinetic, qmax_abs)
    qlow >= qhigh && return 0.0

    n_intervals = (npoints + 8) ÷ 9
    n_gq = 9
    E = kinetic + mass

    dcsint = 0.0
    lqlow = log(qlow)
    lqhigh = log(qhigh)

    for itv in 1:n_intervals
        a = lqlow + (itv - 1) * (lqhigh - lqlow) / n_intervals
        b = lqlow + itv * (lqhigh - lqlow) / n_intervals
        xref, wref = gauss_quad_coefficients(n_gq)
        dx = b - a
        @inbounds for j in 1:n_gq
            xi = exp(a + dx * xref[j])
            wi = wref[j] * dx * xi

            val = if dcs_func == :bremsstrahlung
                dcs_bremsstrahlung_ssr(Z, A, mass, kinetic, xi)
            elseif dcs_func == :pair_production
                dcs_pair_production_ssr(Z, A, mass, kinetic, xi)
            elseif dcs_func == :photonuclear
                dcs_photonuclear_drss(Z, A, mass, kinetic, xi)
            else
                0.0
            end

            y = val * xi
            mode >= 1 && (y *= xi)
            mode >= 2 && (y *= xi)
            dcsint += y * wi
        end
    end

    dcsint *= AVOGADRO_NUMBER / (A * 1e-3)
    dcsint /= E

    return dcsint
end

# ============================================================================
# Legacy / compatibility wrappers
# ============================================================================

function elastic_dcs(Z::Real, A::Real, m::Real, K::T, θ::T) where T<:Real
    gamma = one(T) + K / m
    β² = one(T) - one(T) / (gamma^2)
    p = sqrt(K * (K + 2m))
    M_target = A * 0.931494
    sin_half = sin(θ / 2)
    q = 2 * p * sin_half
    a_TF = BOHR_RADIUS * 0.8853 / Z^(1/3)
    λ_a = HBAR_C / (a_TF * p)
    R_n = 1.2e-15 * A^(1/3)
    λ_n = R_n * p / HBAR_C
    σ_R = (Z * ALPHA_EM * HBAR_C / (2 * p * β² * sin_half^2))^2
    F_atom = 1 / (1 + (sin_half / λ_a)^2)^2
    F_nucl = exp(-2 * (sin_half / λ_n)^2)
    F_spin = 1 - β² * sin_half^2
    return σ_R * F_atom^2 * F_nucl * F_spin * sin(θ)
end

function elastic_path(order::Int, Z::Real, A::Real, m::Real, K::T) where T<:Real
    if order != 0 && order != 1
        return -one(T)
    end
    N = 100
    dθ = π / N
    integral = zero(T)
    for i in 1:N
        θ = (i - 0.5) * dθ
        dσ = elastic_dcs(Z, A, m, K, θ)
        weight = order == 0 ? one(T) : (1 - cos(θ))
        integral += dσ * weight * dθ
    end
    n = AVOGADRO_NUMBER / (A * 1e-3)
    return 1 / (n * integral)
end

end # module Materials
