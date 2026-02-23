"""
    Coulomb

First-principles Coulomb scattering computation, ported from PUMAS C.
Computes transport path, elastic path, and screening parameter tables
by integrating the screened Coulomb DCS analytically via pole reduction.

References:
  - Salvat et al., Phys. Rev. A 36, 467 (1987) — atomic screening
  - Kuraev et al., Phys. Rev. D 89, 116016 (2014) — Born correction
  - Salvat et al., NIMB 316, 144 (2013) — electronic transport
"""
module Coulomb

using ..Constants
using ..Materials: AtomicElement, BaseMaterial,
    dcs_bremsstrahlung_ssr, dcs_pair_production_ssr, dcs_photonuclear_drss

export compute_coulomb_tables!

# ============================================================
# Static data tables from PUMAS (Salvat 1987)
# ============================================================

const SALVAT_PREFACTOR = (
    # prefactor[1][Z] — first amplitude, Z=1..103
    (1.00000e+00,-2.25920e-01, 6.04537e-01, 3.27766e-01,
     2.32684e-01, 1.53676e-01, 9.95750e-02, 6.25130e-02, 3.68040e-02,
     1.88410e-02, 7.44440e-01, 6.42349e-01, 6.00152e-01, 5.15971e-01,
     4.38675e-01, 5.45871e-01, 7.24889e-01, 2.19124e+00, 4.85607e-02,
     5.80017e-01, 5.54340e-01, 1.11950e-02, 3.18350e-02, 1.07503e-01,
     4.97556e-02, 5.11841e-02, 5.00039e-02, 4.73509e-02, 7.70967e-02,
     4.00041e-02, 1.08344e-01, 6.09767e-02, 2.11561e-02, 4.83575e-01,
     4.50364e-01, 4.19036e-01, 1.73438e-01, 3.35694e-02, 6.88939e-02,
     1.17552e-01, 2.55689e-01, 2.69313e-01, 2.20138e-01, 2.75057e-01,
     2.71053e-01, 2.78363e-01, 2.56210e-01, 2.27100e-01, 2.49215e-01,
     2.15313e-01, 1.80560e-01, 1.30772e-01, 5.88293e-02, 4.45145e-01,
     2.70796e-01, 1.72814e-01, 1.94726e-01, 1.91338e-01, 1.86776e-01,
     1.66461e-01, 1.62350e-01, 1.58016e-01, 1.53759e-01, 1.58729e-01,
     1.45327e-01, 1.41260e-01, 1.37360e-01, 1.33614e-01, 1.29853e-01,
     1.26659e-01, 1.28806e-01, 1.30256e-01, 1.38420e-01, 1.50030e-01,
     1.60803e-01, 1.72164e-01, 1.83411e-01, 2.23043e-01, 2.28909e-01,
     2.09753e-01, 2.70821e-01, 2.37958e-01, 2.28771e-01, 1.94059e-01,
     1.49995e-01, 9.55262e-02, 3.19155e-01, 2.40406e-01, 2.26579e-01,
     2.17619e-01, 2.41294e-01, 2.44758e-01, 2.46231e-01, 2.55572e-01,
     2.53567e-01, 2.43832e-01, 2.41898e-01, 2.44050e-01, 2.40237e-01,
     2.34997e-01, 2.32114e-01, 2.27937e-01, 2.29571e-01),
    # prefactor[2][Z]
    (0.00000e+00, 1.22592e+00, 3.95463e-01, 6.72234e-01,
     7.67316e-01, 8.46324e-01, 9.00425e-01, 9.37487e-01, 9.63196e-01,
     9.81159e-01, 2.55560e-01, 3.57651e-01, 3.99848e-01, 4.84029e-01,
     5.61325e-01,-5.33329e-01,-7.54809e-01,-2.28520e+00, 7.75935e-01,
     4.19983e-01, 4.45660e-01, 6.83176e-01, 6.75303e-01, 7.16172e-01,
     6.86632e-01, 6.99533e-01, 7.14201e-01, 7.29404e-01, 7.95083e-01,
     7.59034e-01, 7.48941e-01, 7.15671e-01, 6.70932e-01, 5.16425e-01,
     5.49636e-01, 5.80964e-01, 7.25336e-01, 7.81581e-01, 7.20203e-01,
     6.58088e-01, 5.82051e-01, 5.75262e-01, 5.61797e-01, 5.94338e-01,
     6.11921e-01, 6.06653e-01, 6.50520e-01, 6.15496e-01, 6.43990e-01,
     6.11497e-01, 5.76688e-01, 5.50366e-01, 5.48174e-01, 5.54855e-01,
     6.52415e-01, 6.84485e-01, 6.38429e-01, 6.46684e-01, 6.55810e-01,
     7.05677e-01, 7.13311e-01, 7.20978e-01, 7.28385e-01, 7.02414e-01,
     7.42619e-01, 7.49352e-01, 7.55797e-01, 7.61947e-01, 7.68005e-01,
     7.73365e-01, 7.52781e-01, 7.32428e-01, 7.09596e-01, 6.87141e-01,
     6.65932e-01, 6.46849e-01, 6.30598e-01, 6.17575e-01, 6.11402e-01,
     6.00426e-01, 6.42829e-01, 6.30789e-01, 6.21959e-01, 6.10455e-01,
     6.03147e-01, 6.05994e-01, 6.23324e-01, 6.56665e-01, 6.42246e-01,
     6.24013e-01, 6.30394e-01, 6.29816e-01, 6.31596e-01, 6.49005e-01,
     6.53604e-01, 6.43738e-01, 6.48850e-01, 6.70318e-01, 6.76319e-01,
     6.65571e-01, 6.88406e-01, 6.94394e-01, 6.82014e-01)
)

const SALVAT_EXPONENT = (
    # exponent[1][Z]
    (1.11728e+00, 5.52725e+00, 2.81741e+00, 4.54302e+00,
     5.99006e+00, 8.04043e+00, 1.08122e+01, 1.48233e+01, 2.14001e+01,
     3.49994e+01, 4.12050e+00, 4.72663e+00, 5.14051e+00, 5.84918e+00,
     6.67070e+00, 6.37029e+00, 6.21183e+00, 5.54701e+00, 3.02597e+01,
     6.32184e+00, 6.63280e+00, 9.97569e+01, 4.25330e+01, 1.89587e+01,
     3.18642e+01, 3.18251e+01, 3.29153e+01, 3.47580e+01, 2.53264e+01,
     4.03429e+01, 2.01922e+01, 2.91996e+01, 6.24873e+01, 8.78242e+00,
     9.33480e+00, 9.91420e+00, 1.71659e+01, 5.52077e+01, 3.13659e+01,
     2.20537e+01, 1.42403e+01, 1.40442e+01, 1.59176e+01, 1.43137e+01,
     1.46537e+01, 1.46455e+01, 1.55878e+01, 1.69141e+01, 1.61552e+01,
     1.77931e+01, 1.98751e+01, 2.41540e+01, 3.99955e+01, 1.18053e+01,
     1.65915e+01, 2.23966e+01, 2.07637e+01, 2.12350e+01, 2.18033e+01,
     2.39492e+01, 2.45984e+01, 2.52966e+01, 2.60169e+01, 2.54973e+01,
     2.75466e+01, 2.83460e+01, 2.91604e+01, 2.99904e+01, 3.08345e+01,
     3.16806e+01, 3.13526e+01, 3.12166e+01, 3.00767e+01, 2.86302e+01,
     2.75684e+01, 2.65861e+01, 2.57339e+01, 2.29939e+01, 2.28644e+01,
     2.44080e+01, 2.09409e+01, 2.29872e+01, 2.37917e+01, 2.66951e+01,
     3.18397e+01, 4.34890e+01, 2.00150e+01, 2.45012e+01, 2.56843e+01,
     2.65542e+01, 2.51930e+01, 2.52522e+01, 2.54271e+01, 2.51526e+01,
     2.55959e+01, 2.65567e+01, 2.70360e+01, 2.72673e+01, 2.79152e+01,
     2.86446e+01, 2.93353e+01, 3.01040e+01, 3.02650e+01),
    # exponent[2][Z]
    (1.00000e+00, 2.39924e+00, 6.62463e-01, 9.85154e-01,
     1.21347e+00, 1.49129e+00, 1.76868e+00, 2.04035e+00, 2.30601e+00,
     2.56621e+00, 8.71798e-01, 1.00247e+00, 1.01529e+00, 1.17314e+00,
     1.34102e+00, 2.55169e+00, 3.38827e+00, 4.56873e+00, 3.12426e+00,
     1.00935e+00, 1.10227e+00, 4.12865e+00, 3.94043e+00, 3.06375e+00,
     3.78110e+00, 3.77161e+00, 3.79085e+00, 3.82989e+00, 3.39276e+00,
     3.94645e+00, 3.47325e+00, 4.12525e+00, 4.95015e+00, 1.69671e+00,
     1.79002e+00, 1.88354e+00, 3.11025e+00, 4.28418e+00, 4.24121e+00,
     4.03254e+00, 2.97020e+00, 2.86107e+00, 3.36719e+00, 2.73701e+00,
     2.71828e+00, 2.61549e+00, 2.74124e+00, 3.08408e+00, 2.88189e+00,
     3.29372e+00, 3.80921e+00, 4.61191e+00, 5.91318e+00, 1.79673e+00,
     2.69645e+00, 3.45951e+00, 3.46574e+00, 3.48193e+00, 3.50982e+00,
     3.51987e+00, 3.55603e+00, 3.59628e+00, 3.63834e+00, 3.73639e+00,
     3.72882e+00, 3.77625e+00, 3.82444e+00, 3.87344e+00, 3.92327e+00,
     3.97271e+00, 4.09040e+00, 4.20492e+00, 4.24918e+00, 4.24261e+00,
     4.23412e+00, 4.19992e+00, 4.14615e+00, 3.73461e+00, 3.69138e+00,
     3.96429e+00, 3.24563e+00, 3.62172e+00, 3.77959e+00, 4.25824e+00,
     4.92848e+00, 5.85205e+00, 2.90906e+00, 3.55241e+00, 3.79223e+00,
     4.00437e+00, 3.67795e+00, 3.63966e+00, 3.61328e+00, 3.43021e+00,
     3.43474e+00, 3.59089e+00, 3.59411e+00, 3.48061e+00, 3.50331e+00,
     3.61870e+00, 3.55697e+00, 3.58685e+00, 3.64085e+00),
    # exponent[3][Z]
    (1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
     1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
     1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
     1.00000e+00, 1.67534e+00, 1.85964e+00, 2.04455e+00, 7.32637e-01,
     1.00000e+00, 1.00000e+00, 1.00896e+00, 1.05333e+00, 1.00137e+00,
     1.12787e+00, 1.16064e+00, 1.19152e+00, 1.22089e+00, 1.14261e+00,
     1.27594e+00, 1.00643e+00, 1.18447e+00, 1.35819e+00, 1.00000e+00,
     1.00000e+00, 1.00000e+00, 7.17673e-01, 8.57842e-01, 9.47152e-01,
     1.01806e+00, 1.01699e+00, 1.05906e+00, 1.15477e+00, 1.10923e+00,
     1.12336e+00, 1.43183e+00, 1.14079e+00, 1.26189e+00, 9.94156e-01,
     1.14781e+00, 1.28288e+00, 1.41954e+00, 1.54707e+00, 1.00000e+00,
     6.81361e-01, 8.07311e-01, 8.91057e-01, 9.01112e-01, 9.10636e-01,
     8.48620e-01, 8.56929e-01, 8.65025e-01, 8.73083e-01, 9.54998e-01,
     8.88981e-01, 8.96917e-01, 9.04803e-01, 9.12768e-01, 9.20306e-01,
     9.28838e-01, 1.00717e+00, 1.09456e+00, 1.16966e+00, 1.23403e+00,
     1.29699e+00, 1.35350e+00, 1.40374e+00, 1.44284e+00, 1.48856e+00,
     1.53432e+00, 1.11214e+00, 1.23735e+00, 1.25338e+00, 1.35772e+00,
     1.46828e+00, 1.57359e+00, 7.20714e-01, 8.37599e-01, 9.33468e-01,
     1.02385e+00, 9.69895e-01, 9.82474e-01, 9.92527e-01, 9.32751e-01,
     9.41671e-01, 1.01827e+00, 1.02554e+00, 9.66447e-01, 9.74347e-01,
     1.04137e+00, 9.90568e-01, 9.98878e-01, 1.04473e+00)
)

const NUCLEAR_RADII = (
    # rN[Z=0..119] in fm.  Index 119 is fictitious "Rockium"
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
)

# ============================================================
# Core Coulomb scattering functions
# ============================================================

"""
Nuclear RMS radius in meters.  Z is integer atomic number, A is mass number.
"""
function nuclear_radius(Z::Real, A::Real)
    iN = Int(Z)  # 0-based C index maps to 1-based Julia via +1
    nN = length(NUCLEAR_RADII)
    if iN >= nN - 1
        iN = nN - 2
    elseif iN == 1 && A >= 1.5
        iN = 0      # hydrogen isotopes
    elseif iN == 11 && A == 22.0
        iN = nN - 1 # Rockium
    end
    # NUCLEAR_RADII is 1-based in Julia; C index iN maps to Julia index iN+1
    return NUCLEAR_RADII[iN + 1] * 1e-15
end

"""
Spin correction factor for the Coulomb DCS.
"""
function spin_factor(mass::Float64, kinetic::Float64)
    e = kinetic + mass
    return kinetic * (e + mass) / (e * e)
end

"""
CM frame parameters: kinetic0 (CM kinetic energy), gamma_CM, tau.
Returns (kinetic0, fCM) where fCM = (gamma_CM, tau).
"""
function frame_parameters(Z::Float64, A::Float64, mass::Float64, kinetic::Float64)
    Ma = Z * PROTON_MASS + (A - Z) * NEUTRON_MASS
    M2 = (mass + Ma)^2
    sCM12i = 1.0 / sqrt(M2 + 2.0 * Ma * kinetic)
    gamma_CM = (kinetic + mass + Ma) * sCM12i
    kinetic0 = (kinetic * Ma + mass * (mass + Ma)) * sCM12i - mass
    if kinetic0 < 1e-9
        kinetic0 = 1e-9
    end
    etot = kinetic + mass + Ma
    betaCM2 = kinetic * (kinetic + 2.0 * mass) / (etot * etot)
    rM2 = (mass / Ma)^2
    tau = sqrt(rM2 * (1.0 - betaCM2) + betaCM2)
    return kinetic0, (gamma_CM, tau)
end

"""
Macroscopic cross-section normalisation factor in kg/m^2.
"""
function normalisation(Z::Float64, A::Float64, mass::Float64,
                       kinetic::Float64, kinetic0::Float64)
    p2 = kinetic * (kinetic + 2.0 * mass)
    p02 = kinetic0 * (kinetic0 + 2.0 * mass)
    d = ALPHA_EM * Z * HBAR_C * (kinetic + mass)
    return A * 1e-3 * p2 * p02 / (d * d * π * AVOGADRO_NUMBER)
end

"""
Compute atomic + nuclear screening parameters.
Returns (n_parameters, amplitude[1:n], screening[1:n+1])
where screening[end] is the nuclear screening term.
"""
function screening_parameters(Z::Float64, A::Float64, mass::Float64,
                              kinetic::Float64, kinetic0::Float64)
    # KTV correction (Kuraev et al. 2014)
    aZE = ALPHA_EM * Z * (kinetic + mass)
    p2 = kinetic * (kinetic + 2.0 * mass)
    zeta2 = aZE * aZE / p2
    local f::Float64
    if zeta2 < 1.0
        f = zeta2 * (1.0 / (1.0 + zeta2) +
            0.20205690315959424 -
            0.03692775514336999 * zeta2 +
            0.008349277381922926 * zeta2 * zeta2)
    else
        r2i = 1.0 / (1.0 + zeta2)
        phi = -atan(sqrt(zeta2))
        f = -0.5 * log(r2i) +
            0.5 * (1.0 - r2i) +
            (1.0 / 12.0) * (1.0 - cos(2 * phi) * r2i) -
            (1.0 / 120.0) * (1.0 - cos(4 * phi) * r2i * r2i) +
            (1.0 / 252.0) * (1.0 - cos(6 * phi) * r2i * r2i * r2i)
    end
    ktv = exp(2.0 * f)

    p02 = kinetic0 * (kinetic0 + 2.0 * mass)
    d = 0.25 * HBAR_C * HBAR_C / p02 * ktv

    iZ = clamp(Int(Z) - 1, 0, 102)  # 0-based
    idx = iZ + 1  # 1-based Julia index

    # Determine number of atomic terms
    local n::Int
    amplitude = zeros(Float64, 4)
    screening = zeros(Float64, 5)

    if SALVAT_EXPONENT[3][idx] == 1.0
        if iZ == 0
            n = 1
            amplitude[1] = 1.0
        else
            n = 2
            amplitude[1] = SALVAT_PREFACTOR[1][idx]
            amplitude[2] = 1.0 - amplitude[1]
        end
    else
        n = 3
        amplitude[1] = SALVAT_PREFACTOR[1][idx]
        amplitude[2] = SALVAT_PREFACTOR[2][idx]
        amplitude[3] = 1.0 - (amplitude[1] + amplitude[2])
    end

    for i in 1:n
        a = SALVAT_EXPONENT[i][idx] / BOHR_RADIUS
        screening[i] = d * a * a
    end

    # Nuclear screening
    RN = nuclear_radius(Z, A)
    screening[n + 1] = 12.0 * d / (RN * RN)

    n_total = n + 1  # includes nuclear term
    return n_total, amplitude, screening
end

"""
Pole reduction of the Coulomb DCS for analytic integration.
n_parameters includes the nuclear term (last screening entry).
Returns (a, b, c) coefficient vectors.
"""
function pole_reduction(n_parameters::Int, amplitude::Vector{Float64},
                        screening::Vector{Float64})
    n = n_parameters - 1  # number of atomic terms
    a = zeros(Float64, max(n, 1))
    b = zeros(Float64, max(n, 1))
    c = zeros(Float64, 4)

    # Pole reduction without nuclear term
    for i in 1:n
        Ai = amplitude[i]
        Bi = screening[i]
        b[i] = Ai * Ai
        for j in (i+1):n
            Aj = amplitude[j]
            Bj = screening[j]
            d = 2.0 * Ai * Aj / (Bj - Bi)
            a[i] += d
            a[j] -= d
        end
    end

    # Nuclear factors
    N = screening[n + 1]
    Sa = zeros(Float64, 4)
    Sb = zeros(Float64, 4)
    for i in 1:n
        Bi = screening[i]
        x = 1.0 - Bi / N
        r = x
        for j in 1:4
            Sa[j] += a[i] / r
            r *= x
            Sb[j] += b[i] / r
        end
    end

    tmp = 1.0
    for i in 1:4
        c[i] = ((4 - (i - 1)) * Sb[4 - (i - 1)] / N - Sa[4 - (i - 1)]) * tmp
        tmp *= N
    end

    # Update atomic pole factors for nuclear term
    for i in 1:n
        Bi = screening[i]
        d = 1.0 - Bi / N
        d4 = d * d * d * d
        a[i] = (a[i] - 4.0 * b[i] / (N - Bi)) / d4
        b[i] = b[i] / d4
    end

    return a, b, c
end

"""
Compute order 0 and 1 transport coefficients by summing pole contributions.
Returns (cs0, cs1_times_2).
"""
function transport_coefficients(mu_val::Float64, fspin::Float64,
                                n_parameters::Int, screening::Vector{Float64},
                                a::Vector{Float64}, b::Vector{Float64},
                                c::Vector{Float64})
    S = fspin
    mu = mu_val
    n = n_parameters - 1

    cs0 = 0.0
    cs1 = 0.0
    for i in 1:n
        alp = screening[i]
        r = mu / (mu + alp)
        L = log(1.0 + mu / alp)
        I0 = r / alp
        J0 = L
        I1 = L - r
        J1 = -alp * L
        I2 = mu + alp * (r - 2.0 * L)
        J2 = alp * (alp * L - mu)

        cs0 += a[i] * (J0 - S * J1) + b[i] * (I0 - S * I1)
        cs1 += a[i] * (J1 - S * J2) + b[i] * (I1 - S * I2)
    end

    # Nuclear term
    N = screening[n + 1]
    r = mu / (mu + N)
    L = log(1.0 + mu / N)
    I0 = r / N
    J0 = L
    I1 = L - r
    J1 = -N * L
    I2 = mu + N * (r - 2.0 * L)
    J2 = N * (N * L - mu)

    rn = 1.0 / (1.0 + mu / N)
    K0 = (1.0 - rn) * (1.0 + rn) / (2.0 * N * N)
    L0 = (1.0 - rn) * ((1.0 + rn)^2 - rn) / (3.0 * N^3)
    K1 = I0 - N * K0
    L1 = K0 - N * L0
    K2 = I1 - N * K1
    L2 = K1 - N * L1

    cs0 += c[1] * (J0 - S * J1) +
           c[2] * (I0 - S * I1) +
           c[3] * (K0 - S * K1) +
           c[4] * (L0 - S * L1)
    cs1 += c[1] * (J1 - S * J2) +
           c[2] * (I1 - S * I2) +
           c[3] * (K1 - S * K2) +
           c[4] * (L1 - S * L2)

    return cs0, 2.0 * cs1
end

"""
Restricted cross section integrated from mu to 1.
"""
function restricted_cs(mu::Float64, fspin::Float64, n_parameters::Int,
                       screening::Vector{Float64}, a::Vector{Float64},
                       b::Vector{Float64}, c::Vector{Float64})
    if mu >= 1.0 || mu >= 1e6 * screening[n_parameters]
        return 0.0
    end
    cs0_full, _ = transport_coefficients(1.0, fspin, n_parameters, screening, a, b, c)
    cs0_mu, _ = transport_coefficients(mu, fspin, n_parameters, screening, a, b, c)
    return cs0_mu > cs0_full ? 0.0 : cs0_full - cs0_mu
end

# ============================================================
# Root finder (Ridder's method, from PUMAS math_find_root)
# ============================================================

"""
Find root of f(x) = 0 in [xa, xb] using Ridder's method.
Returns the root or the best estimate if max iterations reached.
"""
function ridder_find_root(f, xa::Float64, xb::Float64,
                          fa::Float64, fb::Float64;
                          xtol::Float64=0.0, rtol::Float64=1e-6,
                          maxiter::Int=100)
    if fa * fb > 0
        return 0.0
    end
    fa == 0 && return xa
    fb == 0 && return xb

    tol = xtol + rtol * min(abs(xa), abs(xb))

    xn = 0.0
    for _ in 1:maxiter
        dm = 0.5 * (xb - xa)
        xm = xa + dm
        fm = f(xm)
        sgn = fb > fa ? 1.0 : -1.0
        dn = sgn * dm * fm / sqrt(fm * fm - fa * fb)
        sgn2 = dn > 0.0 ? 1.0 : -1.0
        dn = abs(dn)
        dm_abs = abs(dm) - 0.5 * tol
        if dn < dm_abs
            dm_abs = dn
        end
        xn = xm - sgn2 * dm_abs
        fn = f(xn)
        if fn * fm < 0.0
            xa = xn; fa = fn
            xb = xm; fb = fm
        elseif fn * fa < 0.0
            xb = xn; fb = fn
        else
            xa = xn; fa = fn
        end
        if fn == 0.0 || abs(xb - xa) < tol
            return xn
        end
    end
    return xn
end

# ============================================================
# Electronic transport (Salvat et al. NIMB 316 2013)
# ============================================================

"""
Sternheimer aS parameter for a material.
Simplified computation: aS = I * 1e9 / wp where
wp = 28.816 * sqrt(ZoA * density_gcc).
This is the ratio I / geometric_mean(oscillator_levels),
approximated assuming oscillator levels scale with the plasma frequency.
"""
function sternheimer_aS(ZoA::Float64, I_gev::Float64, density_kgm3::Float64)
    density_gcc = density_kgm3 * 1e-3
    wp_eV = 28.816 * sqrt(ZoA * density_gcc)
    I_eV = I_gev * 1e9
    aS = I_eV / wp_eV
    return aS
end

"""
First transport cross-section for electronic multiple scattering.
Returns inverse transport path length in kg/m^2.
"""
function transverse_transport_electronic(ZoA::Float64, I::Float64, aS::Float64,
                                         mass::Float64, kinetic::Float64, nu::Float64)
    momentum2 = kinetic * (kinetic + 2.0 * mass)
    E = kinetic + mass
    Wr = 2.0 * ELECTRON_MASS * momentum2 /
         (mass^2 + ELECTRON_MASS * (ELECTRON_MASS + 2.0 * E))
    Wmax = min(nu, Wr)

    beta2 = momentum2 / (E * E)
    J = log(aS * Wmax / I) - beta2 * Wmax / Wr +
        0.25 * Wmax^2 / (E * E) + 1.0 / aS

    gamma = E / mass
    lQ = log(1.0 + 2.0 * Wr / ELECTRON_MASS)
    Delta = ALPHA_EM / (2π) * (log(2.0 * gamma) - lQ / 3.0) * lQ^2

    return 2π * ELECTRON_RADIUS^2 * ELECTRON_MASS^2 *
           AVOGADRO_NUMBER * ZoA / (beta2 * 1e-3 * momentum2) * (J + Delta)
end

# ============================================================
# Radiative DCS transport integrals
# ============================================================

"""
Gauss-Legendre quadrature nodes and weights for n points on [a, b].
"""
function gauss_legendre(n::Int, a::Float64, b::Float64)
    nodes, weights = _gl_nodes_weights(n)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid .+ half .* nodes
    w = half .* weights
    return x, w
end

# Precomputed 20-point Gauss-Legendre on [-1,1]
const _GL20_NODES = (
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
    -0.5108670019508271, -0.3737060887154196, -0.2277858511416451,
    -0.0765265211334973,
     0.0765265211334973,  0.2277858511416451,  0.3737060887154196,
     0.5108670019508271,  0.6360536807265150,  0.7463319064601508,
     0.8391169718222188,  0.9122344282513259,  0.9639719272779138,
     0.9931285991850949)

const _GL20_WEIGHTS = (
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
    0.1527533871307259,
    0.1527533871307259, 0.1491729864726037, 0.1420961093183821,
    0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
    0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
    0.0176140071391521)

function _gl_nodes_weights(n::Int)
    if n == 20
        return collect(_GL20_NODES), collect(_GL20_WEIGHTS)
    end
    error("Only 20-point GL quadrature is precomputed")
end

"""
Compute the angular deflection integral for a DCS process:
  ∫ θ² × dσ/dq dq  (as first transport cross section)

Uses log-spaced Gauss quadrature matching PUMAS's approach.
dcs_func(Z, A, m, K, q) returns dσ/dq in m²/GeV.
"""
function dcs_transport_integral(dcs_func, Z::Float64, A::Float64,
                                mass::Float64, kinetic::Float64,
                                qmin::Float64, qmax::Float64)
    if qmin >= qmax || qmin <= 0.0
        return 0.0
    end

    E = kinetic + mass
    p2 = kinetic * (kinetic + 2.0 * mass)

    lqmin = log(qmin)
    lqmax = log(qmax)
    nodes, weights = gauss_legendre(20, lqmin, lqmax)

    result = 0.0
    for i in eachindex(nodes)
        q = exp(nodes[i])
        dcs_val = dcs_func(Z, A, mass, kinetic, q)
        if dcs_val > 0.0
            theta2 = q^2 / p2
            result += dcs_val * theta2 * q * weights[i]
        end
    end

    return result
end

"""
Bremsstrahlung transport integral (inverse transport path contribution).
cutoff is fractional (0 to 1): integrate from 0 to cutoff*K.
"""
function dcs_bremsstrahlung_transport(Z::Float64, A::Float64, mass::Float64,
                                      kinetic::Float64, cutoff::Float64)
    sqrte = 1.6487212707
    Z13 = Z^(1.0/3.0)
    qmax_phys = kinetic + mass * (1.0 - 0.75 * sqrte * Z13)
    qmax = min(cutoff * kinetic, qmax_phys)
    qmin = 1e-3 * qmax
    (qmin >= qmax || qmin <= 0.0) && return 0.0
    return dcs_transport_integral(dcs_bremsstrahlung_ssr, Z, A, mass, kinetic, qmin, qmax)
end

"""
Pair production transport integral.
"""
function dcs_pair_production_transport(Z::Float64, A::Float64, mass::Float64,
                                       kinetic::Float64, cutoff::Float64)
    qmin_phys = 4.0 * ELECTRON_MASS
    sqrte = 1.6487212707
    Z13 = Z^(1.0/3.0)
    qmax_phys = kinetic + mass * (1.0 - 0.75 * sqrte * Z13)
    qmax = min(cutoff * kinetic, qmax_phys)
    qmin = max(qmin_phys, 1e-3 * qmax)
    (qmin >= qmax || qmin <= 0.0) && return 0.0
    return dcs_transport_integral(dcs_pair_production_ssr, Z, A, mass, kinetic, qmin, qmax)
end

"""
Photonuclear transport integral.
"""
function dcs_photonuclear_transport(Z::Float64, A::Float64, mass::Float64,
                                    kinetic::Float64, cutoff::Float64)
    qmin_phys = 0.0
    qmax = cutoff * kinetic
    qmin = max(qmin_phys, 1e-3 * qmax)
    (qmin >= qmax || qmin <= 0.0) && return 0.0
    return dcs_transport_integral(dcs_photonuclear_drss, Z, A, mass, kinetic, qmin, qmax)
end

"""
Compute per-element radiative soft scattering contribution (inverse transport path).
Returns (invlb1_csda, invlb1_mixed).
"""
function compute_element_soft_scattering(Z::Float64, A::Float64, mass::Float64,
                                         kinetic::Float64, cutoff::Float64)
    invlb1_csda = 0.0
    invlb1_mixed = 0.0

    invlb1_csda  += dcs_bremsstrahlung_transport(Z, A, mass, kinetic, 1.0)
    invlb1_mixed += dcs_bremsstrahlung_transport(Z, A, mass, kinetic, cutoff)

    invlb1_csda  += dcs_pair_production_transport(Z, A, mass, kinetic, 1.0)
    invlb1_mixed += dcs_pair_production_transport(Z, A, mass, kinetic, cutoff)

    invlb1_csda  += dcs_photonuclear_transport(Z, A, mass, kinetic, 1.0)
    invlb1_mixed += dcs_photonuclear_transport(Z, A, mass, kinetic, cutoff)

    return invlb1_csda, invlb1_mixed
end

# ============================================================
# Main table computation
# ============================================================

struct CoulombElementData
    n_parameters::Int
    amplitude::Vector{Float64}
    screening::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    fspin::Float64
    fCM::Tuple{Float64, Float64}
    norm::Float64    # normalisation factor (1/norm gives macroscopic units)
    d2::Float64      # CM→Lab factor squared
end

"""
    compute_coulomb_tables!(material, mass, cutoff, energies,
        transport_path, elastic_path, screening_param, cross_section)

Compute scattering tables from first principles for a single material.
Fills transport_path, elastic_path, and screening_param arrays.
cross_section is NOT modified (computed separately from DEL DCS).

Arguments:
  - material: BaseMaterial with elements and fractions
  - mass: projectile mass in GeV
  - cutoff: fractional cutoff for mixed mode
  - energies: kinetic energy grid in GeV
  - csda_range: CSDA range table (kg/m²) for the material
  - ZoA, I_gev, density: material properties for electronic transport
  - transport_path, elastic_path, screening_param: output arrays
"""
function compute_coulomb_tables!(material::BaseMaterial, mass::Float64,
                                 elastic_ratio::Float64, cutoff::Float64,
                                 energies::Vector{Float64},
                                 csda_range::Vector{Float64},
                                 ZoA::Float64, I_gev::Float64, density::Float64,
                                 transport_path::Vector{Float64},
                                 elastic_path::Vector{Float64},
                                 screening_param::Vector{Float64})
    n_energies = length(energies)
    aS = sternheimer_aS(ZoA, I_gev, density)

    for ie in 1:n_energies
        K = energies[ie]
        if K <= 0.0
            transport_path[ie] = 0.0
            elastic_path[ie] = 0.0
            screening_param[ie] = 0.0
            continue
        end

        # Per-element Coulomb computation
        n_elements = length(material.elements)
        elem_data = Vector{CoulombElementData}(undef, n_elements)
        cs_m = 0.0   # total macroscopic cross section
        cs1_m = 0.0  # first transport cross section
        A_min = Inf   # minimum screening parameter
        cs_A = 0.0

        for iel in 1:n_elements
            elem = material.elements[iel]
            frac = material.fractions[iel]
            Z = Float64(elem.Z)
            Ae = Float64(elem.A)

            kinetic0, fCM = frame_parameters(Z, Ae, mass, K)
            fs = spin_factor(mass, K)
            n_params, amp, scr = screening_parameters(Z, Ae, mass, K, kinetic0)
            a_coeff, b_coeff, c_coeff = pole_reduction(n_params, amp, scr)
            G0, G1 = transport_coefficients(1.0, fs, n_params, scr, a_coeff, b_coeff, c_coeff)
            norm = normalisation(Z, Ae, mass, K, kinetic0)
            norm_frac = frac / norm

            cs_m += norm_frac * G0
            d = 1.0 / (fCM[1] * (1.0 + fCM[2]))  # gamma_CM * (1 + tau)
            d2 = d * d
            cs1_m += norm_frac * G1 * d2

            for j in 1:(n_params - 1)
                Aj = scr[j]
                if Aj < A_min
                    A_min = Aj
                end
            end
            cs_A += norm_frac

            elem_data[iel] = CoulombElementData(
                n_params, amp, scr, a_coeff, b_coeff, c_coeff,
                fs, fCM, norm_frac, d2)
        end

        cs_A /= A_min * (A_min + 1.0)

        # Hard scattering mean free path
        range_i = csda_range[ie] > 0.0 ? 1.0 / csda_range[ie] : 0.0
        cs_h = max(cs1_m, range_i) / elastic_ratio
        if cs_h < 1.0 / EHS_PATH_MAX
            cs_h = 1.0 / EHS_PATH_MAX
        end

        # Find cutoff angle mu0
        max_mu0 = 0.5 * (1.0 - cos(MAX_SOFT_ANGLE * π / 180.0))
        local mu0::Float64
        local lb_h::Float64

        if cs_h < cs_m
            zeta = cs_h / cs_A
            mu_max = A_min * (1.0 - zeta) / (A_min + zeta)
            if mu_max > max_mu0
                mu_max = max_mu0
            end

            objective = function(mu)
                cs_tot = 0.0
                for iel in 1:n_elements
                    ed = elem_data[iel]
                    cs_tot += ed.norm * restricted_cs(mu, ed.fspin, ed.n_parameters,
                                                      ed.screening, ed.a, ed.b, ed.c)
                end
                return cs_tot - cs_h
            end

            fmax = objective(mu_max)
            fmin = cs_m - cs_h
            mu0 = ridder_find_root(objective, 0.0, mu_max, fmin, fmax;
                                   xtol=1e-6 * mu_max, rtol=1e-6, maxiter=100)
            if mu0 > max_mu0
                mu0 = max_mu0
            end
            cs_h += objective(mu0)
            lb_h = cs_h <= 1.0 / EHS_PATH_MAX ? EHS_PATH_MAX : 1.0 / cs_h
        else
            mu0 = 0.0
            lb_h = 1.0 / cs_m
        end

        screening_param[ie] = mu0

        # Store Lb = lb_h * p² (PUMAS convention: divide by p² at runtime)
        p2 = K * (K + 2.0 * mass)
        elastic_path[ie] = lb_h * p2

        # Compute soft scattering (inverse first transport path)
        invlb1 = 0.0
        for iel in 1:n_elements
            ed = elem_data[iel]
            _, G1 = transport_coefficients(mu0, ed.fspin, ed.n_parameters,
                                           ed.screening, ed.a, ed.b, ed.c)
            invlb1 += ed.norm * ed.d2 * G1
        end

        # Add radiative transport contributions
        invlb1_rad = 0.0
        for iel in 1:n_elements
            elem = material.elements[iel]
            frac = material.fractions[iel]
            Z = Float64(elem.Z)
            Ae = Float64(elem.A)
            _, invlb1_mixed = compute_element_soft_scattering(Z, Ae, mass, K, cutoff)
            invlb1_rad += invlb1_mixed * frac
        end

        # Add electronic transport
        nu = cutoff * K
        invlb1_elec = transverse_transport_electronic(ZoA, I_gev, aS, mass, K, nu)

        transport_path[ie] = (invlb1 + invlb1_rad + invlb1_elec) > 0.0 ?
            1.0 / (invlb1 + invlb1_rad + invlb1_elec) : 0.0
    end
end

end # module Coulomb
