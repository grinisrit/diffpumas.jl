"""
    Constants

Physics constants used by PUMAS, in SI-like units with GeV for energy.
"""
module Constants

export ALPHA_EM, HBAR_C, BOHR_RADIUS, MUON_C_TAU, TAU_C_TAU, LARMOR_FACTOR
export ELECTRON_MASS, ELECTRON_RADIUS, MUON_MASS, TAU_MASS
export PROTON_MASS, NEUTRON_MASS, PION_MASS, AVOGADRO_NUMBER
export DEFAULT_CUTOFF, DEFAULT_ELASTIC_RATIO, DEFAULT_ACCURACY
export STEP_MIN, EHS_PATH_MAX, BMC_ALPHA, EPSILON_X
export N_DEL_PROCESSES, N_LARMOR_ORDERS, MAX_SOFT_ANGLE

# Physics constants

"""Fine-structure constant α_em ≈ 1/137"""
const ALPHA_EM = 7.2973525693e-3

"""Planck constant ℏc in GeV⋅m"""
const HBAR_C = 1.973269804e-16

"""Bohr radius a₀ in m"""
const BOHR_RADIUS = 0.529177210903e-10

"""Muon decay length cτ in m"""
const MUON_C_TAU = 658.654

"""Tau decay length cτ in m"""
const TAU_C_TAU = 87.03e-6

"""Larmor magnetic factor in m⁻¹ GeV/c T⁻¹"""
const LARMOR_FACTOR = 0.299792458

"""Electron mass in GeV/c²"""
const ELECTRON_MASS = 0.510998910e-3

"""Electron classical radius in m"""
const ELECTRON_RADIUS = 2.817940285e-15

"""Muon mass in GeV/c²"""
const MUON_MASS = 0.10565839

"""Tau mass in GeV/c²"""
const TAU_MASS = 1.77682

"""Proton mass in GeV/c²"""
const PROTON_MASS = 0.938272

"""Neutron mass in GeV/c²"""
const NEUTRON_MASS = 0.939565

"""Pion mass in GeV/c²"""
const PION_MASS = 0.13957018

"""Avogadro's number in mol⁻¹"""
const AVOGADRO_NUMBER = 6.02214076e23

# Tuning parameters

"""Default cutoff between CEL and DELs (5%)"""
const DEFAULT_CUTOFF = 5e-2

"""Default ratio for elastic scattering (5%)"""
const DEFAULT_ELASTIC_RATIO = 5e-2

"""Default accuracy for MC integration (1%)"""
const DEFAULT_ACCURACY = 1e-2

"""Minimum step size in m"""
const STEP_MIN = 1e-7

"""Maximum path length for EHS events in kg/m²"""
const EHS_PATH_MAX = 1e9

"""BMC differential cross section exponent"""
const BMC_ALPHA = 2.0

"""Grammage ratio for small steps in CSDA mode"""
const EPSILON_X = 3e-3

"""Number of DEL processes (Bremsstrahlung, pair production, photonuclear, delta rays)"""
const N_DEL_PROCESSES = 4

"""Order of expansion for magnetic deflection in CSDA"""
const N_LARMOR_ORDERS = 8

"""Maximum deflection angle for soft scattering in degrees"""
const MAX_SOFT_ANGLE = 1.0

end # module Constants

