"""
    GaisserFlux

Atmospheric muon flux models based on semi-analytical formulas.
Implements Gaisser's and GCCLY models.

This module is a direct Julia translation of:
- pumas/examples/pumas/flux.h
- pumas/examples/pumas/flux.c

References:
- Gaisser: PDG ch.30 (https://pdglive.lbl.gov)
- GCCLY: Guan et al. (https://arxiv.org/abs/1509.06176)
- Charge ratio: CMS (https://arxiv.org/abs/1005.5332)
"""
module GaisserFlux

using ..Constants: MUON_MASS

export flux_gaisser, flux_gccly, charge_fraction, cos_theta_star

"""
    charge_fraction(charge)

Fraction of the muon flux for a given charge.
Uses a constant charge ratio of 1.2766.

Ref: CMS (https://arxiv.org/abs/1005.5332)

# Arguments
- `charge`: Muon charge (-1, 0, or +1). Use 0 for total flux.

# Returns
Fraction of flux for the given charge.
"""
function charge_fraction(charge::T) where T<:Real
    charge_ratio = T(1.2766)
    
    if charge < zero(T)
        return one(T) / (one(T) + charge_ratio)
    elseif charge > zero(T)
        return charge_ratio / (one(T) + charge_ratio)
    else
        return one(T)
    end
end

"""
    cos_theta_star(cos_theta)

Volkova's parameterization of cos(θ*) for Earth curvature correction.

This accounts for the fact that the effective atmospheric depth depends
on the zenith angle in a non-trivial way due to Earth's curvature.

# Arguments
- `cos_theta`: Cosine of the zenith angle at observation point.

# Returns
Effective cos(θ*) for flux calculation.
"""
function cos_theta_star(cos_theta::T) where T<:Real
    # Volkova parameterization coefficients
    # Note: Using tuple (p0, p1, p2, p3, p4) matching C array indices
    p0, p1, p2, p3, p4 = T(0.102573), T(-0.068287), T(0.958633), T(0.0407253), T(0.817285)
    
    # Exact match to C implementation (no clamping - C doesn't clamp):
    # cs2 = (cos_theta^2 + p[0]^2 + p[1]*pow(cos_theta, p[2]) + p[3]*pow(cos_theta, p[4])) / 
    #       (1 + p[0]^2 + p[1] + p[3])
    cs2 = (cos_theta^2 + p0^2 + 
           p1 * cos_theta^p2 + 
           p3 * cos_theta^p4) / 
          (one(T) + p0^2 + p1 + p3)
    
    return cs2 > zero(T) ? sqrt(cs2) : zero(T)
end

"""
    flux_gaisser(cos_theta, kinetic_energy, charge)

Gaisser's flux model for atmospheric muons at sea level.

Ref: see e.g. ch.30 of the PDG (https://pdglive.lbl.gov)

# Arguments
- `cos_theta`: Cosine of the zenith angle
- `kinetic_energy`: Muon kinetic energy in GeV
- `charge`: Muon charge (-1, 0, or +1)

# Returns
Differential flux in m⁻² s⁻¹ sr⁻¹ GeV⁻¹
"""
function flux_gaisser(cos_theta::T, kinetic_energy::T, charge::T) where T<:Real
    # Total energy
    Emu = kinetic_energy + T(MUON_MASS)
    
    # Critical energies for pion and kaon contributions
    ec = T(1.1) * Emu * cos_theta
    rpi = one(T) + ec / T(115.0)   # Pion critical energy ~115 GeV
    rK = one(T) + ec / T(850.0)    # Kaon critical energy ~850 GeV
    
    # Gaisser formula: 1.4E+03 * E^(-2.7) * (1/rpi + 0.054/rK)
    return T(1.4e3) * Emu^(-T(2.7)) * (one(T) / rpi + T(0.054) / rK) * 
           charge_fraction(charge)
end

"""
    flux_gccly(cos_theta, kinetic_energy, charge)

Guan et al. (GCCLY) parameterization of sea level atmospheric muon flux.

This is an improved parameterization that provides better agreement with
measurements, especially at large zenith angles.

Ref: https://arxiv.org/abs/1509.06176

# Arguments
- `cos_theta`: Cosine of the zenith angle
- `kinetic_energy`: Muon kinetic energy in GeV
- `charge`: Muon charge (-1, 0, or +1)

# Returns
Differential flux in m⁻² s⁻¹ sr⁻¹ GeV⁻¹
"""
function flux_gccly(cos_theta::T, kinetic_energy::T, charge::T) where T<:Real
    Emu = kinetic_energy + T(MUON_MASS)
    cs = cos_theta_star(cos_theta)
    
    # GCCLY modification factor
    gccly_factor = (one(T) + T(3.64) / (Emu * cs^T(1.29)))^(-T(2.7))
    
    return gccly_factor * flux_gaisser(cs, kinetic_energy, charge)
end

end # module GaisserFlux

