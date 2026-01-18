"""
    Pumas

Shared utilities for DiffPumas example scripts.

This module provides common functions used across multiple examples:
- `zenith_to_elevation`: Angle conversion
- `load_or_create_physics`: Physics table loading/creation
- `compute_gaisser_flux_grid`: Analytical Gaisser flux computation
- `sample_energy_loguniform`: Log-uniform energy sampling (as in PUMAS geometry.c)
"""
module Pumas

using ..Constants
using ..Physics
using ..Loader
using ..GaisserFlux: flux_gaisser

export zenith_to_elevation, load_or_create_physics, compute_gaisser_flux_grid
export compute_gaisser_flux_integrated, sample_energy_loguniform

"""
    zenith_to_elevation(zenith)

Convert zenith angle to elevation angle (both in degrees).
    elevation = 90° - zenith

Convention:
- zenith = 0°   → elevation = 90°  (vertical, muons from directly above)
- zenith = 90°  → elevation = 0°   (horizontal, muons from horizon)
"""
zenith_to_elevation(zenith::Float64) = 90.0 - zenith

"""
    load_or_create_physics(dump_path; verbose=true)

Load physics tables from dump file, or create new ones if not found.

# Arguments
- `dump_path`: Path to the .pumas binary dump file
- `verbose`: Print status messages (default: true)

# Returns
- PhysicsTables object, or nothing if loading fails
"""
function load_or_create_physics(dump_path::String; verbose::Bool=true)
    if isfile(dump_path)
        verbose && println("Loading physics from binary dump...")
        verbose && println("  Dump file: $(dump_path)")
        physics = load_physics(dump_path)
        if physics !== nothing
            verbose && println("✓ Physics loaded successfully")
            return physics
        end
    end
    
    verbose && println("Physics dump not found, creating new physics tables...")
    physics = create_physics(MUON; n_energies=200, K_min=1e-3, K_max=1e9)
    mkpath(dirname(dump_path))
    save_physics(physics, dump_path)
    verbose && println("✓ Physics tables created and saved to $(dump_path)")
    return physics
end

"""
    compute_gaisser_flux_grid(zeniths, energies)

Compute Gaisser flux (analytical, no simulation) for a grid of zenith angles and energies.
This represents the atmospheric muon flux at sea level with no rock.

# Arguments
- `zeniths`: Vector of zenith angles in degrees
- `energies`: Vector of energy values in GeV

# Returns
- 2D array: flux[zenith_idx, energy_idx] in GeV⁻¹ m⁻² s⁻¹ sr⁻¹
"""
function compute_gaisser_flux_grid(zeniths::Vector{Float64}, energies::Vector{Float64})
    n_z = length(zeniths)
    n_e = length(energies)
    
    flux_grid = zeros(n_z, n_e)
    
    for (z_idx, zenith) in enumerate(zeniths)
        # Convert zenith angle (degrees) to cos(θ)
        # θ = 0° → cos(θ) = 1 (vertical, maximum flux)
        # θ = 90° → cos(θ) = 0 (horizontal, minimum flux)
        cos_theta = cosd(zenith)
        
        for (e_idx, energy) in enumerate(energies)
            # Gaisser flux for total charge (charge=0)
            flux_grid[z_idx, e_idx] = flux_gaisser(cos_theta, energy, 0.0)
        end
    end
    
    return flux_grid
end

"""
    compute_gaisser_flux_integrated(zeniths; energy_min=1e-3, energy_max=1e9, n_points=100)

Compute integrated Gaisser flux over energy range for each zenith angle.
Uses numerical integration with log-uniform sampling.

# Arguments
- `zeniths`: Vector of zenith angles in degrees
- `energy_min`: Minimum energy in GeV (default: 1e-3)
- `energy_max`: Maximum energy in GeV (default: 1e9)
- `n_points`: Number of integration points (default: 100)

# Returns
- Vector: integrated flux for each zenith angle in m⁻² s⁻¹ sr⁻¹
"""
function compute_gaisser_flux_integrated(zeniths::Vector{Float64};
                                          energy_min::Float64 = 1e-3,
                                          energy_max::Float64 = 1e9,
                                          n_points::Int = 100)
    n_z = length(zeniths)
    flux_integrated = zeros(n_z)
    
    # Log-uniform energy grid for integration
    log_e_min = log10(energy_min)
    log_e_max = log10(energy_max)
    log_energies = range(log_e_min, log_e_max, length=n_points+1)
    
    for (z_idx, zenith) in enumerate(zeniths)
        cos_theta = cosd(zenith)
        
        # Numerical integration using trapezoidal rule in log-space
        integral = 0.0
        for i in 1:n_points
            e1 = 10.0^log_energies[i]
            e2 = 10.0^log_energies[i+1]
            e_mid = sqrt(e1 * e2)  # Geometric mean
            de = e2 - e1
            
            # Gaisser flux at midpoint times bin width
            flux_mid = flux_gaisser(cos_theta, e_mid, 0.0)
            integral += flux_mid * de
        end
        
        flux_integrated[z_idx] = integral
    end
    
    return flux_integrated
end

"""
    sample_energy_loguniform(energy_min, energy_max, rng)

Sample a kinetic energy uniformly in log-space and return (energy, weight).

This matches the PUMAS geometry.c sampling:
    rk = log(energy_max / energy_min)
    kf = energy_min * exp(rk * random)
    wf = kf * rk

The weight compensates for the non-uniform sampling PDF.

# Arguments
- `energy_min`: Minimum energy in GeV
- `energy_max`: Maximum energy in GeV
- `rng`: Random number generator

# Returns
- `(energy, weight)`: Sampled energy and its MC weight
"""
function sample_energy_loguniform(energy_min::Float64, energy_max::Float64, rng)
    if energy_max <= energy_min
        # Point estimate - no energy integration
        return energy_min, 1.0
    end
    
    rk = log(energy_max / energy_min)
    u = rand(rng)
    kf = energy_min * exp(rk * u)
    wf = kf * rk
    
    return kf, wf
end

end # module Pumas