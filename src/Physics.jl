"""
    Physics

Physics tables for Monte Carlo transport.
Provides tabulated and interpolated physics properties.

Now supports loading precise tables from PUMAS material files (dE/dx tables).
"""
module Physics

using ..Constants
using ..Types
using ..Materials
using ..Materials: compute_dcs_integral
using ..DEDXLoader
using ..Coulomb: compute_coulomb_tables!
using LinearAlgebra
using ChainRulesCore
using Random

export PhysicsSettings, PhysicsTables
export create_physics, create_composite_table
export property_range, property_stopping_power
export property_kinetic_energy, property_proper_time
export property_cross_section, property_transport_path
export property_straggling, property_elastic_path
export property_screening, property_spin_factor, mixture_ZoA
export interpolate_table, get_material_index
export load_physics_from_dedx

"""
    PhysicsSettings

Configuration for physics tables.
"""
struct PhysicsSettings
    cutoff::Float64           # Relative cutoff between soft/hard energy losses
    elastic_ratio::Float64    # Ratio for elastic scattering
    bremsstrahlung::String    # Model name
    pair_production::String   # Model name
    photonuclear::String      # Model name
    dedx_path::String         # Path to PUMAS materials dE/dx files
end

function PhysicsSettings(;
    cutoff::Float64 = DEFAULT_CUTOFF,
    elastic_ratio::Float64 = DEFAULT_ELASTIC_RATIO,
    bremsstrahlung::String = "SSR",
    pair_production::String = "SSR",
    photonuclear::String = "DRSS",
    dedx_path::String = PUMAS_MATERIALS_PATH
)
    PhysicsSettings(cutoff, elastic_ratio, bremsstrahlung, pair_production, photonuclear, dedx_path)
end

"""
    MaterialTable{T}

Tabulated physics for a single material.
"""
struct MaterialTable{T<:Real}
    name::String
    density::T                    # Reference density in kg/m³
    I::T                          # Mean excitation energy in GeV
    ZoA::T                        # Effective Z/A
    
    # Energy grid
    energies::Vector{T}           # Kinetic energies in GeV
    log_energies::Vector{T}       # Precomputed log(energies) for fast interpolation
    
    # CSDA tables
    csda_stopping_power::Vector{T}    # GeV/(kg/m²)
    csda_range::Vector{T}             # kg/m²
    csda_proper_time::Vector{T}       # kg/m²
    
    # Mixed tables (with cutoff)
    mixed_stopping_power::Vector{T}
    mixed_range::Vector{T}
    
    # Straggling (energy loss variance)
    straggling::Vector{T}             # GeV²/(kg/m²)
    
    # Component stopping powers for mixed mode
    ionization::Vector{T}             # GeV/(kg/m²)
    bremsstrahlung::Vector{T}         # GeV/(kg/m²)
    pair_production::Vector{T}        # GeV/(kg/m²)
    photonuclear::Vector{T}           # GeV/(kg/m²)
    
    # Other properties
    cross_section::Vector{T}          # m²/kg
    transport_path::Vector{T}         # kg/m²
    elastic_path::Vector{T}           # kg/m²
    elastic_cutoff::Vector{T}         # rad
    magnetic_rotation::Vector{T}      # rad⋅kg/m³/T
    
    # Coulomb scattering parameters
    screening_parameter::Vector{T}    # Molière screening angle
    spin_factor::Vector{T}            # Spin correction factor
end

"""
    PhysicsTables{T}

Complete physics tables for all materials.
"""
struct PhysicsTables{T<:Real}
    particle::Particle
    mass::T                       # Particle mass in GeV/c²
    ctau::T                       # Decay length in m
    settings::PhysicsSettings
    materials::Dict{String, Int}  # Material name -> index mapping
    tables::Vector{MaterialTable{T}}
end

"""
    create_energy_grid(n_energies, K_min, K_max)

Create logarithmic energy grid.
"""
function create_energy_grid(n_energies::Int, K_min::T, K_max::T) where T<:Real
    return T[K_min * (K_max / K_min)^((i-1)/(n_energies-1)) for i in 1:n_energies]
end

"""
    coulomb_spin_factor(mass, kinetic)

Compute the spin correction factor for Coulomb scattering.
Matches PUMAS coulomb_spin_factor().
"""
function coulomb_spin_factor(mass::T, kinetic::T) where T<:Real
    e = kinetic + mass
    return kinetic * (e + mass) / (e * e)
end

"""
    create_material_table_from_dedx(dedx_data, material, particle, mass, settings)

Create a MaterialTable from loaded PUMAS dE/dx data.
This provides precise physics tables matching PUMAS exactly.
"""
function create_material_table_from_dedx(dedx::DEDXData, material::BaseMaterial, 
                                          particle::Particle, mass::T,
                                          settings::PhysicsSettings) where T<:Real
    n = length(dedx.kinetic_energy)
    energies = T.(dedx.kinetic_energy)
    
    # Direct from PUMAS tables
    csda_stopping = T.(dedx.stopping_power)
    csda_range = T.(dedx.csda_range)
    ionization = T.(dedx.ionization)
    brems = T.(dedx.bremsstrahlung)
    pair = T.(dedx.pair_production)
    photo = T.(dedx.photonuclear)
    
    # Compute proper time by integration
    csda_time = zeros(T, n)
    for i in 2:n
        K = energies[i]
        K_prev = energies[i-1]
        gamma = one(T) + K / mass
        gamma_prev = one(T) + K_prev / mass
        β = sqrt(one(T) - one(T) / gamma^2)
        β_prev = sqrt(one(T) - one(T) / gamma_prev^2)
        
        # Average velocity factor
        gamma_mid = (gamma + gamma_prev) / 2
        β_mid = (β + β_prev) / 2
        
        # Proper time integral: dτ = dK / (dE/dX * β * γ)
        dedx_avg = (csda_stopping[i] + csda_stopping[i-1]) / 2
        dK = K - K_prev
        csda_time[i] = csda_time[i-1] + dK / (dedx_avg * β_mid * gamma_mid)
    end
    
    # Mixed stopping power and straggling via per-element DCS integration.
    # mixed_dEdX = CSDA_dEdX - Σ hard DEL energy loss above cutoff
    # straggling = Σ per-element electronic straggling (from ionisation DCS mode=2)
    cutoff = settings.cutoff
    mixed_stopping = zeros(T, n)
    mixed_range = zeros(T, n)
    straggling = zeros(T, n)
    npts_mix = 180

    for i in 1:n
        K = Float64(energies[i])
        hard_loss = 0.0
        strag_val = 0.0
        if K >= 1e-3
            for (elem, frac) in zip(material.elements, material.fractions)
                for proc in (:bremsstrahlung, :pair_production, :photonuclear)
                    hard_loss += frac * compute_dcs_integral(proc, elem.Z, elem.A, elem.I,
                        Float64(mass), K, Float64(cutoff), 1.0, 1; npoints=npts_mix)
                end
                hard_loss += frac * compute_dcs_integral(:ionisation, elem.Z, elem.A, elem.I,
                    Float64(mass), K, Float64(cutoff), 1.0, 1; npoints=npts_mix)
                strag_val += frac * compute_dcs_integral(:ionisation, elem.Z, elem.A, elem.I,
                    Float64(mass), K, 0.0, Float64(cutoff), 2; npoints=npts_mix)
            end
        end
        mixed_stopping[i] = max(csda_stopping[i] - T(hard_loss), T(1e-20))
        straggling[i] = T(strag_val)
    end

    mixed_range[1] = energies[1] / mixed_stopping[1]
    for i in 2:n
        dK = energies[i] - energies[i-1]
        dedx_avg = (mixed_stopping[i] + mixed_stopping[i-1]) / 2
        mixed_range[i] = mixed_range[i-1] + dK / dedx_avg
    end
    
    # Cross-section for hard DEL events
    # Computed via Gauss quadrature DCS integration per element (matching PUMAS compute_cel_and_del)
    cross_section = zeros(T, n)
    npts = 180
    for i in 1:n
        K = Float64(energies[i])
        K < 1e-3 && continue
        cs_total = 0.0
        for (elem, frac) in zip(material.elements, material.fractions)
            for proc in (:bremsstrahlung, :pair_production, :photonuclear)
                cs_total += frac * compute_dcs_integral(proc, elem.Z, elem.A, elem.I,
                    Float64(mass), K, Float64(cutoff), 1.0, 0; npoints=npts)
            end
            cs_total += frac * compute_dcs_integral(:ionisation, elem.Z, elem.A, elem.I,
                Float64(mass), K, Float64(cutoff), 1.0, 0; npoints=npts)
        end
        cross_section[i] = T(cs_total)
    end
    
    # Coulomb scattering tables — computed from first principles
    transport_path = zeros(T, n)
    elastic_path_arr = zeros(T, n)
    screening_param = zeros(T, n)
    spin_factor_arr = zeros(T, n)
    magnetic_rotation = zeros(T, n)
    elastic_cutoff_arr = zeros(T, n)

    compute_coulomb_tables!(material, Float64(mass),
        settings.elastic_ratio, settings.cutoff,
        Float64.(energies), Float64.(csda_range),
        Float64(dedx.ZoA), Float64(material.I), Float64(material.density),
        transport_path, elastic_path_arr, screening_param)

    for i in 1:n
        K = energies[i]
        gamma = one(T) + K / mass
        β² = one(T) - one(T) / gamma^2
        β = sqrt(β²)
        p = sqrt(K * (K + 2mass))
        spin_factor_arr[i] = coulomb_spin_factor(mass, K)
        mu0 = screening_param[i]
        elastic_cutoff_arr[i] = mu0 < T(1e-8) ? T(2) * sqrt(max(mu0, zero(T))) :
            acos(max(one(T) - T(2) * mu0, -one(T)))
        magnetic_rotation[i] = LARMOR_FACTOR / (β * p)
    end
    
    # Precompute log energies for fast interpolation
    log_energies = log.(energies)
    
    return MaterialTable{T}(
        material.name, material.density, material.I, T(dedx.ZoA),
        energies, log_energies, csda_stopping, csda_range, csda_time,
        mixed_stopping, mixed_range, straggling,
        ionization, brems, pair, photo,
        cross_section, transport_path, elastic_path_arr, elastic_cutoff_arr,
        magnetic_rotation, screening_param, spin_factor_arr
    )
end

"""
    create_material_table_empirical(material, particle, mass, energies, settings)

Create tabulated physics for a material using empirical formulas (fallback).
"""
function create_material_table_empirical(material::BaseMaterial, particle::Particle, 
                                         mass::T, energies::Vector{T},
                                         settings::PhysicsSettings) where T<:Real
    n = length(energies)
    
    # Precompute log energies for fast interpolation
    log_energies = log.(energies)
    
    # Initialize arrays
    csda_stopping = zeros(T, n)
    csda_range = zeros(T, n)
    csda_time = zeros(T, n)
    mixed_stopping = zeros(T, n)
    mixed_range = zeros(T, n)
    straggling = zeros(T, n)
    ionization = zeros(T, n)
    brems = zeros(T, n)
    pair = zeros(T, n)
    photo = zeros(T, n)
    cross_section = zeros(T, n)
    transport_path = zeros(T, n)
    elastic_path_arr = zeros(T, n)
    elastic_cutoff = zeros(T, n)
    magnetic_rotation = zeros(T, n)
    screening_param = zeros(T, n)
    spin_factor_arr = zeros(T, n)
    
    # Get material properties
    Z_eff = sum(e.Z * f for (e, f) in zip(material.elements, material.fractions))
    A_eff = sum(e.A * f for (e, f) in zip(material.elements, material.fractions))
    ZoA = material.ZoA
    
    # Compute component stopping powers, cross-sections, etc. via full DCS integration
    cutoff = settings.cutoff
    npts_emp = 180
    x_low = 1e-6   # lower integration bound as fraction of K

    for i in 1:n
        K_f = Float64(energies[i])
        K_t = energies[i]
        gamma = one(T) + K_t / mass
        β² = one(T) - one(T) / (gamma^2)
        β = sqrt(β²)
        p = sqrt(K_t * (K_t + 2mass))

        # Per-element DCS integration for full energy loss (mode=1, x: 0..1)
        ion_val = 0.0; brem_val = 0.0; pair_val = 0.0; photo_val = 0.0
        hard_loss = 0.0; cs_total = 0.0; strag_val = 0.0
        if K_f >= 1e-6
            for (elem, frac) in zip(material.elements, material.fractions)
                Zf = Float64(elem.Z); Af = Float64(elem.A); If = Float64(elem.I)
                mf = Float64(mass)
                ion_val += frac * compute_dcs_integral(:ionisation, Zf, Af, If, mf, K_f, x_low, 1.0, 1; npoints=npts_emp)
                brem_val += frac * compute_dcs_integral(:bremsstrahlung, Zf, Af, If, mf, K_f, x_low, 1.0, 1; npoints=npts_emp)
                pair_val += frac * compute_dcs_integral(:pair_production, Zf, Af, If, mf, K_f, x_low, 1.0, 1; npoints=npts_emp)
                photo_val += frac * compute_dcs_integral(:photonuclear, Zf, Af, If, mf, K_f, x_low, 1.0, 1; npoints=npts_emp)

                for proc in (:bremsstrahlung, :pair_production, :photonuclear, :ionisation)
                    hard_loss += frac * compute_dcs_integral(proc, Zf, Af, If, mf, K_f, Float64(cutoff), 1.0, 1; npoints=npts_emp)
                    cs_total += frac * compute_dcs_integral(proc, Zf, Af, If, mf, K_f, Float64(cutoff), 1.0, 0; npoints=npts_emp)
                end
                strag_val += frac * compute_dcs_integral(:ionisation, Zf, Af, If, mf, K_f, 0.0, Float64(cutoff), 2; npoints=npts_emp)
            end
        end

        # Electronic stopping from Fano model
        elec_stop = Float64(electronic_stopping_power(material, Float64(mass), K_f))
        ionization[i] = T(elec_stop)
        brems[i] = T(brem_val)
        pair[i] = T(pair_val)
        photo[i] = T(photo_val)
        csda_stopping[i] = T(elec_stop + brem_val + pair_val + photo_val)
        mixed_stopping[i] = max(csda_stopping[i] - T(hard_loss), T(1e-20))
        straggling[i] = T(strag_val)
        cross_section[i] = T(cs_total)

        spin_factor_arr[i] = coulomb_spin_factor(mass, K_t)
        magnetic_rotation[i] = LARMOR_FACTOR / (β * p)
    end

    # Compute ranges by integration
    csda_range[1] = energies[1] / csda_stopping[1]
    mixed_range[1] = energies[1] / mixed_stopping[1]
    for i in 2:n
        dK = energies[i] - energies[i-1]
        dedx_avg = (csda_stopping[i] + csda_stopping[i-1]) / 2
        csda_range[i] = csda_range[i-1] + dK / dedx_avg
        dedx_mixed_avg = (mixed_stopping[i] + mixed_stopping[i-1]) / 2
        mixed_range[i] = mixed_range[i-1] + dK / dedx_mixed_avg
    end

    # Compute proper time by integration
    for i in 2:n
        K = energies[i]
        K_prev = energies[i-1]
        gamma = one(T) + K / mass
        gamma_prev = one(T) + K_prev / mass
        β = sqrt(one(T) - one(T) / gamma^2)
        β_prev = sqrt(one(T) - one(T) / gamma_prev^2)
        gamma_mid = (gamma + gamma_prev) / 2
        β_mid = (β + β_prev) / 2
        dedx_avg = (csda_stopping[i] + csda_stopping[i-1]) / 2
        dK = K - K_prev
        csda_time[i] = csda_time[i-1] + dK / (dedx_avg * β_mid * gamma_mid)
    end

    # Coulomb scattering tables — computed from first principles
    ZoA_val = Float64(ZoA)
    compute_coulomb_tables!(material, Float64(mass),
        settings.elastic_ratio, settings.cutoff,
        Float64.(energies), Float64.(csda_range),
        ZoA_val, Float64(material.I), Float64(material.density),
        transport_path, elastic_path_arr, screening_param)

    for i in 1:n
        mu0 = screening_param[i]
        elastic_cutoff[i] = mu0 < T(1e-8) ? T(2) * sqrt(max(mu0, zero(T))) :
            acos(max(one(T) - T(2) * mu0, -one(T)))
    end
    
    return MaterialTable{T}(
        material.name, material.density, material.I, ZoA,
        energies, log_energies, csda_stopping, csda_range, csda_time,
        mixed_stopping, mixed_range, straggling,
        ionization, brems, pair, photo,
        cross_section, transport_path, elastic_path_arr, elastic_cutoff,
        magnetic_rotation, screening_param, spin_factor_arr
    )
end

"""
    create_composite_table(composite, component_tables, component_fractions, particle, mass, settings)

Build a `MaterialTable` for a composite material by weighted combination of
already-computed component tables, matching PUMAS C `compute_composite_tables`.

Stopping power, straggling, cross-section, and per-process stopping powers are
mass-fraction-weighted sums.  Range and proper time are reintegrated from the
combined stopping power.  Coulomb scattering is computed from the flattened
atomic composition via `compute_coulomb_tables!`.
"""
function create_composite_table(composite::CompositeMaterial,
                                component_tables::Vector{MaterialTable{T}},
                                component_fractions::Vector{Float64},
                                particle::Particle, mass::T,
                                settings::PhysicsSettings) where T<:Real

    ref = component_tables[1]
    n = length(ref.energies)
    energies = copy(ref.energies)
    log_energies = copy(ref.log_energies)

    csda_stopping = zeros(T, n)
    mixed_stopping = zeros(T, n)
    straggling_arr = zeros(T, n)
    cross_section = zeros(T, n)
    ionization = zeros(T, n)
    brems = zeros(T, n)
    pair = zeros(T, n)
    photo = zeros(T, n)

    for (tbl, f) in zip(component_tables, component_fractions)
        fT = T(f)
        @inbounds for i in 1:n
            csda_stopping[i] += fT * tbl.csda_stopping_power[i]
            mixed_stopping[i] += fT * tbl.mixed_stopping_power[i]
            straggling_arr[i] += fT * tbl.straggling[i]
            cross_section[i] += fT * tbl.cross_section[i]
            ionization[i] += fT * tbl.ionization[i]
            brems[i] += fT * tbl.bremsstrahlung[i]
            pair[i] += fT * tbl.pair_production[i]
            photo[i] += fT * tbl.photonuclear[i]
        end
    end

    csda_range = zeros(T, n)
    csda_range[1] = energies[1] / csda_stopping[1]
    for i in 2:n
        dK = energies[i] - energies[i-1]
        dedx_avg = (csda_stopping[i] + csda_stopping[i-1]) / 2
        csda_range[i] = csda_range[i-1] + dK / dedx_avg
    end

    mixed_range = zeros(T, n)
    mixed_range[1] = energies[1] / mixed_stopping[1]
    for i in 2:n
        dK = energies[i] - energies[i-1]
        dedx_avg = (mixed_stopping[i] + mixed_stopping[i-1]) / 2
        mixed_range[i] = mixed_range[i-1] + dK / dedx_avg
    end

    csda_time = zeros(T, n)
    for i in 2:n
        K = energies[i]
        K_prev = energies[i-1]
        gamma = one(T) + K / mass
        gamma_prev = one(T) + K_prev / mass
        beta = sqrt(one(T) - one(T) / gamma^2)
        beta_prev = sqrt(one(T) - one(T) / gamma_prev^2)
        gamma_mid = (gamma + gamma_prev) / 2
        beta_mid = (beta + beta_prev) / 2
        dedx_avg = (csda_stopping[i] + csda_stopping[i-1]) / 2
        dK = K - K_prev
        csda_time[i] = csda_time[i-1] + dK / (dedx_avg * beta_mid * gamma_mid)
    end

    flat_mat = flatten_to_base_material(composite)
    density = flat_mat.density
    I_val = flat_mat.I
    ZoA_val = flat_mat.ZoA

    transport_path = zeros(T, n)
    elastic_path_arr = zeros(T, n)
    screening_param = zeros(T, n)
    spin_factor_arr = zeros(T, n)
    magnetic_rotation = zeros(T, n)
    elastic_cutoff_arr = zeros(T, n)

    compute_coulomb_tables!(flat_mat, Float64(mass),
        settings.elastic_ratio, settings.cutoff,
        Float64.(energies), Float64.(csda_range),
        Float64(ZoA_val), Float64(I_val), Float64(density),
        transport_path, elastic_path_arr, screening_param)

    for i in 1:n
        K = energies[i]
        gamma = one(T) + K / mass
        beta_sq = one(T) - one(T) / gamma^2
        beta = sqrt(beta_sq)
        p = sqrt(K * (K + 2mass))
        spin_factor_arr[i] = coulomb_spin_factor(mass, K)
        mu0 = screening_param[i]
        elastic_cutoff_arr[i] = mu0 < T(1e-8) ? T(2) * sqrt(max(mu0, zero(T))) :
            acos(max(one(T) - T(2) * mu0, -one(T)))
        magnetic_rotation[i] = LARMOR_FACTOR / (beta * p)
    end

    return MaterialTable{T}(
        composite.name, density, I_val, T(ZoA_val),
        energies, log_energies, csda_stopping, csda_range, csda_time,
        mixed_stopping, mixed_range, straggling_arr,
        ionization, brems, pair, photo,
        cross_section, transport_path, elastic_path_arr, elastic_cutoff_arr,
        magnetic_rotation, screening_param, spin_factor_arr
    )
end

"""
    create_physics(particle; kwargs...)

Create physics tables for a given particle type.
Automatically loads PUMAS dE/dx tables if available.
Composites get their own precomputed tables (matching PUMAS C).
"""
function create_physics(particle::Particle = MUON;
                        n_energies::Int = 200,
                        K_min::Float64 = 1e-3,  # GeV
                        K_max::Float64 = 1e9,   # GeV
                        materials::Vector{BaseMaterial} = [STANDARD_ROCK, AIR],
                        composites::Vector{CompositeMaterial} = CompositeMaterial[],
                        settings::PhysicsSettings = PhysicsSettings())
    
    mass, ctau = if particle == MUON
        MUON_MASS, MUON_C_TAU
    else
        TAU_MASS, TAU_C_TAU
    end
    
    material_dict = Dict{String, Int}()
    tables = MaterialTable{Float64}[]
    
    particle_sym = particle == MUON ? :muon : :tau
    
    for (i, mat) in enumerate(materials)
        material_dict[mat.name] = i
        
        dedx_path = find_dedx_file(mat.name, particle_sym; base_path=settings.dedx_path)
        
        if isfile(dedx_path)
            try
                dedx = load_dedx_file(dedx_path)
                @info "Loaded PUMAS tables for $(mat.name) from $dedx_path"
                push!(tables, create_material_table_from_dedx(dedx, mat, particle, mass, settings))
            catch e
                @warn "Failed to load PUMAS tables for $(mat.name): $e. Using empirical formulas."
                energies = create_energy_grid(n_energies, K_min, K_max)
                push!(tables, create_material_table_empirical(mat, particle, mass, energies, settings))
            end
        else
            @warn "PUMAS dE/dx file not found for $(mat.name) at $dedx_path. Using empirical formulas."
            energies = create_energy_grid(n_energies, K_min, K_max)
            push!(tables, create_material_table_empirical(mat, particle, mass, energies, settings))
        end
    end
    
    # Build precomputed tables for composites (matching PUMAS C)
    for comp in composites
        comp_tables = MaterialTable{Float64}[]
        for base_mat in comp.components
            idx = get(material_dict, base_mat.name, nothing)
            if idx === nothing
                error("Component material \"$(base_mat.name)\" of composite " *
                      "\"$(comp.name)\" not found in base materials. " *
                      "Available: $(collect(keys(material_dict)))")
            end
            push!(comp_tables, tables[idx])
        end
        comp_idx = length(tables) + 1
        material_dict[comp.name] = comp_idx
        @info "Building composite table for $(comp.name) " *
              "($(join(["$(c.name) $(round(f*100;digits=1))%" for (c,f) in zip(comp.components, comp.fractions)], " + ")))"
        push!(tables, create_composite_table(comp, comp_tables, comp.fractions,
                                              particle, mass, settings))
    end
    
    return PhysicsTables{Float64}(particle, mass, ctau, settings, material_dict, tables)
end


"""
    get_material_index(physics, name)

Get material index from name.
"""
function get_material_index(physics::PhysicsTables, name::String)
    return get(physics.materials, name, -1)
end

"""
    interpolate_table(x, xs, ys)

Linear interpolation with logarithmic x-axis.
Zygote-compatible implementation.

PERFORMANCE: Uses binary search, @inline, and @inbounds for speed.
Note: For hot paths, prefer interpolate_table_fast with precomputed log values.
"""
@inline function interpolate_table(x::T, xs::Vector{T}, ys::Vector{T}) where T<:Real
    n = length(xs)
    
    # Clamp to table bounds
    @inbounds if x <= xs[1]
        return ys[1]
    elseif x >= xs[n]
        return ys[n]
    end
    
    # Binary search for interval (much faster than linear search)
    log_x = log(x)
    lo, hi = 1, n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if log_x < log(xs[mid])
            hi = mid
        else
            lo = mid
        end
    end
    i = lo
    
    # Linear interpolation in log-space for x, linear for y
    @inbounds begin
        log_xi = log(xs[i])
        log_xi1 = log(xs[i+1])
        t = (log_x - log_xi) / (log_xi1 - log_xi)
        return ys[i] + t * (ys[i+1] - ys[i])
    end
end

"""
    interpolate_table_fast(x, log_x, xs, log_xs, ys)

Optimized linear interpolation using precomputed log values.
Avoids expensive log() calls in the binary search loop.

PERFORMANCE: ~5x faster than interpolate_table for hot paths.
"""
@inline function interpolate_table_fast(x::T, log_x::T, xs::Vector{T}, log_xs::Vector{T}, ys::Vector{T}) where T<:Real
    n = length(xs)
    
    # Clamp to table bounds
    @inbounds if x <= xs[1]
        return ys[1]
    elseif x >= xs[n]
        return ys[n]
    end
    
    # Binary search using precomputed log values - O(log n) with no log() calls
    lo, hi = 1, n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if log_x < log_xs[mid]
            hi = mid
        else
            lo = mid
        end
    end
    i = lo
    
    # Linear interpolation using precomputed log values
    @inbounds begin
        t = (log_x - log_xs[i]) / (log_xs[i+1] - log_xs[i])
        return ys[i] + t * (ys[i+1] - ys[i])
    end
end

# Make interpolate_table differentiable
function ChainRulesCore.rrule(::typeof(interpolate_table), x::T, xs::Vector{T}, ys::Vector{T}) where T
    y = interpolate_table(x, xs, ys)
    
    function interpolate_pullback(Δy)
        n = length(xs)
        
        @inbounds if x <= xs[1] || x >= xs[n]
            return (NoTangent(), zero(T), NoTangent(), NoTangent())
        end
        
        # Binary search for interval (same as forward pass)
        log_x = log(x)
        lo, hi = 1, n
        @inbounds while lo < hi - 1
            mid = (lo + hi) >> 1
            if log_x < log(xs[mid])
                hi = mid
            else
                lo = mid
            end
        end
        i = lo
        
        # Derivative w.r.t. x
        @inbounds begin
            Δlog_xs = log(xs[i+1]) - log(xs[i])
            dy_dlogx = (ys[i+1] - ys[i]) / Δlog_xs
            dlogx_dx = one(T) / x
        end
        
        return (NoTangent(), Δy * dy_dlogx * dlogx_dx, NoTangent(), NoTangent())
    end
    
    return y, interpolate_pullback
end

"""
    property_range(physics, mode, material, energy)

Get CSDA range for given energy.
"""
@inline function property_range(physics::PhysicsTables{T}, mode::EnergyLossMode,
                        material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    range_table = mode == ENERGY_LOSS_MIXED ? table.mixed_range : table.csda_range
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, range_table)
end

"""
    property_stopping_power(physics, mode, material, energy)

Get stopping power for given energy.
"""
@inline function property_stopping_power(physics::PhysicsTables{T}, mode::EnergyLossMode,
                                 material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    dedx_table = mode == ENERGY_LOSS_MIXED ? table.mixed_stopping_power : table.csda_stopping_power
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, dedx_table)
end

"""
    property_straggling(physics, material, energy)

Get energy straggling variance for given energy.
Returns Ω in GeV²/(kg/m²).

The straggling represents the variance of energy loss per unit grammage.
"""
@inline function property_straggling(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.straggling)
end

"""
    property_kinetic_energy(physics, mode, material, range)

Get kinetic energy for given range (inverse of range function).

PERFORMANCE: Uses binary search for O(log n) lookup.
"""
@inline function property_kinetic_energy(physics::PhysicsTables{T}, mode::EnergyLossMode,
                                 material::Int, range::T) where T<:Real
    @inbounds table = physics.tables[material]
    range_table = mode == ENERGY_LOSS_MIXED ? table.mixed_range : table.csda_range
    
    # Inverse interpolation
    n = length(range_table)
    @inbounds if range <= range_table[1]
        return table.energies[1]
    elseif range >= range_table[n]
        return table.energies[n]
    end
    
    # Binary search for interval (range is monotonically increasing)
    lo, hi = 1, n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if range < range_table[mid]
            hi = mid
        else
            lo = mid
        end
    end
    i = lo
    
    @inbounds begin
        t = (range - range_table[i]) / (range_table[i+1] - range_table[i])
        # Avoid log/exp: use linear interpolation in log-space more efficiently
        log_E_i = log(table.energies[i])
        log_E_i1 = log(table.energies[i+1])
        log_E = log_E_i + t * (log_E_i1 - log_E_i)
    end
    
    return exp(log_E)
end

"""
    property_proper_time(physics, mode, material, energy)

Get integrated proper time for given energy.
"""
@inline function property_proper_time(physics::PhysicsTables{T}, mode::EnergyLossMode,
                              material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.csda_proper_time)
end

"""
    property_proper_time(physics, mode, mix::MaterialMixture, energy)

Mixture proper time from cached integrated table.
"""
@inline function property_proper_time(physics::PhysicsTables{T}, mode::EnergyLossMode,
                              mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_proper_time(physics, mode, single_material(mix), energy)
    end
    mtbl = get_mixture_table(physics, mix)
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, mtbl.energies, mtbl.log_energies, mtbl.csda_proper_time)
end

"""
    property_cross_section(physics, material, energy)

Get total cross-section for hard events.
"""
@inline function property_cross_section(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.cross_section)
end

"""
    property_transport_path(physics, material, energy)

Get transport mean free path for soft scattering.
"""
@inline function property_transport_path(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.transport_path)
end

"""
    property_elastic_path(physics, material, energy)

Get elastic scattering mean free path.
"""
@inline function property_elastic_path(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.elastic_path)
end

"""
    property_screening(physics, material, energy)

Get Coulomb screening parameter.
"""
@inline function property_screening(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.screening_parameter)
end

"""
    property_spin_factor(physics, material, energy)

Get spin correction factor for Coulomb scattering.
"""
@inline function property_spin_factor(physics::PhysicsTables{T}, material::Int, energy::T) where T<:Real
    @inbounds table = physics.tables[material]
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, table.energies, table.log_energies, table.spin_factor)
end

"""
    property_component_stopping(physics, material, energy, component)

Get component stopping power (ionization, bremsstrahlung, pair, photonuclear).
"""
@inline function property_component_stopping(physics::PhysicsTables{T}, material::Int, 
                                      energy::T, component::Symbol) where T<:Real
    @inbounds table = physics.tables[material]
    
    comp_table = if component == :ionization
        table.ionization
    elseif component == :bremsstrahlung
        table.bremsstrahlung
    elseif component == :pair_production
        table.pair_production
    elseif component == :photonuclear
        table.photonuclear
    else
        error("Unknown component: $component")
    end
    
    return interpolate_table(energy, table.energies, comp_table)
end

# =============================================================================
# Mixture-aware property functions
# =============================================================================
# Continuous properties (stopping power, straggling, cross-section) are
# fraction-weighted sums computed at query time.
#
# Range, inverse-range, and proper time require integrated tables.
# A MixtureTable is built lazily the first time one of these is needed
# for a given mixture, then cached globally.  The integration uses the
# same trapezoidal rule as create_composite_table / create_material_table_empirical,
# so the result is identical to a precomputed composite table.
# =============================================================================

"""
    MixtureTable{T}

Cached integrated tables for a `MaterialMixture`, built from the reference
energy grid of the underlying `PhysicsTables`.  Makes `property_range`,
`property_kinetic_energy`, and `property_proper_time` for mixtures exact
(same procedure as for precomputed composite / base material tables).
"""
struct MixtureTable{T<:Real}
    energies::Vector{T}
    log_energies::Vector{T}
    csda_range::Vector{T}
    mixed_range::Vector{T}
    csda_proper_time::Vector{T}
end

const _MIXTURE_TABLE_CACHE = Dict{Tuple{UInt64, Vector{Int}, Vector{Float64}}, MixtureTable{Float64}}()
const _MIXTURE_TABLE_LOCK  = ReentrantLock()

"""
    _mixture_cache_key(physics, mix)

Deterministic cache key: (objectid of physics, materials, fractions).
"""
@inline function _mixture_cache_key(physics::PhysicsTables{T}, mix::MaterialMixture) where T
    return (objectid(physics), mix.materials, mix.fractions)
end

"""
    build_mixture_table(physics, mix)

Build CSDA/mixed range and proper-time tables for a `MaterialMixture` by
trapezoidal integration of the mixture stopping power over the reference
energy grid — exactly the same procedure used for precomputed composite
and base-material tables.
"""
function build_mixture_table(physics::PhysicsTables{T}, mix::MaterialMixture) where T<:Real
    ref = physics.tables[1]
    n = length(ref.energies)
    energies = ref.energies
    log_energies = ref.log_energies
    mass = physics.mass

    csda_range = zeros(T, n)
    mixed_range = zeros(T, n)
    csda_time = zeros(T, n)

    csda_stop_prev = property_stopping_power(physics, ENERGY_LOSS_CSDA, mix, energies[1])
    mixed_stop_prev = property_stopping_power(physics, ENERGY_LOSS_MIXED, mix, energies[1])

    csda_range[1] = energies[1] / csda_stop_prev
    mixed_range[1] = energies[1] / mixed_stop_prev

    for i in 2:n
        K = energies[i]
        K_prev = energies[i-1]
        dK = K - K_prev

        csda_stop = property_stopping_power(physics, ENERGY_LOSS_CSDA, mix, K)
        mixed_stop = property_stopping_power(physics, ENERGY_LOSS_MIXED, mix, K)

        csda_range[i] = csda_range[i-1] + dK / ((csda_stop + csda_stop_prev) / 2)
        mixed_range[i] = mixed_range[i-1] + dK / ((mixed_stop + mixed_stop_prev) / 2)

        gamma = one(T) + K / mass
        gamma_prev = one(T) + K_prev / mass
        beta = sqrt(one(T) - one(T) / gamma^2)
        beta_prev = sqrt(one(T) - one(T) / gamma_prev^2)
        dedx_avg = (csda_stop + csda_stop_prev) / 2
        csda_time[i] = csda_time[i-1] + dK / (dedx_avg * ((beta + beta_prev) / 2) * ((gamma + gamma_prev) / 2))

        csda_stop_prev = csda_stop
        mixed_stop_prev = mixed_stop
    end

    return MixtureTable{T}(copy(energies), copy(log_energies),
                           csda_range, mixed_range, csda_time)
end

"""
    get_mixture_table(physics, mix)

Retrieve or build the cached `MixtureTable` for a given mixture.
Thread-safe via a ReentrantLock.
"""
function get_mixture_table(physics::PhysicsTables{T}, mix::MaterialMixture) where T<:Real
    key = _mixture_cache_key(physics, mix)
    tbl = get(_MIXTURE_TABLE_CACHE, key, nothing)
    tbl !== nothing && return tbl::MixtureTable{Float64}
    lock(_MIXTURE_TABLE_LOCK) do
        tbl = get(_MIXTURE_TABLE_CACHE, key, nothing)
        tbl !== nothing && return tbl::MixtureTable{Float64}
        tbl = build_mixture_table(physics, mix)
        _MIXTURE_TABLE_CACHE[key] = tbl
        return tbl
    end
end

"""
    property_stopping_power(physics, mode, mix::MaterialMixture, energy)

Mixture stopping power: weighted sum of per-material stopping powers.
    dEdX_mix = sum(f_i * dEdX_i(E))
"""
@inline function property_stopping_power(physics::PhysicsTables{T}, mode::EnergyLossMode,
                                 mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_stopping_power(physics, mode, single_material(mix), energy)
    end
    dedx = zero(T)
    @inbounds for j in eachindex(mix.materials)
        dedx += T(mix.fractions[j]) * property_stopping_power(physics, mode, mix.materials[j], energy)
    end
    return dedx
end

"""
    property_straggling(physics, mix::MaterialMixture, energy)

Mixture straggling variance: weighted sum of per-material straggling.
"""
@inline function property_straggling(physics::PhysicsTables{T}, mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_straggling(physics, single_material(mix), energy)
    end
    s = zero(T)
    @inbounds for j in eachindex(mix.materials)
        s += T(mix.fractions[j]) * property_straggling(physics, mix.materials[j], energy)
    end
    return s
end

"""
    property_cross_section(physics, mix::MaterialMixture, energy)

Mixture cross-section: weighted sum of per-material cross-sections.
"""
@inline function property_cross_section(physics::PhysicsTables{T}, mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_cross_section(physics, single_material(mix), energy)
    end
    xs = zero(T)
    @inbounds for j in eachindex(mix.materials)
        xs += T(mix.fractions[j]) * property_cross_section(physics, mix.materials[j], energy)
    end
    return xs
end

"""
    property_transport_path(physics, mix::MaterialMixture, energy)

Mixture transport path: inverse of weighted sum of inverse transport paths.
    1/λ_mix = sum(f_i / λ_i)
"""
@inline function property_transport_path(physics::PhysicsTables{T}, mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_transport_path(physics, single_material(mix), energy)
    end
    inv_path = zero(T)
    @inbounds for j in eachindex(mix.materials)
        lp = property_transport_path(physics, mix.materials[j], energy)
        if lp > zero(T)
            inv_path += T(mix.fractions[j]) / lp
        end
    end
    return inv_path > zero(T) ? one(T) / inv_path : T(1e9)
end

"""
    property_screening(physics, mix::MaterialMixture, energy)

Mixture screening parameter: fraction-weighted sum of per-material
screening parameters.  This replaces the old dominant-material
approximation and matches how a precomputed composite table stores
its screening parameter (computed from the flattened composition).
"""
@inline function property_screening(physics::PhysicsTables{T}, mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_screening(physics, single_material(mix), energy)
    end
    mu0 = zero(T)
    @inbounds for j in eachindex(mix.materials)
        mu0 += T(mix.fractions[j]) * property_screening(physics, mix.materials[j], energy)
    end
    return max(mu0, T(1e-12))
end

"""
    property_spin_factor(physics, mix::MaterialMixture, energy)

Mixture spin factor: fraction-weighted sum of per-material spin factors.
"""
@inline function property_spin_factor(physics::PhysicsTables{T}, mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_spin_factor(physics, single_material(mix), energy)
    end
    sf = zero(T)
    @inbounds for j in eachindex(mix.materials)
        sf += T(mix.fractions[j]) * property_spin_factor(physics, mix.materials[j], energy)
    end
    return sf
end

"""
    mixture_ZoA(physics, mix::MaterialMixture)

Effective Z/A for a mixture: fraction-weighted sum of per-material Z/A.
Used for the effective target mass in CM-frame scattering calculations.
"""
@inline function mixture_ZoA(physics::PhysicsTables{T}, mix::MaterialMixture) where T<:Real
    if is_single_material(mix)
        @inbounds return physics.tables[single_material(mix)].ZoA
    end
    zoa = zero(T)
    @inbounds for j in eachindex(mix.materials)
        zoa += T(mix.fractions[j]) * physics.tables[mix.materials[j]].ZoA
    end
    return zoa
end

"""
    property_range(physics, mode, mix::MaterialMixture, energy)

Mixture range from cached integrated table — same procedure as for
precomputed composite / base-material tables.
"""
@inline function property_range(physics::PhysicsTables{T}, mode::EnergyLossMode,
                        mix::MaterialMixture, energy::T) where T<:Real
    if is_single_material(mix)
        return property_range(physics, mode, single_material(mix), energy)
    end
    mtbl = get_mixture_table(physics, mix)
    range_table = mode == ENERGY_LOSS_MIXED ? mtbl.mixed_range : mtbl.csda_range
    log_energy = log(energy)
    return interpolate_table_fast(energy, log_energy, mtbl.energies, mtbl.log_energies, range_table)
end

"""
    property_kinetic_energy(physics, mode, mix::MaterialMixture, range)

Inverse range lookup for mixture via cached integrated table — same
binary-search + log-space interpolation as for single materials.
"""
@inline function property_kinetic_energy(physics::PhysicsTables{T}, mode::EnergyLossMode,
                                 mix::MaterialMixture, range::T) where T<:Real
    if is_single_material(mix)
        return property_kinetic_energy(physics, mode, single_material(mix), range)
    end
    mtbl = get_mixture_table(physics, mix)
    range_table = mode == ENERGY_LOSS_MIXED ? mtbl.mixed_range : mtbl.csda_range

    n = length(range_table)
    @inbounds if range <= range_table[1]
        return mtbl.energies[1]
    elseif range >= range_table[n]
        return mtbl.energies[n]
    end

    lo, hi = 1, n
    @inbounds while lo < hi - 1
        mid = (lo + hi) >> 1
        if range < range_table[mid]
            hi = mid
        else
            lo = mid
        end
    end
    i = lo

    @inbounds begin
        t = (range - range_table[i]) / (range_table[i+1] - range_table[i])
        log_E = mtbl.log_energies[i] + t * (mtbl.log_energies[i+1] - mtbl.log_energies[i])
    end

    return exp(log_E)
end

"""
    sample_mixture_material(physics, mix, energy, rng)

Sample which material an interaction occurs in, weighted by
fraction * cross-section. Returns the material index.
"""
@inline function sample_mixture_material(physics::PhysicsTables{T}, mix::MaterialMixture,
                                          energy::T, rng::AbstractRNG) where T<:Real
    if is_single_material(mix)
        return single_material(mix)
    end
    # Compute weighted cross-sections
    n = length(mix.materials)
    xs_total = zero(T)
    @inbounds for j in 1:n
        xs_j = T(mix.fractions[j]) * property_cross_section(physics, mix.materials[j], energy)
        xs_total += xs_j
    end
    if xs_total <= zero(T)
        # Fallback: sample proportional to fraction alone
        u = rand(rng)
        cumulative = zero(T)
        @inbounds for j in 1:n
            cumulative += T(mix.fractions[j])
            if u <= cumulative
                return mix.materials[j]
            end
        end
        return mix.materials[n]
    end
    # Sample from cumulative distribution
    u = rand(rng) * xs_total
    cumulative = zero(T)
    @inbounds for j in 1:n
        cumulative += T(mix.fractions[j]) * property_cross_section(physics, mix.materials[j], energy)
        if u <= cumulative
            return mix.materials[j]
        end
    end
    return mix.materials[n]
end

export sample_mixture_material

end # module Physics
