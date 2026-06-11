"""
    Tomography

Absorption muography tomography for DiffPumas: a differentiable CSDA forward
model that maps a per-cell rock/water field `w` (water fraction in `[0, 0.9]`)
to the directional muon flux reaching a deep detector, plus a **sparse**
sensitivity (Jacobian) assembled with reverse-mode AD, and inverse solvers that
recover `w` from the observed flux.

This is the reusable core extracted from `examples/lvd_tomography.jl`. The example
keeps all geometry building (TetGen / voxel meshing, ray tracing, topography) and
plotting; this module owns the physics forward model, the AD Jacobian, the
inverse solvers, the image-quality metrics, and the resolution estimator.

# Modality note
The inverse solvers here target **transmission/absorption** muography: the
observable is the directional muon flux/rate, and the unknown is the per-cell
rock/water proportion, recovered through flux attenuation. The scattering
tomography methods surveyed in `docs/muon_tomography_overview.md`
(μTRec, mPoCA) require per-muon *tracking* (incoming + outgoing trajectories,
multiple-Coulomb-scattering angle, per-muon momentum) and are a different
modality — they are intentionally **out of scope** here. The algebraic (SART)
and statistical (MLEM / Bayesian-MAP) families *do* transfer to the absorption
problem and are provided.

# Performance
The Jacobian `J = ∂flux/∂w` is extremely sparse: bin `b` only depends on the
handful of cells its ray crosses. We therefore differentiate each bin only over
its on-path cells (`directional_flux_and_grad_csda`) and assemble a
`SparseMatrixCSC`, optionally threaded across bins. Mixture stopping powers are
evaluated as explicit linear combinations of single-material lookups
(numerically identical to `MaterialMixture`, but allocation-free and friendly to
Zygote), so the AD hot path allocates no `MaterialMixture` and stays type-stable.
"""
module Tomography

using ..Types: MaterialMixture, State, Vec3
using ..Physics: property_stopping_power, property_range, property_straggling,
                 ENERGY_LOSS_CSDA, ENERGY_LOSS_MIXED
using ..Geometry: compute_decay_weight_from_path, locals_air_at_altitude,
                  transport_backward_step, transport_backward_step_full,
                  transport_backward_step_mixed
using ..GaisserFlux: flux_gccly
using ..Pumas: sample_energy_loguniform

using Random
using Statistics
using LinearAlgebra
using SparseArrays
using Zygote

export MaterialConfig, SiteConfig, EnergySample, CellSegment, DirectionalPath
export sample_energy_set, direction_vector
export cell_density, cell_stopping_power
export compute_directional_flux_csda, directional_flux_and_grad_csda
export assemble_forward_and_jacobian
export build_cell_properties_for_mc, compute_directional_flux_mc
export CsdaCorrection, set_csda_correction!, get_csda_correction, calibrate_csda_correction
export SmoothnessPrior, laplacian_gradient, apply_laplacian_smoothing
export EdgePrior, edge_gradient, edge_hessian
export sart_reconstruct, mlem_reconstruct, gradient_descent_reconstruct, gauss_newton_reconstruct
export make_csda_operator
export laplace_posterior, null_model_tests, profile_inventory
export mse, rmse, snr_metric, recon_snr, cnr, psnr, ssim_metric, reconstruction_report
export point_spread_recovery, resolution_vs_depth, resolution_map, radial_fwhm

# --- CSDA stepping constants (kept identical to examples/lvd_tomography.jl) ---
const MAX_WATER_FRACTION = 0.9
const CSDA_MAX_RELATIVE_GAIN = 0.02
const CSDA_MAX_STEP_M = 60.0
const MC_STEP_MIN_M = 1e-6
const MC_STEP_EPSILON_M = 1e-7

# ===========================================================================
# Differentiable CSDA fluctuation/hard-loss correction (matches full MC)
#
# Pure CSDA overpredicts the deep-detector flux by ~40% because it propagates a
# deterministic energy with the *mean* dE/dx and ignores the stochastic physics
# the MC keeps: (i) energy-loss straggling (variance Ω, GeV²/(kg/m²)) and
# (ii) catastrophic hard radiative events (brem/pair/photonuclear) whose mean
# loss is `dedx_CSDA − dedx_MIXED`. Both push the required surface energy up the
# steeply-falling spectrum, suppressing the transmitted flux. We fold them into
# the per-step weight as a differentiable, MC-calibrated factor
#   exp(−κ_s · Ω · dX / E²  −  κ_h · (dedx_CSDA − dedx_MIXED) · dX / E),
# with `dX` the step grammage and `E` the step's mean energy. The coefficients
# (κ_s, κ_h) are fit to converged MC by `calibrate_csda_correction`. Because the
# factor lives inside the propagators, the segment `rrule`s differentiate it
# automatically (it depends on `w` through density, Ω and the stopping powers).
# ===========================================================================

mutable struct CsdaCorrection
    enabled::Bool
    # MC-calibrated multiplicative flux correction applied per energy sample, split
    # into two PHYSICALLY DISTINCT effects (see `csda_var_correction`):
    #   exp( +α · ½b(b+1) · V/E²   −   κ_hard · H/E ).
    # The first term is the soft-straggling flux ENHANCEMENT: a Gaussian surface-
    # energy spread of variance V, folded against a locally-power-law surface flux of
    # spectral index b, enhances the transmitted flux by ½b(b+1)·V/E² (the power-law
    # convolution moment). b is COMPUTED from the flux model, so the coefficient is
    # parameter-free physics and α (`kappa_strag`) is only a near-unity calibration
    # multiplier (default 1). The second term is the catastrophic-hard-loss DEFICIT
    # (brem/pair/photonuclear removed stochastically by MIXED/MC but continuous in
    # CSDA), κ_hard fitted. Splitting enhancement (+, pinned) from deficit (−, fitted)
    # removes the κ_strag↔κ_hard degeneracy of the old two-suppression form.
    kappa_strag::Float64       # α: near-unity multiplier on the computed straggling enhancement
    kappa_hard::Float64        # weight on H/E   (catastrophic hard-loss deficit)
    # Smooth geometric residual `exp(a + b·logL + c·logL² + d·cosθ)` (L = rock path
    # length, m; θ = zenith), fit to the convolution-corrected MC residual to absorb
    # any leftover depth/angle structure. w-independent, so it rescales flux and its
    # gradient by the same factor.
    resid_a::Float64
    resid_b::Float64
    resid_c::Float64       # weight on (logL)²
    resid_d::Float64       # weight on cosθ
end

# Backward-compatible constructor (resid_c = resid_d = 0).
CsdaCorrection(en, ks, kh, a, b) = CsdaCorrection(en, ks, kh, a, b, 0.0, 0.0)

const CSDA_CORRECTION = Ref(CsdaCorrection(false, 0.0, 0.0, 0.0, 0.0))

"""
    set_csda_correction!(; enabled=true, kappa_strag=0.0, kappa_hard=0.0,
                         resid_a=0.0, resid_b=0.0, resid_c=0.0, resid_d=0.0)

Install the differentiable CSDA spectral-convolution correction used by the surface
flux, plus the geometric residual factor. Disabled by default (pure CSDA).
See [`calibrate_csda_correction`](@ref).
"""
function set_csda_correction!(; enabled::Bool = true, kappa_strag::Float64 = 0.0,
                              kappa_hard::Float64 = 0.0,
                              resid_a::Float64 = 0.0, resid_b::Float64 = 0.0,
                              resid_c::Float64 = 0.0, resid_d::Float64 = 0.0)
    CSDA_CORRECTION[] = CsdaCorrection(enabled, kappa_strag, kappa_hard,
                                       resid_a, resid_b, resid_c, resid_d)
    return CSDA_CORRECTION[]
end

get_csda_correction() = CSDA_CORRECTION[]

# ===========================================================================
# Shared types
# ===========================================================================

struct MaterialConfig
    rock_material::Int
    water_material::Int
    air_material::Int
    porous_material::Int
    rock_density::Float64
    water_density::Float64
    porous_density::Float64
    porous_top_thickness::Float64
end

"""
    SiteConfig(detector_elevation, primary_altitude)

Site geometry the forward model needs but that is otherwise carried by the
topography in the example. `detector_elevation` is the detector altitude in
metres ASL (used for air-density-vs-altitude); `primary_altitude` is the height
above the detector at which the cosmic-ray primary flux is evaluated (m).
"""
struct SiteConfig
    detector_elevation::Float64
    primary_altitude::Float64
end

struct EnergySample
    energy::Float64
    charge::Float64
    weight::Float64
end

struct CellSegment
    cell_idx::Int
    distance::Float64
end

struct DirectionalPath
    zenith::Float64
    azimuth::Float64
    segments::Vector{CellSegment}
    distance_in_volume::Float64
    surface_distance::Float64
    remaining_rock_distance::Float64
    surface_exit_z::Float64
    valid::Bool
end

# ===========================================================================
# Small helpers
# ===========================================================================

@inline function direction_vector(zenith_deg::Float64, azimuth_deg::Float64)
    theta_rad = deg2rad(zenith_deg)
    phi_rad = deg2rad(azimuth_deg)
    dh = sin(theta_rad)
    return (dh * sin(phi_rad), dh * cos(phi_rad), cos(theta_rad))
end

function sample_energy_set(n_samples::Int,
                           energy_min::Float64,
                           energy_max::Float64,
                           seed::Int)
    rng = MersenneTwister(seed)
    samples = Vector{EnergySample}(undef, n_samples)
    for i in 1:n_samples
        kf, w = sample_energy_loguniform(energy_min, energy_max, rng)
        charge = rand(rng) > 0.5 ? 1.0 : -1.0
        samples[i] = EnergySample(Float64(kf), charge, Float64(w))
    end
    return samples
end

# ===========================================================================
# Differentiable density / stopping power as explicit functions of w
#
# These reproduce `continuous_cell_mixture_and_density` + the mixture-weighted
# `property_stopping_power(::MaterialMixture)` EXACTLY (a fraction-weighted sum
# of single-material values), but without constructing a MaterialMixture, so the
# AD hot path is allocation-free and Zygote differentiates w directly.
# ===========================================================================

@inline function is_shallow_porous(shallow::Bool, matcfg::MaterialConfig)
    return shallow && matcfg.porous_material > 0 && matcfg.porous_top_thickness > 0.0
end

# Water mixture parameter `w ∈ [0, 1]`:
#   (1-w)·rock + w·water  (water lightens the cell; w=0 is pure standard rock).
@inline function cell_density(w, shallow::Bool, matcfg::MaterialConfig)
    if is_shallow_porous(shallow, matcfg)
        return 0.5 * (1.0 - w) * matcfg.porous_density +
               0.5 * (1.0 - w) * matcfg.rock_density +
               w * matcfg.water_density
    end
    return (1.0 - w) * matcfg.rock_density + w * matcfg.water_density
end

@inline function cell_stopping_power(physics, w, shallow::Bool, matcfg::MaterialConfig, energy)
    if is_shallow_porous(shallow, matcfg)
        sp_p = property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.porous_material, energy)
        sp_r = property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.rock_material, energy)
        sp_w = property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.water_material, energy)
        return 0.5 * (1.0 - w) * sp_p + 0.5 * (1.0 - w) * sp_r + w * sp_w
    end
    sp_r = property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.rock_material, energy)
    sp_w = property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.water_material, energy)
    return (1.0 - w) * sp_r + w * sp_w
end

# Rock hard-loss stopping power (dedx_CSDA − dedx_MIXED ≥ 0) — the mean energy
# carried by catastrophic radiative events that MIXED mode resolves stochastically.
@inline function rock_hardloss(physics, matcfg::MaterialConfig, energy)
    return property_stopping_power(physics, ENERGY_LOSS_CSDA, matcfg.rock_material, energy) -
           property_stopping_power(physics, ENERGY_LOSS_MIXED, matcfg.rock_material, energy)
end

# Flux suppression from the *accumulated* surface-energy spread. The steeply
# falling surface spectrum, folded against the energy straggling, is suppressed
# by ≈ exp(−κ_s · V/E_s² − κ_h · H/E_s), where V = ∫Ω(E(X))dX is the variance of
# the surface energy built up along the column (GeV²), H = ∫(dedx_csda−dedx_mixed)dX
# the integrated hard-loss (GeV), and E_s the surface energy. V and H are
# accumulated with the *real* energy profile (not a single-point proxy), which
# is what makes the deficit nearly column-independent the way the MC is.
# Local differential spectral index b = −d lnΦ/d lnE of the SURFACE flux at
# (E, cosθ), from a well-conditioned symmetric log-difference of `flux_gccly`.
# Fully differentiable in E (Zygote-safe — `flux_gccly` is the same flux used in
# the forward) and charge-independent (the charge fraction cancels in the
# log-derivative). b runs ≈2 (E∼10 GeV) → 3.7 (E≳1 TeV), so the straggling
# enhancement coefficient ½b(b+1) runs ≈3 → 8.7.
@inline function local_spectral_index(E, cos_theta)
    η = 1.0e-2
    fp = flux_gccly(cos_theta, E * (one(E) + η), one(E))
    fm = flux_gccly(cos_theta, E * (one(E) - η), one(E))
    return -(log(fp) - log(fm)) / (2η)
end

# MC-calibrated CSDA→MC flux correction, exp( +α·½b(b+1)·V/E²  −  κ_h·H/E ):
#   • soft energy-loss straggling → ENHANCEMENT. The steeply-falling surface
#     spectrum convolved with the Gaussian surface-energy spread of variance V
#     (GeV²) enhances the transmitted flux by ½b(b+1)·V/E², with b the local
#     spectral index (computed, not fitted) and α (`kappa_strag`) a near-unity
#     calibration multiplier.
#   • catastrophic hard radiative losses → DEFICIT exp(−κ_h·H/E), H the integrated
#     (dedx_csda − dedx_mixed) hard-loss (GeV), κ_h fitted to MC.
# Because the enhancement coefficient is physical (pinned) and only the hard-loss
# κ_h is fitted, the two terms are no longer degenerate.
@inline function csda_var_correction(energy_surface, V, H, cos_theta)
    c = CSDA_CORRECTION[]
    c.enabled || return one(typeof(energy_surface))
    e = max(energy_surface, 1.0)
    b = local_spectral_index(e, cos_theta)
    c_strag = 0.5 * b * (b + one(b))
    return exp(c.kappa_strag * c_strag * V / (e * e) - c.kappa_hard * max(H, 0.0) / e)
end

# Geometric rock path length (m), w-independent: cells + rock remainder.
@inline function path_rock_length(path::DirectionalPath)
    L = path.remaining_rock_distance
    @inbounds for s in path.segments
        L += s.distance
    end
    return L
end

# Smooth geometric residual multiplier `exp(a + b·logL + c·logL² + d·cosθ)`;
# 1.0 when no residual coefficients are set. w-independent (rescales flux and its
# gradient equally), so it is safe for the AD Jacobian.
@inline function csda_geom_multiplier(path::DirectionalPath)
    c = CSDA_CORRECTION[]
    (c.enabled && (c.resid_a != 0.0 || c.resid_b != 0.0 ||
                   c.resid_c != 0.0 || c.resid_d != 0.0)) || return 1.0
    logL = log(max(path_rock_length(path), 1.0))
    cosθ = cosd(path.zenith)
    return exp(c.resid_a + c.resid_b * logL + c.resid_c * logL * logL + c.resid_d * cosθ)
end

# ===========================================================================
# CSDA segment propagation
# ===========================================================================

"""
    propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy, weight)

Backward CSDA through one rock/water cell parametrised by water fraction `w`.
Identical arithmetic to the original `propagate_csda_uniform_segment` on a
`MaterialMixture`, but with `w`-explicit density / stopping power so the segment
is differentiable w.r.t. `w` and allocates nothing.
"""
function propagate_csda_cell_segment(physics, w, shallow::Bool, matcfg::MaterialConfig,
                                     distance::Float64, energy, weight)
    density = cell_density(w, shallow, matcfg)
    remaining = distance
    while remaining > 1e-8
        dedx0 = cell_stopping_power(physics, w, shallow, matcfg, energy)
        (!isfinite(dedx0) || dedx0 <= 0.0) && return (energy, zero(weight), false)

        grammage_rel = CSDA_MAX_RELATIVE_GAIN * max(energy, 1e-6) / dedx0
        grammage_step = min(remaining * density, max(grammage_rel, 1e-6), CSDA_MAX_STEP_M * density)
        ds = min(remaining, grammage_step / max(density, 1e-12))
        ds <= 0.0 && return (energy, zero(weight), false)

        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = cell_stopping_power(physics, w, shallow, matcfg, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || energy_next >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = cell_stopping_power(physics, w, shallow, matcfg, energy_next)
        weight = weight * (dedx_next / max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        energy = energy_next
        remaining -= ds
    end
    return energy, weight, true
end

"""
    propagate_csda_uniform_segment(physics, material::Int, density, distance, energy, weight)

Backward CSDA through a fixed single material (rock remainder, or any non-`w`
segment). Type-stable on `material::Int`.
"""
function propagate_csda_uniform_segment(physics, material::Int, density::Float64,
                                        distance::Float64, energy, weight)
    remaining = distance
    while remaining > 1e-8
        dedx0 = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy)
        (!isfinite(dedx0) || dedx0 <= 0.0) && return (energy, zero(weight), false)

        grammage_rel = CSDA_MAX_RELATIVE_GAIN * max(energy, 1e-6) / dedx0
        grammage_step = min(remaining * density, max(grammage_rel, 1e-6), CSDA_MAX_STEP_M * density)
        ds = min(remaining, grammage_step / max(density, 1e-12))
        ds <= 0.0 && return (energy, zero(weight), false)

        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || energy_next >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy_next)
        weight = weight * (dedx_next / max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        energy = energy_next
        remaining -= ds
    end
    return energy, weight, true
end

"""
    propagate_csda_air_segment(physics, air_material, site, start_z, cos_theta, distance, energy, weight)

Backward CSDA through the atmosphere with altitude-dependent air density.
"""
function propagate_csda_air_segment(physics, air_material::Int, site::SiteConfig,
                                    start_z::Float64, cos_theta::Float64,
                                    distance::Float64, energy, weight)
    remaining = distance
    traversed = 0.0
    while remaining > 1e-8
        altitude_mid = site.detector_elevation + start_z +
                       cos_theta * (traversed + 0.5 * min(remaining, CSDA_MAX_STEP_M))
        density = locals_air_at_altitude(altitude_mid).density
        dedx0 = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy)
        (!isfinite(dedx0) || dedx0 <= 0.0) && return (energy, zero(weight), false)

        grammage_rel = CSDA_MAX_RELATIVE_GAIN * max(energy, 1e-6) / dedx0
        ds = min(remaining, CSDA_MAX_STEP_M, grammage_rel / max(density, 1e-12))
        ds = max(ds, 1e-3)

        altitude_mid = site.detector_elevation + start_z + cos_theta * (traversed + 0.5 * ds)
        density = locals_air_at_altitude(altitude_mid).density
        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || energy_next >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_next)
        weight = weight * (dedx_next / max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        energy = energy_next
        traversed += ds
        remaining -= ds
    end
    return energy, weight, true
end

# ===========================================================================
# Directional CSDA flux + sparse on-path gradient
# ===========================================================================

# Core single-direction CSDA flux as a function of the on-path water fractions.
# `wsel` is indexed via `seg_pos` for each segment so the whole computation
# depends only on the small on-path vector (enables a sparse reverse pass).
function _directional_flux_core(physics, matcfg::MaterialConfig, site::SiteConfig,
                                path::DirectionalPath, energy_samples::Vector{EnergySample},
                                cos_theta::Float64, seg_pos::Vector{Int},
                                seg_shallow::Vector{Bool}, seg_dist::Vector{Float64}, wsel)
    flux_sum = zero(eltype(wsel)) + 0.0
    nseg = length(seg_pos)
    @inbounds for sample in energy_samples
        energy = sample.energy
        weight = one(eltype(wsel)) * 1.0
        Vacc = zero(eltype(wsel)) + 0.0   # ∫Ω(E)dX  (surface-energy variance)
        Hacc = zero(eltype(wsel)) + 0.0   # ∫(dedx_csda−dedx_mixed)dX (hard-loss)
        ok = true
        for k in 1:nseg
            e_in = energy
            energy, weight, ok = propagate_csda_cell_segment(
                physics, wsel[seg_pos[k]], seg_shallow[k], matcfg, seg_dist[k], energy, weight)
            ok || break
            e_rep = 0.5 * (e_in + energy)
            gx = seg_dist[k] * cell_density(wsel[seg_pos[k]], seg_shallow[k], matcfg)
            Vacc += property_straggling(physics, matcfg.rock_material, e_rep) * gx
            Hacc += rock_hardloss(physics, matcfg, e_rep) * gx
        end
        if ok && path.remaining_rock_distance > 1e-8
            e_in = energy
            energy, weight, ok = propagate_csda_uniform_segment(
                physics, matcfg.rock_material, matcfg.rock_density,
                path.remaining_rock_distance, energy, weight)
            if ok
                e_rep = 0.5 * (e_in + energy)
                gx = path.remaining_rock_distance * matcfg.rock_density
                Vacc += property_straggling(physics, matcfg.rock_material, e_rep) * gx
                Hacc += rock_hardloss(physics, matcfg, e_rep) * gx
            end
        end
        if ok
            air_distance = max(0.0, site.primary_altitude - path.surface_exit_z) / max(cos_theta, 1e-3)
            energy, weight, ok = propagate_csda_air_segment(
                physics, matcfg.air_material, site, path.surface_exit_z, cos_theta,
                air_distance, energy, weight)
        end
        if ok
            corr = csda_var_correction(energy, Vacc, Hacc, cos_theta)
            flux_single = weight * corr * flux_gccly(cos_theta, energy, sample.charge)
        else
            flux_single = zero(eltype(wsel))
        end
        flux_sum += 2.0 * sample.weight * flux_single
    end
    return csda_geom_multiplier(path) * flux_sum / length(energy_samples)
end

# Precompute the on-path cell list and per-segment index/shallow/distance arrays.
function _path_layout(path::DirectionalPath, shallow_flags::AbstractVector{Bool})
    segs = path.segments
    path_cells = Int[]
    posmap = Dict{Int,Int}()
    seg_pos = Vector{Int}(undef, length(segs))
    seg_shallow = Vector{Bool}(undef, length(segs))
    seg_dist = Vector{Float64}(undef, length(segs))
    @inbounds for (k, s) in enumerate(segs)
        p = get(posmap, s.cell_idx, 0)
        if p == 0
            push!(path_cells, s.cell_idx)
            p = length(path_cells)
            posmap[s.cell_idx] = p
        end
        seg_pos[k] = p
        seg_shallow[k] = shallow_flags[s.cell_idx]
        seg_dist[k] = s.distance
    end
    return path_cells, seg_pos, seg_shallow, seg_dist
end

"""
    compute_directional_flux_csda(physics, shallow_flags, matcfg, site, path, water_fractions, energy_samples)

Forward CSDA flux for one angular bin (value only). `shallow_flags` is a
per-cell `Bool` vector, `water_fractions` a per-cell `Float64` vector.
"""
function compute_directional_flux_csda(physics, shallow_flags::AbstractVector{Bool},
                                       matcfg::MaterialConfig, site::SiteConfig,
                                       path::DirectionalPath,
                                       water_fractions::AbstractVector{<:Real},
                                       energy_samples::Vector{EnergySample})
    path.valid || return NaN
    cos_theta = cosd(path.zenith)
    cos_theta <= 0.0 && return NaN
    # Direct scalar indexing of the full field (no array mutation), so this is
    # itself Zygote-differentiable over the full vector; used as the forward
    # value and as an independent full-vector AD reference in tests.
    flux_sum = zero(eltype(water_fractions)) + 0.0
    @inbounds for sample in energy_samples
        energy = sample.energy
        weight = 1.0
        Vacc = zero(eltype(water_fractions)) + 0.0
        Hacc = zero(eltype(water_fractions)) + 0.0
        ok = true
        for seg in path.segments
            e_in = energy
            energy, weight, ok = propagate_csda_cell_segment(
                physics, water_fractions[seg.cell_idx], shallow_flags[seg.cell_idx],
                matcfg, seg.distance, energy, weight)
            ok || break
            e_rep = 0.5 * (e_in + energy)
            gx = seg.distance * cell_density(water_fractions[seg.cell_idx],
                                             shallow_flags[seg.cell_idx], matcfg)
            Vacc += property_straggling(physics, matcfg.rock_material, e_rep) * gx
            Hacc += rock_hardloss(physics, matcfg, e_rep) * gx
        end
        if ok && path.remaining_rock_distance > 1e-8
            e_in = energy
            energy, weight, ok = propagate_csda_uniform_segment(
                physics, matcfg.rock_material, matcfg.rock_density,
                path.remaining_rock_distance, energy, weight)
            if ok
                e_rep = 0.5 * (e_in + energy)
                gx = path.remaining_rock_distance * matcfg.rock_density
                Vacc += property_straggling(physics, matcfg.rock_material, e_rep) * gx
                Hacc += rock_hardloss(physics, matcfg, e_rep) * gx
            end
        end
        if ok
            air_distance = max(0.0, site.primary_altitude - path.surface_exit_z) / max(cos_theta, 1e-3)
            energy, weight, ok = propagate_csda_air_segment(
                physics, matcfg.air_material, site, path.surface_exit_z, cos_theta,
                air_distance, energy, weight)
        end
        if ok
            corr = csda_var_correction(energy, Vacc, Hacc, cos_theta)
            flux_single = weight * corr * flux_gccly(cos_theta, energy, sample.charge)
        else
            flux_single = zero(eltype(water_fractions))
        end
        flux_sum += 2.0 * sample.weight * flux_single
    end
    return csda_geom_multiplier(path) * flux_sum / length(energy_samples)
end

"""
    directional_flux_and_grad_csda(physics, shallow_flags, matcfg, site, path, water_fractions, energy_samples)
        -> (flux, cellids, values)

Forward flux plus the **sparse** gradient `∂flux/∂w` restricted to the cells the
ray crosses. `cellids[k]` is a cell index and `values[k] = ∂flux/∂w[cellids[k]]`.
Uses a single Zygote reverse pass over the small on-path vector.
"""
function directional_flux_and_grad_csda(physics, shallow_flags::AbstractVector{Bool},
                                        matcfg::MaterialConfig, site::SiteConfig,
                                        path::DirectionalPath,
                                        water_fractions::AbstractVector{<:Real},
                                        energy_samples::Vector{EnergySample})
    if !path.valid
        return (NaN, Int[], Float64[])
    end
    cos_theta = cosd(path.zenith)
    if cos_theta <= 0.0
        return (NaN, Int[], Float64[])
    end
    path_cells, seg_pos, seg_shallow, seg_dist = _path_layout(path, shallow_flags)
    if isempty(path_cells)
        f = _directional_flux_core(physics, matcfg, site, path, energy_samples,
                                   cos_theta, seg_pos, seg_shallow, seg_dist, Float64[])
        return (f, Int[], Float64[])
    end
    wsel = Float64[water_fractions[c] for c in path_cells]
    fobj = w -> _directional_flux_core(physics, matcfg, site, path, energy_samples,
                                       cos_theta, seg_pos, seg_shallow, seg_dist, w)
    res = Zygote.withgradient(fobj, wsel)
    flux = res.val
    grad = res.grad[1]
    values = grad === nothing ? zeros(Float64, length(path_cells)) : collect(Float64, grad)
    return (flux, path_cells, values)
end

"""
    assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site, paths, water_fractions,
                                  energy_samples; n_cells, threaded=true)
        -> (flux::Vector{Float64}, J::SparseMatrixCSC{Float64,Int})

Forward flux for every bin plus the sparse Jacobian `J[bin, cell] = ∂flux/∂w`.
Bins are independent; with `threaded=true` they run on `Threads.@threads`, each
writing only its own preallocated slot (lock-free).
"""
function assemble_forward_and_jacobian(physics, shallow_flags::AbstractVector{Bool},
                                       matcfg::MaterialConfig, site::SiteConfig,
                                       paths::Vector{DirectionalPath},
                                       water_fractions::AbstractVector{<:Real},
                                       energy_samples::Vector{EnergySample};
                                       n_cells::Int, threaded::Bool = true)
    n_bins = length(paths)
    flux = fill(NaN, n_bins)
    cols = Vector{Vector{Int}}(undef, n_bins)
    vals = Vector{Vector{Float64}}(undef, n_bins)

    do_bin = function (b)
        f, cellids, gradvals = directional_flux_and_grad_csda(
            physics, shallow_flags, matcfg, site, paths[b], water_fractions, energy_samples)
        flux[b] = f
        cols[b] = cellids
        vals[b] = gradvals
    end

    if threaded
        Threads.@threads for b in 1:n_bins
            do_bin(b)
        end
    else
        for b in 1:n_bins
            do_bin(b)
        end
    end

    I = Int[]
    J = Int[]
    V = Float64[]
    for b in 1:n_bins
        isassigned(cols, b) || continue
        cellids = cols[b]
        gradvals = vals[b]
        @inbounds for t in eachindex(cellids)
            v = gradvals[t]
            if isfinite(v) && v != 0.0
                push!(I, b)
                push!(J, cellids[t])
                push!(V, v)
            end
        end
    end
    Jmat = sparse(I, J, V, n_bins, n_cells)
    return flux, Jmat
end

# ===========================================================================
# Monte Carlo cross-check (stochastic transport, for AD-vs-MC validation)
# ===========================================================================

function mc_cell_mixture_and_density(w::Float64, shallow::Bool, matcfg::MaterialConfig)
    if is_shallow_porous(shallow, matcfg)
        if w <= 1e-12
            return MaterialMixture(matcfg.porous_material), matcfg.porous_density
        end
        porous_frac = 0.5 * (1.0 - w)
        rock_frac = 0.5 * (1.0 - w)
        mix = MaterialMixture(
            [matcfg.porous_material, matcfg.rock_material, matcfg.water_material],
            [porous_frac, rock_frac, w])
        density = porous_frac * matcfg.porous_density +
                  rock_frac * matcfg.rock_density + w * matcfg.water_density
        return mix, density
    end
    if w <= 1e-12
        return MaterialMixture(matcfg.rock_material), matcfg.rock_density
    end
    mix = MaterialMixture([matcfg.rock_material, matcfg.water_material], [1.0 - w, w])
    density = (1.0 - w) * matcfg.rock_density + w * matcfg.water_density
    return mix, density
end

function build_cell_properties_for_mc(shallow_flags::AbstractVector{Bool},
                                      water_fractions::Vector{Float64},
                                      matcfg::MaterialConfig)
    materials = Vector{MaterialMixture}(undef, length(water_fractions))
    densities = Vector{Float64}(undef, length(water_fractions))
    cache = Dict{Tuple{Bool,Float64}, Tuple{MaterialMixture,Float64}}()
    for idx in eachindex(water_fractions)
        key = (shallow_flags[idx], water_fractions[idx])
        if !haskey(cache, key)
            cache[key] = mc_cell_mixture_and_density(water_fractions[idx], key[1], matcfg)
        end
        materials[idx], densities[idx] = cache[key]
    end
    return materials, densities
end

function initial_state(energy_final::Float64, charge::Float64, zenith::Float64, azimuth::Float64)
    dir = direction_vector(zenith, azimuth)
    return State{Float64}(
        charge = charge, energy = energy_final, distance = 0.0, grammage = 0.0,
        time = 0.0, weight = 1.0, position = Vec3{Float64}(0.0, 0.0, 0.0),
        direction = Vec3{Float64}(-dir[1], -dir[2], -dir[3]), decayed = false)
end

function propagate_mc_uniform_segment(physics, state::State{Float64}, material, density::Float64,
                                      distance::Float64, rng::AbstractRNG;
                                      straggling::Bool, energy_threshold_low::Float64,
                                      scattering::Bool = false)
    remaining = distance
    energy_limit = 1e12
    while remaining > 1e-8 && state.energy < energy_limit - eps(Float64)
        (state.weight <= 0.0 || !isfinite(state.weight)) && break
        current_limit = state.energy < energy_threshold_low - eps(Float64) ? energy_threshold_low : energy_limit
        Xi = property_range(physics, ENERGY_LOSS_CSDA, material, state.energy)
        max_step = 0.01 * Xi / max(density, 1e-6)
        step_size = clamp(min(remaining + MC_STEP_EPSILON_M, max_step), MC_STEP_MIN_M, remaining)
        if state.energy < energy_threshold_low - eps(Float64)
            if straggling
                state, _ = transport_backward_step_full(physics, state, material, density, step_size, rng;
                    mode = :straggled, scattering = scattering, energy_limit = current_limit)
            else
                state = transport_backward_step(physics, state, material, density, step_size;
                    rng = nothing, straggling = false)
            end
        else
            if straggling
                state = transport_backward_step_mixed(physics, state, material, density, step_size, rng;
                    energy_limit = current_limit)
            else
                state = transport_backward_step(physics, state, material, density, step_size;
                    rng = nothing, straggling = false)
            end
        end
        remaining -= step_size
    end
    return state
end

function propagate_mc_air_segment(physics, state::State{Float64}, air_material::Int, site::SiteConfig,
                                  distance::Float64, cos_theta::Float64, rng::AbstractRNG;
                                  straggling::Bool, energy_threshold_low::Float64,
                                  scattering::Bool = false)
    remaining = distance
    energy_limit = 1e12
    while remaining > 1e-8 && state.energy < energy_limit - eps(Float64)
        (state.weight <= 0.0 || !isfinite(state.weight)) && break
        current_limit = state.energy < energy_threshold_low - eps(Float64) ? energy_threshold_low : energy_limit
        altitude = site.detector_elevation + state.position[3]
        density = locals_air_at_altitude(altitude).density
        Xi = property_range(physics, ENERGY_LOSS_CSDA, air_material, state.energy)
        max_step = 0.01 * Xi / max(density, 1e-8)
        max_step = min(max_step, 0.05 * 12e3 / max(cos_theta, 0.05))
        step_size = clamp(min(remaining + MC_STEP_EPSILON_M, max_step), MC_STEP_MIN_M, remaining)
        if state.energy < energy_threshold_low - eps(Float64)
            if straggling
                state, _ = transport_backward_step_full(physics, state, air_material, density, step_size, rng;
                    mode = :straggled, scattering = scattering, energy_limit = current_limit)
            else
                state = transport_backward_step(physics, state, air_material, density, step_size;
                    rng = nothing, straggling = false)
            end
        else
            if straggling
                state = transport_backward_step_mixed(physics, state, air_material, density, step_size, rng;
                    energy_limit = current_limit)
            else
                state = transport_backward_step(physics, state, air_material, density, step_size;
                    rng = nothing, straggling = false)
            end
        end
        remaining -= step_size
    end
    return state
end

"""
    compute_directional_flux_mc(physics, shallow_flags, matcfg, site, path, cell_materials,
                                cell_densities, energy_samples, seed; straggling, energy_threshold_low)
        -> (flux, sigma)

Stochastic backward-MC flux for one direction. Used to finite-difference the
flux w.r.t. a cell's water fraction and cross-check the CSDA AD gradient.
"""
function compute_directional_flux_mc(physics, matcfg::MaterialConfig, site::SiteConfig,
                                     path::DirectionalPath,
                                     cell_materials::Vector{MaterialMixture},
                                     cell_densities::Vector{Float64},
                                     energy_samples::Vector{EnergySample}, seed::Int;
                                     straggling::Bool, energy_threshold_low::Float64,
                                     scattering::Bool = false)
    path.valid || return (NaN, NaN)
    cos_theta = cosd(path.zenith)
    cos_theta <= 0.0 && return (NaN, NaN)
    rng = MersenneTwister(seed)
    w_sum = 0.0
    w2_sum = 0.0
    for sample in energy_samples
        state = initial_state(sample.energy, sample.charge, path.zenith, path.azimuth)
        for seg in path.segments
            state = propagate_mc_uniform_segment(physics, state, cell_materials[seg.cell_idx],
                cell_densities[seg.cell_idx], seg.distance, rng;
                straggling = straggling, energy_threshold_low = energy_threshold_low,
                scattering = scattering)
            (state.weight <= 0.0 || !isfinite(state.weight)) && break
        end
        if state.weight > 0.0 && isfinite(state.weight) && path.remaining_rock_distance > 1e-8
            state = propagate_mc_uniform_segment(physics, state, MaterialMixture(matcfg.rock_material),
                matcfg.rock_density, path.remaining_rock_distance, rng;
                straggling = straggling, energy_threshold_low = energy_threshold_low,
                scattering = scattering)
        end
        if state.weight > 0.0 && isfinite(state.weight)
            air_distance = max(0.0, site.primary_altitude - path.surface_exit_z) / max(cos_theta, 1e-3)
            state = propagate_mc_air_segment(physics, state, matcfg.air_material, site,
                air_distance, cos_theta, rng;
                straggling = straggling, energy_threshold_low = energy_threshold_low,
                scattering = scattering)
        end
        flux_single = 0.0
        if state.weight > 0.0 && isfinite(state.weight) &&
           state.position[3] >= site.primary_altitude - 1.0
            flux_single = state.weight * flux_gccly(cos_theta, state.energy, sample.charge)
        end
        wi = 2.0 * sample.weight * flux_single
        w_sum += wi
        w2_sum += wi * wi
    end
    n = length(energy_samples)
    flux = w_sum / max(n, 1)
    variance = (w2_sum / max(n, 1) - flux^2) / max(1, n - 1)
    sigma = sqrt(max(0.0, variance))
    return flux, sigma
end

"""
    calibrate_csda_correction(physics, shallow_flags, matcfg, site, paths,
                              water_fractions, energy_samples, mc_fluxes, bin_indices;
                              kh_grid, refine, fit_geometric, alpha_strag,
                              kh_anchor, kh_anchor_weight) -> (alpha_strag, kappa_hard, stats)

Fit the CSDA→MC correction so the corrected CSDA flux matches the full-MC reference
`mc_fluxes` on the calibration bins `bin_indices`. The straggling enhancement
coefficient is PHYSICAL (½b(b+1), computed per sample inside `csda_var_correction`),
scaled by the near-unity multiplier `alpha_strag` (pinned, default 1); only the
catastrophic-hard-loss `κ_hard` is fitted — a WELL-POSED 1-D search, in contrast to
the old degenerate 2-D (κ_strag,κ_hard) grid. Minimises `mean(log(flux_csda/flux_mc)²)`
by a coarse-to-fine 1-D grid, optionally with a ridge `kh_anchor_weight·(κ_h−kh_anchor)²`
that pins κ_h toward a previous-pass value (recursion damping). Installs the best
coefficients via [`set_csda_correction!`] and returns `(alpha_strag, κ_hard, stats)`
with `stats.rel_before`/`stats.rel_after` = mean |1 − csda/mc|.
"""
function calibrate_csda_correction(physics, shallow_flags::AbstractVector{Bool},
                                   matcfg::MaterialConfig, site::SiteConfig,
                                   paths::Vector{DirectionalPath},
                                   water_fractions::AbstractVector{<:Real},
                                   energy_samples::Vector{EnergySample},
                                   mc_fluxes::AbstractVector{<:Real},
                                   bin_indices::AbstractVector{Int};
                                   kh_grid = range(0.0, 80.0; length = 41),
                                   refine::Int = 4,
                                   fit_geometric::Bool = true,
                                   alpha_strag::Float64 = 1.0,
                                   kh_anchor::Union{Nothing,Float64} = nothing,
                                   kh_anchor_weight::Float64 = 0.0,
                                   ridge_geom::Float64 = 1.0e-3)
    @assert length(mc_fluxes) == length(bin_indices)
    valid = [i for i in eachindex(bin_indices) if isfinite(mc_fluxes[i]) && mc_fluxes[i] > 0]
    isempty(valid) && error("calibrate_csda_correction: no positive MC reference fluxes")
    logmc = [log(mc_fluxes[i]) for i in valid]
    bins = [bin_indices[i] for i in valid]

    # When fit_geometric=false the material-INDEPENDENT geometric residual is held
    # fixed at its current (pass-1) value and only the hard-loss κ_hard is re-fit.
    # Used by the self-consistent recursion so the geometric term cannot drift to
    # absorb the recovered material's mean absorption.
    cur = get_csda_correction()
    fa, fb, fc, fd = fit_geometric ? (0.0, 0.0, 0.0, 0.0) :
                     (cur.resid_a, cur.resid_b, cur.resid_c, cur.resid_d)
    # α (kappa_strag) is the PINNED near-unity multiplier on the computed straggling
    # enhancement; only κ_hard varies in the fit.
    set_corr(kh) = set_csda_correction!(enabled = true, kappa_strag = alpha_strag, kappa_hard = kh,
                                        resid_a = fa, resid_b = fb, resid_c = fc, resid_d = fd)
    # ridge that pins κ_h toward a supplied anchor (0 weight ⇒ inert); scaled by the
    # log-objective magnitude so the weight is dimensionless-comparable.
    anchor_pen(kh) = (kh_anchor === nothing || kh_anchor_weight <= 0.0) ? 0.0 :
                     kh_anchor_weight * (kh - kh_anchor)^2
    function objective(kh)
        set_corr(kh)
        s = 0.0; n = 0
        for (j, b) in enumerate(bins)
            f = compute_directional_flux_csda(physics, shallow_flags, matcfg, site,
                                              paths[b], water_fractions, energy_samples)
            (isfinite(f) && f > 0) || continue
            d = log(f) - logmc[j]; s += d * d; n += 1
        end
        return n == 0 ? Inf : s / n + anchor_pen(kh)
    end
    function rel_only()
        acc = 0.0; n = 0
        for (j, b) in enumerate(bins)
            f = compute_directional_flux_csda(physics, shallow_flags, matcfg, site,
                                              paths[b], water_fractions, energy_samples)
            (isfinite(f) && f > 0) || continue
            acc += abs(1.0 - f / exp(logmc[j])); n += 1
        end
        return n == 0 ? NaN : acc / n
    end

    # Baseline "before": pure CSDA (no straggling enhancement, no hard-loss deficit).
    set_csda_correction!(enabled = true, kappa_strag = 0.0, kappa_hard = 0.0,
                         resid_a = fa, resid_b = fb, resid_c = fc, resid_d = fd)
    rel_before = rel_only()

    # Coarse-to-fine 1-D grid over κ_hard (well-posed; α pinned at alpha_strag).
    best_kh, best_obj = 0.0, Inf
    khg = collect(Float64, kh_grid)
    for _ in 0:refine
        for kh in khg
            o = objective(kh)
            (o < best_obj) && (best_obj = o; best_kh = kh)
        end
        dkh = length(khg) > 1 ? (khg[2] - khg[1]) : 0.5
        khg = collect(range(max(0.0, best_kh - dkh), best_kh + dkh; length = 9))
    end
    best_ks = alpha_strag

    # --- geometric residual: fit exp(a + b·logL + c·logL² + d·cosθ) to the
    # convolution-corrected residual by RIDGED linear least-squares (the ridge keeps
    # the logL² term from chasing a few outlier bins). w-independent, so it is
    # AD-safe. Skipped (held frozen at the pass-1 value) when fit_geometric=false. ---
    resid_a, resid_b, resid_c, resid_d = fa, fb, fc, fd
    if fit_geometric
        set_csda_correction!(enabled = true, kappa_strag = best_ks, kappa_hard = best_kh)
        feats = Vector{Float64}[]; ys = Float64[]
        for (j, b) in enumerate(bins)
            f = compute_directional_flux_csda(physics, shallow_flags, matcfg, site,
                                              paths[b], water_fractions, energy_samples)
            (isfinite(f) && f > 0) || continue
            logL = log(max(path_rock_length(paths[b]), 1.0))
            push!(feats, [1.0, logL, logL * logL, cosd(paths[b].zenith)])
            push!(ys, logmc[j] - log(f))          # log(mc / csda_κ): what the multiplier must supply
        end
        nfit = length(ys)
        resid_a = resid_b = resid_c = resid_d = 0.0
        if nfit >= 4
            A = reduce(vcat, transpose.(feats))   # nfit × 4
            # ridge only the non-constant features (keep the offset unpenalised)
            R = Diagonal([0.0, ridge_geom, ridge_geom, ridge_geom])
            coef = (A' * A + R) \ (A' * ys)        # ridged least-squares (a, b, c, d)
            resid_a, resid_b, resid_c, resid_d = coef[1], coef[2], coef[3], coef[4]
        elseif nfit > 0
            resid_a = sum(ys) / nfit               # constant offset only
        end
    end

    set_csda_correction!(enabled = true, kappa_strag = best_ks, kappa_hard = best_kh,
                         resid_a = resid_a, resid_b = resid_b,
                         resid_c = resid_c, resid_d = resid_d)
    acc = 0.0; n = 0
    for (j, b) in enumerate(bins)
        f = compute_directional_flux_csda(physics, shallow_flags, matcfg, site,
                                          paths[b], water_fractions, energy_samples)
        (isfinite(f) && f > 0) || continue
        acc += abs(1.0 - f / exp(logmc[j])); n += 1
    end
    rel_after = n == 0 ? NaN : acc / n
    stats = (rel_before = rel_before, rel_after = rel_after, mse_log = best_obj,
             n_bins = length(bins), kappa_strag = best_ks, kappa_hard = best_kh,
             resid_a = resid_a, resid_b = resid_b, resid_c = resid_c, resid_d = resid_d)
    return best_ks, best_kh, stats
end

include("Tomography_rules.jl")
include("Tomography_inverse.jl")

end # module Tomography
