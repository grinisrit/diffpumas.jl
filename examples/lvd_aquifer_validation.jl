#!/usr/bin/env julia
"""
lvd_aquifer_validation.jl — CSDA+AD vs full-MC+FD on a real LVD line of sight.

Builds one Gran Sasso LVD line of sight from the `nm_c.inc` slant-rock table,
slices the whole rock overburden into a mesh of `N_CELLS` equal cells along the
ray, and places a rock/water aquifer in the middle cell (≈ halfway between the
detector and the surface). It then compares, for the aquifer cell:

  * flux:  direct CSDA          vs  full stochastic backward MC
  * dFlux/dw:  reverse-mode AD  vs  MC finite difference

Both paths share the same material model, energy samples and geometry, so any
residual difference is the physical CSDA-vs-straggling/fluctuation effect, not a
modelling mismatch. Run with straggling on and off to separate the two.

Usage:
    julia --project=. examples/lvd_aquifer_validation.jl [zenith_deg] [azimuth_deg]
"""

const lvd_aquifer_validation = nothing

using DiffPumas
using DiffPumas.Tomography
using DiffPumas.Physics: get_material_index
using DiffPumas.Pumas: load_or_create_physics
using Printf
using Statistics

module LVDTopo
include(joinpath(@__DIR__, "lvd_muography.jl"))
end

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")
const DEFAULT_MDF  = joinpath(@__DIR__, "data", "materials.xml")

# --- tunables ---------------------------------------------------------------
const N_CELLS          = 10        # mesh cells along the rock overburden
const AQUIFER_CELL     = 5         # ≈ halfway (cell 5 spans 40–50 % of the depth)
const W_BASE           = 0.30      # baseline water fraction in the aquifer cell
const FD_DELTA         = 0.20      # finite-difference half-step in w (wide => clean signal)
# The reverse-mode AD tape grows with n_samples × n_transport_steps, so the AD
# gradient runs on a small *converged* energy quadrature (production tomography
# uses ~8). The CSDA-vs-MC physics comparison uses a large *shared* set so the
# two are on identical quadrature and the MC variance is small (CSDA value-only,
# no tape, is cheap at any size).
const N_AD              = 256      # energy samples for the AD + CSDA-self-FD check
const N_BIG             = 4000     # energy samples shared by CSDA-value-FD and MC-FD
const ENERGY_MIN_GEV    = 1.0      # detector-energy sampling range
const ENERGY_MAX_GEV    = 1.0e5
const THRESHOLD_GEV     = 100.0    # mixed/straggled MC mode switch
const SEED              = 20260601

# Pick a non-underground LVD line of sight with a sensible rock overburden.
function choose_line_of_sight(dgsm, zenith_req, azimuth_req)
    R, ug = LVDTopo.nmap_lookup(dgsm, zenith_req, azimuth_req)
    if !ug && isfinite(R) && 800.0 <= R <= 4000.0
        return zenith_req, azimuth_req, R
    end
    @warn "Requested direction unusable (R=$R m, underground=$ug); scanning φ=0 line"
    for θ in 5.0:1.0:55.0
        Rr, u = LVDTopo.nmap_lookup(dgsm, θ, 0.0)
        if !u && isfinite(Rr) && 800.0 <= Rr <= 4000.0
            return θ, 0.0, Rr
        end
    end
    error("No usable LVD line of sight found")
end

function build_lvd_path(zenith_deg, azimuth_deg, R_slant)
    cell_len = R_slant / N_CELLS
    segments = [CellSegment(i, cell_len) for i in 1:N_CELLS]
    surface_exit_z = R_slant * cosd(zenith_deg)   # local height where the ray leaves the rock
    return DirectionalPath(zenith_deg, azimuth_deg, segments,
                           R_slant,            # distance_in_volume (whole overburden meshed)
                           R_slant,            # surface_distance
                           0.0,                # remaining_rock_distance (none: fully meshed)
                           surface_exit_z,
                           true)
end

fd_bounds() = (max(0.0, W_BASE - FD_DELTA), min(0.9, W_BASE + FD_DELTA))

with_aquifer(w_aq) = (w = zeros(Float64, N_CELLS); w[AQUIFER_CELL] = w_aq; w)

# --- AD correctness: reverse-mode AD vs deterministic CSDA-self FD --------
# Small, converged energy quadrature. Both are pure CSDA (no MC), so this is an
# exact check that the AD gradient equals ∂flux_csda/∂w to ~machine precision.
function check_ad(physics, matcfg, site, path, shallow, samples_ad)
    lo, hi = fd_bounds()
    flux, cells, grad = directional_flux_and_grad_csda(
        physics, shallow, matcfg, site, path, with_aquifer(W_BASE), samples_ad)
    k = findfirst(==(AQUIFER_CELL), cells)
    grad_ad = k === nothing ? 0.0 : grad[k]
    f_lo = compute_directional_flux_csda(physics, shallow, matcfg, site, path, with_aquifer(lo), samples_ad)
    f_hi = compute_directional_flux_csda(physics, shallow, matcfg, site, path, with_aquifer(hi), samples_ad)
    grad_csda_fd = (f_hi - f_lo) / (hi - lo)
    return (; flux, grad_ad, grad_csda_fd)
end

# --- CSDA-vs-MC physics on the large shared quadrature --------------------
# CSDA value (no tape, cheap at any size) and full stochastic MC use the SAME
# energy samples; MC lo/hi reuse the seed (common random numbers) so the FD
# difference cancels most of the MC variance.
function run_case(physics, matcfg, site, path, shallow, samples::Vector{EnergySample}; straggling::Bool)
    lo, hi = fd_bounds()
    w0, w_lo, w_hi = with_aquifer(W_BASE), with_aquifer(lo), with_aquifer(hi)

    fcsda0  = compute_directional_flux_csda(physics, shallow, matcfg, site, path, w0,   samples)
    fcsda_l = compute_directional_flux_csda(physics, shallow, matcfg, site, path, w_lo, samples)
    fcsda_h = compute_directional_flux_csda(physics, shallow, matcfg, site, path, w_hi, samples)
    grad_csda_fd = (fcsda_h - fcsda_l) / (hi - lo)

    mats0, dens0 = build_cell_properties_for_mc(shallow, w0,   matcfg)
    matsl, densl = build_cell_properties_for_mc(shallow, w_lo, matcfg)
    matsh, densh = build_cell_properties_for_mc(shallow, w_hi, matcfg)
    f0, s0 = compute_directional_flux_mc(physics, matcfg, site, path, mats0, dens0,
        samples, SEED; straggling=straggling, energy_threshold_low=THRESHOLD_GEV)
    fl, sl = compute_directional_flux_mc(physics, matcfg, site, path, matsl, densl,
        samples, SEED; straggling=straggling, energy_threshold_low=THRESHOLD_GEV)
    fh, sh = compute_directional_flux_mc(physics, matcfg, site, path, matsh, densh,
        samples, SEED; straggling=straggling, energy_threshold_low=THRESHOLD_GEV)

    grad_mc_fd = (fh - fl) / (hi - lo)
    # √(σ_lo²+σ_hi²) is an UPPER bound: lo/hi share random numbers, so the common
    # part cancels in the difference and the true FD error is smaller.
    sigma_grad = sqrt(sl^2 + sh^2) / (hi - lo)

    return (; flux_csda=fcsda0, flux_mc=f0, sigma_mc=s0, grad_csda_fd, grad_mc_fd,
              sigma_grad, lo, hi, straggling)
end

function report(label, r)
    flux_rel = 100 * abs(r.flux_mc - r.flux_csda) / max(abs(r.flux_csda), 1e-300)
    grad_rel = 100 * abs(r.grad_mc_fd - r.grad_csda_fd) / max(abs(r.grad_csda_fd), 1e-300)
    n_sigma_flux = abs(r.flux_mc - r.flux_csda) / max(r.sigma_mc, 1e-300)
    n_sigma_grad = abs(r.grad_mc_fd - r.grad_csda_fd) / max(r.sigma_grad, 1e-300)
    println("--- $label (straggling=$(r.straggling)) ---")
    println(@sprintf("  flux_csda     = %.6e", r.flux_csda))
    println(@sprintf("  flux_mc       = %.6e ± %.2e   (%.2f %%, %.2f σ)",
                     r.flux_mc, r.sigma_mc, flux_rel, n_sigma_flux))
    println(@sprintf("  grad_csda(FD) = %.6e", r.grad_csda_fd))
    println(@sprintf("  grad_mc(FD)   = %.6e ± %.2e   (%.2f %%, %.2f σ)   [FD on w∈%.2f..%.2f]",
                     r.grad_mc_fd, r.sigma_grad, grad_rel, n_sigma_grad, r.lo, r.hi))
    println()
end

function main()
    zenith_req  = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 30.0
    azimuth_req = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0

    println("=" ^ 70)
    println(" LVD aquifer validation: CSDA+AD vs full-MC+FD")
    println("=" ^ 70)

    physics = load_or_create_physics(DEFAULT_DUMP; mdf_path = DEFAULT_MDF)
    physics === nothing && error("Failed to load physics tables")

    rock  = get_material_index(physics, "StandardRock")
    water = get_material_index(physics, "Water")
    air   = get_material_index(physics, "Air")
    (rock == -1 || water == -1 || air == -1) &&
        error("Need StandardRock, Water and Air in the dump")

    matcfg = MaterialConfig(rock, water, air, -1,
        Float64(physics.tables[rock].density),
        Float64(physics.tables[water].density),
        0.0, 0.0)
    site = SiteConfig(Float64(LVDTopo.DETECTOR_ELEVATION), Float64(LVDTopo.GAISSER_HEIGHT))

    dgsm = LVDTopo.load_dgsm(LVDTopo.NMC_PATH)
    zenith, azimuth, R = choose_line_of_sight(dgsm, zenith_req, azimuth_req)
    path = build_lvd_path(zenith, azimuth, R)
    shallow = falses(N_CELLS)
    samples_ad  = sample_energy_set(N_AD,  ENERGY_MIN_GEV, ENERGY_MAX_GEV, SEED)
    samples_big = sample_energy_set(N_BIG, ENERGY_MIN_GEV, ENERGY_MAX_GEV, SEED)

    println()
    println(@sprintf("Line of sight:   θ = %.1f°, φ = %.1f°", zenith, azimuth))
    println(@sprintf("Rock overburden: R = %.1f m slant  (%.1f m vertical equiv.)",
                     R, R * cosd(zenith)))
    println(@sprintf("Mesh:            %d cells of %.1f m; aquifer in cell %d (depth %.0f-%.0f m along ray)",
                     N_CELLS, R / N_CELLS, AQUIFER_CELL,
                     (AQUIFER_CELL - 1) * R / N_CELLS, AQUIFER_CELL * R / N_CELLS))
    println(@sprintf("Aquifer:         w_base = %.2f  (rock density %.3f, water density %.3f g/cm³)",
                     W_BASE, matcfg.rock_density, matcfg.water_density))
    println(@sprintf("Quadrature:      AD %d samples; CSDA-vs-MC %d shared samples; seed %d",
                     N_AD, N_BIG, SEED))
    println(@sprintf("Detector energy: %.1e – %.1e GeV (log-uniform)", ENERGY_MIN_GEV, ENERGY_MAX_GEV))
    println()

    # 0) AD correctness: reverse-mode AD vs deterministic CSDA central diff.
    t = time()
    ad = check_ad(physics, matcfg, site, path, shallow, samples_ad)
    ad_rel = 100 * abs(ad.grad_csda_fd - ad.grad_ad) / max(abs(ad.grad_ad), 1e-300)
    println("--- AD correctness check ($(N_AD)-sample CSDA, deterministic) ---")
    println(@sprintf("  grad_csda(AD) = %.6e", ad.grad_ad))
    println(@sprintf("  grad_csda(FD) = %.6e   (AD vs FD mismatch: %.4f %%)", ad.grad_csda_fd, ad_rel))
    println(@sprintf("  (%.1f s)\n", time() - t))

    # 1) Straggling OFF: MC should reproduce the CSDA mean almost exactly.
    t = time()
    r_off = run_case(physics, matcfg, site, path, shallow, samples_big; straggling=false)
    report("no-straggling sanity", r_off)
    println(@sprintf("  (%.1f s)\n", time() - t))

    # 2) Straggling ON: full stochastic transport; reveals the physical
    #    fluctuation bias on a multi-km path.
    t = time()
    r_on = run_case(physics, matcfg, site, path, shallow, samples_big; straggling=true)
    report("full stochastic MC", r_on)
    println(@sprintf("  (%.1f s)", time() - t))

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
