#!/usr/bin/env julia
# Gate G0c (S7 protocol), validation-driver half.
#
#   julia --project=. dev/g0c_validation_path.jl G0C_DIR PHYSICS_DUMP
#
# `G0C_DIR` is the output of `lvd_tomography.jl --reconstruction-sweep --protocol-stage g0c`: the 12 dry
# sweep rays (zenith 10, 30, 50 degrees; 4 azimuths each) as Julia objects and the values the sweep code
# path computed for them (Phi_CSDA from the sweep operator, Phi_MC from the sweep template routine,
# 16384 histories). This script evaluates the SAME ray objects with the code path of
# dev/validate_surrogate_v2.jl (its physics/site/material set-up, the historical dry correction file, the
# frozen material coefficients, its CSDA call and its Monte Carlo call, 16384 histories, own seeds) and
# compares:
#   pass: Phi_CSDA agrees to 1e-6 relative on every ray and Phi_MC agrees within 3 combined MC standard
#         errors on every ray; the rock path lengths of both ray sets are then reported.
# Exit code 0 on pass, 5 on failure.
include(joinpath(@__DIR__, "validate_surrogate_v2.jl"))
using Serialization
using DelimitedFiles

function g0c_validation_path(argv = ARGS)
    length(argv) == 2 || error("usage: G0C_DIR PHYSICS_DUMP")
    dir, dump = abspath.(argv)
    # The calls below must be the ones the validation driver makes; assert their text in its source.
    src = read(joinpath(@__DIR__, "validate_surrogate_v2.jl"), String)
    for needle in ("en=sample_energy_set(nsamples,1e-3,1e9,sd)",
                   "compute_directional_flux_mc(ph,mat,site,path,mats,dens,en,sd+1;straggling=true,scattering=false,energy_threshold_low=100.)",
                   "install_dry(joinpath(@__DIR__,\"..\",\"docs\",\"revision_evidence\",\"corrected_validation\",\"calibration.toml\"))",
                   "compute_directional_flux_csda(ph,shallow,mat,site,p,w,es)",
                   "mats,dens=build_cell_properties_for_mc(shallow,w,mat)")
        occursin(needle, src) || error("validate_surrogate_v2.jl no longer contains `$needle`; update the gate")
    end
    rays = Serialization.deserialize(joinpath(dir, "sweep_rays.jls"))
    sweep = readdlm(joinpath(dir, "sweep_values.tsv"), '\t', Any; header = true)[1]
    length(rays.paths) == size(sweep, 1) || error("ray list and sweep values differ in length")
    nmc = rays.histories
    ph, mat, site, em, vol, _, es = v2state(dump)          # the validation driver's set-up (its own volume is not used for the rays)
    shallow = falses(rays.n_cells); w = zeros(rays.n_cells)
    install_dry(joinpath(@__DIR__, "..", "docs", "revision_evidence", "corrected_validation", "calibration.toml"))
    material = TOML.parsefile(joinpath(@__DIR__, "..", "docs", "revision_evidence", "surrogate_v2", "production", "frozen_corrections.toml"))["material"]
    mats, dens = build_cell_properties_for_mc(shallow, w, mat)
    rows = NamedTuple[]
    for (b, p) in enumerate(rays.paths)
        set_csda_material_correction!(enabled = false)
        before = compute_directional_flux_csda(ph, shallow, mat, site, p, w, es)
        set_csda_material_correction!(linear = material["linear"], quadratic = material["quadratic"])
        after = compute_directional_flux_csda(ph, shallow, mat, site, p, w, es)
        sd = 2029000000 + 10000b                            # not a training or test seed of the validation run
        en = sample_energy_set(nmc, 1e-3, 1e9, sd)
        mcv, sigv = compute_directional_flux_mc(ph, mat, site, p, mats, dens, en, sd + 1; straggling = true, scattering = false, energy_threshold_low = 100.0)
        csda_s, mc_s, sig_s = Float64(sweep[b, 5]), Float64(sweep[b, 6]), Float64(sweep[b, 7])
        push!(rows, (bin = rays.bins[b], zenith = p.zenith, azimuth = p.azimuth, rock_path_m = Float64(sweep[b, 4]),
            csda_sweep = csda_s, csda_validation = after, csda_validation_material_off = before,
            csda_rel_diff = abs(after - csda_s) / csda_s, mc_sweep = mc_s, mc_validation = mcv, sigma_sweep = sig_s, sigma_validation = sigv,
            mc_diff_in_se = abs(mcv - mc_s) / sqrt(sig_s^2 + sigv^2),
            ratio_sweep = mc_s / csda_s, ratio_validation = mcv / after))
    end
    csv_rows(joinpath(dir, "validation_values.csv"), rows)
    max_csda = maximum(r.csda_rel_diff for r in rows); max_mc = maximum(r.mc_diff_in_se for r in rows)
    max_mat = maximum(abs(r.csda_validation - r.csda_validation_material_off) / r.csda_validation for r in rows)
    pass = max_csda <= 1e-6 && max_mc <= 3.0
    # rock path lengths of the sweep dry rays and of the README's dry held-out rays (zenith 75, three azimuths)
    readme = filter(p -> p.valid, precompute_paths(vol, em, [75.0], [37.5, 157.5, 277.5]))
    rock(p) = sum(seg.distance for seg in p.segments; init = 0.0) + p.remaining_rock_distance
    rr = [rock(p) for p in readme]; rs = [r.rock_path_m for r in rows]
    sw_all = TOML.parsefile(joinpath(dir, "sweep_rock.toml"))["all_valid_bins"]
    verdict = Dict{String,Any}("passed" => pass, "max_csda_rel_diff" => max_csda, "max_mc_diff_in_combined_se" => max_mc,
        "max_material_factor_effect_on_dry_csda" => max_mat, "n_rays" => length(rows), "histories" => nmc,
        "median_ratio_sweep" => median(r.ratio_sweep for r in rows), "median_ratio_validation" => median(r.ratio_validation for r in rows),
        "rock_path_g0c_rays_m" => [minimum(rs), maximum(rs)], "rock_path_all_sweep_bins_m" => [sw_all["min_m"], sw_all["max_m"]],
        "rock_path_readme_dry_heldout_rays_m" => rr, "readme_dry_heldout_rays_n" => length(rr),
        "validation_driver_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "validate_surrogate_v2.jl")))))
    open(joinpath(dir, "g0c_verdict.toml"), "w") do io; TOML.print(io, verdict; sorted = true); end
    println("G0c: max relative CSDA difference $(max_csda) (limit 1e-6); max MC difference $(round(max_mc; digits = 3)) combined SE (limit 3); " *
            "material factor effect on dry CSDA $(max_mat); passed=$pass")
    println("G0c: median MC/CSDA  sweep path $(verdict["median_ratio_sweep"])  validation path $(verdict["median_ratio_validation"])")
    println("G0c: rock path [m]: g0c rays $(minimum(rs)) - $(maximum(rs)); all sweep bins $(sw_all["min_m"]) - $(sw_all["max_m"]); README dry held-out rays $(rr)")
    pass || println("G0c FAILED: the drivers differ in a setting not in the gate table; return to the reviewer, every MC arm would measure a configuration mismatch")
    return pass ? 0 : 5
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(g0c_validation_path())
end
