# Final-field deterministic energy quadrature check

Completed 2026-10-23 at10:21:37UTC, exit0, source/runtime unchanged; CPU20,1CPU/2GiB,46.4s. The600-cell production geometry, coordinate order, final5-pass field, allowed support and frozen final correction were verified against recorded inputs. This is a deterministic integration convergence check, not Monte Carlo agreement or independent AD validation. The512-node result is a comparator, not truth.

Seven azimuth0 directions (zenith0,30,45,60,75,85,86 degrees), dry and actual final fields, correction off/on, and32/64/128/256/512 log-midpoint nodes give140 raw measurements. Each comparison below takes the maximum absolute relative difference against512 over28 flux cases and20 derivative cases. Relative denominators are the absolute512 comparator values; undefined/zero-denominator cases are flagged in CSV, not replaced by a noise floor.

| Nodes | Maximum flux difference (%) | Maximum selected derivative difference (%) |
|---:|---:|---:|
|32|0.103348|1.567560|
|64|0.031091|0.831123|
|128|0.019196|0.499601|
|256|0.002179|0.226702|

The derivative is a finite-difference component with respect to one allowed cell's water mass fraction: the longest crossed allowed-support cell, selected separately per ray. It is not an AD derivative. Central differences are used inside the box and second-order one-sided differences at its boundary, h=1e-4. The maximum512-node h-versus-h/2 relative change is0.003740%. No stochastic uncertainty bars apply.

The worst32-node derivative case is zenith60 degrees, azimuth0, actual final field, correction enabled, cell266 at water fraction0; second-order forward stencil [0,0.0001,0.0002]. Derivatives are1.32912190243818e-5 (32) and1.3086086764902578e-5 (512), absolute difference2.0513225947922263e-7, relative difference1.5675599831%. This denominator is nonzero. The same case gives the maximum32-node flux difference,0.1033479699%.

At85/86 degrees no crossed cell lies in the allowed fitted support: only flux is compared. Their32-node maximum flux differences are0.09430%/0.01420%; this provides no high-angle fitted-gradient claim. All seven paths are valid, including the external dry-rock continuation. Source-boundary convention sensitivity is a separate experiment.

Thus the earlier homogeneous-field/up-to75-degree result below1% does not establish a general derivative bound: the actual final-field selected component reaches1.57% at32 nodes.64 nodes are below1% in this tested subset. Current production32-node settings were not tuned or changed against this check.

Provenance: run_manifest.json, execution.json, runtime_observation.json and end_status.json bind input/source/runtime identities and raw output hashes. run.jl and launch.py are the exact external scripts. summary.json was derived after the end audit; sha256.json separately binds this derived summary and the complete curated bundle. No active runtime source was edited.
