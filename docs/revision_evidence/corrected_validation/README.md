# Corrected fixed-path stochastic validation (v5)

Completed 2026-10-23 01:55:46 UTC with exit 0 and unchanged recorded source, physics dump, runtime and input hashes. See `execution.json`, the immutable run manifest, `config.toml`, and `end_status.json` for exact commands, seeds, sample counts and audits. This is stochastic energy-loss transport along fixed straight paths; scattering and boundary retracing are outside this API.

The suite uses 32 deterministic midpoint CSDA nodes, 1,024 MC samples, eight independent replicate streams, 24 selected ray/cell pairs, and zeniths 0/15/30/45/60/75 degrees. Frozen dry calibration uses a separate training stream and 24 training rays through 45 degrees. The 60/75-degree checks therefore extrapolate the calibration angular domain. The fitted hard-loss coefficient is at its lower bound zero, with the straggling coefficient pinned to one; four geometric coefficients are fitted subsequently. This sequential fit does not establish the physical absence of hard interactions.

## Results and limitations

- All 144 requested rays are valid. Independent reconstruction reproduced all 24 selected pairs across 24 joint strata from 81 populated strata and 1,009 nonzero Jacobian entries. Selected absolute sensitivities range 3.31e-5 to 8.75e-4; the population reaches 5.22e-9, so the near-zero tail is not validated by these selected comparisons.
- Automatic versus same-surrogate finite differences at step 1e-6 have median relative error 4.64e-9 and maximum 2.43e-6. This is deterministic consistency, not agreement with the stochastic physical reference.
- For matched-energy MC versus CSDA secants, 20/72 conditional replicate comparisons exceed twice their estimated standard error. Median relative discrepancies at uniform water fractions 0/0.3/0.8 are -27.8%/-27.4%/-31.4%. Eight replicates do not calibrate confidence coverage.
- The 21 held-out rays use the same frozen global correction for dry and unseen spatially varying wet predictions. Median dry flux bias is +10.8%; median wet bias is +27.8%, with maximum wet bias +37.4%. All wet biases are positive.
- The constructed wet field produces a strong response: median MC wetting increment is 7.13 times dry flux. The paired response bias is 12.6% to 40.6% of that increment, median 30.4%; it does not overwhelm or reverse this particular strong signal. Every paired bias-change mean exceeds twice its conditional eight-replicate standard error. This is a material surrogate response limitation, not a validated water-recovery accuracy result.

Dry/wet estimates within a replicate share energy and transport seeds to retain their covariance. Replicate streams are independent of one another and of training calibration. Reported conditional standard errors omit calibration-coefficient uncertainty and cross-bin covariance; relative ratios are descriptive and have no calibrated ratio confidence interval. No validation-driven retuning was performed. `heldout_assessment.json` records per-ray means and paired standard errors computed from `held_out_closure.csv`.

Main five-stage production and nonlinear robustness were launched later into new output roots. Their completion and convergence must be assessed separately; this suite does not prove either.
