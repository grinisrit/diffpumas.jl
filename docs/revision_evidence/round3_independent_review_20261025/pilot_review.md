# Independent completed pilot assessment

**Execution and validation: PASS. Scientific screening outcome: REJECT.** These are separate conclusions. The original production correction remains unchanged.

All 65 producer source/input bindings, runtime identity, output hashes and predraw plan/helper bindings pass. All 48 heldout cases have eight unique planned replicates and both arms; the raw tables contain 768 flux rows, 768 derivative rows, 288 local derivative checks and 36 training rows. Pairing, seed formulas, correction-factor arithmetic and column-fraction coverage pass. An independent vectorized implementation reproduces the locked decision and 864 case statistics; maximum absolute difference from the frozen analysis is 4.51e-16. The curated assessment equals a fresh execution of the frozen analysis exactly.

The preselected numerical verification cases were frozen before heldout inspection: bin 1/field 1 and bin 12/field 4, replicate 1. Serial replay of their 8192 total dry/wet/low/high MC histories reproduces all eight reference fluxes bitwise. This reuses original seeds for verification; it supplies no new independent replicate. Both deterministic arms' fluxes, Jacobians, finite-difference ladders and matched-energy secants also reproduce. The bounded reviewer run completes cleanly with unchanged source/runtime and verified output hashes.

The material-ratio median absolute residual changes from 2.73133% to 2.71873%, a relative median gain of 0.46145% with descriptive jackknife SE 29.0589 percentage points. The required 30% gain is unresolved. Absolute flux median bias increases from 10.5956% to 11.4585%; selected finite-contrast derivative median absolute bias increases from 4.84937e-5 to 5.05680e-5. Under the locked descriptive screen, 42 of 48 absolute-flux cases and 13 of 48 derivative cases worsen by more than twice their paired-change SE; these failures determine rejection. This is not a calibrated confidence-coverage claim.

Six arm/case local derivative checks are unresolved larger-step crossing candidates, while 90 ladders pass the local consistency criterion. The six failures agree at the smallest step and are not treated as proof of incorrect AD or as the substantive rejection cause. Training column fractions span approximately [0.025, 0.15]; heldouts span [0.0228050, 0.1218551], with one below-range case and none above. Column fractions include remaining dry rock in the total mass denominator and exclude air.

No new model, refit, tolerance change, additional independent draw or coefficient adoption is justified by this result. It does not repair the historical dry discrepancy or establish physical water-recovery accuracy.

Detailed evidence: `pilot_independent_analysis.json`, `pilot_provenance_check.json`, `pilot_locked_summary_replay.json`, `pilot_recheck_frozen.json`, and `pilot_replay_run/results/review.toml` in this directory.
