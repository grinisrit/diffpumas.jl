# Saved-count Poisson likelihood-ratio reference

All 48 postprocess outcomes completed with clean source/runtime audits on 2026-10-24 at 22:22 UTC. No new Poisson draws, stochastic transport, SART solves or numerical reconstruction were performed. The complete compact output, including calibration/test score arrays and count-source pointers, is preserved in `examples/data/poisson_likelihood_reference_20261025_retry01`.

For each of four frozen acceptance profiles, six predeclared exposure factors and two known alternatives, the statistic is

```text
L = sum_m n_m log(rate_alt_m / rate_null_m) - E sum_m (rate_alt_m - rate_null_m).
```

The null rates are `p*dry_flux`. The first-order alternative uses the original floating-point recipe `p*dry_flux + p*linear_signal`; the finite-CSDA alternative uses `p*finite_lens_flux`. These are known-simple-hypothesis reference detectors, not claims of physical optimality. The finite surrogate remains conditional and biased relative to stochastic transport. Total integrated exposure `E` is not an expected count.

For factors 0.5, 1 and 2, calibration/test null and first-order signal populations come from the original first-order experiment; finite signals come from its clean extension. At factors 4, 8 and 16, all populations come from the extension. The helper checks source end records and hashes; template/profile/bin identity; exact rate vectors, exposure metadata, count format and byte size; and nonnegative counts. Count arrays are memory-mapped in 128-draw chunks under CPU 20, a 2 GiB address-space limit, BLAS one thread and the shared Python postprocessing lock. It rechecks all 274 adopted input identities after evaluation. All 100 wrapper-bound output files were independently rehashed without a mismatch.

Each alternative has its own calibration-null type-7 empirical quantile at `1 - 0.0013498980316301`; rejection uses strict `L > threshold`. Held-out false-positive and power intervals are marginal Clopper–Pearson 95% intervals conditional on the realized fixed threshold. They are not simultaneous over 48 outcomes and do not propagate threshold uncertainty. With only 3,000 held-out nulls, these results do not certify a true 3-sigma false-positive rate. Observed null exceedances range from 1/3,000 to 14/3,000; this finite-sample calibration can differ materially from its nominal target.

At the equal-bin first-order Gaussian planning exposure, expected null counts are only **9.856**. The known-first-order LLR gives **395/600 = 65.83%** power (conditional 95% interval 61.88–69.63%), and the finite-CSDA LLR gives **122/600 = 20.33%** (17.18–23.78%). Changing the statistic to a Poisson reference therefore does not turn this planning point into 95% power. At twice that exposure the corresponding powers are 98.83% and 63.17%. At four times it the finite-CSDA power is 96.33%, with interval 94.50–97.69%; no certified 95%-power crossing or interpolation is claimed.

Five regression tests passed, including an independent `scipy.stats.poisson.logpmf` oracle, a 128-draw chunk boundary, identical hypotheses/strict threshold behavior, metadata and byte-size rejection, and numeric integral CSV bin IDs. The first attempt failed before score evaluation because the profile CSV writes bin IDs as `1.0`; the parser now accepts finite exact-integer numeric IDs and rejects fractional IDs. That failed attempt is retained separately in `examples/data/poisson_likelihood_reference_20261025`.

The low-count known-template rationale is consistent with [Ofek and Zackay (2018), AJ 155(4), 169](https://doi.org/10.3847/1538-3881/aab265). The [university author record](https://weizmann.elsevierpure.com/en/publications/optimal-matched-filter-in-the-low-number-count-poisson-noise-regi/) verifies the 2018 journal metadata; a 2017 preprint date should not be used as the journal year. The exact simple-hypothesis formula above and its independent numerical oracle define this implementation.
