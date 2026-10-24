# Finite training-MC material-coefficient dispersion

Eight predeclared independent replicates each evaluate the original 12 training rays through dry, 0.2 and 0.6 uniform fields at 8192 histories per ray/field (2,359,296 histories). The complete run ended cleanly at 22:37:28 UTC on 24 October 2026, with source/runtime unchanged. The two-parameter ridge fit (1e-6) and dry correction are fixed. Within each ray, the three fields share energy proposals and transport RNG initialization; state-dependent draw consumption can desynchronize transport histories. Rays and replicates have distinct seed blocks.

The coefficient sample mean is (-0.4806073, 0.4021105), versus original (-0.5903848, 0.5777418); sample standard deviations are (0.0300223, 0.0360478). The eight linear coefficients span [-0.5207592, -0.4210887], and quadratic coefficients span [0.3271351, 0.4324011]; the original point lies outside both ranges. Eight empirical ranges are not confidence bounds. The full empirical covariance is retained in the TOML and independently checked Python summary. These are repeated finite-MC estimator outcomes, not a Bayesian posterior or calibrated intervals. The original correction remains unchanged; the ensemble mean is not substituted into the production fit.

At the fixed recovered field, pointwise relative sample standard deviations across eight replicates have these median / maximum values over the 690 fitted directions:

| Quantity | Median | Maximum |
|---|---:|---:|
| Flux | 0.22170% | 0.42405% |
| Sum of absolute supported-cell Jacobian | 0.46570% | 0.73102% |
| Above sum divided by flux | 0.30133% | 0.38108% |

The ensemble-mean flux shift relative to the original calibration is separately 0.77509% median and 1.38774% maximum; it is not the sample dispersion or an estimate of truth bias. All 1003 covered rays are retained. The dry-column flux anchor is exact, while its derivative may change. The q derivative includes remaining dry-rock mass and excludes air. Closed-factor propagation was checked against the full transport/Jacobian operator on three directions for two predeclared replicate coefficients, with maximum relative flux/Jacobian discrepancies 3.64e-16 / 7.14e-16.

This quantifies only finite training-MC variation of the two material coefficients, conditional on fixed dry calibration and field. It does not propagate the six dry-correction coefficients, observation uncertainty, terrain/spectrum/acceptance errors, structural surrogate discrepancy, or inverse-solution uncertainty. It does not repair the remaining held-out approximation bias or establish water recovery. Final-field columns remain below the wet-training interval. Derivative sums are not resolution, Fisher information, or confidence bounds.

A separate original-ray verification replays dry/0.2/0.6 with original energy seed 2027112401, transport seed 2027112402 and 8192 histories. All three fluxes match the original stored MC values bitwise; the verification ended cleanly at 22:38:53 UTC. Original MC-tail and energy-sampler source regions, nine transport dependencies, original runtime and original saved-output hashes are bound explicitly. This verification is not a ninth replicate. The original saved training table also reproduces its fitted coefficients exactly.

`independent_summary.json` recomputes sample covariance and pointwise dispersions from the CSV using NumPy. Raw per-ray replicate outcomes, seeds, marginal MC errors, coefficient fits, formula checks and execution hashes are retained. The harmless Kaleido timeout warning concerns an unused figure service; no figures were requested by this run.

`contrast_audit/` compares all 24 original log MC wet/dry contrasts with the eight-replicate means and sample dispersions. The fixed-design ridge influence recovers the original-minus-ensemble coefficient shift to floating precision. Several rays contribute with mixed signs; no ray is removed or retuned. Standardized contrast differences are descriptive ratios to eight-sample SDs, not z-tests.
