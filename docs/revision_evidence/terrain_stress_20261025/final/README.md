# Completed conditional terrain stress

All ten original sequential cases completed on 2026-10-24 at 23:10 UTC. The canonical audit rehashed **71 per-case output files** and verified the original scientific command, CPU 0–7 / 16 GiB allocation, Julia runtime, and every preregistered source/input binding for every case. No numerical cases were parallelized or relocated. Rejected/prepared orchestration experiments are historical operational evidence only, separately archived one directory above.

The six actual refits and four ±1% fixed-field diagnostics are separated in `terrain_refit_comparison.csv` and `terrain_fixed_field_diagnostics.csv`. The figure and JSON are review candidates, with exact output hashes in `canonical_provenance.json`.

| Fit screen | Assumed peak height change | Field norm change versus matching zero | Sector RMS log residual | Projected-gradient infinity norm |
|---|---:|---:|---:|---:|
| Original 690-bin | 0% | 0% | 3.39204 | 0.00331190 |
| Original 690-bin | −5% | 0.01915% | 3.42494 | 0.00116776 |
| Original 690-bin | +5% | 0.01866% | 3.35834 | 0.00175740 |
| All 1,003 directions | 0% | 0% | 3.26155 | 0.00058590 |
| All 1,003 directions | −5% | 0.01810% | 3.29409 | 0.00045784 |
| All 1,003 directions | +5% | 0.01716% | 3.22828 | 0.00004158 |

**All six refits fail the requested 1e-6 stationarity tolerance.** These are finite-budget solver outcomes, not converged terrain sensitivity or uncertainty bounds. The original-screen zero continuation left the saved field unchanged. The all-direction zero continuation changed that old screened field by **77.24%** in relative norm; this is a separate screen/continuation sensitivity and is not caused by a terrain perturbation. Each ±5% field comparison uses its own matching-screen zero.

The ±5% scales are assumed map-height scenarios, not measured terrain errors. They realize at most 32.19 m of map-node elevation change, 44.25 m of ray slant change under thinning and 41.49 m under thickening. Bilinear interpolation changes 18 off-sector rays, by at most 9.20 m of slant distance. Under +5%, two rays change their sets of traversed supported cells. The 600-cell grid and 147-cell parameter-support mask remain fixed; ray paths, surface exit height, remaining dry rock and air columns are consistently retraced. The sector contains 34 covered directions, of which 11 enter the original screen and all 34 enter the all-direction fit.

Residuals are `log(s*flux/observation)`. The sector RMS is an **unweighted** RMS over the same 34 directions, not the fitted pseudo-Huber objective. In fixed-field diagnostics, flux uses the original saved main field and normalization `s` is reprofiled under that case's declared screen/DEM weights. In refit diagnostics, both field and profiled normalization come from that finite-budget solve. Raw relative flux changes exclude normalization. The four ±1% cases have no refit: their generic source `refit_log_residual` column repeats the unchanged-field calculation and must not be compared with an optimized all-screen zero as a refit response. They are excluded from refit/field/KKT panels and the six-refit table.

Small within-screen field changes under these assumed height scenarios do not establish an accurate DEM or identify the cause of the residual sector. The much larger screen sensitivity and failed stationarity tests remain material limits. No independent terrain data, physical inventory claim, confidence interval, or actual terrain-error attribution is supplied by this experiment.

In the figure, connecting lines are visual guides between predeclared scenarios, not certified interpolation. Crosses on the ±1% fixed-field diagnostic points mean stationarity was not tested; all six actual refit crosses fail the requested tolerance. The independent source/result/visual review is recorded in `../independent_review.json`.
