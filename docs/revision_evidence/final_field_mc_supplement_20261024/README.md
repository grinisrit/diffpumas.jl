# Supplemental same-field fixed-path MC closure

All 60 previously omitted fitted directions were evaluated at 8192 histories per ray, without rerunning optimization or modifying the original 630 records. The clean run ended 24 October 2026 22:07:04 UTC. Source/runtime identities were unchanged. Dry and final deterministic predictions reproduce all 1003 saved values bitwise; original 630 MC/SE/model/observation/scale values are preserved exactly in the combined table.

The combined table covers **690/690 fitted directions**, and 690/1003 covered directions. The 313 excluded directions remain unchecked. At the original fitted scale, dry CSDA / wet CSDA / wet MC relative residuals are **0.625620 / 0.410702 / 0.408765**. On the new 60 directions alone, wet CSDA / MC residuals are **0.642854 / 0.656396**. These different subsets must not be compared as a controlled improvement.

The original energy proposals are recreated with seed 819. New transport seeds are 16000000 + original bin index, disjoint from the original 900042 + covered-position seeds. Shared proposals induce cross-bin dependence. Per-ray MC errors are marginal sampling diagnostics, not calibration or total uncertainty. All finite zero estimates would have been retained and flagged; none occurred. Log-profiled scales clip arguments at 1e-300 and do not minimize linear relative residuals.

This is complete same-field **fixed-path energy-loss closure on the fitted subset**, not independent material recovery, angular-scattering validation, full-sky coverage, or groundwater discovery. The first attempt failed before any new MC because of an imported-name collision; it was corrected to Base.intersect and the final attempt used a fresh source-bound output root.
