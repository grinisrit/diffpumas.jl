# Final production completion and source transition

The main calculation completed successfully on 2026-10-23 at 18:11 UTC with its source audit unchanged; its five-pass fit reached the configured tolerance. The ten robustness scenarios completed at 17:16 UTC; most stop at their three-pass cap, which does not establish convergence. Reporting completed at 18:12 UTC.

These curated records retain the original computed source identities. `source_transition.json` records the later application of the reviewed shallow-mixture correction and reporting wording patches. The corrected optional porous branch was disabled in these computations (zero porous thickness). Numeric production was not rerun after this change. Historical generated prose retains the original labels; current source now qualifies heuristic scores and conditional diagnostics.

The complete main and robustness run directories are retained under `examples/data/lvd_revision_corrected_20261023` and `examples/data/revision_robustness_corrected_20261023`, including their original run manifests and audit records.

The complete no-air and cutoff900 fit records are retained under `examples/data/revision_boundary_fit_20261023` and `examples/data/revision_cutoff_fit_20261023`. These sensitivity fits use separate settings; they are not additional convergence demonstrations for the main fit.

## Final integrated package tests

After the correction and reporting patches, full package suites passed **634/634** on Julia 1.10.12 (128.4s) and Julia 1.11.7 (182.8s), including 77 actual-helper shallow-recipe checks. Julia 1.10 used the established isolated environment because the root Julia 1.11 manifest references StyledStrings. Logs, resolved 1.10 dependencies and the initial pre-assertion environment failure are retained in `tests/`. Tests establish deterministic recipe consistency and package regressions, not optional-layer stochastic accuracy.
