# Final-field stochastic-closure coverage diagnostic

The 1003-ray deterministic check completed cleanly on 24 October 2026. All saved dry/final predictions and Jacobian primal values match bitwise; the 1003 × 600 sparse Jacobian has 5453 stored entries. No additional stochastic histories were evaluated in this diagnostic.

The existing MC subset checks 630/690 fitted directions. Those directions capture 99.3971% of the fitted rays’ summed absolute derivatives over 147 supported parameters, but 87.4349% of summed absolute derivatives divided by flux. The 60 omitted fitted rays span zenith 62–78 degrees and slant 1959.93–4457.22 m (median 3025.51 m). These unweighted derivative sums are neither resolution nor Fisher information, acceptance coverage, or physical uncertainty.

Fitted rays intersect 94 of the 147 supported cells; MC-checked rays intersect 90. Cells 205, 258, 351 and 362 are intersected by fitted rays but not by MC-checked rays. The 53 support cells not crossed by any fitted ray have zero fitted local Jacobian; 45 have fitted fraction above 0.001 through prior/coupling rather than direct ray constraints. This does not determine their physical composition.

Three selected omitted components were checked by deterministic finite differences. Maximum relative discrepancy falls from 2.8063e-4 at step 1e-4 to 9.0719e-5 at step 5e-5. This selected diagnostic is not full-domain or stochastic-gradient validation.

Dry-reference values remain separately named. Source/runtime/input hashes and clean completion are recorded in execution.json, end_status.json and configuration.toml. All 600 cells are retained in final_field_cell_coverage.csv. The independent arithmetic check recomputes fractions from the ray CSV. A later supplemental MC run, if performed, must retain a distinct evidence boundary.
