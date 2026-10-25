# Independent round 3 acceptance gates

Review date: 2026-10-24 UTC. No numerical execution or manuscript edits performed by this review at registration.

The corrected main endpoint is objective 100.59269073659428, branch projected-gradient infinity norm 0.003311897033113187, normalization 3542719.29102373, 147 supported cells, and 690 fit bins. The recovery parity record reports exact prediction/objective/gradient restoration. The dry control has objective 98.53429563400051 and branch residual 0.008991907742836217. Neither meets the requested 1e-6 tolerance. The dry coordinate audit only checks cell 275: finite left/right slopes have opposite signs. This does not certify stationarity or establish a mathematical limiting corner.

## Convergence diagnostic and polishing

1. Bind source, physics dump, frozen correction, baseline field, support, fit mask, weights, graph and absolute lambda by hashes before execution. Restore the exact objective, profile normalization and predictions at baseline. Do not infer a new prior weight from the endpoint objective.
2. Preserve the fixed pseudo-Huber log objective, robust scale 0.3, absolute lambda 3.954930197858131, original support/mask and bounds. Reprofile normalization exactly at each trial. Verify unsupported coordinates and all box constraints after each displacement.
3. Record repeated baseline objective evaluations and full-precision trial objective differences. Define a numerical resolution floor; slopes dominated by subtraction error are unresolved.
4. Evaluate both feasible sides over a decreasing step ladder, with distances after projection recorded. At bounds only feasible sides are meaningful. Coordinate slopes and selected AD branches are separate quantities.
5. Test coupled feasible directions as diagnostics, recording deterministic construction or full displacement, evaluation counts and actual objective values. Explicitly report covered coordinates and direction families. A finite sample of coordinate or joint directions does not establish nonsmooth stationarity.
6. For polishing accept only finite, feasible decreases of the exact frozen objective that exceed numerical resolution. Record every trial, rejection, accepted update, budget exhaustion and termination reason. Branch Armijo rejection, small updates and objective stagnation do not establish convergence.
7. Independently restore the final objective and diagnostic values. A claim of generalized constrained stationarity needs an appropriate explicit nonsmooth condition and evidence beyond finite sampled directions; otherwise report finite-budget diagnostic/polish outcomes.

## Low-fraction calibration pilot

Freeze design, training population, heldout population, seeds, estimator, acceptance thresholds, exclusions and handling failed or zero-survivor rays before new draws. Freeze fitted coefficients before examining heldouts. Use independent fresh random streams for heldout validation and retain all declared rows.

Report absolute dry and wet flux, paired wet/dry ratio with shared-denominator covariance, and derivatives against independently estimated transport reference derivatives. Model AD versus finite difference is a separate implementation check, including a step ladder and feasible one-sided dry checks. Specify the differentiated field amplitude for nonuniform fields.

Define proposed 30% median ratio-residual improvement and nonworsening criteria mathematically before launch, including uncertainty, denominator floors and failed-row handling. Four independent replicate estimates may leave comparisons unresolved; unresolved does not mean passing. Existing dry-baseline discrepancy remains visible even if ratios improve. No learned model is required.

## Execution review

The proposed first convergence pass is diagnostic only. Its stated approximately 200 evaluations undercounts 96 coordinate evaluations plus 140 pairwise evaluations before other directions. Require an explicit accurate evaluation cap and deterministic truncation coverage. Runnable code and commands remain pending independent review before numerical launch. Review resources are CPUs 16–19, at most 8 GiB; no heavy job launched.

The revised diagnostic resolves this count: at most 43 directions, six steps, 258 evaluations; pairs are the 14 signed adjacent-top-cell exchanges rather than all 28 unordered pairs. Reviewed `dev/convergence_round3_diagnostic.jl`, `docs/revision_evidence/convergence_round3_20261025/proposed_command.json` and `launch.sh`: PASS for diagnostic execution only. The command uses the canonical corrected main baseline and production correction, 32 midpoint nodes, explicit forward-mode segment tangents, enabled material factor, CPUs 0–7, 16 GiB, and a 3600-second timeout within the provenance wrapper. This approves the scoped experiment, not its future scientific conclusion. Actual output review and any independent objective recomputation remain pending.
