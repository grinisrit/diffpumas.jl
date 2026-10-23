# Reviewer revision: COMPHY-D-26-00704

The working revision is on `codex/reviewer-revision-20261023`. The attached
reviews were used as evidence; no submission or correspondence was sent.

## Software changes

- Package installation no longer needs an unregistered TriangleIntersect package.
  The existing local implementation is bundled, the loader API is unambiguous,
  and material-data paths are read at runtime.
- Deterministic density derivatives avoid RNG branches in Zygote. The public MC
  result explicitly identifies its gradient as that of the sample-averaged
  deterministic CSDA flux. It does not claim a stochastic-transport derivative.
- Loader entry points honor the same complete material-definition file, including
  custom-material caches. Existing dumps retain their contents.
- Tetrahedral examples use one-based TetGen indices and physical ray lengths.
  Edge intersections have a bounded barycentric tolerance; the grid remains the
  default for the broad manually reconstructed Gran Sasso terrain.
- CSDA integration uses deterministic log-midpoint energy quadrature by default.
  Its charge-summed convention is tested against explicit positive/negative pairs.
  Monte Carlo still uses random physical charges and rejects quadrature nodes.
- Measured fits export actual masks, calibration histories, termination status,
  common-subset MC comparisons and coverage. Fixed normalization now participates
  in the nonlinear fit rather than being substituted only after optimization.

Before the later inverse and transport audits, the full suite passed **259/259** checks on Julia 1.10.12 / Zygote 0.7.13
and Julia 1.11.7 / Zygote 0.7.10. The standalone geometry/CLI regressions pass
29/29. Curated numerical evidence and versioned test logs are in
`docs/revision_evidence/`. GPT-6.1 Sol agents implemented and independently reviewed
the changes; review found the custom-material cache issue, which was fixed and tested.

## Reproduction

Generate the pinned physics dump with `bash dev/create_revision_physics.sh`.
Its input tables, source revision, licenses and checksums are under
`examples/data/revision_physics/`. See `revision-physics-comparison.md` for the
differences from the historical dump; imported stopping powers are identical,
while some derived scattering and straggling arrays use different material recipes.

The production command template is (use a new output directory after the transport correction):

```bash
bash dev/run_lvd_paper.sh \
  --output-dir examples/data/lvd_revision_corrected_NEW_RUN \
  --threads 20 --memory-gib 48 --recursions 6 --recur-tol 0.02
```

The launcher runs topography, main reconstruction, reconstruction sweep, detection
and contrast sweep sequentially, stopping on error. Each invocation records hashes
of source files and inputs. CSDA uses 32 midpoint nodes explicitly. The main
process has 20 CPU affinity slots and a 48 GiB address-space limit. Other bounded
experiments use overlapping CPU affinities and smaller limits. Address-space
limits are per process, not aggregate cgroup limits; process-tree RSS is inspected
during the run. Do not start additional large jobs without checking the total
30% CPU/RAM budget on `research_vm_roland`.

The bounded validation driver is `examples/revision_suite.jl`; nonlinear measured
refits use `dev/run_revision_robustness.sh`. Their outputs are separate from the
archived paper data. The quadrature probe reads an explicitly archived calibration
from `dev/revision_inputs/`; this is a frozen diagnostic, not a new fit at every
quadrature resolution.

## Interpretation and outstanding work

The three-ray quadrature check found substantial bias in the archived eight-sample
energy rule. This does not directly quantify historical reconstruction errors,
because its calibration was refitted. Production results must be recomputed and
reconciled before numerical claims are retained.

The new derivative suite distinguishes deterministic differentiation accuracy from
CSDA-versus-MC model discrepancy. Its 24 ray-cell pairs do not cover the full
near-horizontal LVD domain. Four MC replicates provide limited precision, and
calibration uncertainty is not propagated. Small-grid timing is not evidence of
arbitrary large-scale capacity.

The legacy angular table is associated with older Aglietta papers, but its precise
publication, units, exposure and acceptance have not been verified. The manual
topographic extraction is not an independent surveyed DEM. These gaps prevent a
fully calibrated experimental significance or water-inventory claim.

The companion manuscript contains a review-response matrix, numerical figures and
the full Consensus literature report. Its archived reconstruction and detection
figures have been improved visually, but remain explicitly marked pending fresh
production results. The revised manuscript is a working draft.

## Subsequent correctness audits

Production and nonlinear robustness are stopped pending correction of a stochastic transport defect. The fresh common-input C comparison reveals a 45.34% Julia deficit at 800 m thickness and 75 degrees zenith, sustained at45–50% across three seeds. Julia's high-energy mixed helper uses deterministic CSDA without hard-loss sampling. The old general equivalence claim is withdrawn. Evidence is preserved under `revision_evidence/transport_before_fix/`. Stochastic calibration and closure must be recomputed after correction; these observations do not invalidate deterministic AD–finite-difference consistency.

The inverse audit corrected MAP trial acceptance, smoothness curvature (2λL for R=λΣedges differences²), fixed aquifer coordinates throughout optimization, and the affine intercept of frozen models. Edge-prior posterior curvature is distinguished from the IRLS optimizer majorizer. Targeted tests pass 36/36 on both Julia versions with independent Sol6.1 mathematical review; the full package tests pass 295/295 on each version. Aborted earlier reconstruction attempts are diagnostic evidence only.


### Inverse solver audit evidence

The regularization convention is `R = lambda * sum((w_i-w_j)^2)` over unique
undirected edges, with gradient `2*lambda*L*w` and Hessian `2*lambda*L`.
Prior constructors normalize adjacency once to a symmetric, duplicate-free graph.
Gauss--Newton trial acceptance uses the complete weighted data-plus-prior objective.
The `free` mask restricts both the solve and trial updates; fixed coordinates retain
their projected initial values and their boundary contributions to the prior.
The aquifer wrapper masks the forward input as well as Jacobian columns.

Frozen models retain their affine intercept. Gradient-descent histories describe
the returned state, and its projected-gradient stopping rule includes the prior.
Huber optimization uses an IRLS curvature majorizer, while posterior diagnostics
use the true local Huber curvature away from its knots. The supplied experimental
errors remain undocumented, so variance-rescaled objective scores and conditional
intervals are not validated discovery significances or coverage guarantees.

A deterministic nonlinear regression verifies identical accepted fields and
residual histories for full-model and prediction-only Gauss--Newton line searches,
with eight versus seven Jacobian evaluations. A saturated constrained case needs
only the initial Jacobian; repeated clamped candidates are cached. This comparison
is between two implementations of the corrected objective, not between the old
incorrect acceptance objective and the correction.

Full and targeted test logs, independent review/oracle results, unchanged before/
after source hashes, and the stopped pre-fix attempt are preserved in
`revision_evidence/solver_audit/`. The earlier robustness run was administratively
terminated before completing a recursion; it is not a nonlinear robustness result.
Production and robustness remain paused until the stochastic transport correction
is separately validated.


### Corrected validation and recomputation, 2026-10-23

The earlier transport pause described above is historical. After the corrected
transport passed both full package suites (557 assertions on Julia 1.10 and 1.11),
the expanded v5 validation completed with exit zero and unchanged input/source
audits. Its raw evidence and independent selection reconstruction are preserved
in `revision_evidence/corrected_validation/`. Deterministic derivative consistency
is strong, but matched stochastic sensitivities and held-out wet fluxes retain
material surrogate discrepancies: median wet flux bias +27.8% (maximum +37.4%),
and median paired wetting-response bias 30.4% of the actual MC increment. These
results limit accuracy claims and were not used to retune the frozen correction.

The user-authorized five-stage production and nonlinear robustness runs began
at 06:53 UTC in new corrected output directories, with aggregate limits of
20 CPUs and 48 GiB address space. Their outputs remain pending until execution,
source audits, and fit convergence have been checked; launch is not completion.
