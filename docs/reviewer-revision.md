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

The final full suite passes **259/259** checks on Julia 1.10.12 / Zygote 0.7.13
and Julia 1.11.7 / Zygote 0.7.10. The standalone geometry/CLI regressions pass
25/25. Curated numerical evidence and versioned test logs are in
`docs/revision_evidence/`. GPT-6.1 Sol agents implemented and independently reviewed
the changes; review found the custom-material cache issue, which was fixed and tested.

## Reproduction

Generate the pinned physics dump with `bash dev/create_revision_physics.sh`.
Its input tables, source revision, licenses and checksums are under
`examples/data/revision_physics/`. See `revision-physics-comparison.md` for the
differences from the historical dump; imported stopping powers are identical,
while some derived scattering and straggling arrays use different material recipes.

The production command is:

```bash
bash dev/run_lvd_paper.sh \
  --output-dir examples/data/lvd_revision_midpoint_20261023 \
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
