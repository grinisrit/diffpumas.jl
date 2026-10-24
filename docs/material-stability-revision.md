# Material calibration and fixed-objective reconstruction

The material correction and fixed-objective solver are opt-in alternatives to the archived October 23 calculation. The segment-adjoint backend now defaults to local forward-mode differentiation; the old finite-difference backend remains explicitly selectable for reproduction. The material correction passed the prespecified independent transport checks below. Matched reconstruction comparisons are in progress; passing software tests alone does not validate stochastic transport derivatives or recovered material fields.

## Frozen material correction

`DiffPumas.Tomography` exports `CsdaMaterialCorrection`, `get_csda_material_correction`, `set_csda_material_correction!`, `calibrate_csda_material_correction`, `save_csda_corrections`, and `load_csda_corrections!`.

The corrected deterministic flux is multiplied by `exp(linear*q + quadratic*q^2)`, where `q` is the water mass divided by total rock/cell mass along the prescribed ray. The denominator includes the dry remainder outside the voxel volume; air is excluded. Both numerator and denominator are differentiated. The factor is exactly one for dry fields and when disabled. It describes an empirical composition correction, not pore saturation or an identified stochastic energy-loss mechanism.

Fit the two coefficients to paired MC material contrasts with a fixed dry correction. The calibration API accepts explicit training fields, directions, dry references and a field-by-direction MC matrix. No observation from a held-out field, direction or random stream should enter calibration. Inspect design rank, conditioning, training range and independent flux/derivative discrepancies before use. A single column-composition feature does not fully describe the order of materials along a ray.

```julia
using DiffPumas.Tomography
set_csda_material_correction!(enabled=false) # historical behavior
# After fitting and validating a candidate:
save_csda_corrections("corrections.toml"; metadata=provenance)
metadata = load_csda_corrections!("corrections.toml")
```

The version-1 TOML artifact stores both the original `CsdaCorrection` and the separate material coefficients. It preserves the original serialized struct layout. The example checks recorded physics, source altitude, rock density, quadrature nodes and energy bounds when those metadata are provided. A valid file format does not guarantee that a new geometry or composition lies inside its validation domain.

## Fixed robust reconstruction

The example's `--papermatch-fit-mode stable` uses a fixed pseudo-Huber objective on log-flux residuals, with fixed screen/slant weights and a fixed smoothness prior. Normalization is profiled through a convex scalar solve at each candidate field, or pinned with `--papermatch-abs-scale`. The full field objective is still nonlinear and can have multiple stationary solutions.

The residual Jacobian includes the implicit derivative of profiled normalization. Active-set Gauss–Newton holds outward-pointing bound coordinates fixed. Every accepted projected step must be a descent step and satisfy Armijo decrease in the same objective. After exhausted Newton searches, a bounded, diagonally scaled projected-gradient step supplies a separate descent safeguard. Rejected trials use flux-only evaluations; accepted candidates still receive a full-model objective check and a fresh Jacobian. `lvd_solver_progress.csv` records live progress, `lvd_solver_history.csv` the final history, and `lvd_solver_status.txt` whether the stationarity tolerance was met. Small parameter changes alone do not establish stationarity or uniqueness.

Useful options:

```text
--correction-artifact corrections.toml
--papermatch-fit-mode stable
--papermatch-initial-water 0.15
--papermatch-gn-iters 100
--papermatch-gradient-tol 1e-6
--papermatch-log-scale 0.3
```

An artifact freezes all correction coefficients, bypassing recursive recalibration. Stable mode uses one optimization with a fixed objective. The initial field and the calibration field are different concepts. `--solver-mode stable` separately selects active-set GN for the synthetic benchmark; it does not change the synthetic objective into the measured-data robust objective.

`--papermatch-absolute-prior-weight` fixes the graph penalty coefficient directly. Without that option, the coefficient is scaled from the dry robust Jacobian. The latter changes when normalization or observation weights change, so matched controls must explicitly retain the same absolute coefficient to isolate those changes. Both the actual coefficient and the automatic reference scale are written to the solver status.

Stable full runs omit legacy count-space uncertainty/profile products. `lvd_stable_curvature.csv` instead describes the local robust residual Gauss–Newton curvature. It is not a calibrated posterior covariance or a detection significance.

## Segment derivatives and stationarity

The transport pullbacks now use local ForwardDiff tangents inside the outer Zygote reverse pass. This removes the finite-difference approximation from the default segment derivative. `set_adjoint_derivative_mode!(:finite_difference)` explicitly restores the legacy backend; changing `set_adjoint_fd_step` alone does not select it. The default mode is `:forwarddiff`.

A nearly cancelling data/prior gradient exposed a 0.737% error in one legacy segment derivative that could reverse the total objective-gradient sign. The applied-library regression now agrees with a local whole-flux finite difference to about 1.9e-9 relative error, with the primal flux bitwise unchanged. A separate [fixed-case replay](revision_evidence/surrogate_v2/forwarddiff_replay/README.md) retained all 96 stored flux values and 1,920 paired CSDA secants bitwise. Across 120 selected derivative cases its maximum AD/finite-difference relative discrepancy was 3.644e-8 at the fixed outer step 1e-6. These are deterministic derivative checks, not evidence that CSDA reproduces stochastic transport. The full package suite passed 731 tests on both Julia 1.10.12 and 1.11.7.

The derivative follows the executed smooth branch. Interpolation knots, adaptive transport decisions and cutoffs can still create genuine nonsmooth points. A tiny objective decrease or step does not certify the configured classical first-order KKT tolerance. Final solver status, directional diagnostics and comparisons across initializations must remain explicit even when execution succeeds.

## Recorded experiments

The completed [independent production validation](revision_evidence/surrogate_v2/production/README.md) reduced median absolute relative flux bias from 24.983% to 12.937% across 48 held-out ray/field cases. Median finite-contrast derivative bias fell from 25.254% to 21.373%; comparisons exceeding twice their estimated conditional Monte Carlo standard error fell from 16 to 6. The original validation's same-surrogate AD/finite-difference check reached 1.287e-8 with its recorded backend and step; the subsequent default-backend replay is reported separately above. Every tested field type improved in median flux bias, including the localized lens.

These are conditional average improvements. The maximum flux bias remains 59.936%, 43/48 flux comparisons still exceed twice their estimated standard error, and the 15-degree derivative subgroup worsens in median relative bias. The uncertainty excludes coefficient fitting and physical-model uncertainty. The selected derivatives emphasize sensitive crossed cells, not the entire Jacobian. Historical replay and independent testing are reported separately.

- `dev/validate_surrogate_v2.jl`: prespecified material training and unseen-field/direction/seed checks. Production uses 8192 histories per training reference and eight independent 1024-history test replicates. Includes a localized field with dry surroundings and boundary-aware derivative differences.
- `dev/replay_surrogate_v2.jl`: comparisons against the already inspected October 23 MC references. This is a historical comparison, not a blind test.
- `dev/run_stability_v2.py`: matched frozen-artifact starts at 0, 0.15 and 0.3, normalization controls and a screen-disabled fit; checks stationarity, common observations and field differences. Default is plan-only; `--execute` starts jobs.
- `dev/run_revision_job.py`: fresh output directory, bounded CPU affinity/address space, source/runtime identities and completion hashes. Launch inside tmux. Coordinate aggregate allocations across jobs; per-process address-space limits are not aggregate cgroup memory limits.

The October 24 full production and 12-case matched stability study remain in progress at this checkpoint. Their common absolute prior coefficient is 3.954930197858131. The durable production queue recomputes main, detection, contrast and exposure-sweep results from the frozen correction, then exports audited figures. The separate control queue includes three initial fields, normalization changes of ±5% and the disabled angular screen for both dry-only and material-corrected operators. A failed stationarity test remains a scientific result; an execution or provenance failure stops the affected queue. Final-field column-domain and quadrature checks are separate audits. Archived October 23 measured-field and sensitivity numbers are not results of this new configuration.

The source spectrum/altitude convention, uncertain LVD table provenance and acceptance, prescribed straight paths, geological support and observation model remain separate limitations. Improvements to these two numerical components do not resolve those physical ambiguities.
