# Convergence round 3 preregistration

Frozen main corrected endpoint: committed `material_main_recovery_20261024/completed/main/data`, objective 100.59269073659428, selected-branch projected gradient 0.003311897033113187, profiled normalization 3542719.29102373. No paper or core source changes.

The read-only helper checks saved field feasibility and unsupported zeros; saved support, screen and correction hashes; fixed robust scale, normalization and absolute prior; saved predictions, objective, branch gradient and normalization. A separate full value-only endpoint computation estimates reproducibility. Changes smaller than max(10 times repeat discrepancy, 100 eps(objective)) remain unresolved. This noise screen is not an error bound.

At most 43 directions and 258 objective-only trials: eight largest projected-gradient coordinates each feasible sign, projected gradient, diagonal GN, three damped GN directions, eight deterministic random directions (seed 20261025), and seven adjacent pairs among the top eight in both signs. Directions are normalized in infinity norm after tangent-cone projection; actual trials are projected into the frozen box. All directions use steps 1e-3 through 1e-8. Full direction vectors, actual displacement norms, objective/data/prior values, profile scale, one-sided secants, selected-branch predictions and resolution flags are retained. The direction count may shrink only for identically zero feasible directions, reported explicitly.

The diagnostic cannot establish generalized stationarity, even if all samples rise. Finite secants do not prove derivatives exist. Apparent corners will be traced to adaptive CSDA step-regime changes, logarithmic interpolation intervals, segment stopping and hard branch conditions; source inspection alone identifies candidates, not confirmed causes.

Launch requires independent code review and root authorization. Allocation: CPU0-7, 16 GiB address space, Julia 8 threads, BLAS1, tmux on research_vm_roland. Expected 5-30 minutes, external 60-minute wall cap. Exact argv saved proposed_command.json. A timeout is an incomplete diagnostic, never a stationarity result.

After evidence review a separate bounded polishing plan may use exact-objective descent along sampled feasible directions, with explicit attempts/rejections and unchanged physics, quadrature, prior, robust objective and support. No tolerance relabeling or smoothing is authorized by this preregistration.
