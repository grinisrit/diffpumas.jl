# Convergence investigation, round 3

The unchanged objective admits small validated decreases. This round does not establish generalized stationarity. The final selected-branch tolerance 1e-6 status is recorded explicitly in final_summary.json. Original baseline sources and fitted fields were preserved; new results and manuscript additions are supplemental evidence.

| Endpoint | Objective | Selected-branch projected gradient infinity norm |
|---|---:|---:|
| Frozen main | 100.59269073659428 | 0.003311897033 |
| First polish | 100.59269066369612 | 0.003545350203 |
| Revised polish | 100.59269064139075 | 0.003671222111 |

The diagnostic checked 43 finite feasible directions at six step sizes (258 exact objective trials), including coordinate, coupled, damped GN and seeded random directions. Twenty-two trials decrease the objective above the numerical repeatability screen. These finite secants are not a generalized stationarity test. The independent value-only baseline repeated identically; the numerical screen is 1.4210854715202004e-12, not a rigorous error bound.

The first polisher accepted 35 exact decreases in 827 trials using 20 full gradient assemblies. Its full endpoint objective and gradient were independently reproduced by the reviewer. One six-gradient auxiliary QP remained materially suboptimal despite a small weight update; a Frank-Wolfe gap exposed this limitation. The revised method uses exact pairwise simplex transfers and records an explicit finite-QP optimality gap.

The revised pass eliminated 53 supported cells absent from all retained fitted rays by exact conditional graph-prior minimization, leaving 94 supported cells crossed by fitted rays. All eight unseen components were anchored. Fitted fluxes and profile normalization remain exactly unchanged for this harmonic-only step. Its objective change is -3.5967673284176271e-11; unseen prior-gradient infinity residual is 3.5527137e-15. The explicit reduced derivative is Tᵀg, checked against smooth toy finite differences. Every physical sample checks full-field radius and harmonic feasibility.

Two revised sampling radii use 21 gradients each (base, 12 random reduced neighbors and both signs of four dominant coordinates), comparing six-gradient and enriched finite hulls at the same center. The revised pass accepted 2 exact decreases in 8 line trials using 45 full assemblies. Total objective decrease from the frozen endpoint is 9.5203532168852689e-08. Termination is the preregistered finite budget. Finite sampled hull residuals and their auxiliary QP gaps do not certify the Clarke hull or constrained stationarity.

Tracing the largest baseline branch-gradient coordinate, cell 375, localized the numerical corner. At perturbations ±1e-8, only bin 831, energy node 16, final air step 1177 changes an interpolation interval; adaptive step counts, limiter regimes and air floors do not change. The actually used Air table (material index 3) has knots 25000, 30000 and 35000 GeV at indices 118:120. Its predictor/next-energy interpolation crosses the 30 TeV knot. The profiled robust envelope-score calculation attributes 99.9529283% of the finite 1e-8 objective slope jump to this ray. This is finite-scale mechanism localization, not a derivative-limit proof or a convergence certificate. Raw trace, source-parity checks, score algebra and input hashes are retained in trace_mechanism_assessment.json.

Tests passed: 11 first-direction checks; 13 exact pair-QP cases, 12 actual saved-gradient QP checks and 8 harmonic/explicit reduced-derivative checks; independent SciPy comparisons of all three saved convex subproblems. Every traced flux matches its production counterpart within relative 1e-13. Final forward/J evaluations reproduce incrementally cached objectives and predictions. Source/runtime/output hash verification for all five bounded jobs is recorded in final_summary.json. Independent reviewer evidence is separate under round3_independent_review_20261025.

All physical jobs ran sequentially in tmux on research_vm_roland, CPU0-7, at most 16 GiB address space, eight Julia threads and BLAS1. Raw execution records contain exact argv, Julia binary identity, source/input hashes and status; no tolerance was relabeled, no objective was smoothed, and no core transport source or primary fitted result was changed. Manuscript additions are supplementary only. No further polishing is authorized in this round. Remaining stationarity limitations must accompany any downstream calibration pilot.
