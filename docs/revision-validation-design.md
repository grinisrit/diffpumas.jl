# Next bounded validation design

Stochastic production remains paused pending the corrected transport reference gate.
Old v4 outputs remain historical diagnostics; these changes do not replace them.

The revision suite keeps scaling and acceptance on their original zenith domain
below 60 degrees. `--validation-zeniths 0,15,30,45,60,75` controls the separate
Jacobian and held-out validation pool. Seeded joint-quartile selection is followed
by replacing overrepresented zenith cases to include each requested valid zenith
with nonzero sensitivity. The case budget must cover the requested zenith count.
`validation_coverage.csv` records requested/valid rays, selected rays/pairs and
population nonzero entries by zenith; `selected_pairs.csv` records the actual
pairs. The independent audit supports both this policy and historical v4 policy.
This is selected coverage, not domain-wide validation.

Each held-out dry/wet pair shares energy and transport seeds. Different replicate
pairs and training calibration use separate streams. The frozen global correction
predicts both fields without a per-bin refit. Raw paired absolute flux-bias changes
are stored, and the plotting summary estimates their mean and standard error from
replicate differences, retaining covariance. Relative bias changes compare ratios
of replicate means; no ratio uncertainty is asserted. Calibration uncertainty is
not propagated. Few replicates remain a coarse precision diagnostic.

The additional dry comparison adds one transport evaluation per held-out wet
replicate: at 17 selected rays and four replicates this is 68 additional calls,
compared with 576 low/high calls for 24 cases, three fractions and four replicates.
A larger angular pool adds deterministic Jacobian work; high-slant transport cost
must be measured after the physics gate. Samples/replicates remain configurable;
64/128 sample defaults are exploratory, not a demonstrated production budget.
The next quadrature convergence probe must use the corrected physics and new
frozen calibration, including 60/75-degree directions, before choosing production
energy nodes and MC budgets.

Deterministic verification: real-topography tracing at 60/75 degrees in four
azimuths yielded eight valid nonempty paths. Configuration and seeded angular
coverage regressions exercise the new policy without stochastic transport.
