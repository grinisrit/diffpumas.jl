# Deferred reporting wording correction

This patch is reviewed but **not applied** while production source audits are active. It changes comments, docstrings and explanatory report/console labels; numerical algorithms, public API names and CSV columns are preserved. The text qualifies weighted-residual amplitude scans, heuristic null-model scores, local fluctuation proxies and the centroid-based resolution diagnostic.

The paper parser in commit `83c2705` accepts legacy, current nominal and planned reporting labels. Its fixture tests preserve numerical values and explicit non-likelihood scope, including data-bundle materialization. This prevents loss of fitted-field markers or stored null-model diagnostics when the truthful labels replace historical strings. Synthetic Poisson detection thresholds are a distinct task and are not relabelled by this patch.

Apply only after all production source audits and strict report builders finish. Recheck patch applicability, run integrated tests and retain the computation's original source identity. No new numerical results are established by this wording correction.

The separate `deferred-detection-wording.patch` qualifies the synthetic detection benchmark: local-Gaussian Fisher estimates, finite first-order templates, and fixed-reference bounded-SART extrapolation do not certify exact false-positive rates or power. It preserves numerical expressions and machine result lines while adding explanatory limitations. Independent source review and a combined apply-check with both shallow-mixture patches and the original reporting patch passed. None of these patches is applied.
