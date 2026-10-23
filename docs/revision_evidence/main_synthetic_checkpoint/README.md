# Main synthetic checkpoint — full main stage still running

The initial synthetic reconstruction returned and its report/field/mesh files
were closed before the measured-data fit began. This snapshot preserves those
completed subtask outputs and a checkpoint of the still-growing main log. It
does NOT claim completion of the measured fit, closure or five-stage pipeline.
Original manifests/execution metadata retain their absolute source identities.

Use the exact printed values in `lvd_tomography_inverse.txt` as authoritative:
GN RMSE **9.9908e-2**, SSIM **0.5696**. Console messages have lower precision;
CSV fields have six decimal places and were not used to invent extra metric
digits. Earlier chat GN values 0.099911/0.5703 were transcription errors, not
evidence of a changing report or an independently recomputed metric.
`reported_metrics_and_scope.json` parses all four finalized report rows and
records this distinction explicitly.

GD has slightly lower RMSE than GN in this realization; both improve only
modestly over MLEM, whose SSIM is higher. The height band 1054–1406 m has lower
MLEM RMSE than either nonlinear solver. These results supersede the stronger
archived ranking claims; differing objectives/priors/budgets and the finite
MC template prevent attributing the comparison to Hessian use alone.

The report header `regularisation weight = 0.0` is a generic CLI setting, NOT
a no-prior claim: frozen source selects nonzero curvature/column-sum-scaled
priors when this input is zero. The GN count of 13 denotes stored history
entries including its initial state; the actual iteration cap is 12, with no
separate convergence certificate exported.

All 21,600 finite synthetic rays lie at 0 ≤ theta < 60 degrees; they do not
cover the full measured sky. Coordinates and nominal feature bands are heights
above the detector. RMS mesh values describe the initial rich synthetic field
and only the top 150 ranked cell centroids. The selected-12 sigma scores remain
descriptive because endpoint covariance and other uncertainty terms are omitted.
