# ML surrogate comparison notes — discussion only

The user clarified that the requested scope is literature discussion and
comparison with the existing method. No learned surrogate, training run or ML
benchmark is proposed as a completed revision result. The filename records the
earlier planning request; this document is a comparison note, not an execution
plan. Installation's focused literature audit supplies the paper-specific
citations and separates published work from preprints.

## What is implemented here

The forward surrogate is a physics-based continuous-slowing-down (CSDA)
transport calculation using material stopping-power tables, prescribed straight
ray paths and deterministic energy quadrature. Its derivatives are obtained
through the implemented numerical/AD rules. A small parametric correction is
calibrated against stochastic energy-loss MC. It is therefore neither an
uncalibrated first-principles prediction nor a trained neural transport model.

Calibration has an offline data cost: the current production configuration uses
48 dry training directions with 8,192 MC samples per direction, while the
600-cell robustness fits use 24 directions at the same budget. Measured-fit
recursions recalibrate through the current reconstructed field, retaining the
specified geometric residual convention. These finite-dimensional coefficient
fits are data fitting; describing the whole approach as needing no training or
reference simulation would be misleading. The initial straggling coefficient
is pinned and the hard-loss coefficient is constrained; the geometric residual
is fitted sequentially, not by a joint unrestricted neural optimization.

## Comparison with learned surrogates

| Question | Current corrected-CSDA method | Learned surrogate discussion |
|---|---|---|
| Representation | Stopping-power physics and a small fitted correction on fixed paths. | A learned flux residual, transition kernel, generator or scalar objective can represent different targets; these are not interchangeable. |
| Data requirement | Explicit MC calibration directions/fields; repeated calibration has a measurable cost. | Reference simulation plus training, model selection and validation are needed. Reuse across geometries/materials is an empirical transfer question. |
| Differentiability | Derivative of the implemented surrogate, with piecewise/numerical rules and tested step sensitivity. | A differentiable network gives its own derivative; correct flux values alone do not validate material gradients. |
| Physics scope | Current directional reference includes stochastic energy losses on prescribed rays, without scattered boundary retracing. | A model trained on the same reference inherits this scope. Scattering-image or detector-objective surrogates do not establish deep-overburden density-Jacobian accuracy. |
| Cost | Physics evaluations, Jacobians, calibration MC and nonlinear fits must all be counted. | Dataset generation, training/search, uncertainty assessment, inference and gradients must all be counted; an inference-only speedup is not an end-to-end amortization result. |

The literature examples audited by installation have distinct tasks: learned
scattering-muography generation, local detector-objective surrogates, and neural
radiation-transport kernels. They motivate possible extensions, but do not
benchmark this LVD inverse problem. TomOpt-style differentiable detector
optimization also differs from learning the transport itself. Paper language
should identify each target rather than treating all uses of AD as ML.

## What the present evidence allows us to say

V5 demonstrates strong AD/finite-difference consistency for the same surrogate,
but substantial stochastic-reference discrepancy remains: the held-out wet
flux median bias is +27.8% (maximum +37.4%), and median paired wetting-response
bias is 30.4% of the MC wetting increment. A more expressive learned residual
might reduce error within its training distribution; these results do not
establish that it would generalize, improve inverse recovery, or remove an
incorrect source-boundary convention. Both approaches can learn or calibrate a
shared reference bias. The no-air run is a conditional boundary-convention
control, not an altitude-corrected physical truth.

A credible future comparison would keep whole rays, whole material fields and
independent energy/transport streams out of training and model selection.
Splitting individual histories from one simulated ray/field across train and
test sets would leak information. Held-out wet fields and independently paired
MC finite differences would test transfer and gradients separately. MC sampling
error, fitted-calibration uncertainty and learned-model uncertainty are distinct;
conditional replicate SE alone does not establish their combined coverage.

Positive flux can be imposed by a log-flux residual representation, but zero or
poorly sampled flux requires explicit treatment. Monotonicity constraints must
be conditional on the variable being changed: replacing dense rock by water at
fixed geometry can increase transmission, whereas changing composition at
fixed grammage also changes stopping powers. No global sign constraint should
be asserted without specifying that domain. Source spectrum, boundary altitude,
angular coverage, table version and material parameterization must be fixed and
recorded in either comparison.

No ML speed, accuracy, uncertainty or amortization result is claimed here.
