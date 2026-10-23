# Improvements motivated by the completed revision

The five main stages and all ten robustness scenarios completed on 2026-10-23, with successful source audits. These recommendations are proposals for subsequent work, not claims that the present revision has implemented or validated them. The completed calculation is a conditional method demonstration; it does not establish a groundwater inventory or calibrated discovery significance.

## 1. Reduce forward-model transfer error

The frozen dry calibration has a median wet-field flux bias of +27.8% and a maximum absolute bias of 37.4% on the tested directions. In contrast, the final-field 32-versus-512-node quadrature control differs by at most 0.10335% in flux and 1.56756% in selected finite-difference components. The quadrature comparator is not exact truth, but this evidence prioritizes material-transfer correction over simply increasing integration nodes.

First reconcile the source spectrum and its altitude with the transported air column. Then evaluate a small material-conditioned correction across dry, intermediate and wet fields, slant thickness and zenith. Reserve whole fields, directions and random seeds for validation. Score both flux error and paired stochastic derivative error; a good flux fit alone is insufficient. A differentiable correction remains a candidate until it improves held-out flux and derivative errors without sacrificing stability. A trained ML implementation is outside this revision's agreed scope; the paper already discusses learned surrogates and their costs.

## 2. Establish convergence and identifiability

Seven of ten robustness fits exhaust the three-pass limit. Initial calibration at water fraction 0.3 changes the recovered field by 19.04%; normalization shifts of −5% and +5% change it by 12.00% and 5.49%. These are full-grid relative L2 differences from the separate robustness baseline, not confidence intervals. Continue selected baseline and alternate-initialization fits to a verified projected-gradient/KKT and objective criterion; use multiple starts. Profile or jointly estimate normalization under an explicit observation model, and report correlations with composition parameters.

Removing residual screening produces a zero field with a much worse common-bin residual. Raising the support cutoff to 900 m changes the field by 41.59%, partly by forcing removed cells to zero. Compare candidates on fixed held-out observations and fixed common spatial domains. Test an explicit contamination or heavy-tailed observation model instead of interpreting one residual-selected solution as a physical detection. Choose screening rules without examining the held-out outcomes.

## 3. Resolve the measurement model before physical interpretation

Establish the exact Aglietta publication/table, units, event sample, exposure, rebinning and detector acceptance. Verify terrain against an independent elevation model and propagate plausible terrain perturbations. If the scientific target is pore saturation, replace the current rock-replacement mass-mixture parameterization with a model that specifies porosity and a fixed solid skeleton; the current fitted mass fraction cannot be relabelled as groundwater volume.

Acceptance and incident-flux nuisance estimation has a muography precedent: Lesparre, Gibert and Marteau jointly invert telescope acceptance and flux [1]. This motivates testing nuisance parameters here; it does not validate a joint model for the presently undocumented table. The [publisher record](https://academic.oup.com/gji/article/188/2/490/584718) confirms the study scope.

## 4. Separate sampling noise, model error and detection claims

Repeat the synthetic benchmark with independent MC templates and increased histories at informative rays. The current 1024-history template has median marginal sampling error 7.95%, which is not propagated through the inverse weights. Use independent template replicates to distinguish the exposure plateau from solver error, and preserve paired comparisons across solvers. Equalize objectives, priors and stopping criteria for any claim about the benefit of nonlinear updates alone.

For detection, generate finite-contrast alternative fluxes and null/alternative observations at each candidate exposure, then rerun the bounded reconstruction. Report uncertainty in false-positive rate and power. Current nominal matched-filter/SART comparisons use first-order signals and fixed-reference extrapolation. SART values of 100 in the completed headline and contrast-0.8 outputs hit the search floor and are upper bounds on the crossing within that surrogate calculation, not exact exposure requirements or exact gains.

Align uncertainty calculations with the objective actually optimized, including its nuisance variables and weighting scheme. Test coverage in replicated synthetic experiments before naming inverse curvature a posterior covariance. Reduced parameterizations and explicit approximation-error likelihoods are relevant background [2], but their performance in another tomography problem is not transferable validation.

## Literature checked through Consensus

[1] N. Lesparre, D. Gibert and J. Marteau (2012), [Bayesian dual inversion of experimental telescope acceptance and integrated flux for geophysical muon tomography](https://consensus.app/papers/bayesian-dual-inversion-of-experimental-telescope-lesparre-gibert/3182ca3fa9df5e859104eff76b79da79/?utm_source=chatgpt), *Geophysical Journal International* 188, 490–497. DOI: 10.1111/j.1365-246X.2011.05268.x. Consensus recorded 22 citations at retrieval.

[2] G. Meles, N. Linde and S. Marelli (2022), [Bayesian tomography with prior-knowledge-based parametrization and surrogate modeling](https://consensus.app/papers/bayesian-tomography-with-priorknowledgebased-meles-linde/4036adab4a3c510b97c64e97e09735b7/?utm_source=chatgpt). DOI: 10.1093/gji/ggac214. Consensus recorded 12 citations at retrieval; this fetched record omitted journal metadata. The application is ground-penetrating-radar travel-time tomography, not muon transport; the [authors’ manuscript](https://arxiv.org/abs/2201.02444) provides a primary-source cross-check.

Raw fetched records are preserved in `revision_evidence/improvement_literature_20261023.json`. Numerical evidence is in the paper's completed main/validation/sensitivity archives and the final production reports; the recommendations above are our interpretation of that evidence.
