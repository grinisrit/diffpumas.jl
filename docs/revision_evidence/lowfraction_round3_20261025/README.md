# Fresh low-fraction physics-correction pilot

The preregistered candidate is rejected by this screening experiment. The original production coefficients remain unchanged; no coefficient adoption, heldout retuning, or automatic second run follows.

The run started 2026-10-25 at 00:15:13UTC after independently validated convergence investigation, and completed cleanly 00:43:42UTC (28m29s). Source, runtime, and recorded inputs remained unchanged. It used CPU 8–15, 16 GiB maximum, 8 Julia threads, and a 5400 s timeout with 30 s kill grace. The separate convergence investigation did not establish optimizer stationarity; that is not a claim made by this pilot.

Two material coefficients were fitted to completely new training directions with 4096 histories per dry/wet field at uniform cell mass fractions 0.025, 0.075, 0.15. The frozen coefficients are linear −0.48271728856140567,quadratic 0.5339669456974949. Twelve new heldout directions and four new fields used 1024 histories in eight independent replicates, with paired dry/wet/low/high energy and initial transport seeds within each case/replicate. Training and heldout streams are independent. Full design, seeds, exact source and analysis are retained in frozen_inputs/; no new trained ML model was used.

The material correction multiplies flux by exp(linear×q+quadratic×q²), where q is actual whole-column water mass fraction including remaining dry rock and excluding air. Training q spans 0.025–0.15; heldout q spans 0.02280504749–0.12185510780. One of 48 heldout cases is below training support; none is above. Geometry and q for every field/ray, dry anchor, and derivative endpoints are retained.

| Endpoint | Frozen production control | Low-fraction candidate |
|---|---:|---:|
| Median absolute material-ratio residual |2.73133%|2.71873%|
| Maximum absolute material-ratio residual |20.3602%|21.7296%|
| Median absolute flux residual |10.5956%|11.4585%|
| Maximum absolute flux residual |22.1491%|23.6819%|
| Median absolute finite-contrast derivative bias |4.84937e-5|5.05680e-5|

The material-ratio median gain is 0.46145%, with descriptive paired jackknife SE 29.0589 percentage points; the required 30% gain is unresolved. Absolute-flux bias worsens beyond twice paired SE in 42/48 cases; derivative bias worsens by that criterion in 13/48 cases. These reference-based endpoint failures determine rejection. Derivative median-bias gain is−2.07425e-6 with SE 5.87649e-6, failing the deliberately stronger screening requirement of improvement beyond 2 SE. All case outcomes and uncertainty flags remain in pilot/assessment.json. Eight-replicate conditional jackknife SEs do not guarantee confidence coverage, and do not propagate calibration uncertainty.

Local derivative consistency has 90 stable arm/case ladders and 6 larger-step crossing candidates. Maximum AD/FD discrepancy is 1.93348e-4; those six cases agree at the smallest tested step. Per-step errors and tags are reported. These are unresolved local checks, potentially reflecting known interpolation kinks, and are not counted as proof of incorrect AD or substantive calibration rejection.

The historical dry correction is unchanged. It uses alpha 1/hard 0 plus exp(a+b logL+c logL²+d cosθ), calibrated on 0–45° and 1193.13–2297.33m slants. The previous worst dry ray 75°/277.5° has slant 4752.06775m, outside both supports; its material multiplier is 1. The bounded deterministic audit finds only −0.07932% change between 32 and 256 energy nodes on that ray (maximum absolute 0.09051% over 18 rays). Those observed changes are insufficient to account for the previous59.936% discrepancy;256 nodes are not truth and no asymptotic guarantee or causal attribution follows. Source altitude remains 30 km; its physical adequacy is unresolved. No global dry refit is adopted.

pilot/ retains raw training and heldout data, frozen coefficients, structured fit diagnostics, q-domain summary, predraw identities, wrapper execution/end records, frozen assessment, and analysis provenance. frozen_inputs/ preserves exact helper/analysis/plan/launcher/test snapshots with hashes. dry_diagnostic/ preserves the successful preliminary audit and exact driver recovered with SHA256 matching its original execution identity. The first diagnostic attempt failed parsing before calculations; its immutable failure record and exact hash-matching failed source are retained under dry_diagnostic_failed_attempt/. Old records were not rewritten.

Transport scope is fixed prescribed rays with stochastic energy loss and scattering disabled. This experiment does not establish water-recovery accuracy or repair the dry baseline. Synthetic summary tests cover zero-bias candidates, paired identity, denominator failure, and finite-step interpretation; they pass without physics simulations.

The [independent completed pilot assessment](../round3_independent_review_20261025/pilot_review.md) records execution/validation PASS separately from scientific screening REJECT. It verifies all 65 producer bindings, raw tables and 864 case statistics, exact frozen-summary reproduction, and preselected 8192-history serial MC replay with bitwise agreement for all eight reference fluxes.
