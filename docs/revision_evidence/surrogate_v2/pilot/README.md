# Material correction v2 pilot: implementation smoke test

Completed 2026-10-24 with 128 MC samples and four independent replicate streams. The unchanged source, runtime and raw output hashes were independently checked against `end_status.json`. `derived_provenance.json` binds the separately computed summary to the immutable raw rows and summary helper.

The frozen historical dry correction is retained. Two empirical material coefficients are fitted only to paired dry/wet contrasts at uniform water fractions 0.2 and 0.6 on 12 training directions. The 12 test directions, four fields and test streams are excluded from fitting. Two smooth spatial fields and one localized 0.6 lens with zeros elsewhere supplement a uniform 0.35 field. Source altitude remains 30 km for this controlled comparison; the unresolved atmospheric/source-boundary assumption is not repaired here.

The pilot is not a final adoption gate and no coefficient, ridge, split or acceptance retuning follows its results. Final production repeats the preregistered experiment with independent streams, 8,192 training samples per field/direction, and 1,024 samples in eight test replicates.

Aggregate median absolute relative flux bias changes from 21.62% to 11.14%; maximum from 69.56% to 65.75%. Localized-field median changes from 27.03% to 21.16%. Paired-energy derivative median absolute relative bias changes from 41.78% to 35.66%, with 11 versus 9 of 48 comparisons exceeding twice their conditional replicate standard error. Derivative medians worsen in test fields 1 and 3; weak MC derivatives produce unstable relative ratios. Maximum AD versus same-surrogate finite-difference relative error is 1.25e-8. These observations establish implementation consistency and motivate the already specified precision increase, not physical derivative accuracy or water-recovery accuracy.

Conditional standard errors omit fitted-coefficient uncertainty and cross-bin covariance. The few replicates do not calibrate confidence coverage. Scattering and boundary retracing remain outside this fixed-path transport reference.
