# Fixed-case ForwardDiff precision replay

Completed with source, runtime, reference-input and raw-output hash checks unchanged. The actual segment-adjoint backend is `forwarddiff`; the recorded legacy FD step 1e-6 is explicitly inactive. No MC transport, coefficient refitting, cell reselection or reference replacement occurs.

All 96 stored deterministic full-flux predictions (before/after material correction across 48 held-out ray/field cases) and all 1,920 stored paired-energy CSDA secants (before/after across 384 new-test and 576 historical replicate rows) reproduce bitwise under the new source. The immutable MC references and their producer source hashes remain those of the original runs; this replay does not relabel them as newly generated.

The exact same 48 new held-out cell cases and 72 historical ray/cell/fraction pairs are checked with local AD versus physical-box-aware finite differences at 1e-4 through 1e-8. At the original fixed 1e-6 check step, all 120 pass the 1e-4 relative-error gate. Maximum relative error is 3.6441713391958555e-8 overall (3.003444189323168e-8 on the 48 new cases; 3.6441713391958555e-8 on the 72 historical cases). Full step-sweep rows are retained.

This substantiates unchanged selected-case primal predictions and numerical derivative consistency after replacing step-sensitive segment pullbacks with local forward-mode tangents. It differentiates the executed piecewise smooth deterministic discretization. Adaptive/interpolation knots, remaining surrogate/MC response bias, calibration uncertainty and the fixed 30 km source-boundary convention remain separate limitations. See the production and historical evidence for unchanged physical-reference discrepancies; no water-recovery accuracy claim follows from these AD checks.
