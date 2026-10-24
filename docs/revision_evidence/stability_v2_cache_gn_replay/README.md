# One real GN outer-attempt cache replay

Completed 2026-10-24 13:05:42 UTC; single CPU20/5GiB. All source/input and output digests independently rechecked. Saved dual dry0 final field, same fixed objective/prior, copied production algorithm with only unique name/trace instrumentation. Both variants start with fresh default LM damping; this is not the original fit history or a new convergence result.

All 124 complete candidate vectors, residual predictions, objectives, directional derivatives, LM/halving indices and final solution/history/gradient diagnostics match exactly. No trial was accepted at this endpoint. A separate small accepted-step fixture is required.

The bounded serial predecessor cache used 91,094 ray evaluations and 33,278 exact hits out of 124,372 callback ray opportunities (26.76% avoided). Full model/J computations remain uncached and excluded from those callback counts. Observed elapsed costs were 704.63s baseline and 469.29s cached; they include initial full model/J and new function/JIT specialization, so their ratio is NOT a steady-state speed claim. Warm paired callback microbenchmark is archived separately.

This replay uses the serial predecessor cache under an explicitly immutable physical context. It is reference evidence only, not a certificate for the guarded threaded successor. The latter must exercise the exact frozen entry replacement closure, both artifacts, and threaded/context guard gates before deployment.
