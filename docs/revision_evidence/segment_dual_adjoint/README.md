# Local forward-mode CSDA adjoint evidence

The prototype files are archived experiments, not package code. They add
process-local dual overloads and replace rules only inside the test process.
The recorded input field and physics hashes are in `adversarial_result.toml`.
The production implementation is commit `c5d2c59`, with primal-only comparisons
and deterministic left ties added after independent review. The original
prototype did not harden every adaptive tie; it must not be included into a
production run.

The saved adversarial case is bin 916, cell 275, water fraction
0.11471926542607168. Its flux is bitwise unchanged. The prototype local tangent
3.1569060697022277e-6 agrees with whole-flux differences at h=1e-7 and h=1e-8,
whereas the former h=1e-6 internal numerical adjoint gave 3.1801601296528428e-6.
This correction matters near objective stationarity because data and prior
terms nearly cancel. It is a deterministic differentiation diagnosis, not an
MC derivative validation or a reconstructed-material accuracy result.

The 500-evaluation timing is a short warmed local microbenchmark at one segment,
not end-to-end throughput or a general speedup guarantee. Central FD differences
at some interpolation/adaptive knots remain large (including the reported local
energy/weight derivative discrepancies); a selected-branch tangent is not a
classical derivative at a nondifferentiable knot. An initial uniform/air check
at exactly 100 GeV failed central-FD assertions; the successful smooth-branch
check uses 123.4567 GeV instead. The package regressions separately test explicit
interpolation endpoint/interior-knot conventions, seed-invariant branch ties,
valid zero weight and zero-distance identity.

`focused_tool_transcript.txt` transcribes actual interactive command output.
There was no formal wrapper `end_status.json` for these focused tests, so none
has been invented. Full integrated tests and independent applied-library replay
will have their own formal run audits. `provenance.json` binds these archived
files, relevant runtime source hashes and the prototype input files.
