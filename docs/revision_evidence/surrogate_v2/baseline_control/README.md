# Historical baseline reproduction control

All 21 historical validation rays were recomputed at both the dry and the original spatial wet field, using the frozen dry correction with the new material correction disabled. All 42 flux predictions reproduce their stored historical values exactly: maximum absolute relative difference 0.0 (preregistered numeric check tolerance 1e-10). Source, runtime and output hashes pass the bounded launcher's checks.

This isolates the new material multiplier as the change in the historical before/after comparison. It does not validate the 30 km atmospheric source boundary, fixed-path geometry, or stochastic physical accuracy. `driver_as_executed.jl` preserves exact executed bytes; its relative includes assume the original path in `execution.json` under `examples/data/`, and the recorded working directory is the repository root.
