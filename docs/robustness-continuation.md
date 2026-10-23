# Resource-only continuation of nonlinear robustness

`dev/continue_revision_robustness.py` prepares and sequentially runs the remaining
robustness scenarios after the complete five-stage main pipeline has exited.
It never kills or signals a process. The parent/operator first stops the old
robustness driver safely; this helper refuses live drivers or Julia workers,
including a main driver between stages.

Validate without writing files:

```bash
python3 -B dev/continue_revision_robustness.py \
  examples/data/revision_robustness_corrected_20261023 \
  --main-run examples/data/lvd_revision_corrected_20261023 --dry-run
```

Execution uses the same command without `--dry-run`. The default allocation is
20 CPUs (0–19) and 36 GiB address space, subject to `run_bounded.py`'s host/cgroup
limits. An incomplete existing scenario causes refusal. Only after the recorded
root is proven stopped and owned may the operator explicitly add
`--preserve-incomplete`: the entire incomplete directory is moved under a new
`continuations/<UTC>/preserved_incomplete/` before its replacement is created.
Nothing is overwritten or treated as a completed fit.

The main prerequisite requires its successful execution/source audit, all five
nonempty reports, all five successful stage markers, and the final launcher
completion marker. A main failure must be diagnosed separately. This helper does
not infer aggregate host idleness beyond those specific recorded research jobs;
the operator must keep unrelated allocations within the shared resource budget.

The ten scenario names and numerical command tails come from the recorded
baseline configuration plus the original fixed scenario overrides. Both ±5%
normalization variants derive from the verified completed baseline predictions.
Only CPU count, memory limit and the initial Julia wrapper may change; the actual
Julia binary/version/hash, sources, dump, environment identity, numerical
arguments and seeds must match. Original completed four-CPU cases are retained
and skipped only after identity and output hashes pass. Completed continued
cases additionally require their prior continuation audit, shim and invariants.

Each continuation records an immutable root manifest, self/helper/shim and input
hashes, main evidence hashes, prior/new resource allocation, per-case identity
invariants, actual worker affinity, and an aggregate end audit. A worker whose
observed affinity is missing or wrong is not marked complete, and no next case
starts after it exits. There is no automatic process termination. The latest
successful aggregate audit explicitly binds all ten retained identities and
output hashes, including cases completed by an earlier partial continuation.

Execution completion is separate from scientific convergence. Existing
`lvd_recursion_status.txt`, histories, and comparison metrics remain the evidence
for convergence and robustness; extra resources do not change the fitted
algorithm or validate its accuracy.
