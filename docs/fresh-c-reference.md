# Fresh flat-column C reference

The archived `examples/data/flux_comparison.{xml,csv}` have no recorded C build,
seed, or sample-count provenance. The fresh driver never reads or overwrites
them. Its C implementation is the official `niess/pumas` release tag `v1.2`,
commit `529976ef51bd420f37ec63b1d3f615784d35b626` (library version 1.2.3).

PUMAS v1.1 was tried first because the pinned stopping-power files identify that
version. Its unmodified geometry example crashes on this host during Coulomb
transport: UBSan identifies a 16-byte workspace-alignment violation. Official
v1.2 contains the alignment fix. This is a newly pinned reference comparison,
not a reconstruction of the unknown archived C executable.

Both languages use `examples/data/revision_physics/materials.xml` and the three
pinned upstream muon stopping-power files. The C initializer imports those
existing files with default `update=0`; the preparation utility checks that their
hashes remain unchanged. C and Julia have separate binary serialization formats.
Their dumps are therefore generated separately from the same physical inputs.
Cutoff and elastic ratio are 0.05, radiative models SSR/SSR/DRSS, rock density
2650 kg/m³, primary altitude 1000 m, and the mode switch is 100 GeV. Below it,
straggling and scattering are enabled; above it, mixed loss without scattering
is used. This tests the two transport implementations; it does not assume that
their complete internal tables or numerical algorithms are identical.

The only adaptations to the official C geometry example select an explicit dump,
MC sample count and seed, retain 17-digit output, and compute sampling errors for
open sky instead of forcing them to zero. The generated C geometry disables the
air magnetic field because the Julia flat transport kernel consumes only the
atmospheric density. The generated C surface-flux wrapper aligns its rounded
muon mass constant (0.10566 GeV) to Julia's 0.10565839 GeV. Both source patches
and hashes are retained; the official PUMAS physics library is unchanged. This
compares transport using a common parameterized surface spectrum, rather than
independently validating that surface spectrum or magnetic transport. The Julia driver uses the same
log-uniform energy and equal-charge proposals through its public transport API.
The same integer seed and sample count are recorded for both languages; their
random streams are independent implementations, not common random histories.
Each point resets its language's seed, so estimates across grid points can be
correlated. Per-point z-scores do not constitute 102 independent hypothesis tests.
The reported standard error is the population second moment divided by the MC
sample count. No model/systematic uncertainty is added in this nominal comparison.

Preparation, using an isolated source checkout and an external build directory:

```bash
git clone --branch v1.2 https://github.com/niess/pumas.git /absolute/source
python3 dev/run_bounded.py --cpus 2 --memory-gib 4 -- \
  cmake -S /absolute/source -B /absolute/build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPUMAS_BUILD_EXAMPLES=ON
python3 dev/run_bounded.py --cpus 2 --memory-gib 4 -- \
  cmake --build /absolute/build --parallel 2
python3 dev/run_bounded.py --cpus 2 --memory-gib 4 -- \
  python3 dev/prepare_pumas_reference.py --source /absolute/source \
  --build /absolute/build --output /absolute/reference
```

The preparation script refuses an existing dump before modifying any reference
files and verifies the exact source commit. The comparison runner verifies the
current executable, shared-library, and C-dump hashes against that provenance
before simulation. It records hashes of
the original/adapted geometry, compiled executable/library, C dump, and physical
inputs, together with the compiler and physics-creation commands. Select new
output directories; the utilities refuse to replace existing dumps/results.

Generate a **new** Julia dump from the pinned inputs with the corrected parser,
radiative integrals, and versioned recipes:

```bash
JULIA_BIN=/path/to/julia1.10 bash dev/create_revision_physics.sh /absolute/new-julia-physics
```

Legacy Julia dumps support deterministic use but lack the recipes required for
the corrected stochastic kernel. Julia dumps generated on 1.10 can be loaded on
1.11; regenerating on the target runtime avoids Serialization compatibility
limits. The corrected comparison uses source-energy cutoff 1e9 GeV in both
languages; `--source-energy-max` can select a lower common cutoff. The historical
pre-fix comparisons used 1e12 GeV. Those outputs are preserved, and the cutoff
control/incident-tail bound is recorded separately.

Run a three-point smoke test before the full 102-point grid:

```bash
python3 dev/run_bounded.py --cpus 2 --memory-gib 8 -- \
  python3 dev/run_flat_comparison.py --reference /absolute/reference \
  --julia-dump /absolute/new-julia-physics/materials.pumas \
  --output /absolute/smoke --scope smoke --samples 100
python3 dev/run_bounded.py --cpus 2 --memory-gib 8 -- \
  python3 dev/run_flat_comparison.py --reference /absolute/reference \
  --julia-dump /absolute/new-julia-physics/materials.pumas \
  --output /absolute/full --scope full --samples 10000 --seed 42
```

For shared-host work, an outer coordinator can put `taskset -c 24,25` immediately
after the bounded launcher's `--`. The default limits are per process, inherited
by children; they do not create an aggregate memory cgroup.

The full grid is 0:50:800 m and zeniths 0°,15°,30°,45°,60°,75°, integrating
1e-3 to 1e9 GeV. C and Julia each simulate 1.02 million histories at the default
sample count. Outputs contain per-language CSVs, individual C logs, the Julia
log, input/driver hashes, MC errors, relative differences, and a difference z-score
using the independent MC errors. A completed marker is written only after both
languages finish and the archived XML hash is checked again.
