# Backward stochastic transport and reproducible tables

Fresh physics objects retain the actual MDF elemental mass fractions, including
flattened composites, in immutable recipes. `save_physics` writes a version-2
envelope with canonical SHA256 fingerprints of recipes and numeric physics,
and the radiative-table algorithm version. `load_physics` verifies both.
The physics struct itself retains its old layout. A legacy raw serialized dump
can still be used for deterministic calculations; stochastic transport requires
regeneration from the original MDF and stopping tables into a **new path**.
Adjacent XML and material names are never used to guess missing compositions.
Julia Serialization is not backwards portable across all Julia releases: generate
with Julia 1.10 for use on 1.10/1.11, or regenerate on the target runtime. Canonical
fingerprints do not remove that underlying wire-format restriction.

`MixedTransport.prepare_mixed_kernels(physics, materials)` validates the physics
once and builds independent kernel snapshots for a batch. Registry entries use
`objectid(physics.tables)` plus a verified weak identity, with locked access;
structurally equal vectors do not alias. Copy/deepcopy does not inherit recipes.
Contexts must be used with their original physics and material. Do not mutate
physics or contexts during a batch: low-energy fluctuation/scattering routines
also read the original physics arrays. Preparing a new batch detects mutations.

For restricted continuous stopping power S(K) and total hard rate sigma(K), the
straight-line MIXED kernel integrates range R=integral dK/S, optical depth
H=integral sigma*dK/S, and proper path T=integral m*dK/[p(K)*S]. It samples an
exponential increment of H, competes with the remaining material path and energy
ceiling, and consumes every event-truncated piece of the requested segment.
Continuous backward weight is S(K_before_event)/S(K_start), times weighted decay
exp(-deltaT/(density*ctau)). An instantaneous event contributes no path or time.
Integrals use a refined logarithmic grid (eight subdivisions per source interval)
and trapezoidal quadrature; this is numerical integration of the package's
interpolated tables, not bitwise reproduction of C's PCHIP implementation.

For an event ending at Kf, propose nu with density proportional to nu^-2 over the
physical interval, and set Ki=Kf/(1-nu). The event multiplier is

    Ki * sum_element,process[f_element * atoms_per_kg * raw_dsigma_dq(Ki,Ki-Kf)]
    / [pdf(nu) * (1-nu) * sigma(Kf)]

Raw differential models are SSR bremsstrahlung/pair production, DRSS photonuclear,
and electronic ionisation. The target element/process is drawn from the same
summed differential kernel, including electronic events. At the low-energy
zero-rate onset Kt, the official regularised rate and null-CEL mixture are used;
the hard proposal additionally divides by its non-null mixture probability.
Below 100GeV, the existing straggled continuous approximation uses a step-local
endpoint rate majorant and one-candidate thinning, as in the official stepping
algorithm. The same adjoint event multiplier is then applied. This low-energy
scheme remains a numerical stepping approximation and requires C controls;
passing a high-energy test alone does not establish full MC equivalence.

The separate [radiative angular-moment limitation](coulomb-angular-moment-limitation.md)
records residual Coulomb scattering-table differences, including a sampled
20.5% transport-path discrepancy. It is not an equivalent bound on flux error.

The flat `Geometry` API follows deflected directions across layer boundaries.
The tomography `DirectionalPath` API instead transports energy on an already
computed path and rejects `scattering=true`: it cannot retrace cell/topography
boundaries after deflections. Its spectrum uses the original path direction.
Segment loops subtract the actual travelled distance after events or mode limits.

The default source-energy ceiling in flat geometry is the smaller of 1e9GeV and
the supported table maximum. `source_energy_max` can choose a lower ceiling;
explicit requests above the grid fail. A particle reaching the ceiling before
the source surface contributes zero. Fixed-path tomography uses the same ceiling.
This is a finite source-energy domain, distinct from the detector-energy proposal
range. Corrected C comparisons explicitly use the same 1e9GeV source cutoff;
pre-fix archived comparisons used 1e12GeV. Their cutoff effect must be measured
before treating them as otherwise identical controls.

Rebuild pinned inputs without overwriting existing evidence:

```bash
JULIA_BIN=/path/to/julia-1.10 bash dev/create_revision_physics.sh NEW_DIRECTORY
```

The historical dumps and pre-fix comparison CSVs are retained. A regenerated dump
changes restricted stopping powers, hard rates, and derived moments after the
radiative-unit correction. Restoring the first imported 1 MeV row adds one grid
node; the remaining imported CSDA records keep their original input values.
