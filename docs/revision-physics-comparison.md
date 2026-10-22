# Historical and pinned revision physics tables

This inspection compared the existing binary dumps without changing either one.
The numerical checks used Julia 1.11.7, two Julia threads, one BLAS thread, and the
working source on 2026-10-22. They establish numerical differences and consistency
with the supplied inputs; they do not establish transport accuracy against PUMAS C
or recover an undocumented historical generation command.

## Identified inputs

| File | SHA-256 |
|---|---|
| `examples/data/materials.pumas` | `3b94e176fcff9b29480768be72c2eeaf5527bc9484738bbf58a448078dd752f3` |
| `examples/data/revision_physics/materials.pumas` | `5179ed2823c2b5c98cf347d9cc390c7d3df92349f719edcc626f668e01c1edf3` |
| Both copies of `materials.xml` | `ca280fac5e83c74985adb322d39fefbd6b55feb9727956140cd7d60160dd13fb` |
| `src/Physics.jl` at inspection | `68606bf6b1c724bdb694b20188bdef4afd055da24bafe866f51974e1d9d55b0b` |
| `src/DEDXLoader.jl` at inspection | `ec390ac5a3f8f5a54651cfaa4c1e8d2d997628eabdc5e231a64668a248b0619f` |

The revision directory records the upstream `niess/pumas-materials` commit as
`d00fd31e5365b699b2d077232313f0c36343a837`, contains checksums for its three muon
text tables and MDF, and includes the generation command and loader log in
`provenance.json` and `loader.log`. The text headers identify PUMAS v1.1, with
SSR bremsstrahlung, SSR pair production, and DRSS photonuclear models. This is
recorded input provenance; this inspection did not independently authenticate the
historical dump against an upstream Git commit.

The historical dump stores `/home/roland/devspace/pumas-materials` as its data
path. That path alone does not prove how or when its tables were created. Both
dumps record cutoff 0.05, elastic ratio 0.05, and the same model-name strings.
Model-name metadata alone does not prove that every implementation detail matches
the named upstream model.

## Imported stopping power and range

Both dumps contain StandardRock, Air, Water, WetRock, and PorousWetRock. Every
material has 192 energy nodes spanning 0.0012 to 1e9 GeV. The energy grid, CSDA
stopping power and range, and imported ionization, bremsstrahlung, pair-production,
and photonuclear arrays compare exactly between corresponding tables. Thus the
historical dump is numerically consistent with the same supplied PUMAS stopping
and range data; describing this comparison as "old empirical tables versus new
PUMAS tables" would be incorrect.

At the requested energies, CSDA stopping power is identical in both dumps:

| Material | 1 GeV | 10 GeV | 100 GeV | 1000 GeV |
|---|---:|---:|---:|---:|
| StandardRock | 0.0001813 | 0.0002193 | 0.0002752 | 0.0006609 |
| Water | 0.0002107 | 0.0002507 | 0.0003022 | 0.0006035 |
| Air | 0.0002020 | 0.0002651 | 0.0003245 | 0.0006212 |

Units are GeV per kg/m². StandardRock CSDA ranges at these energies are also
identical: 5518, 49060, 407600, and 2450000 kg/m².

The supplied text tables include a first 1 MeV row. The current parser starts two
lines after the units line, skipping that first row in the supplied format. This
explains the observed 1.2 MeV minimum and 192 retained nodes in both dumps. This
inspection did not change that behavior or regenerate either dump.

## Derived transport tables differ

`create_material_table_from_dedx` imports CSDA arrays, then computes mixed stopping
power, straggling, hard-event cross sections, and scattering tables using the
supplied material's elemental recipe. These derived arrays differ between dumps.

The historical Water excitation energy is 75 eV; the revision MDF specifies
79.7 eV. Air is 85.7 versus approximately 85.6686 eV. StandardRock excitation
energy is 136.4 eV in both. The built-in StandardRock recipe contains eight real
elements; the MDF uses one effective element with Z=11, A=22.

As an independent numerical fingerprint, recomputing the ionization straggling
integral at 10 GeV with the current built-in material recipes exactly reproduces
the historical entries for all three base materials. Recomputing it with the
parsed MDF recipes exactly reproduces the revision entries. This strongly
supports differing material definitions as the explanation; it does not recover
the historical source revision or generation command.

| StandardRock quantity | New/old change at 1 GeV | 10 GeV | 100 GeV | 1000 GeV |
|---|---:|---:|---:|---:|
| Mixed stopping power | −0.00828% | −0.04540% | +0.03279% | +0.60085% |
| Straggling variance per grammage | +0.97224% | +0.97224% | +0.97224% | +0.97224% |
| Transport path | +7.31054% | +7.14640% | +7.05586% | +6.67036% |

At the same energies Water straggling changes by −0.01510%, and Air by
−0.000869%. Their transport-path changes are approximately +0.016% to +0.024%.
Cross sections, mixed ranges, and other scattering entries also differ.
These percentages describe tables, not predicted flux errors.

Uncorrected CSDA calculations use the unchanged stopping arrays and effectively
unchanged densities, so they should agree up to roundoff for matching inputs.
Stochastic transport and straggling-corrected CSDA use changed derived arrays;
their old and new predictions require separate comparison. Seeded stochastic
histories can diverge after a branch changes, so equal seeds do not imply equal
individual histories after changing tables.

The two `load_or_create_physics` entry points previously differed when creating
from the same MDF: the particle overload parsed base materials and composites,
while the filename overload used built-in base materials and parsed only
composites. The filename creation path now delegates to the full-MDF particle
loader. `test/test_loader_parity.jl` checks equality of every material-table field
using small independent temporary dumps. Existing dumps retain their contents.

## What the stochastic gradient checks measure

The revision suite's AD versus CSDA finite-difference comparisons use the same
fixed energy quadrature. These check derivatives of that discrete deterministic
forward solver, including its frozen calibration when enabled. They do not by
themselves establish convergence of the energy integral or equality to a
stochastic transport derivative.

The initial v3 run used 32 fixed, seeded CSDA energies and 256 independent MC
energies per replicate, sampled log-uniformly from 1e−3 to 1e9 GeV. Comparing an
MC secant with the fixed-CSDA secant therefore combines quadrature differences
with transport-model differences. Comparing the MC secant directly with local AD
also includes finite-step truncation: the MC interval has half-width 0.05 in the
interior and is one-sided at zero water fraction. The CSV output separately
records the interval and the deterministic secant.

Replicate dispersion estimates MC uncertainty conditional on the fixed CSDA
quadrature and any fitted calibration. It omits uncertainty of that quadrature
and calibration. If the two per-energy gradient variances were comparable, a
32-sample CSDA quadrature standard error would be about sqrt(256/32)=2.83 times
that of one 256-sample MC replicate, or sqrt(8R) times the standard error of an
average of R MC replicates. At R=4 that factor is 5.66. These are sample-count
scalings under an explicit equal-variance assumption, not measured error bars.

For Julia 1.11.7 and seed 20261022, the actual fixed 32-energy set contains zero
energies between 1 and 10 TeV, and has a gap from approximately 188.9 GeV to
20.33 TeV. Random-number streams can differ between Julia versions. This
illustrates why MC standard errors alone cannot certify quadrature convergence.
A paired MC/CSDA secant using each replicate's same energies, and a separate
energy-quadrature convergence or repeated-seed study, would distinguish these
uncertainties more directly.

## Measured energy-quadrature probe

`dev/energy_quadrature_probe.jl` measures three LVD rays at zeniths 0°, 30°, and
45°, all at azimuth 0°. The geometric rock lengths are respectively 1390.58,
1237.04, and 1546.47 m. It uses the pinned revision dump, rock density 2710 kg/m³,
the suite's voxel grid with eight depth layers and 1.5 km horizontal spacing,
and homogeneous water fractions 0 and 0.3. The local derivative is with respect
to the longest intersected cell on each ray. All other cells stay fixed.

The deterministic comparison uses equally spaced midpoints in log energy over
1e−3–1e9 GeV, with both charges at every energy. A rule with N energy nodes has
2N particles. Charge normalization follows the implementation: `flux_gccly`
returns charge-specific flux, with positive/negative ratio 1.2766. Seeded
sampling proposes either charge with probability one-half and multiplies its
charge-specific contribution by two. The paired rule uses the same factor and
averages both charges, so it estimates the charge-summed integral. A formula
that applies this factor two to an already charge-summed surface flux would
instead double-count charge normalization.

The table gives ranges across the six ray/field combinations, relative to the
256-node paired midpoint comparator. The frozen calibration is exactly the
existing v3 calibration, without refitting for a different quadrature. Its exact
TOML is preserved at `dev/revision_inputs/quadrature_frozen_calibration.toml`;
it originated from the v3 dry-field training with a fixed 32-particle seeded
CSDA quadrature (seed 20261022). The probe reads this preserved input directly
and does not require the v3 output directory. Both copies have SHA-256
`650b27e6374c16275c8b00fe7e4933052e1bcf9f00622aac4ec6dc66310838c3`.

| Quadrature | Correction | Relative flux difference | Relative local derivative difference |
|---|---|---:|---:|
| 8 seeded particles, seed 42 | Disabled | +95.53% to +102.69% | +90.57% to +101.91% |
| 32 seeded particles, seed 20261022 | Disabled | +7.62% to +11.15% | +8.19% to +11.85% |
| 8 seeded particles, seed 42 | Frozen v3 | +95.33% to +102.16% | +90.72% to +101.93% |
| 32 seeded particles, seed 20261022 | Frozen v3 | +8.15% to +11.39% | +8.46% to +11.89% |
| 32 paired energy nodes, 64 particles | Either mode | within 0.078% | within 0.92% |
| 128 paired energy nodes, 256 particles | Either mode | within 0.017% | within 0.30% |

Local derivatives use a step of 1e−4: second-order right differences at zero
fraction and central differences at 0.3. Halving the step at the 256-node
comparator changes these derivatives by at most 0.00352%, substantially less
than the seeded-quadrature differences. The comparator is not exact truth;
the reported 128-to-256 differences document its observed convergence.

These measurements show that an eight-particle energy quadrature is insufficient
for percent-level integral or derivative accuracy on these paths. They do not
measure the complete historical reconstruction pipeline with its own refitted
calibration, nor establish an error bound for every direction. Calibration can
absorb quadrature bias on training data, so it is not a substitute for a
quadrature-convergence check. A deterministic charge-summed rule is justified
by this comparison. The implemented `csda_energy_quadrature` uses one CSDA
transport per log-midpoint node, with charge-zero surface-flux summation and a
half Jacobian. Tests confirm its equality to explicit positive/negative pairs
for flux and derivatives; MC rejects these nonphysical charge-zero integration
nodes. Changing the production quadrature requires rerunning calibration and
reported results with the selected rule.

Actual values and inspection configuration are saved in
`examples/data/revision_quadrature_20261023/{measurements,comparison}.csv` and
`config.toml`. Reproduce the probe with:

```bash
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. dev/energy_quadrature_probe.jl
```

## Reproduce the inspections

Run from the repository root. These commands read existing dumps only:

```bash
sha256sum examples/data/materials.pumas examples/data/revision_physics/materials.pumas
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. -e '
using DiffPumas
old = load_physics("examples/data/materials.pumas")
new = load_physics("examples/data/revision_physics/materials.pumas")
for (name, p) in (("old", old), ("new", new))
    println(name, " ", p.settings)
    for t in p.tables
        println(t.name, " ", length(t.energies), " ", extrema(t.energies), " I=", t.I)
    end
end
for name in ("StandardRock", "Water", "Air"), E in (1., 10., 100., 1000.)
    i, j = get_material_index(old, name), get_material_index(new, name)
    a = property_stopping_power(old, ENERGY_LOSS_CSDA, i, E)
    b = property_stopping_power(new, ENERGY_LOSS_CSDA, j, E)
    println((name=name, energy=E, old=a, new=b, relative_change=b/a-1))
end
for a in old.tables
    b = new.tables[get_material_index(new, a.name)]
    println(a.name, ": differing fields = ",
        [f for f in fieldnames(typeof(a)) if getfield(a,f) != getfield(b,f)])
end'
```

To reproduce the straggling fingerprint, parse the MDF with
`DiffPumas.Loader.parse_mdf`, and for each base-material recipe sum
`fraction * DiffPumas.Materials.compute_dcs_integral(:ionisation, element.Z,
element.A, element.I, Float64(physics.mass), 10.0, 0.0,
physics.settings.cutoff, 2; npoints=180)` over its elements. Compare with
`DiffPumas.Physics.property_straggling(physics, material_index, 10.0)`.
