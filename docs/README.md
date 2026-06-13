# DiffPumas.jl Documentation

DiffPumas.jl is a Julia transport and muography research package inspired by
PUMAS. It focuses on high-energy muon and tau propagation, backward Monte Carlo
flux estimates, differentiable flux calculations with Zygote, and absorption
muography/tomography workflows.

The package is not just a table lookup wrapper. It contains:

- Physics tables for muons and taus in rock, air, water, and material mixtures.
- CSDA, mixed, and straggled energy-loss paths.
- Backward transport through simple and site-specific geometries.
- Automatic differentiation of flux with respect to material density or per-cell
  water fraction.
- Uncertainty helpers for Monte Carlo noise and first-pass transport
  systematics.
- Gran Sasso LVD examples for topography, paper checks, sensitivity volumes,
  and synthetic inversion studies.

## Documentation Map

Start here when you are new to the repo:

- [Getting started](getting-started.md): installation, environment setup,
  physics-table loading, first flux calculation, and testing.
- [Physics and transport](physics-and-transport.md): materials, physics tables,
  energy-loss modes, backward Monte Carlo, atmospheric flux, and uncertainty
  budgets.
- [Differentiable flux](differentiable-flux.md): Zygote workflows, direct CSDA
  gradients, density sensitivities, and common AD constraints.
- [Muography and tomography](muography-and-tomography.md): flat muography,
  LVD topography, directional paths, Jacobians, reconstruction solvers, and
  validation checks.
- [Examples guide](examples.md): runnable commands for the scripts in
  `examples/`, including quick/low-statistics variants.
- [API orientation](api-reference.md): exported types and functions grouped by
  workflow, with notes on which APIs are stable enough for examples.

## Package Layout

The important directories are:

- `src/`: package modules for types, physics tables, materials, transport,
  geometry, uncertainty, plotting, Turtle-style topography helpers, and
  tomography.
- `examples/`: runnable research scripts and command-line examples.
- `examples/data/`: material definitions, LVD benchmark assets, generated
  physics dumps, and example outputs.
- `test/`: unit and regression tests for constants, types, physics lookup,
  differentiability, uncertainty budgets, and tomography.
- `diffmuontomoLVD/`: manuscript/figure material for the LVD muon tomography
  study.

## Minimal Workflow

Use the package environment from the repository root:

```bash
cd /path/to/diffpumas.jl
julia --project=.
```

In Julia:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)
rock_idx = get_material_index(physics, "StandardRock")

range_100gev = property_range(physics, ENERGY_LOSS_CSDA, rock_idx, 100.0)
println("100 GeV muon CSDA range in standard rock: $range_100gev kg/m^2")
```

For command-line examples, generate or load a materials dump first:

```bash
julia --project=. examples/loader_example.jl \
  --mdf examples/data/materials.xml \
  --dump examples/data/materials.pumas
```

Then run a flux example:

```bash
julia --project=. examples/diff_flux.jl \
  --dump examples/data/materials.pumas \
  --thickness 100 \
  --zenith-max 45 \
  --samples 100
```

## Units

DiffPumas.jl uses the following conventions in user-facing APIs:

- Kinetic energy: GeV.
- Distance and thickness: m.
- Grammage and range: kg/m^2.
- Density: kg/m^3.
- Angles in examples and high-level APIs: degrees.
- Flux: generally m^-2 s^-1 sr^-1, or GeV^-1 m^-2 s^-1 sr^-1 for differential
  atmospheric flux parameterizations.
- Water fraction in tomography: dimensionless, clipped to the interval used by
  the solver, usually `[0, 0.9]`.

## Choosing a Workflow

Use `create_physics` and low-level `property_*` functions when you want physics
table lookups, interpolation checks, or unit tests.

Use `compute_flux`, `compute_flux_uncertainty`, or `examples/diff_flux.jl` when
you want a scalar flux for a two-layer rock/air geometry.

Use `compute_flux_differentiable_csda`, `compute_flux_gradient_csda`, or
`directional_flux_and_grad_csda` when you need gradients. These paths avoid
stochastic control flow and are the preferred route for inversion and sensitivity
studies.

Use `examples/flat_muography.jl` for synthetic flat/cubic aquifer studies, and
`examples/lvd_muography.jl` or `examples/lvd_tomography.jl` for Gran Sasso LVD
topography and tomography workflows.

## Development Checks

Run the test suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

For documentation-only edits, tests are not always necessary, but examples in
this documentation should stay consistent with exported APIs and script help
text.
