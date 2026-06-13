# Getting Started

This guide gets you from a fresh checkout to a working physics-table lookup and
a first muon flux calculation.

## Requirements

DiffPumas.jl targets Julia 1.10 or newer. The project environment is declared in
`Project.toml` and includes dependencies such as Zygote, ChainRulesCore,
PlotlyJS, TetGen, TriangleIntersect, StaticArrays, and ArgParse.

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Use the same project environment for tests and examples:

```bash
julia --project=.
julia --project=. examples/loader_example.jl --help
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Load the Package

Start a Julia REPL in the project:

```bash
julia --project=.
```

Then load DiffPumas:

```julia
using DiffPumas

println(version())
pumas_info()
```

`pumas_info()` prints a short summary of supported particles and transport
modes.

## Create Physics Tables In Memory

For quick checks, create tables directly:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

rock_idx = get_material_index(physics, "StandardRock")
air_idx = get_material_index(physics, "Air")

@show rock_idx air_idx
@show property_range(physics, ENERGY_LOSS_CSDA, rock_idx, 10.0)
@show property_stopping_power(physics, ENERGY_LOSS_CSDA, rock_idx, 10.0)
```

Use direct creation in tests or small notebooks. For repeated example runs, use
the dump workflow below so physics table construction is cached.

## Create or Reuse a Physics Dump

Most examples use `examples/data/materials.xml` and cache tables in a `.pumas`
dump:

```bash
julia --project=. examples/loader_example.jl \
  --mdf examples/data/materials.xml \
  --dump examples/data/materials.pumas \
  --particle muon
```

The loader prints available material names and sample CSDA ranges. Subsequent
examples can reuse the dump:

```bash
julia --project=. examples/diff_flux.jl \
  --dump examples/data/materials.pumas \
  --thickness 100 \
  --density 2650 \
  --zenith-max 45 \
  --samples 100
```

If the dump is missing, scripts that call `DiffPumas.Pumas.load_or_create_physics`
will usually rebuild it from the MDF path they provide.

## First Flux Calculation

The high-level two-layer geometry is rock below an exponential atmosphere. The
detector sits at `z = 0`, rock extends upward to `rock_thickness`, and the
primary-flux surface is above that.

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

flux, sigma = compute_flux(
    physics,
    2650.0,  # rock density, kg/m^3
    100.0,   # rock thickness, m
    45.0,    # elevation angle, degrees
    1e-3,    # detector energy min, GeV
    1e6;     # detector energy max, GeV
    n_samples = 100,
    seed = 42,
    straggling = true,
    scattering = true,
    energy_threshold_low = 100.0,
)

println("flux = $flux +/- $sigma")
```

For a deterministic AD-friendly CSDA value at one final energy:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

flux = compute_flux_differentiable_csda(
    physics,
    2650.0,  # density, kg/m^3
    100.0,   # thickness, m
    45.0,    # elevation, degrees
    10.0,    # final kinetic energy, GeV
    1.0,     # charge
)

println(flux)
```

## First Density Gradient

The direct CSDA path is the most convenient route for a density sensitivity:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

flux, dflux_drho = compute_flux_gradient_csda(
    physics,
    2650.0,  # density, kg/m^3
    100.0,   # thickness, m
    45.0,    # elevation, degrees
    10.0,    # final kinetic energy, GeV
    1.0,     # charge
)

println("flux = $flux")
println("dflux/drho = $dflux_drho")
```

For an integrated flux over sampled energies, see `examples/diff_flux.jl`, which
pre-samples random values before differentiating so the AD target is
deterministic.

## Run the Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests cover:

- Physical constants and basic types.
- Material and physics-table lookup.
- Transport state updates.
- Zygote gradients through differentiable transport paths.
- Uncertainty-budget combination math.
- Tomography forward-model and reconstruction helpers.

## Common Problems

If `using DiffPumas` fails, instantiate the environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If an example cannot find `StandardRock`, `Air`, or `Water`, rebuild the dump
from `examples/data/materials.xml` with `examples/loader_example.jl`.

If a stochastic example is slow, lower `--samples`, `--n-angles`,
`--mc-samples`, or run only a script part with `--part`. Low-statistics runs are
for smoke tests and API checks; they are not physics-quality results.

If a Zygote gradient is unexpectedly `nothing` or very slow, make sure the code
path is deterministic and avoids random sampling inside the differentiated
function. Prefer direct CSDA helpers for gradients.
