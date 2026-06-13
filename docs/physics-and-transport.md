# Physics And Transport

This page explains the main physical objects in DiffPumas.jl and how they fit
together in flux calculations.

## Core Concepts

DiffPumas.jl separates physics tables from transport:

- `PhysicsTables` hold particle mass, decay length, energy grids, material
  tables, and interpolated properties.
- Materials describe composition and density.
- `State` carries the transported particle state.
- Geometry helpers decide which medium the particle is in.
- Transport functions update state using CSDA, mixed, or straggled energy loss.
- Flux helpers combine transport with an atmospheric muon flux model.

The package supports `MUON` and `TAU`, but most examples focus on muons.

## Particles

```julia
using DiffPumas

muon_physics = create_physics(MUON)
tau_physics = create_physics(TAU)

@show muon_physics.mass
@show tau_physics.mass
```

The constants module exports useful physical constants such as `MUON_MASS`,
`TAU_MASS`, `ELECTRON_MASS`, `ALPHA_EM`, and decay lengths.

## Materials

Built-in materials include standard rock, air, and water:

```julia
using DiffPumas

@show STANDARD_ROCK.density
@show AIR.density
@show WATER.density

dedx = electronic_stopping_power(STANDARD_ROCK, MUON_MASS, 10.0)
println("Electronic stopping power at 10 GeV: $dedx")
```

When physics tables are loaded from an MDF file, material names can be resolved
with `get_material_index`:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

rock_idx = get_material_index(physics, "StandardRock")
air_idx = get_material_index(physics, "Air")

println("rock=$rock_idx air=$air_idx")
```

## Material Mixtures

`MaterialMixture` represents runtime mixtures by mass fraction. This is used in
aquifer and tomography workflows where a cell can be part rock and part water.

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

rock_idx = get_material_index(physics, "StandardRock")
water_idx = get_material_index(physics, "Water")

pure_rock = MaterialMixture(rock_idx)
wet_cell = MaterialMixture([rock_idx, water_idx], [0.7, 0.3])

range_pure = property_range(physics, ENERGY_LOSS_CSDA, pure_rock, 10.0)
range_wet = property_range(physics, ENERGY_LOSS_CSDA, wet_cell, 10.0)

println("pure rock range = $range_pure kg/m^2")
println("wet mixture range = $range_wet kg/m^2")
```

Mixture fractions should describe material composition. Density is still supplied
separately to transport and geometry routines, often as the mass-fraction
weighted density:

```julia
rho_rock = 2650.0
rho_water = 1000.0
water_fraction = 0.3
rho_eff = (1 - water_fraction) * rho_rock + water_fraction * rho_water
```

## Physics Tables

Tables are generated over a kinetic-energy grid:

```julia
using DiffPumas

energies = create_energy_grid(80, 1e-2, 1e6)
physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)
```

Common table lookups:

```julia
rock_idx = get_material_index(physics, "StandardRock")
energy = 100.0

range = property_range(physics, ENERGY_LOSS_CSDA, rock_idx, energy)
dedx = property_stopping_power(physics, ENERGY_LOSS_CSDA, rock_idx, energy)
kinetic = property_kinetic_energy(physics, ENERGY_LOSS_CSDA, rock_idx, range)

println((range = range, dedx = dedx, kinetic = kinetic))
```

The main energy-loss modes are:

- `ENERGY_LOSS_CSDA`: continuously slowing down approximation.
- `ENERGY_LOSS_MIXED`: continuous soft losses plus hard stochastic losses.
- `ENERGY_LOSS_STRAGGLED`: mixed transport with electronic energy straggling.

For gradients, start with CSDA unless you have a specific reason to differentiate
a more complex path.

## Particle State

`State` is immutable and AD-friendly. Use `update_state` rather than mutating a
field.

```julia
using DiffPumas

state = State{Float64}(
    charge = 1.0,
    energy = 100.0,
    position = Vec3(0.0, 0.0, 0.0),
    direction = Vec3(0.0, 0.0, -1.0),
)

state2 = update_state(state; energy = 95.0)
@show state.energy state2.energy
```

Fields use SI-like transport units:

- `energy`: GeV.
- `distance`: m.
- `grammage`: kg/m^2.
- `time`: m/c.
- `weight`: dimensionless Monte Carlo weight.
- `position`: m.
- `direction`: unit vector.

## Low-Level Transport

For a single uniform step through rock:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)
rock_idx = get_material_index(physics, "StandardRock")

state = State{Float64}(
    charge = 1.0,
    energy = 100.0,
    position = Vec3(0.0, 0.0, 0.0),
    direction = Vec3(0.0, 0.0, -1.0),
)

new_state, event = transport_with_density(
    physics,
    state,
    rock_idx,
    2650.0,  # density, kg/m^3
    1.0,     # step distance, m
)

println("energy: $(state.energy) -> $(new_state.energy)")
println("grammage: $(new_state.grammage)")
println("event: $event")
```

Most users should prefer higher-level flux helpers, but low-level transport is
useful for tests and for building custom geometry callbacks.

## Two-Layer Backward Flux

`compute_flux` evaluates a detector-level flux by sampling final energies and
charges, then transporting particles backward through rock and air.

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

flux, sigma = compute_flux(
    physics,
    2650.0,  # rock density, kg/m^3
    250.0,   # rock thickness, m
    30.0,    # elevation, degrees
    1e-3,    # energy min, GeV
    1e6;     # energy max, GeV
    n_samples = 500,
    seed = 123,
    straggling = true,
    scattering = true,
    energy_threshold_low = 100.0,
)

println("flux = $flux +/- $sigma")
```

The corresponding geometry type is `TwoLayerGeometry`:

```julia
rock_idx = get_material_index(physics, "StandardRock")
air_idx = get_material_index(physics, "Air")

geometry = TwoLayerGeometry{Float64}(250.0, 2650.0, rock_idx, air_idx)
```

Some examples add an optional shallow porous layer through extra
`TwoLayerGeometry` fields or `compute_flux` keyword arguments:

```julia
porous_idx = get_material_index(physics, "PorousWetRock")

flux, sigma = compute_flux(
    physics,
    2650.0,
    1000.0,
    45.0,
    1e-3,
    1e6;
    n_samples = 200,
    porous_material = porous_idx,
    porous_density = 2200.0,
    porous_thickness = 100.0,
)
```

## Atmospheric Flux

The exported atmospheric flux helpers are:

- `flux_gaisser(cos_theta, energy, charge)`.
- `flux_gccly(cos_theta, energy, charge)`.
- `charge_fraction`.
- `cos_theta_star`.

Example:

```julia
using DiffPumas

cos_theta = cosd(30.0)
energy = 100.0

plus_flux = flux_gccly(cos_theta, energy, 1.0)
minus_flux = flux_gccly(cos_theta, energy, -1.0)

println("mu+ flux = $plus_flux")
println("mu- flux = $minus_flux")
```

## Transport Uncertainty Budgets

`estimate_transport_uncertainty` is generic: provide a function that accepts a
`TransportVariation` and returns `(value, sigma_mc)`.

```julia
using DiffPumas

function evaluate(variation)
    value = 10.0
    value += variation.straggling ? 0.4 : -0.2
    value += variation.scattering ? 0.2 : -0.1
    value += variation.energy_threshold_low / 1000.0
    sigma_mc = 0.05
    return value, sigma_mc
end

budget = estimate_transport_uncertainty(
    evaluate;
    straggling = true,
    scattering = true,
    energy_threshold_low = 100.0,
    threshold_factors = (0.5, 2.0),
)

println("value = $(budget.value)")
println("MC sigma = $(budget.sigma_mc)")
println("systematic sigma = $(budget.sigma_syst)")
println("total sigma = $(budget.sigma_total)")
println("relative total = $(relative_uncertainty_percent(budget.value, budget.sigma_total))%")
```

For two-layer flux, use the convenience wrapper:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

budget = compute_flux_uncertainty(
    physics,
    2650.0,
    100.0,
    45.0,
    1e-3,
    1e6;
    n_samples = 200,
    straggling = true,
    scattering = true,
    energy_threshold_low = 100.0,
)

println("flux = $(budget.value)")
println("total sigma = $(budget.sigma_total)")
```

The systematic model currently scans:

- Straggling on/off.
- Scattering on/off.
- `energy_threshold_low` multiplied by the configured threshold factors.

These are first-pass model-systematic envelopes, not a complete detector or
geology uncertainty model.

## Performance Notes

Use fewer samples for smoke tests and documentation examples. Increase samples
for physics runs.

Keep random sampling outside differentiated functions. For AD, pre-sample
energies and charges, then differentiate a deterministic reduction over the
sample set.

Reuse physics dumps for command-line examples. Table generation is useful during
development, but loading a dump is much faster for repeated studies.

When scanning many angles or cells, prefer the direct CSDA forward model for
sensitivity assembly and reserve full stochastic MC for validation or final
comparison.
