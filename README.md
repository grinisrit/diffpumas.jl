# DiffPumas.jl

A Julia port of the [PUMAS](https://github.com/niess/pumas) library for transporting high energy muons and taus, with **automatic differentiation support** via [Zygote.jl](https://github.com/FluxML/Zygote.jl).

**PUMAS** = **P**hysics **U**tility for **MU**on **A**nd tau **S**imulations

## Features

- **Differentiable Transport**: Compute gradients of flux w.r.t. material density using automatic differentiation
- **Monte Carlo Methods**: Forward and backward Monte Carlo transport
- **Energy Loss Modes**: CSDA, mixed, and straggled energy loss
- **Scattering**: Multiple Coulomb scattering with nuclear form factors
- **Radiative Processes**: Bremsstrahlung, pair production, photonuclear interactions
- **Particle Types**: Muon (μ±) and tau (τ±) leptons

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/your-repo/DiffPumas.jl")
```

Or in development mode:

```julia
] dev /path/to/diffpumas.jl
```

## Quick Start

```julia
using DiffPumas

# Create physics tables for muon transport
physics = create_physics(MUON)

# Run backward Monte Carlo flux calculation
result = run_backward_mc(
    physics;
    rock_thickness = 100.0,    # meters
    rock_density = 2650.0,     # kg/m³
    elevation = 45.0,          # degrees
    energy_min = 1.0,          # GeV
    n_samples = 10000,
    compute_gradient = true    # Enable ∂flux/∂ρ calculation
)

println("Flux: $(result.flux) ± $(result.sigma) m⁻² s⁻¹ sr⁻¹")
println("∂flux/∂density: $(result.gradient)")
```

## Differentiable Flux Calculation

The key feature of DiffPumas.jl is computing gradients of the transmitted muon flux with respect to material properties. This is useful for:

- **Sensitivity analysis**: Understanding how flux depends on rock density
- **Inversion problems**: Inferring material properties from flux measurements
- **Optimization**: Finding optimal detector placement or shielding configurations

```julia
using DiffPumas
using Zygote

# Create physics
physics = create_physics(MUON)

# Define flux as function of density
flux_fn(ρ) = compute_flux_differentiable(
    physics, ρ, 100.0, 45.0, 10.0, 1.0
)

# Compute gradient
ρ = 2650.0
grad = gradient(flux_fn, ρ)[1]
println("∂flux/∂ρ = $grad")
```

## Module Structure

The package is organized into several modules:

| Module | Description |
|--------|-------------|
| `Constants` | Physical constants (masses, decay lengths, etc.) |
| `Types` | Core data structures (State, Locals, Medium, etc.) |
| `Materials` | Material properties and DCS calculations |
| `Physics` | Physics tables and interpolation |
| `Transport` | Differentiable transport algorithms |
| `Context` | Simulation context management |
| `Loader` | Smart materials loader with caching |
| `Geometry` | Example geometry (two-layer rock+air) |

## Examples

### Loader Example (equivalent to `pumas/examples/pumas/loader.c`)

```julia
using DiffPumas

# Load or create physics tables
physics = load_or_create_physics(MUON; dump_path="materials.pumas")

# Print summary
print_physics_summary(physics)

# Look up properties
rock_idx = get_material_index(physics, "StandardRock")
range = property_range(physics, ENERGY_LOSS_CSDA, rock_idx, 100.0)
println("Range at 100 GeV: $range kg/m²")
```

### Geometry Example (equivalent to `pumas/examples/pumas/geometry.c`)

```julia
using DiffPumas

# Create physics
physics = create_physics(MUON)

# Run backward MC with gradient
result = run_backward_mc(
    physics;
    rock_thickness = 100.0,
    elevation = 45.0,
    energy_min = 1.0,
    energy_max = 100.0,
    n_samples = 10000,
    compute_gradient = true
)

println("Flux: $(result.flux) m⁻² s⁻¹ sr⁻¹")
println("∂flux/∂ρ: $(result.gradient)")
```

### Direct Gradient Computation

```julia
using DiffPumas
using Zygote

physics = create_physics(MUON)

# Compute flux and gradient for specific parameters
flux, grad = compute_flux_gradient(
    physics,
    2650.0,  # rock density (kg/m³)
    100.0,   # rock thickness (m)
    45.0,    # elevation (°)
    10.0,    # final energy (GeV)
    1.0      # charge
)

println("Flux: $flux")
println("Gradient: $grad")
```

## Physics Models

### Energy Loss

- **CSDA**: Continuously Slowing Down Approximation
- **Mixed**: Soft continuous + hard stochastic losses
- **Straggled**: Mixed + electronic energy straggling

### Differential Cross-Sections

| Process | Model |
|---------|-------|
| Bremsstrahlung | SSR (Sandrock et al., 2019) |
| Pair Production | SSR |
| Photonuclear | DRSS (Dutta et al., 2001) |
| Elastic | Salvat (2013) with nuclear form factors |

### Atmospheric Muon Flux

The `flux_gccly` function implements a parameterization of the atmospheric muon flux based on the Gaisser formula with zenith angle corrections.

## Running Tests

```julia
] test DiffPumas
```

Or run the test file directly:

```julia
include("test/runtests.jl")
```

## Running Examples

```bash
julia --project=. examples/loader_example.jl
julia --project=. examples/geometry_example.jl [ROCK_THICKNESS] [ELEVATION] [ENERGY_MIN] [ENERGY_MAX]
```

## API Reference

### Core Types

```julia
State{T}        # Monte Carlo particle state
Locals{T}       # Local medium properties
Medium{T}       # Propagation medium definition
PhysicsTables{T}# Tabulated physics data
```

### Key Functions

```julia
create_physics(particle)                    # Create physics tables
transport_with_density(physics, state, ...)  # Differentiable transport
compute_flux(physics, density, ...)          # Compute flux
compute_flux_gradient(physics, density, ...) # Compute flux + gradient
```

## References

1. **PUMAS**: V. Niess et al., "PUMAS: A Unified Tool for Muon Propagation" - [GitHub](https://github.com/niess/pumas)
2. **Energy Loss**: P.D.G. Review of Particle Physics
3. **Elastic Scattering**: F. Salvat, "PENELOPE-2014: A code system for Monte Carlo simulation of electron and photon transport"
4. **Bremsstrahlung**: A. Sandrock et al., ICRC 2019
5. **Photonuclear**: S.R. Dutta et al., Phys. Rev. D63 (2001) 094020

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This package is a Julia port of the original PUMAS C library by Valentin Niess (CNRS/IN2P3, LPC Clermont).
