# DiffPumas

**DiffPumas** is a Julia implementation of the [PUMAS](https://github.com/niess/pumas) and [TURTLE](https://github.com/niess/turtle) C libraries, upgraded to support **differentiable programming** through [Zygote.jl](https://github.com/FluxML/Zygote.jl).

## Overview

DiffPumas provides a Julia-native version of:

- **PUMAS** (Semi-Analytical Muons -or taus- Propagation, backwards): Particle transport library for muon and tau leptons using forward or backward Monte Carlo methods
- **TURTLE** (Topographic Utilities for tRansporting parTicules over Long rangEs): Library for long-range transport of Monte-Carlo particles through topography using Digital Elevation Models (DEMs)

By leveraging Zygote.jl's automatic differentiation capabilities, DiffPumas enables:

- **Gradient-based optimization** of physical parameters
- **Sensitivity analysis** of particle transport simulations
- **Differentiable physics** workflows that integrate seamlessly with machine learning
- **Parameter inference** through gradient descent on simulation results

## Features

- Pure Julia implementation for better integration with the Julia ecosystem
- Automatic differentiation support for all core simulation functions
- Thread-safe particle transport algorithms
- Support for topographic transport and coordinate transformations
- Fast ray-triangle intersections for raytracing (via [TriangleIntersect.jl](https://github.com/JuliaGeometry/TriangleIntersect.jl))
- Tetrahedral meshing and 3D Delaunay/Voronoi tessellation (via [TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl))
- Interactive visualization with [PlotlyJS.jl](https://github.com/JuliaPlots/PlotlyJS.jl)
- Compatible with Flux.jl and other Julia ML frameworks

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/diffpumas.jl")

# TriangleIntersect.jl is added automatically, but if needed manually:
Pkg.develop(url="https://github.com/JuliaGeometry/TriangleIntersect.jl.git")
```

Or add it as a dependency in your `Project.toml`:

```toml
[deps]
DiffPumas = "b2257c3e-0f2b-464b-8aa8-308d82d5db5b"
```

## Usage

```julia
using DiffPumas
using Zygote

# Example: Differentiable particle transport
# (API to be implemented)
```

## Dependencies

- `Zygote.jl` - Automatic differentiation library for differentiable programming
- `TriangleIntersect.jl` - Fast ray-triangle intersections for raytracing, essential for particle transport and topography intersection calculations
- `TetGen.jl` - Tetrahedral mesh generation and 3D Delaunay/Voronoi tessellation, useful for geometric modeling and topography discretization
- `PlotlyJS.jl` - Interactive plotting library for visualizing simulation results and gradients

## Related Projects

- [PUMAS](https://github.com/niess/pumas) - Original C library for muon/tau propagation
- [TURTLE](https://github.com/niess/turtle) - Original C library for topographic particle transport

## License

This project is licensed under the same terms as the original PUMAS and TURTLE libraries (GNU LGPLv3).

