"""
    DiffPumas

A Julia port of the PUMAS library for transporting high energy muons and taus.
Designed for automatic differentiation with Zygote.jl.

PUMAS = Physics Utility for MUon And tau Simulations

This package provides:
- Monte Carlo transport of muons/taus through various media
- Physics tables for energy loss, scattering, and cross-sections
- Automatic differentiation support for flux calculations w.r.t. density
- Material definition and management
- Geometry modeling utilities

# Key Features
- Differentiable transport algorithms using Zygote.jl
- Support for both forward and backward Monte Carlo
- CSDA and mixed energy loss modes
- Multiple scattering with Coulomb interactions
- Radiative processes (bremsstrahlung, pair production, photonuclear)

# Example Usage

```julia
using DiffPumas

# Create physics tables for muon transport
physics = create_physics(MUON)

# Run backward Monte Carlo with gradient computation
result = run_backward_mc(
    physics;
    rock_thickness = 100.0,
    rock_density = 2650.0,
    elevation = 45.0,
    energy_min = 1.0,
    n_samples = 10000,
    compute_gradient = true
)

# Access results
println("Flux: \$(result.flux) ± \$(result.sigma)")
println("∂flux/∂density: \$(result.gradient)")
```

# References
- Original PUMAS: https://github.com/niess/pumas
- V. Niess et al., "PUMAS: A Unified Tool for Muon Propagation"
"""
module DiffPumas

# Re-export standard library modules used
using LinearAlgebra
using Random
using StaticArrays

# Re-export dependencies for user convenience
using Zygote
export gradient, pullback, jacobian

using PlotlyJS
export plot, Plot, Layout, scatter, scatter3d, bar, heatmap, surface

using TetGen
export tetrahedralize, RawTetGenIO, facetlist!

using TriangleIntersect
export Point, Ray, Triangle, Intersection, intersect

# Include submodules in dependency order
include("Constants.jl")
include("Types.jl")
include("Materials.jl")
include("DEDXLoader.jl")
include("Physics.jl")
include("Transport.jl")
include("Context.jl")
include("GaisserFlux.jl")
include("Straggling.jl")
include("Loader.jl")
include("Geometry.jl")
include("Plotting.jl")

# Import and re-export from submodules
using .Constants
using .Types
using .Materials
using .DEDXLoader
using .Physics
using .Transport
using .Context
using .GaisserFlux
using .Straggling
using .Loader
using .Geometry
using .Plotting

# Export Constants
export ALPHA_EM, HBAR_C, BOHR_RADIUS, MUON_C_TAU, TAU_C_TAU
export ELECTRON_MASS, ELECTRON_RADIUS, MUON_MASS, TAU_MASS
export PROTON_MASS, NEUTRON_MASS, PION_MASS, AVOGADRO_NUMBER

# Export Types
export Particle, MUON, TAU
export EnergyLossMode, ENERGY_LOSS_DISABLED, ENERGY_LOSS_CSDA, ENERGY_LOSS_MIXED, ENERGY_LOSS_STRAGGLED
export DecayMode, DECAY_DISABLED, DECAY_WEIGHTED, DECAY_RANDOMISED
export DirectionMode, DIRECTION_FORWARD, DIRECTION_BACKWARD
export ScatteringMode, SCATTERING_DISABLED, SCATTERING_MIXED
export Event, EVENT_NONE, EVENT_LIMIT_ENERGY, EVENT_LIMIT_DISTANCE
export EVENT_LIMIT_GRAMMAGE, EVENT_LIMIT_TIME, EVENT_MEDIUM
export EVENT_VERTEX_BREMSSTRAHLUNG, EVENT_VERTEX_PAIR_CREATION
export EVENT_VERTEX_PHOTONUCLEAR, EVENT_VERTEX_DELTA_RAY
export EVENT_VERTEX_COULOMB, EVENT_VERTEX_DECAY, EVENT_WEIGHT
export Process, PROCESS_BREMSSTRAHLUNG, PROCESS_PAIR_PRODUCTION, PROCESS_PHOTONUCLEAR
export Step, STEP_CHECK, STEP_RAW
export Property, PROPERTY_CROSS_SECTION, PROPERTY_STOPPING_POWER, PROPERTY_RANGE
export State, Locals, Medium, ContextMode, ContextLimit, Vec3
export update_state, LocalsCallback, has_event, combine_events

# Export Materials
export AtomicElement, BaseMaterial, CompositeMaterial
export ELEMENTS, MATERIALS, STANDARD_ROCK, AIR
export electronic_stopping_power, electronic_density_effect
export elastic_dcs, elastic_path, electronic_dcs
export dcs_bremsstrahlung_ssr, dcs_pair_production_ssr, dcs_photonuclear_drss

# Export DEDXLoader
export DEDXData, load_dedx_file, find_dedx_file, PUMAS_MATERIALS_PATH

# Export Physics
export PhysicsSettings, PhysicsTables, MaterialTable
export create_physics, create_energy_grid, get_material_index
export property_range, property_stopping_power, property_kinetic_energy
export property_proper_time, property_cross_section, property_transport_path
export interpolate_table

# Export Transport
export transport_csda_uniform, transport_csda_magnetic
export transport_single_medium, transport_with_density
export compute_grammage_step, compute_energy_loss

# Export Context
export SimulationContext, create_context, set_seed!, transport!
export random_uniform, random_exponential

# Export Loader
export load_physics, save_physics, load_or_create_physics
export parse_mdf, print_physics_summary

# Export GaisserFlux
export flux_gaisser, flux_gccly, charge_fraction, cos_theta_star

# Export Straggling
export fluctuate_energy_loss, sample_del_event, sample_ehs_event
export compute_del_cross_section, compute_ehs_mean_free_path
export rotate_direction, box_muller_randn, sample_scattering_angle
export sample_soft_scattering

# Export Geometry
export TwoLayerGeometry, create_geometry_context
export compute_flux, compute_flux_differentiable, compute_flux_gradient
export run_backward_mc, locals_rock, locals_air

# Export Plotting
export plot_trajectories, plot_transport_path

# Version information
const VERSION = v"0.2.0"

"""
    version()

Return the version of DiffPumas.
"""
version() = VERSION

"""
    pumas_info()

Print information about DiffPumas.
"""
function pumas_info()
    println("═" ^ 60)
    println(" DiffPumas.jl - Differentiable Particle Transport")
    println("═" ^ 60)
    println()
    println(" Version: $(VERSION)")
    println(" Based on: PUMAS v1.2.3")
    println()
    println(" Supported particles:")
    println("   • Muon (μ±)")
    println("   • Tau (τ±)")
    println()
    println(" Key features:")
    println("   • Automatic differentiation via Zygote.jl")
    println("   • Forward and backward Monte Carlo")
    println("   • CSDA, mixed, and straggled energy loss")
    println("   • Multiple Coulomb scattering")
    println("   • Radiative processes")
    println()
    println(" Usage: using DiffPumas")
    println("        physics = create_physics(MUON)")
    println("        result = run_backward_mc(physics)")
    println("═" ^ 60)
end

export version, pumas_info

end # module DiffPumas
