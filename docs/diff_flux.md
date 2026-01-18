# Differentiable Muon Transport

## Abstract

This document describes DiffPumas.jl, a Julia implementation of differentiable muon transport through matter. We implement backward Monte Carlo methods for computing underground muon flux and leverage automatic differentiation (AD) to compute gradients of the flux with respect to material properties. The implementation closely follows the physics of the PUMAS C library while enabling gradient-based optimization through Zygote.jl.

## 1. Introduction

Cosmic ray muons are produced in the upper atmosphere through the decay of charged pions and kaons created in hadronic showers. These muons can penetrate deep underground, making them valuable probes for applications such as muon tomography, underground physics experiments, and geological surveys.

### 1.1 Experimental Geometry

We consider a simplified two-layer geometry consisting of:

1. **Air layer**: From the rock-air interface at altitude $z = h$ to the primary sampling altitude $z = H$ (typically 1000 m)
2. **Rock layer**: From the detector position at $z = 0$ to the rock-air interface at $z = h$

```
                    PRIMARY_ALTITUDE (H = 1000 m)
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                    (exponential density)
                         |
    ──────────────────────────────────────── z = h (rock thickness)
                         |
                    ROCK LAYER
                    (uniform density ρ)
                         |
                         |      ╱ incoming muon
                         |    ╱   at zenith angle θ
                         |  ╱
                         |╱
    ─────────────────────●─────────────────── z = 0 (Detector)
```

A muon arriving at the detector with kinetic energy $K_f$ and zenith angle $\theta$ must have originated in the atmosphere with some higher energy $K_i$, having lost energy traversing the intervening matter.

### 1.2 Backward Monte Carlo

Rather than simulating muons forward from the atmosphere (which would be highly inefficient for deep detectors), we employ backward Monte Carlo: we trace particles backward from the detector through the geometry, accumulating weight factors that account for energy loss and decay probability.

## 2. Atmospheric Muon Flux

### 2.1 The Gaisser Formula

The differential flux of atmospheric muons at sea level is well-described by the modified Gaisser formula. For muons with energy $E$ and zenith angle $\theta$, the flux is:

$$
\frac{d\Phi}{dE\, d\Omega} = \frac{0.14 \cdot E^{-2.7}}{\text{cm}^{2}\, \text{s}\, \text{sr}\, \text{GeV}} \left( \frac{1}{1 + \frac{1.1 E \cos\theta^*}{115\, \text{GeV}}} + \frac{0.054}{1 + \frac{1.1 E \cos\theta^*}{850\, \text{GeV}}} \right)
$$

where:
- $E$ is the muon energy in GeV
- $\theta^*$ is the effective zenith angle accounting for Earth's curvature
- The first term represents pion decay contribution
- The second term represents kaon decay contribution
- The critical energies (115 GeV and 850 GeV) reflect the competition between decay and interaction for the parent mesons

### 2.2 Effective Zenith Angle

For horizontal and near-horizontal muons, the curvature of the Earth becomes important. The effective zenith angle $\theta^*$ is computed as:

$$
\cos\theta^* = \sqrt{\frac{\cos^2\theta + p_1^2 + p_2 \cos^{p_3}\theta + p_4 \cos^{p_5}\theta}{1 + p_1^2 + p_2 + p_4}}
$$

with parameters fitted to detailed atmospheric cascade simulations.

### 2.3 Charge Ratio

The flux contains both $\mu^+$ and $\mu^-$ with a charge ratio:

$$
R = \frac{\Phi(\mu^+)}{\Phi(\mu^-)} \approx 1.2766
$$

This slight excess of positive muons reflects the excess of protons over neutrons in the primary cosmic ray flux and the predominance of $\pi^+$ production.

## 3. Matter Interactions

### 3.1 Air Density Model

The atmosphere is modeled with an exponential density profile:

$$
\rho_{\text{air}}(z) = \rho_0 \exp\left(-\frac{z}{h_s}\right)
$$

where:
- $\rho_0 = 1.205$ kg/m³ is the sea-level density
- $h_s \approx 12$ km is the scale height

The column depth (grammage) traversed by a muon traveling from altitude $z_1$ to $z_2$ at zenith angle $\theta$ is:

$$
X = \int_{z_1}^{z_2} \frac{\rho(z)}{\cos\theta} dz = \frac{\rho_0 h_s}{\cos\theta} \left[ \exp\left(-\frac{z_1}{h_s}\right) - \exp\left(-\frac{z_2}{h_s}\right) \right]
$$

### 3.2 Geomagnetic Field

The geomagnetic field affects muon trajectories through the Lorentz force. We model a simplified uniform field:

$$
\vec{B} = (0, 2 \times 10^{-5}, -4 \times 10^{-5}) \text{ T}
$$

For most applications at energies above a few GeV, the geomagnetic deflection is negligible over the path lengths considered.

### 3.3 Energy Loss Mechanisms

Muons lose energy through several processes:

1. **Ionization** (dominant at low energies): Described by the Bethe-Bloch formula
2. **Bremsstrahlung**: Radiative energy loss in the Coulomb field of nuclei
3. **Pair production**: $\mu \rightarrow \mu e^+ e^-$
4. **Photonuclear interactions**: $\mu N \rightarrow \mu X$

The total stopping power is:

$$
-\frac{dE}{dX} = a(E) + b(E) \cdot E
$$

where $a(E)$ represents ionization losses and $b(E)E$ represents radiative losses. At the critical energy $E_c \approx 500$ GeV in standard rock, radiative and ionization losses are equal.

### 3.4 Muon Decay

Muons decay with proper lifetime $\tau_0 = 2.197 \times 10^{-6}$ s, corresponding to a decay length:

$$
c\tau_0 = 658.654 \text{ m}
$$

The survival probability over a path length $L$ depends on the Lorentz factor:

$$
P_{\text{survival}} = \exp\left(-\frac{L}{\beta\gamma c\tau_0}\right)
$$

where $\beta\gamma = p/m_\mu c$ with $m_\mu = 105.658$ MeV/c².

## 4. Transport Physics

### 4.1 Continuous Slowing Down Approximation (CSDA)

In the CSDA, energy loss is treated as purely deterministic. The range $R(E)$ is defined as:

$$
R(E) = \int_0^E \frac{dE'}{|dE'/dX|}
$$

For backward transport from energy $K_f$ through grammage $\Delta X$, the initial energy is found by inverting:

$$
R(K_i) = R(K_f) + \Delta X
$$

The flux weight (Jacobian) for this transformation is:

$$
w = \frac{|dE/dX|_{K_i}}{|dE/dX|_{K_f}}
$$

This accounts for the change in phase space density as particles slow down.

### 4.2 Energy Straggling

In reality, energy loss fluctuates statistically. At each step, the actual energy loss $\Delta E$ follows a distribution around the mean $\langle\Delta E\rangle = |dE/dX| \cdot \Delta X$.

For thin absorbers, the Vavilov distribution applies:

$$
f(\Delta E) = \frac{1}{\xi} \phi_V\left(\frac{\Delta E - \langle\Delta E\rangle}{\xi}, \kappa, \beta^2\right)
$$

where $\xi$ is the mean energy loss in the absorber and $\kappa$ is the ratio of mean to maximum energy transfer.

In DiffPumas.jl, straggling is implemented using sampling from appropriate distributions, with the mode controlled by energy thresholds:
- **Below threshold** (typically 100 GeV): Full straggling with the STRAGGLED mode
- **Above threshold**: MIXED mode with deterministic continuous energy loss plus stochastic discrete losses

### 4.3 Multiple Coulomb Scattering

As muons traverse matter, they undergo many small-angle Coulomb scatterings. The cumulative effect is described by Molière theory. The RMS scattering angle after traversing thickness $X$ is approximately:

$$
\theta_{\text{rms}} = \frac{13.6 \text{ MeV}}{\beta c p} \sqrt{\frac{X}{X_0}} \left[1 + 0.038 \ln\left(\frac{X}{X_0}\right)\right]
$$

where $X_0$ is the radiation length of the material.

Scattering causes trajectories to deviate from straight lines, affecting both the path length through matter and the apparent arrival direction at the detector.

## 5. Implementation

### 5.1 Validation Against PUMAS

DiffPumas.jl is designed to match the physics of the PUMAS C library (Physics Utility for Muon and tau Active Simulation). Key validation points include:

1. **Physics tables**: Range, stopping power, and proper time tables are loaded from PUMAS binary dumps
2. **Energy thresholds**: The dual-mode transport (STRAGGLED below 100 GeV, MIXED above) matches PUMAS exactly
3. **Flux computation**: Integrated flux values agree within statistical uncertainty

For pure CSDA transport (no straggling, no scattering), the relative difference between DiffPumas.jl and PUMAS C is typically < 0.1%.

### 5.2 Automatic Differentiation with Zygote.jl

[Zygote.jl](https://github.com/FluxML/Zygote.jl) is Julia's source-to-source automatic differentiation system. Unlike numerical differentiation (finite differences) or symbolic differentiation, AD computes exact derivatives by applying the chain rule to the program's computational graph.

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, Zygote computes the gradient:

$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)
$$

in time comparable to evaluating $f$ itself (the "cheap gradient principle").

### 5.3 Direct CSDA for Differentiable Transport

The iterative transport algorithm (step-by-step energy loss with boundary checking) presents challenges for AD:
- Control flow depends on state (energy, position)
- Many small steps accumulate numerical error
- Discrete decisions (which cell, which material) are non-differentiable

We developed a **Direct CSDA** algorithm that computes the flux in closed form:

1. **Precompute geometry**: Trace the ray through cells, recording segment lengths (non-differentiable)
2. **Compute physics**: For each segment, compute energy change and weight using range tables (differentiable)

For a trajectory passing through $N$ segments with grammages $\{X_1, \ldots, X_N\}$:

$$
K_{\text{final}} = R^{-1}\left(R(K_{\text{initial}}) + \sum_{i=1}^N X_i\right)
$$

$$
w_{\text{total}} = \prod_{i=1}^N \frac{|dE/dX|_{K_{i+1}}}{|dE/dX|_{K_i}} \cdot P_{\text{decay},i}
$$

The flux is then:

$$
\Phi = w_{\text{total}} \cdot \Phi_{\text{Gaisser}}(\cos\theta, K_{\text{final}})
$$

This formulation allows Zygote to differentiate through the physics tables (which are implemented with differentiable interpolation) while keeping the geometric ray tracing separate.

## 6. Tessellated Geometry Example

### 6.1 The diff_flux_cube.jl Example

To demonstrate gradient computation with spatially-varying properties, we implemented a tessellated rock volume:

```
                    PRIMARY_ALTITUDE (1000m)
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                         |
    ──────────────────────────────────────── z = h
                         |  ┌─────────────┐
                    ROCK │  │ Cell Grid   │
                    CUBE │  │ (nx×ny×nz)  │
                         │  └─────────────┘
                         |╱ θ (zenith)
    ─────────────────────●─────────────────── z = 0 (Detector)
```

The rock volume is divided into $n_x \times n_y \times n_z$ cells, each with its own density $\rho_{ijk}$.

### 6.2 Ray Tracing Algorithm

For each Monte Carlo sample:

1. **Sample trajectory parameters**: zenith angle $\theta$, azimuth $\phi$, starting position $(x_0, y_0)$
2. **Trace through cells**: Compute which cells the trajectory intersects and the path length in each
3. **Compute flux**: Use Direct CSDA with cell-specific densities
4. **Accumulate gradients**: Zygote differentiates the flux with respect to all cell densities

The ray tracing respects cell boundaries:

```julia
function trace_cube_path(geo, elevation, azimuth, start_x, start_y)
    # Compute direction vector
    # For each cell intersection:
    #   - Find distance to next boundary
    #   - Record (cell_index, distance) segment
    # Return segments and exit position
end
```

### 6.3 Gradient Interpretation

The gradient $\partial\Phi/\partial\rho_{ijk}$ represents the sensitivity of the total flux to density changes in cell $(i,j,k)$. Key observations:

1. **Negative gradients**: Higher density means more energy loss, fewer muons reach the detector
2. **Bottom cells dominate**: All trajectories pass through bottom layers, making them most sensitive
3. **Central cells**: Higher sensitivity than edge cells due to geometric acceptance

These gradients enable:
- **Tomographic inversion**: Reconstruct density distribution from flux measurements
- **Sensitivity analysis**: Identify which regions most affect the measurement
- **Optimization**: Design experiments to maximize information content

## 7. Results and Validation

### 7.1 Flux Comparison

For pure CSDA transport (no straggling, no scattering), the Direct CSDA method matches the iterative detailed transport within 0.03%:

| Method | Integrated Flux (m⁻² s⁻¹ sr⁻¹) |
|--------|-------------------------------|
| Detailed Transport | 2.894 × 10⁻¹ |
| Direct CSDA | 2.895 × 10⁻¹ |
| Relative difference | 0.03% |

### 7.2 Gradient Validation

Gradients computed by Zygote AD agree with finite difference estimates:

$$
\frac{\partial\Phi}{\partial\rho} \approx \frac{\Phi(\rho + \epsilon) - \Phi(\rho - \epsilon)}{2\epsilon}
$$

For typical rock densities (~2650 kg/m³) and thicknesses (~100 m), the gradient is:

$$
\frac{\partial\Phi}{\partial\rho} \sim -10^{-5} \text{ to } -10^{-6} \text{ m}^{-2}\text{s}^{-1}\text{sr}^{-1}\text{(kg/m}^3\text{)}^{-1}
$$

## 8. Conclusions and Future Work

### 8.1 Summary

DiffPumas.jl successfully implements differentiable muon transport by:

1. Accurately reproducing PUMAS physics for flux computation
2. Enabling automatic differentiation through the Direct CSDA formulation
3. Supporting tessellated geometries with per-cell gradients

### 8.2 Limitations and Future Directions

The current Direct CSDA approach assumes deterministic energy loss. To fully match the detailed transport with straggling and scattering, we need to:

1. **Integrate straggling with Zygote**: This requires either:
   - Reparameterization tricks for stochastic sampling
   - Pathwise gradient estimators
   - Score function (REINFORCE) estimators

2. **Handle scattering differentiably**: Scattering changes trajectory directions, coupling geometric and physics computations

3. **Extend to 3D tomography**: Full inversion of density distributions from multi-angle flux measurements

The framework established here provides a foundation for these extensions, enabling gradient-based optimization for muon tomography applications.

## References

1. Gaisser, T.K. (1990). *Cosmic Rays and Particle Physics*. Cambridge University Press.
2. Particle Data Group (2022). Review of Particle Physics. *Prog. Theor. Exp. Phys.* 2022, 083C01.
3. Niess, V. et al. (2018). PUMAS: A portable library for muon transport. *Computer Physics Communications*, 229, 54-67.
4. Innes, M. (2018). Don't Unroll Adjoint: Differentiating SSA-Form Programs. arXiv:1810.07951.

## Appendix: Code Examples

### A.1 Basic Flux Computation

```julia
using DiffPumas

# Load physics tables
physics = load_or_create_physics("materials.pumas")

# Define geometry
geometry = TwoLayerGeometry(
    rock_thickness = 100.0,  # meters
    rock_density = 2650.0,   # kg/m³
    rock_material = 1,       # StandardRock index
    air_material = 2         # Air index
)

# Compute flux at specific energy and angle
flux = compute_flux_single(physics, geometry, 
    energy = 100.0,      # GeV
    elevation = 60.0,    # degrees
    charge = 1.0         # μ+
)
```

### A.2 Gradient Computation with Tessellated Geometry

```julia
using DiffPumas
using Zygote

# Create cube geometry with cells
geo = CubeGeometry(rock_thickness, nx, ny, nz, rock_idx, air_idx)
cell_densities = fill(2650.0, num_cells(geo))

# Compute flux and gradients
flux, gradients = compute_flux_gradient_cube(
    physics, geo, cell_densities,
    zenith_min, zenith_max,
    energy_min, energy_max,
    n_angles, n_samples
)

# gradients[i] = ∂flux/∂ρᵢ for each cell
```
