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
\frac{d\Phi}{dE\, d\Omega} = \frac{0.14 \cdot E^{-2.7}}{\text{cm}^{2}\, \text{s}\, \text{sr}\, \text{GeV}}\left( \frac{1}{1 + \frac{1.1 E \cos\theta^{\ast}}{115\, \text{GeV}}} + \frac{0.054}{1 + \frac{1.1 E \cos\theta^*}{850\, \text{GeV}}} \right)
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
- **Below threshold** (typically 100 GeV): Full straggling with the STRAGGLED mode, including DEL events
- **Above threshold**: MIXED mode with purely deterministic CSDA energy loss (no straggling, no DEL events)

### 4.3 Discrete Energy Loss (DEL) Events

While straggling models the statistical fluctuations in continuous energy loss, **Discrete Energy Loss (DEL) events** represent rare, large energy transfers from individual hard collisions. These are distinct from the many small ionization losses that contribute to straggling.

#### 4.3.1 DEL Processes

DEL events include three radiative processes:

1. **Bremsstrahlung**: $\mu \rightarrow \mu \gamma$ - emission of a high-energy photon in the Coulomb field of a nucleus
2. **Pair production**: $\mu \rightarrow \mu e^+ e^-$ - creation of an electron-positron pair
3. **Photonuclear interactions**: $\mu N \rightarrow \mu X$ - inelastic scattering off nuclei

These processes have cross-sections that scale approximately as $\sigma \propto E^{-1}$ at high energies, making them increasingly important for high-energy muons.

#### 4.3.2 DEL Sampling Algorithm

DEL events are sampled using an acceptance-rejection method:

1. **Cross-section lookup**: Compute the maximum DEL cross-section $\sigma_{\text{DEL}}$ over the energy range of the step
2. **Interaction length**: Sample the interaction grammage $X_{\text{DEL}}$ from an exponential distribution: $X_{\text{DEL}} = -\frac{\ln(\zeta)}{\sigma_{\text{DEL}}}$ where $\zeta \sim \text{Uniform}(0,1)$
3. **Energy at interaction**: Determine the energy $K_{\text{DEL}}$ at the interaction point using the fluctuation ratio from straggling
4. **Acceptance**: Accept the event with probability $r = \sigma_{\text{DEL}}(K_{\text{DEL}}) / \sigma_{\text{DEL}}^{\max}$
5. **Energy transfer**: If accepted, sample the fractional energy transfer $\nu$ from a power-law distribution $P(\nu) \propto \nu^{-\alpha}, \quad \nu \in [\nu_{\text{cut}}, 1]$ where $\alpha = 2$ for backward Monte Carlo

#### 4.3.3 When DEL Events Are Modeled

DEL events are **only modeled in STRAGGLED mode** (energies below ~100 GeV):

- **STRAGGLED mode**: Both straggling and DEL events are sampled
- **MIXED mode**: Only deterministic CSDA energy loss (no DEL events)
- **CSDA mode**: Purely deterministic (no DEL events)

This matches the PUMAS C implementation, where DEL sampling occurs only in `PUMAS_MODE_STRAGGLED`.

#### 4.3.4 Importance in Backward Monte Carlo

DEL events are particularly important in backward Monte Carlo transport because:

1. **Energy boosts**: In backward mode, a DEL event that caused energy loss in forward mode becomes an energy gain, allowing particles to reach higher energies
2. **High zenith angles**: For near-horizontal muons, the long path through the atmosphere makes DEL events more likely, significantly affecting the flux
3. **Rare but large**: While DEL events are rare, they can transfer a large fraction of the muon's energy (up to $\nu \approx 1$), making them crucial for accurate flux computation

The energy after a DEL event in backward mode is:

$$
K_{\text{after}} = \frac{K_{\text{before}}}{1 - \nu}
$$

where $\nu$ is the fractional energy transfer sampled from the power-law distribution.

#### 4.3.5 DEL Angular Scattering

In addition to transferring energy, DEL events also deflect the muon. DiffPumas.jl implements the same process-specific polar angle sampling as PUMAS C. Each function returns $\mu = \tfrac{1}{2}(1 - \cos\theta)$:

**Bremsstrahlung and pair production** use Tsai's double-differential cross-section (DDCS) with nuclear screening. The characteristic angle is set by the minimum momentum transfer:

$$
\mu_0 = \left(\frac{m_\mu \nu}{2 E (E - \nu)}\right)^2
$$

where $\nu = K_i - K_f$ is the energy transfer and $E = K_i + m_\mu$ is the total energy. A rejection sampling method draws $\mu$ from an envelope $\propto 1/(\mu_0 + \mu)^2$ and accepts based on the unscreened PDF and a nuclear form-factor correction parameterised by the nuclear RMS radius $R_N(Z)$.

**Photonuclear events** sample the virtuality $Q^2$ from the photonuclear DDCS envelope (log-uniform in $Q^2$), then convert to $\mu$ via energy-momentum conservation:

$$
\mu = \frac{1}{2}\,\frac{p\,p' + m_\mu^2 - E\,E'}{p\,p'} + \frac{Q^2}{4\,p\,p'}
$$

where $p$, $p'$ are the initial and final momenta and $E' = E - \nu$.

**Ionisation (delta rays)** uses the exact kinematic formula for an electron initially at rest:

$$
\mu = \frac{1}{2}\left(1 - \frac{p^2 - \nu(E + m_e)}{\sqrt{p^2(p^2 + \nu^2 - 2\nu E)}}\right)
$$

**Activation condition.** Matching PUMAS C (`transport_do_del`, line 5746), DEL angular deflection is applied **only when multiple Coulomb scattering (MCS) is enabled** (`scattering=true`). When scattering is disabled, DEL events update the energy but not the direction. This coupling is physically motivated: if soft MCS is already being tracked, the rarer but larger DEL deflections should also be included for a consistent angular distribution.

#### 4.3.6 Elastic Hard Scattering (EHS)

In addition to radiative DEL events, muons can undergo rare large-angle Coulomb scatters off nuclei. These **elastic hard scattering (EHS)** events are distinct from the many soft Coulomb scatterings that are treated collectively as multiple Coulomb scattering (MCS, Section 4.4).

**Sampling.** EHS events are sampled with the same acceptance-rejection approach used for DEL:

1. Compute the EHS mean free path $\lambda_{\text{EHS}}$ at the lower energy of the step. This is derived from the first transport mean free path $\lambda_1$ and the elastic ratio parameter: $\lambda_{\text{EHS}} \approx \lambda_1 / \varepsilon_{\text{elastic}}$
2. Sample the interaction grammage: $X_{\text{EHS}} = -\lambda_{\text{EHS}} \ln(\zeta)$
3. If $X_{\text{EHS}} < \Delta X$ (the step grammage), the event occurs
4. Apply acceptance correction based on the energy-dependent MFP at the actual interaction energy

**Angular distribution.** When an EHS event occurs, the scattering angle $\mu = \tfrac{1}{2}(1 - \cos\theta)$ is sampled from a screened Coulomb (Wentzel) distribution:

$$
\frac{d\sigma}{d\mu} \propto \frac{1}{(A + \mu)^2} \cdot (1 - f_{\text{spin}} \cdot \mu)
$$

where $A = \mu_0 / 4$ is the screening parameter derived from the Thomas-Fermi atomic radius and $f_{\text{spin}}$ is the spin correction factor for the muon. The sampling uses inverse-CDF sampling from the envelope $1/(A + \mu)^2$ with spin-factor rejection.

**Priority.** In each transport step, DEL and EHS are sampled independently. If a DEL event already occurred (truncating the step), EHS is skipped for that step — the DEL takes priority since it already determined the step's interaction grammage. Conversely, if no DEL occurred, EHS is checked over the full step.

**Direction update ordering.** Within the scattering block, deflections are applied in order: (1) DEL angular deflection, (2) EHS angular deflection, (3) soft MCS. All three accumulate on the direction vector before the position update.

### 4.4 Multiple Coulomb Scattering

As muons traverse matter, they undergo many small-angle Coulomb scatterings. The cumulative effect is described by Molière theory. The RMS scattering angle after traversing thickness $X$ is approximately:

$$
\theta_{\text{rms}} = \frac{13.6 \text{ MeV}}{\beta c p} \sqrt{\frac{X}{X_0}} \left[1 + 0.038 \ln\left(\frac{X}{X_0}\right)\right]
$$

where $X_0$ is the radiation length of the material.

Scattering causes trajectories to deviate from straight lines, affecting both the path length through matter and the apparent arrival direction at the detector.

### 4.5 Material Mixtures

#### 4.5.1 Motivation

In many muography scenarios, the traversed medium is not a single pure material. An aquifer, for example, consists of rock with varying degrees of water saturation. Rather than defining a new material table for every possible mixture, DiffPumas.jl supports **runtime material mixtures**: arbitrary combinations of loaded materials, specified by mass fractions.

#### 4.5.2 The `MaterialMixture` Type

A mixture is represented by:

```julia
struct MaterialMixture
    materials::Vector{Int}       # Material indices into PhysicsTables
    fractions::Vector{Float64}   # Mass fractions (sum to 1)
end
```

A convenience constructor wraps a single material index for backward compatibility:

```julia
MaterialMixture(material::Int) = MaterialMixture([material], [1.0])
```

#### 4.5.3 Physics for Mixtures

##### 4.5.3.1 Continuous Processes (CSDA)

For continuous energy loss, the effective stopping power of a mixture is the mass-fraction-weighted sum of per-material stopping powers:

$$
\left(\frac{dE}{dX}\right)_{\text{mix}} = \sum_i f_i \left(\frac{dE}{dX}\right)_i
$$

where $f_i$ is the mass fraction of material $i$. This applies to both total CSDA stopping power and the mixed-mode soft stopping power.

Similarly, straggling variance is additive:

$$
\Omega^2_{\text{mix}} = \sum_i f_i \, \Omega^2_i
$$

Since precomputed range tables $R(E)$ are only available for individual materials, the mixture range is approximated by scaling the dominant material's range by the ratio of stopping powers:

$$
R_{\text{mix}}(E) \approx R_{\text{dom}}(E) \cdot \frac{(dE/dX)_{\text{dom}}}{(dE/dX)_{\text{mix}}}
$$

This is accurate for the small steps (1% of range) used in the transport loops.

##### 4.5.3.2 Discrete Events (DEL, EHS)

For discrete interactions, the total cross-section is the weighted sum:

$$
\sigma_{\text{mix}} = \sum_i f_i \, \sigma_i(E)
$$

When a discrete event occurs, the interacting material is sampled with probability:

$$
P(\text{material } i) = \frac{f_i \, \sigma_i(E)}{\sigma_{\text{mix}}}
$$

The sampled material's cross-section tables are then used to determine the energy transfer, process type (bremsstrahlung, pair production, photonuclear), and other event properties.

##### 4.5.3.3 Scattering

The transport mean free path for soft multiple scattering follows a harmonic-mean weighting:

$$
\frac{1}{\lambda_{\text{mix}}} = \sum_i \frac{f_i}{\lambda_i}
$$

This reflects the additive nature of scattering probabilities.

## 5. Implementation, Tessellated Geometry, and Validation

DiffPumas.jl matches the physics of the PUMAS C library: physics tables (range, stopping power, proper time) are loaded from PUMAS binary dumps, the dual-mode transport (STRAGGLED below 100 GeV, MIXED above) matches exactly, and integrated flux agrees within statistical uncertainty (relative difference typically < 0.1%).

### 5.1 Automatic Differentiation and Direct CSDA

[Zygote.jl](https://github.com/FluxML/Zygote.jl) computes exact gradients in time comparable to evaluating the function (the "cheap gradient principle"). The iterative transport algorithm (step-by-step energy loss) is poorly suited for AD: control flow depends on state, many small steps accumulate error, and discrete decisions (which cell, which material) are non-differentiable. We use a **Direct CSDA** formulation that computes flux in closed form: (1) precompute geometry—trace the ray through cells, recording segment lengths (non-differentiable); (2) compute physics—for each segment, energy change and weight from range tables (differentiable). For $N$ segments with grammages $\{X_1, \ldots, X_N\}$:

$$
K_{\text{final}} = R^{-1}\left(R(K_{\text{initial}}) + \sum_{i=1}^N X_i\right), \quad
w_{\text{total}} = \prod_{i=1}^N \frac{|dE/dX|_{K_{i+1}}}{|dE/dX|_{K_i}} \cdot P_{\text{decay},i}, \quad
\Phi = w_{\text{total}} \cdot \Phi_{\text{Gaisser}}(\cos\theta, K_{\text{final}})
$$

Zygote differentiates through the physics tables (differentiable interpolation); geometric ray tracing stays separate.

### 5.2 Tessellated Geometry Example

Gradient computation with spatially-varying properties uses a tessellated rock cube divided into $n_x \times n_y \times n_z$ cells, each with density $\rho_{ijk}$:

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

For each Monte Carlo sample: sample trajectory parameters ($\theta$, $\phi$, $(x_0, y_0)$ ), trace through cells (path length per cell), compute flux via Direct CSDA, and accumulate gradients $\partial\Phi/\partial\rho_{ijk}$. The gradient gives sensitivity to density changes in cell $(i,j,k)$—negative (higher density → fewer muons), dominated by bottom cells and central regions. Applications: tomographic inversion, sensitivity analysis, experiment optimization.

### 5.3 Validation

For pure CSDA transport (no straggling, no scattering), Direct CSDA matches detailed transport within 0.03%:

| Method | Integrated Flux (m⁻² s⁻¹ sr⁻¹) |
|--------|-------------------------------|
| Detailed Transport | 2.894 × 10⁻¹ |
| Direct CSDA | 2.895 × 10⁻¹ |
| Relative difference | 0.03% |

Zygote gradients agree with finite differences $\partial\Phi/\partial\rho \approx [\Phi(\rho+\epsilon) - \Phi(\rho-\epsilon)]/(2\epsilon)$. For typical rock (~2650 kg/m³, ~100 m thickness), $\partial\Phi/\partial\rho \sim -10^{-5}$ to $-10^{-6}$ m⁻² s⁻¹ sr⁻¹ (kg/m³)⁻¹.

## 6. Material Mixtures and Aquifer Muography Example

Section 4.5 defines runtime material mixtures (type and physics). This section describes the aquifer muography example in `muography.jl`.

### 6.1 Aquifer muography example

Fixed geometry: detector at 1000 m depth (z = 0), rock surface at z = 1000 m, air layer above to the primary sampling altitude. The rock is **layered** (no tessellation); materials are Standard Rock, Water, and Air. The **vadose zone** (z = 900–1000 m) is rock with 10% air; the **aquifer** is a layer (e.g. z = 400–600 m) with a rock–water mixture.

```
                    PRIMARY_ALTITUDE
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                         |
    ──────────────────────────────────────── z = 1000 m (rock surface)
                         |
              VADOSE ZONE (rock + 10% air, z = 900–1000 m)
                         |
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ vadose ends
                         |  ┌─────────────────────────┐
                         │  │ AQUIFER (rock–water mix)│
                         │  └─────────────────────────┘
                         |
                    ROCK |
                         |
                         |      ╱ incoming muon
                         |    ╱   at zenith angle θ
                         |  ╱
                         |╱
    ─────────────────────●─────────────────── z = 0 (Detector, 1000 m depth)
```

**Baseline flux.** Integrated flux vs rock depth and zenith angle (e.g. 0–1000 m in 100 m steps, 0°–60° in 2° steps) with a uniform rock layer per depth, using the two-layer geometry and `compute_flux` as in the basic usage.

**Aquifer scans.** The script runs: water fraction 0% to 90% at fixed depth and aquifer extent; fixed 90% water with increasing aquifer size; aquifer layer moved vertically and along the zenith direction. Flux response to location, size, and water content of the anomaly is illustrated.

**Rock-to-water sweep.** A 200 m thick aquifer (z = 400–600 m) with water fraction swept 0% to 100% in steps; effective density $\rho_{\text{eff}} = \sum_i f_i \rho_i$. Resulting flux curves show the characteristic increase as rock is replaced by lighter water.

**Implementation.** Layers use `MaterialMixture` (Section 4.5.2); segment grammage $X = \rho_{\text{eff}} \cdot L$. Transport applies the mixture physics of Section 4.5.3 (weighted stopping power, straggling, DEL/EHS, scattering) at runtime from single-material tables.

## 7. Conclusions and Future Work

### 7.1 Summary

DiffPumas.jl successfully implements differentiable muon transport by:

1. Accurately reproducing PUMAS physics for flux computation
2. Enabling automatic differentiation through the Direct CSDA formulation
3. Supporting tessellated geometries with per-cell gradients
4. **Supporting runtime material mixtures** with physically correct weighted stopping powers, straggling, cross-sections, and scattering
5. **Full discrete angular scattering** matching PUMAS C: process-specific DEL polar angle sampling (bremsstrahlung/Tsai DDCS, pair production, photonuclear kinematics, ionisation) and elastic hard scattering (EHS) with screened Coulomb/Wentzel angular distribution, both activated alongside soft multiple Coulomb scattering

### 7.2 Limitations and Future Directions

The current Direct CSDA approach assumes deterministic energy loss. To fully match the detailed transport with straggling, DEL events, and scattering, we need to:

1. **Integrate straggling with Zygote**: This requires either:
   - Reparameterization tricks for stochastic sampling
   - Pathwise gradient estimators
   - Score function (REINFORCE) estimators

2. **Handle DEL events differentiably**: DEL events introduce discrete jumps in energy and direction. The stochastic nature of DEL sampling presents similar AD challenges as straggling.

3. **Handle scattering differentiably**: Scattering changes trajectory directions, coupling geometric and physics computations

4. **Extend to 3D tomography**: Full inversion of density distributions from multi-angle flux measurements

5. **Precomputed mixture tables**: For frequently used mixtures, precompute range and energy tables to avoid the dominant-material approximation in range lookups

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
