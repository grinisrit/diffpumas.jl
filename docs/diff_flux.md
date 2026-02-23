# Differentiable Muon Transport

## Abstract

This document describes DiffPumas.jl, a Julia implementation of differentiable muon transport through matter. We implement backward Monte Carlo methods for computing underground muon flux and leverage automatic differentiation (AD) to compute gradients of the flux with respect to material properties. The implementation closely follows the physics of the PUMAS C library while enabling gradient-based optimization through Zygote.jl.

## 1. Introduction

Cosmic ray muons are produced in the upper atmosphere through the decay of charged pions and kaons created in hadronic showers. These muons can penetrate deep underground, making them valuable probes for applications such as muon tomography, underground physics experiments, and geological surveys.

### 1.1 Experimental Geometry

We consider a simplified two-layer geometry consisting of:

1. **Air layer**: From the rock-air interface at altitude $z = h$ to the primary sampling altitude $z = H$ (30 000 m)
2. **Rock layer**: From the detector position at $z = 0$ to the rock-air interface at $z = h$

```
                    PRIMARY_ALTITUDE (H = 30 000 m)
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

**Mean free path.** The EHS mean free path $\lambda_{\text{EHS}}$ is loaded from PUMAS physics tables (the `elastic_path` property), which encodes the momentum-dependent hard-scattering cross-section:

$$
\lambda_{\text{EHS}}(K) = \frac{\Lambda_b(K)}{p^2}
$$

where $\Lambda_b$ is the tabulated hard-scattering path parameter and $p$ is the muon momentum. The tabulated values account for the full multi-element screened Coulomb DCS above the cutoff angle $\mu_0$.

**Sampling.** EHS events are sampled with the same acceptance-rejection approach used for DEL:

1. Look up the EHS mean free path $\lambda_{\text{EHS}}(K)$ at the current energy
2. Sample the interaction grammage: $X_{\text{EHS}} = -\lambda_{\text{EHS}} \ln(\zeta)$, where $\zeta \sim \mathrm{Uniform}(0,1)$
3. If $X_{\text{EHS}} < \Delta X$ (the step grammage), the event occurs
4. Apply acceptance correction: accept with probability $\lambda_{\text{EHS}}(K_{\min}) / \lambda_{\text{EHS}}(K_{\text{actual}})$

**Angular distribution.** When an EHS event occurs, the scattering angle is sampled in the centre-of-mass frame from a screened Coulomb (Wentzel) distribution:

$$
\frac{d\sigma}{d\mu_1} \propto \frac{1}{(A + \mu_1)^2} \cdot (1 - f_{\text{spin}} \cdot \mu_1)
$$

where $A = \mu_0 / 4$ is the screening parameter, $\mu_0$ is the elastic cutoff angle from PUMAS tables, and $f_{\text{spin}} = K(E + m) / E^2$ is the spin correction factor. The sampling uses inverse-CDF sampling from the Wentzel envelope:

$$
\mu_1 = \frac{(A + \mu_0)(A + 1)}{A + 1 - \zeta(1 - \mu_0)} - A, \qquad \zeta \sim \mathrm{Uniform}(0,1)
$$

followed by spin-factor rejection: accept if $U \le 1 - f_{\text{spin}} \mu_1$. The accepted $\mu_1$ is then transformed from CM to Lab frame (Section 4.3.7).

**Competition.** Both DEL and EHS interaction lengths are sampled at the start of each step. The event with the shorter grammage wins; the other is discarded. This matches PUMAS C `transport_limit`, where both are computed upfront and the closer one determines the step's vertex.

**Direction update ordering.** Within the scattering block, deflections are applied in order: (1) DEL angular deflection, (2) EHS angular deflection, (3) soft MCS. All three accumulate on the direction vector before the position update.

#### 4.3.7 EHS Centre-of-Mass to Laboratory Frame Transformation

The Wentzel sampling described above produces a scattering parameter $\mu_1$ in the centre-of-mass (CM) frame. For heavy targets the CM and Lab frames differ appreciably. PUMAS converts via the Lorentz boost.

Let the projectile (muon, mass $m$) scatter off a target nucleus (mass $M$) with lab kinetic energy $K$. The Mandelstam variable is:

$$
s = m^2 + M^2 + 2(K + m)M
$$

The CM kinetic energy is:

$$
K_0 = \frac{s - (m + M)^2}{2\sqrt{s}}
$$

Denote $E_{\mathrm{CM}} = K_0 + m$, $\,p_{\mathrm{CM}} = \sqrt{K_0(K_0 + 2m)}$, and $E_{\mathrm{lab}} = K + m$, $\,p_{\mathrm{lab}} = \sqrt{K(K + 2m)}$. The boost parameters are:

$$
\gamma_{\mathrm{CM}} = \frac{E_{\mathrm{lab}} E_{\mathrm{CM}} + p_{\mathrm{lab}} p_{\mathrm{CM}}}{m\sqrt{s}}, \qquad
\tau = \frac{E_{\mathrm{lab}} p_{\mathrm{CM}} - E_{\mathrm{CM}} p_{\mathrm{lab}}}{p_{\mathrm{lab}} p_{\mathrm{CM}}}
$$

For $\mu_1 > 10^{-6}$ the lab-frame scattering parameter is obtained from the exact formula:

$$
a = \gamma_{\mathrm{CM}}(\tau + 1 - 2\mu_1), \qquad
\cos\hat\theta = \frac{a}{\sqrt{4\mu_1(1 - \mu_1) + a^2}}, \qquad
\mu_{\mathrm{lab}} = \tfrac{1}{2}(1 - \cos\hat\theta)
$$

For small angles ($\mu_1 \le 10^{-6}$) the asymptotic limit avoids cancellation:

$$
\mu_{\mathrm{lab}} = \frac{\mu_1}{\bigl[\gamma_{\mathrm{CM}}(1 + \tau)\bigr]^2}
$$

### 4.4 Multiple Coulomb Scattering

As muons traverse matter, they undergo many small-angle Coulomb scatterings. The cumulative angular deflection is described by the first transport mean free path $\lambda_1$, which encodes the scattering power of the medium.

#### 4.4.1 First Transport Mean Free Path

The inverse first transport mean free path $1/\lambda_1$ (units: m²/kg) sums three contributions:

$$
\frac{1}{\lambda_1} = \frac{1}{\lambda_1^{\text{el}}} + \frac{1}{\lambda_1^{\text{rad}}} + \frac{1}{\lambda_1^{\text{e}}}
$$

- **$1/\lambda_1^{\text{el}}$**: Elastic (nuclear Coulomb) contribution. Computed by PUMAS from the full screened nuclear DCS integrated over scattering angle up to the hard-scattering cutoff $\mu_0$, with CM→Lab transformation and nuclear form factors for each atomic element.
- **$1/\lambda_1^{\text{rad}}$**: Radiative contributions from bremsstrahlung, pair production, and photonuclear soft collisions (below the DEL cutoff).
- **$1/\lambda_1^{\text{e}}$**: Electronic (ionisation) transverse transport, computed from the restricted Møller/Mott cross-section up to the DEL energy cutoff $\nu = \varepsilon K$ (MIXED mode) or $\nu = K$ (CSDA mode).

The mode dependence enters through the electronic and radiative terms: MIXED mode integrates soft processes up to the DEL cutoff $\varepsilon K$, while CSDA mode integrates to the full kinematic limit.

DiffPumas loads the exact PUMAS $\lambda_1$ values via a CSV table produced by the `dump-scattering` utility, avoiding the need to reimplement the full Coulomb DCS integration.

#### 4.4.2 Soft MSC Angle Sampling

At each transport step, the soft multiple scattering deflection is sampled following PUMAS (pumas.c lines 6913–6922). Given a step of distance $\ell$ (metres) through a medium of density $\rho$ (kg/m³):

1. Compute the inverse transport paths at the step's start and end energies:

$$
\frac{1}{\lambda_1^{\text{start}}} = \frac{\rho}{\lambda_1(K_{\text{start}})}, \qquad
\frac{1}{\lambda_1^{\text{end}}} = \frac{\rho}{\lambda_1(K_{\text{end}})}
$$

2. Compute the scattering strength parameter using the trapezoidal rule:

$$
\bar\mu = \tfrac{1}{4}\,\ell\,\Bigl(\lambda_1^{-1,\text{start}} + \lambda_1^{-1,\text{end}}\Bigr)
$$

   The factor $\frac{1}{4}$ arises from the relationship $\langle\mu\rangle = \frac{1}{2}\langle 1-\cos\theta\rangle = \frac{X}{2\lambda_1}$ combined with the trapezoidal averaging factor $\frac{1}{2}$.

3. Clamp: if $\bar\mu > 1$, set $\bar\mu = 1$ (isotropic limit).

4. Sample $\mu$ from the exponential distribution with rejection:

$$
\mu = -\bar\mu\,\ln(U), \quad U \sim \mathrm{Uniform}(0,1)
$$

   Reject and re-sample if $\mu > 1$. This is the exact PUMAS algorithm; for small $\bar\mu$ the rejection probability is negligible.

The sampled $\mu = \frac{1}{2}(1 - \cos\theta)$ is passed to the direction rotation.

#### 4.4.3 Step Size Limitation from Scattering

When scattering is active, PUMAS limits the step size to keep the scattering angle small per step. In addition to the energy-loss-based limit $\ell_{\max} = \varepsilon_{\text{acc}} \cdot R(K) / \rho$, the step is also limited by:

$$
\ell_{\text{scat}} = \frac{\varepsilon_{\text{acc}}}{\rho / \lambda_1(K)}
$$

where $\varepsilon_{\text{acc}} = 0.01$ is the accuracy parameter. The actual step is:

$$
\ell = \min\bigl(\ell_{\text{boundary}},\; \ell_{\max},\; \ell_{\text{scat}}\bigr)
$$

This ensures that $\bar\mu \lesssim \varepsilon_{\text{acc}}$ per step, maintaining the validity of the small-angle approximation.

#### 4.4.4 Direction Rotation

Given a deflection parameter $\mu$ and azimuthal angle $\phi \sim \mathrm{Uniform}(-\pi, \pi)$, the direction vector $\hat{d}$ is rotated as follows.

Compute:

$$
\cos\theta = 1 - 2\mu, \qquad \sin\theta = \sqrt{4\mu(1-\mu)}
$$

Construct an orthonormal basis $(\hat{u}_0, \hat{u}_1, \hat{d})$ where $\hat{u}_0$ is chosen as the co-vector with the largest component of $\hat{d}$ projected out, and $\hat{u}_1 = \hat{u}_0 \times \hat{d}$. Then:

$$
\hat{d}' = \cos\theta\,\hat{d} + \sin\theta\,(\cos\phi\,\hat{u}_0 + \sin\phi\,\hat{u}_1)
$$

This matches PUMAS `step_rotate_direction` exactly.

#### 4.4.5 Mixture Transport Path

For a material mixture with mass fractions $f_i$, the inverse transport path follows harmonic-mean weighting:

$$
\frac{1}{\lambda_{1,\text{mix}}} = \sum_i \frac{f_i}{\lambda_{1,i}}
$$

This reflects the additive nature of scattering probabilities.

### 4.5 Composite Materials

#### 4.5.1 Definition-Time Composites

A `CompositeMaterial` is a named combination of `BaseMaterial`s with mass fractions, defined in the Material Description File (MDF XML). For example:

```xml
<composite name="PorousWetRock">
    <component name="StandardRock" fraction="0.8"/>
    <component name="Air"          fraction="0.1"/>
    <component name="Water"        fraction="0.1"/>
</composite>
```

Composites are resolved at physics-table creation time. Each composite receives its own `MaterialTable` with fully precomputed tables (stopping power, range, proper time, Coulomb scattering parameters), exactly as PUMAS C handles them.

#### 4.5.2 Composite Table Construction

Given a composite with $N_c$ components at mass fractions $f_j$:

**Stopping power.** The composite stopping power is the mass-fraction-weighted sum of component stopping powers:

$$
\left(\frac{dE}{dX}\right)_{\text{comp}} = \sum_j f_j \left(\frac{dE}{dX}\right)_j
$$

This applies to both CSDA and mixed-mode stopping powers. Straggling variance, ionisation, bremsstrahlung, pair production, photonuclear, and total cross-section tables are formed identically.

**Range and proper time.** The composite range table is obtained by trapezoidal integration of $1/(dE/dX)_{\text{comp}}$ over the reference energy grid — the same procedure used for single materials:

$$
R_{\text{comp}}(E_k) = R_{\text{comp}}(E_{k-1}) + \frac{E_k - E_{k-1}}{\tfrac{1}{2}\bigl[(dE/dX)_k + (dE/dX)_{k-1}\bigr]}
$$

**Density.** The composite density follows the PUMAS C harmonic-mean formula:

$$
\rho_{\text{comp}} = \frac{\sum_j f_j}{\sum_j f_j / \rho_j}
$$

**Coulomb scattering.** The composite is flattened to its constituent atomic elements: for each element $e$ shared across components, the elemental fraction is $w_e = \sum_j f_j \cdot w_{e,j}$ (summing over all components that contain element $e$). An effective Z/A and mean excitation energy $I$ are computed from these combined elemental fractions using the Sternheimer–Bragg rule:

$$
(Z/A)_{\text{comp}} = \sum_e \frac{Z_e}{A_e} w_e', \qquad
\ln I_{\text{comp}} = \frac{\sum_e \frac{Z_e}{A_e} w_e' \ln I_e}{(Z/A)_{\text{comp}}}
$$

where $w_e'$ are the normalised elemental fractions. The PUMAS Coulomb scattering routine then computes transport path, elastic path, and screening parameters from this synthetic material.

After construction, a composite material is indistinguishable from a single material in the transport engine — it has its own index in `PhysicsTables` and all table lookups proceed identically.

### 4.6 Material Mixtures

#### 4.6.1 Motivation

In many muography scenarios, the traversed medium is not a single pure material. An aquifer, for example, consists of rock with varying degrees of water saturation. Rather than defining a new composite for every possible combination, DiffPumas.jl supports **runtime material mixtures**: arbitrary combinations of loaded materials (including composites), specified by mass fractions.

#### 4.6.2 The `MaterialMixture` Type

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

#### 4.6.3 Physics for Mixtures

Mixture physics is designed to follow exactly the same procedure as composites and single materials. The key difference is that tables are built lazily at runtime rather than at load time.

##### 4.6.3.1 Continuous Processes (CSDA)

For continuous energy loss, the effective stopping power of a mixture is the mass-fraction-weighted sum of per-material stopping powers:

$$
\left(\frac{dE}{dX}\right)_{\text{mix}} = \sum_i f_i \left(\frac{dE}{dX}\right)_i
$$

where $f_i$ is the mass fraction of material $i$. This applies to both total CSDA stopping power and the mixed-mode soft stopping power.

Similarly, straggling variance is additive:

$$
\Omega^2_{\text{mix}} = \sum_i f_i \, \Omega^2_i
$$

**Integrated range and proper time tables.** Rather than approximating the range from a dominant material, mixtures build their own `MixtureTable` containing fully integrated CSDA and mixed-mode range tables as well as the CSDA proper time table. The integration follows the same trapezoidal procedure used for composites (Section 4.5.2):

$$
R_{\text{mix}}(E_k) = R_{\text{mix}}(E_{k-1}) + \frac{E_k - E_{k-1}}{\tfrac{1}{2}\bigl[(dE/dX)^{\text{mix}}_k + (dE/dX)^{\text{mix}}_{k-1}\bigr]}
$$

These tables are lazily computed on first use and thread-safely cached (keyed by physics instance, material indices, and fractions). Once built, range lookups and energy inversions for a mixture use the same `interpolate_table_fast` and binary-search routines as single materials and composites, giving identical numerical behaviour.

##### 4.6.3.2 Discrete Events (DEL, EHS)

For discrete interactions, the total cross-section is the weighted sum:

$$
\sigma_{\text{mix}} = \sum_i f_i \, \sigma_i(E)
$$

When a discrete event occurs, the interacting material is sampled with probability:

$$
P(\text{material } i) = \frac{f_i \, \sigma_i(E)}{\sigma_{\text{mix}}}
$$

The sampled material's cross-section tables are then used to determine the energy transfer, process type (bremsstrahlung, pair production, photonuclear), and other event properties.

##### 4.6.3.3 Scattering

The soft multiple scattering transport path follows harmonic-mean weighting (Section 4.4.5):

$$
\frac{1}{\lambda_{1,\text{mix}}} = \sum_i \frac{f_i}{\lambda_{1,i}}
$$

For elastic hard scattering (EHS) and its Wentzel sampling, mixtures use **effective screening parameters** computed as fraction-weighted averages of the per-material values:

$$
\mu_{0,\text{mix}}(E) = \sum_i f_i \, \mu_{0,i}(E)
$$

$$
f_{\text{spin},\text{mix}}(E) = \sum_i f_i \, f_{\text{spin},i}(E)
$$

$$
(Z/A)_{\text{mix}} = \sum_i f_i \, (Z/A)_i
$$

The effective $(Z/A)_{\text{mix}}$ determines the target mass $M = A_{\text{eff}} \cdot m_n$ used in the centre-of-mass to laboratory frame transformation. With these three effective parameters, the Wentzel sampling for a mixture (inverse-CDF draw from the screened Coulomb envelope, spin-factor rejection, CM→Lab boost) follows precisely the same algorithm as for a single material or composite (Section 4.3.6).

##### 4.6.3.4 Summary: Mixture vs Composite

| Aspect | Composite | Mixture |
|--------|-----------|---------|
| Definition | MDF XML at load time | Runtime, arbitrary fractions |
| Stopping power | Precomputed table | Weighted sum of component tables |
| Range / proper time | Precomputed table | Lazily integrated, cached table |
| Coulomb (soft MCS) | Precomputed $\lambda_1$ | Harmonic-mean $1/\lambda_1$ |
| EHS screening | Precomputed from flattened elements | Weighted-average $\mu_0$, $f_{\text{spin}}$, $Z/A$ |
| DEL / EHS events | Single-material tables | Weighted cross-section, sampled material |
| Table lookup speed | Direct index | Same (after lazy build) |

In CSDA mode (range, inverse range, proper time) the two approaches are numerically equivalent: both integrate the same mixture stopping power over the same energy grid. The scattering treatment differs slightly — composites use exact atomic-element-level Coulomb computation while mixtures use effective weighted parameters — but for typical geological mixtures the difference is negligible.

## 5. Implementation, Tessellated Geometry, and Validation

DiffPumas.jl matches the physics of the PUMAS C library: physics tables (range, stopping power, proper time) are loaded from PUMAS binary dumps or built from dE/dx data, the dual-mode transport (STRAGGLED below 100 GeV, MIXED above) matches exactly, and integrated flux agrees within statistical uncertainty (relative difference typically < 0.1%). Composite materials defined in MDF XML receive precomputed tables identical to base materials (Section 4.5); runtime mixtures lazily build equivalent tables on first use (Section 4.6).

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
                    PRIMARY_ALTITUDE (30 000 m)
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

## 6. Muography Example (`muography.jl`)

Sections 4.5–4.6 define composites and runtime mixtures. This section describes the muography example script.

### 6.1 Part 1: Baseline Studies

Rock depths span 0–1000 m in 100 m steps; density is read from the physics table. Zenith angles span 0°–60° in 2° steps. All three subscenarios use the two-layer geometry (`TwoLayerGeometry`).

- **1.1 Flux vs zenith angle.** For each depth, compute integrated flux (energy-averaged via log-uniform sampling) at each zenith angle. One curve per depth.
- **1.2 Zenith angle scattering variance.** For each (depth, zenith) pair, run backward MC trajectories and record the final zenith $\theta_f$ at primary altitude. Plot $\mathrm{Var}(\theta_f - \theta_{\text{det}})$ vs zenith, one curve per depth, quantifying the angular spread due to multiple scattering.
- **1.3 Flux vs zenith at fixed energy.** At 1000 m depth, for each of 100 log-spaced energies between $E_{\min}$ and $E_{\max}$, compute flux vs zenith. One curve per energy.

### 6.2 Part 2: Aquifer Detection

Fixed geometry: detector at 1000 m depth (z = 0), rock surface at z = 1000 m, air layer above to PRIMARY_ALTITUDE (30 000 m). Materials: Standard Rock, Water, Air, PorousWetRock (precomputed composite from `materials.xml`).

The top 100 m (z = 900–1000 m) is **porous wet rock**, represented by the `PorousWetRock` composite. The aquifer is a 100 m cube embedded in the rock, containing a three-component `MaterialMixture` of Water, Rock, and PorousWetRock with configurable water fraction.

```
                    PRIMARY_ALTITUDE (30 000 m)
    ════════════════════════════════════════════════════════
                         |
                    AIR LAYER
                         |
    ──────────────────────────────────────── z = 1000 m (rock surface)
                         |
        POROUS WET ROCK (composite, z = 900–1000 m)
                         |
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                         |  ┌─────────────────────────┐
                         │  │ AQUIFER (rock–water mix)│
                         │  └─────────────────────────┘
                         |
                    ROCK |
                         |      ╱ incoming muon
                         |    ╱   at zenith angle θ
                         |  ╱
                         |╱
    ─────────────────────●─────────────────── z = 0 (Detector, 1000 m depth)
```

**Aquifer scans.** The script sweeps: water fraction 0%–90% at fixed depth and extent; fixed 90% water with increasing aquifer size; aquifer moved vertically and along the y-axis (azimuth sampled uniformly). The simulation cube is centred along y (x from −50 to +50 m).

**Implementation.** Cells use `MaterialMixture` (Section 4.6.2); in the shallow+aquifer overlap region the mixture consists of `[PorousWetRock, Rock, Water]`. Transport applies the mixture physics of Section 4.6.3 (integrated range tables, weighted scattering parameters, DEL/EHS with material sampling) at runtime.

## 7. Conclusions and Future Work

### 7.1 Summary

DiffPumas.jl successfully implements differentiable muon transport by:

1. Accurately reproducing PUMAS physics for flux computation
2. Enabling automatic differentiation through the Direct CSDA formulation
3. Supporting tessellated geometries with per-cell gradients
4. **Precomputed composite materials** matching PUMAS C: composites defined in MDF XML receive their own `MaterialTable` with fully integrated range, proper time, and Coulomb scattering tables (Section 4.5)
5. **Runtime material mixtures** with physics aligned to composites: lazily integrated range/proper time tables, weighted-average scattering parameters, and per-material DEL/EHS sampling (Section 4.6)
6. **Full scattering matching PUMAS C**:
   - Soft MSC with exact trapezoidal $\bar\mu = \frac{1}{4}\ell(\lambda_1^{-1,\text{start}} + \lambda_1^{-1,\text{end}})$ formula and rejection sampling
   - EHS with Wentzel sampling, spin rejection, and CM→Lab frame transformation — extended to mixtures via effective screening parameter, spin factor, and Z/A
   - DEL angular deflection (bremsstrahlung/Tsai DDCS, pair production, photonuclear kinematics, ionisation)
   - DEL/EHS competition by interaction length comparison
   - Scattering-aware step size limiting ($\ell_{\text{scat}} = \varepsilon_{\text{acc}} / (\rho/\lambda_1)$)
   - Exact PUMAS transport path tables loaded from C library dump

### 7.2 Limitations and Future Directions

The current Direct CSDA approach assumes deterministic energy loss. To fully match the detailed transport with straggling, DEL events, and scattering, we need to:

1. **Integrate straggling with Zygote**: This requires either:
   - Reparameterization tricks for stochastic sampling
   - Pathwise gradient estimators
   - Score function (REINFORCE) estimators

2. **Handle DEL events differentiably**: DEL events introduce discrete jumps in energy and direction. The stochastic nature of DEL sampling presents similar AD challenges as straggling.

3. **Handle scattering differentiably**: Scattering changes trajectory directions, coupling geometric and physics computations

4. **Extend to 3D tomography**: Full inversion of density distributions from multi-angle flux measurements

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
