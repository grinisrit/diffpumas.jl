# Differentiable Flux

DiffPumas.jl is designed so selected transport and flux calculations can be
differentiated with Zygote. The main gradients used by the examples are:

- Density sensitivity: `d flux / d rho`.
- Cell water-fraction sensitivity: `d flux / d w_i`.
- Jacobians for absorption tomography.

The most reliable AD path is deterministic direct CSDA. Full stochastic Monte
Carlo is used for value estimates and validation, while sensitivities are usually
computed with direct CSDA or finite differences with common random numbers.

## Gradient Returned By `run_backward_mc`

`run_backward_mc(...; compute_gradient=true)` records the sampled detector
energies, charges, and integration weights from its MC run. It differentiates
the average deterministic CSDA flux at exactly those samples, holding them fixed
as density changes. This fixes the scale and energy integration of the density
sensitivity returned by the high-level helper.

The result reports `flux` and `sigma` for the MC estimate, `gradient` for this
CSDA surrogate's derivative, `gradient_flux` for the surrogate's value, and
`gradient_method = :deterministic_csda_sample_average`. The latter two fields
are `nothing` when gradients are disabled. The surrogate and MC values can
differ when stochastic straggling or scattering is enabled; `gradient` is not
an AD derivative through those random transport events. Monte Carlo errors in
`result.sigma` describe the flux estimate, not the uncertainty of its surrogate
gradient. Validate the stochastic density response separately, for example
with repeated finite differences using common random numbers.

## Why Deterministic Paths Matter

Reverse-mode AD differentiates a deterministic program trace. Random draws and
branching stochastic events inside the differentiated function can make gradients
ill-defined, noisy, or very expensive.

Recommended pattern:

1. Create or load physics tables outside the differentiated function.
2. Pre-sample energies, charges, and directions outside the function.
3. Define a pure function of the parameter of interest.
4. Call `Zygote.gradient` or use a DiffPumas helper that does this internally.

## Density Gradient With Direct CSDA

Use `compute_flux_gradient_csda` for a compact scalar example:

```julia
using DiffPumas

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

flux, grad = compute_flux_gradient_csda(
    physics,
    2650.0,  # density, kg/m^3
    100.0,   # rock thickness, m
    45.0,    # elevation, degrees
    10.0,    # final kinetic energy, GeV
    1.0,     # charge
)

println("flux = $flux")
println("dflux/drho = $grad")
println("log sensitivity = $(grad * 2650.0 / flux)")
```

The log sensitivity is often easier to interpret:

```julia
sensitivity = grad * rho / flux
```

It is the fractional flux change per fractional density change.

## Manual Zygote Gradient

The direct helper is just a convenience. You can define your own target function:

```julia
using DiffPumas
using Zygote

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

rock_thickness = 100.0
elevation = 45.0
energy_final = 10.0
charge = 1.0

flux_at_density(rho) = compute_flux_differentiable_csda(
    physics,
    rho,
    rock_thickness,
    elevation,
    energy_final,
    charge,
)

rho0 = 2650.0
flux0 = flux_at_density(rho0)
grad0 = gradient(flux_at_density, rho0)[1]

println((flux = flux0, dflux_drho = grad0))
```

Use this pattern when you need a custom objective, such as a loss function built
from multiple angles or energies.

## Integrated Flux Gradient

For an integrated flux, pre-sample the Monte Carlo quadrature and differentiate
the deterministic sum:

```julia
using DiffPumas
using Random
using Zygote

physics = create_physics(MUON; n_energies = 80, K_min = 1e-2, K_max = 1e6)

energy_min = 1e-3
energy_max = 1e6
n_samples = 128
rng = MersenneTwister(42)

samples = [
    begin
        energy, weight = sample_energy_loguniform(energy_min, energy_max, rng)
        charge = rand(rng) > 0.5 ? 1.0 : -1.0
        (energy, weight, charge)
    end
    for _ in 1:n_samples
]

function integrated_flux(rho)
    total = 0.0
    for (energy, weight, charge) in samples
        total += 2.0 * weight * compute_flux_differentiable_csda(
            physics,
            rho,
            100.0,
            45.0,
            energy,
            charge,
        )
    end
    return total / length(samples)
end

rho0 = 2650.0
flux0 = integrated_flux(rho0)
grad0 = gradient(integrated_flux, rho0)[1]

println((flux = flux0, dflux_drho = grad0))
```

`examples/diff_flux.jl` uses the same idea for its direct-CSDA AD section and
also compares the AD gradient to finite differences.

## Finite-Difference Check

Always check new differentiable transport logic against a finite difference when
the quantity is important:

```julia
rho0 = 2650.0
step = 1.0

fd = (integrated_flux(rho0 + step) - integrated_flux(rho0 - step)) / (2step)
ad = gradient(integrated_flux, rho0)[1]

println("finite difference = $fd")
println("AD = $ad")
println("relative error = $(abs(fd - ad) / max(abs(ad), 1e-30))")
```

Use a step size large enough to avoid interpolation noise but small enough to
measure the local slope. Density steps around `1 kg/m^3` are often a reasonable
starting point for rock-density examples.

## Water-Fraction Gradients

Tomography treats the unknown field as a vector of water fractions `w`. Each cell
is converted into a rock/water material mixture and an effective density.

The main AD helper is:

```julia
flux, cells, grad = directional_flux_and_grad_csda(
    physics,
    shallow_flags,
    material_config,
    site_config,
    path,
    w,
    energy_samples,
)
```

It returns:

- `flux`: directional flux for one angular path.
- `cells`: the cell indices touched by the path.
- `grad`: derivative of the flux with respect to the touched cells' water
  fractions.

The typical pattern is to assemble rows into a sparse forward model:

```julia
flux0, J = assemble_forward_and_jacobian(
    physics,
    shallow_flags,
    material_config,
    site_config,
    paths,
    w0,
    energy_samples,
)
```

Use `J` as a local linearization for reconstruction or sensitivity analysis.

## Direct CSDA Correction

The tomography module includes an optional differentiable correction for direct
CSDA:

```julia
using DiffPumas
using DiffPumas.Tomography

set_csda_correction!(
    enabled = true,
    kappa_strag = 1.0,
    kappa_hard = 0.0,
    resid_a = 0.0,
    resid_b = 0.0,
)

current = get_csda_correction()
@show current.enabled
```

The correction is disabled by default. It is intended for calibrated tomography
workflows where direct CSDA sensitivities are matched to stochastic MC or
measured data. See `examples/lvd_tomography.jl` for the production-style use
case.

## Full MC Versus AD

Use full MC when you need stochastic physics:

- Straggling.
- Discrete energy-loss events.
- Scattering.
- Transport systematic checks.

Use direct CSDA AD when you need:

- Fast gradients.
- Sparse Jacobian assembly.
- Inversion loops.
- Smooth objective functions.

Use common-random-number finite differences to compare them. The validation
script `examples/lvd_aquifer_validation.jl` is built around this comparison:

```bash
julia --project=. examples/lvd_aquifer_validation.jl 30 0
```

It compares CSDA value and AD gradient against full-MC finite differences along a
real LVD line of sight.

## AD Constraints

When adding new differentiable code:

- Keep differentiated functions deterministic.
- Avoid mutation of arrays captured by Zygote unless the code already has custom
  rules.
- Prefer immutable `State` updates through `update_state`.
- Keep random draws outside the differentiated function.
- Add or adjust ChainRulesCore rules when introducing non-Zygote-friendly
  operations.
- Verify against finite differences.
- Run the test suite after changes to transport, interpolation, or custom rules.

## Troubleshooting Gradients

If the gradient is `nothing`, the target may not depend on the parameter, or the
parameter may have been converted to a non-differentiable path.

If the gradient is zero, test a finite-difference perturbation. Some geometries
and energy ranges produce extremely small sensitivity.

If the gradient is slow, reduce the number of samples and make sure you are not
differentiating stochastic MC. Direct CSDA is the intended high-throughput path.

If AD and finite difference disagree, first confirm they use the same model. A
direct CSDA AD gradient should be compared to a direct CSDA finite difference,
not to a straggled/scattered MC finite difference unless the difference itself is
the physics being studied.

### Local tomography segment adjoints

Tomography's adaptive CSDA segments use ForwardDiff tangents inside their
ChainRules pullbacks by default. A unit-weight walk computes the segment energy
map and multiplicative weight factor together; the cell walk seeds water
fraction and incoming energy simultaneously. This avoids a large reverse-mode
tape and finite-difference steps that can straddle interpolation or adaptive-step
branches. Valid zero incoming weight retains both the energy derivative and the
derivative of output weight with respect to input weight.

These are derivatives of the executed, discretised deterministic model. At
interpolation knots and adaptive-step ties a classical derivative need not exist.
Control flow compares primal values, with deterministic left selection for
min/max ties; the table endpoints keep the original clamped, zero-derivative
branch. Invalid stopped paths retain the existing zero-cotangent convention.
Neither exact differentiation of a smooth branch nor finite-difference agreement
establishes accuracy of stochastic material sensitivities.

`get_adjoint_derivative_mode()` returns `:forwarddiff`. For reproducing historical
numerical adjoints, explicitly call
`set_adjoint_derivative_mode!(:finite_difference)` and set the legacy relative
step with `set_adjoint_fd_step!(1e-6)`. The step setter does not select FD mode,
and its value is unused in forward mode. Restore `:forwarddiff` afterwards.
These process-global selectors must only change between derivative evaluations,
never while threaded transport is running. Record the active backend with run
provenance; an archived FD step alone does not describe the new default adjoint.
