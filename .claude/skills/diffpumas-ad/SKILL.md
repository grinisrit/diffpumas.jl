---
name: diffpumas-ad
description: Differentiable flux and Zygote gradients in DiffPumas.jl. Use when changing compute_gradient, ZygoteRules, or sensitivity w.r.t. density.
---
> **Claude Code:** run from this repo root; invoke with `/diffpumas-ad`.

# DiffPumas AD

## Read first

- README section "Differentiable Flux Calculation"

## Patterns

```julia
flux_fn(ρ) = compute_flux_differentiable(physics, ρ, thickness, elevation, ...)
grad = gradient(flux_fn, ρ)[1]
```

## Constraints

- AD must match finite-difference checks in `test/` where present
- `ChainRulesCore` custom rules — add/adjust when introducing non-Zygote-friendly ops
- Do not break `compute_gradient=true` path in `run_backward_mc` without updating tests

## Verify

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Output

- Quantity differentiated (∂flux/∂ρ, etc.)
- Test or numeric spot-check result
