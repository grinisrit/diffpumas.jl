# DiffPumas.jl — differentiable muon/tau transport

Julia port of [PUMAS](https://github.com/niess/pumas) with **Zygote** gradients for flux w.r.t. material density (muography, inversion).

## Layout

| Path | Role |
|------|------|
| `src/` | Physics, transport, MC, differentiable flux |
| `test/` | Unit tests |
| `examples/` | Usage examples |
| `docs/` | Extra documentation |
| `dev/` | Dev utilities |

## Environment

Julia **1.10+** (`Project.toml`). Use project environment:

```bash
julia --project=.
```

## Quick API

```julia
using DiffPumas
physics = create_physics(MUON)
result = run_backward_mc(physics; rock_thickness=100.0, compute_gradient=true)
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## AD notes

- Zygote + ChainRulesCore; test gradients when changing energy loss or scattering
- Mixed materials: mass fractions affect stopping power and straggling — see README

## Skills

- `.cursor/skills/diffpumas-transport/` — MC, physics tables, examples
- `.cursor/skills/diffpumas-ad/` — differentiable flux and Zygote
