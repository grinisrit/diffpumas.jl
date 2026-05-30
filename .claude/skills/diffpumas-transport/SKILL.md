---
name: diffpumas-transport
description: Muon/tau transport and Monte Carlo in DiffPumas.jl. Use for backward/forward MC, physics tables, mixed materials, or examples/.
---
> **Claude Code:** run from this repo root; invoke with `/diffpumas-transport`.

# DiffPumas transport

## Read first

- Root `README.md` — module structure, energy loss modes, mixed materials

## Common entry points

- `create_physics(MUON)` / tau variants
- `run_backward_mc`, `run_forward_mc` (see `src/` and `examples/`)

## When editing physics

1. Identify particle type and energy loss mode (CSDA, mixed, straggled)
2. Run `Pkg.test()` after changes to cross sections or scattering
3. Update `examples/` if public API changes

## Mixed materials

Runtime mixtures (rock–water–air) use mass fractions — weighted stopping power and scattering must stay consistent.

## Output

- Physics assumption changed
- Example command to reproduce flux
- Test result
