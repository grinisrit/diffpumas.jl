# Optional shallow-mixture dry-boundary inconsistency

A source inspection on 2026-10-23 found an inconsistency in the optional shallow
porous branch. No transport simulation was run for this finding, and the frozen
runtime has not been changed.

The branch requires all three conditions in
[`is_shallow_porous`](../src/Tomography.jl#L253): the cell's shallow flag is true,
a porous material is configured, and `porous_top_thickness > 0`.

For this branch, the differentiable helpers use mass fractions

```text
porous = 0.5*(1-w), rock = 0.5*(1-w), water = w
rho = 1 / (porous/rho_porous + rock/rho_rock + water/rho_water)
```

This also applies at `w=0`: the dry background is half porous material and half
rock by mass. See [`cell_density`, lines 263–267](../src/Tomography.jl#L263)
and [`cell_stopping_power`, lines 272–277](../src/Tomography.jl#L272).

The stochastic material builder uses that same recipe for `w > 1e-12`, at
[`mc_cell_mixture_and_density`, lines 753–761](../src/Tomography.jl#L753).
However, its special case at
[lines 750–751](../src/Tomography.jl#L750) returns **pure porous material** for
`w <= 1e-12`. Unless the porous and rock properties happen to coincide, its dry
background therefore differs from the differentiable one and from its own
positive-water limit. The discontinuity concerns density and composition, not
just the parameter's name. Optional-layer calibrations that compare dry and wet
states can be affected.

## Scope of the running production calculations

The active corrected production configuration does not enable this branch:

- The Julia CLI default is `porous_top_thickness = 0.0` in
  [`examples/lvd_tomography.jl`, line 1330](../examples/lvd_tomography.jl#L1330).
- The recorded main launcher uses `COMMON` without `--porous-top` in
  [`dev/run_lvd_paper.sh`, lines 119–128](../dev/run_lvd_paper.sh#L119). The
  robustness launcher likewise omits it at
  [`dev/run_revision_robustness.sh`, lines 15–19](../dev/run_revision_robustness.sh#L15).
- The actual main log
  `examples/data/lvd_revision_corrected_20261023/log_main.txt:20` and robustness
  baseline log
  `examples/data/revision_robustness_corrected_20261023/baseline/run.log:20`
  both record `Porous top: 0.0 m`.
- The expanded validation suite constructs its material configuration with zero
  porous-layer thickness at
  [`examples/revision_suite.jl`, lines 155–156](../examples/revision_suite.jl#L155).

These are configuration and log checks, not evidence that the optional branch
is correct. They establish that this specific inconsistency is inactive in the
recorded production and validation settings. It does not invalidate those
calculations through an unused branch. Future runs that enable the shallow
layer require a fix and targeted verification first.

## Minimal proposed correction and verification

Confirm the intended dry recipe before changing code. To preserve the current
differentiable and positive-water convention, replace the pure-porous shortcut
with the same half-porous/half-rock dry recipe and harmonic density, or use the
continuous three-component recipe with supported zero-weight handling. Changing
the entire branch to another porous model would be a separate modelling choice.

After active source audits finish:

1. Use distinct synthetic porous/rock/water densities and mass stopping powers
   so accidental endpoint equality cannot hide the inconsistency. Compare the
   helper density and mass-weighted properties at `w=0`, below/at/above `1e-12`,
   and a representative positive fraction.
2. Check convergence to the declared dry recipe from positive `w`, including
   stopping power, straggling and hard-loss properties used by the correction.
3. Check that disabling the shallow flag, material or layer thickness retains
   the ordinary two-material behavior.
4. Add a small deterministic transport/finite-difference regression with the
   layer enabled before using it in stochastic calibration. Distinguish this
   deterministic consistency check from validation of stochastic physics.

No source or test changes, new simulations, or claims of a verified fix are
part of this note.
