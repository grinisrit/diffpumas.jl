# Residual radiative angular-moment limitation

The corrected backward energy-loss kernel does not make all Julia scattering
physics identical to PUMAS. This note records a separate, uncorrected limitation
in the Coulomb table construction; no scattering-model rewrite is included in
this revision.

## Verified source differences

In `src/Coulomb.jl`, `dcs_transport_integral` approximates the angular factor by
`theta² = q² / p²` and integrates it against the raw microscopic radiative
`dσ/dq`. Its result remains per atom. `compute_coulomb_tables!` adds the
mass-fraction-weighted result directly to macroscopic elastic and electronic
inverse transport paths, without an atoms-per-kilogram conversion. The radiative
term is therefore dimensionally incompatible with the other terms; it cannot be
regarded as a correctly normalized macroscopic angular moment.

Official PUMAS v1.2, commit
`529976ef51bd420f37ec63b1d3f615784d35b626` (library 1.2.3), instead evaluates an
angular transport kernel. Its
[bremsstrahlung transport integral](https://github.com/niess/pumas/blob/529976ef51bd420f37ec63b1d3f615784d35b626/src/pumas.c#L14145)
uses a Tsai angular integrand and applies `2 N_A / (A * 10^-3)` at line 14206.
Its [photonuclear transport integral](https://github.com/niess/pumas/blob/529976ef51bd420f37ec63b1d3f615784d35b626/src/pumas.c#L15112)
uses an angular moment of the doubly differential cross section and applies the
same macroscopic factor at line 15157. Simply adding a normalization factor to
the Julia surrogate would leave its angular approximation unresolved.
The inspected official `src/pumas.c` SHA256 is
`1fbd2554994ef9bde2a1a5d3207697b17b3e5dc6889296702a246e1697bddd32`.

## Observed table discrepancies and scope

The preserved corrected property probes are:

- `examples/data/revision_flat_corrected_20261023/physics-c.csv`, SHA256
  `0dac60bc06d88c0a8e6b5d329257fa6ad7c5610eda5b8fe4073dd1333116320d`.
- `examples/data/revision_flat_corrected_20261023/physics-julia.csv`, SHA256
  `4393f1dedd54c97d8d8ed6f5462d9ddf4a5e73778f800a44a341238c6297e4a8`.

For the sampled energies 10, 100, 1000, 10000 and 100000 GeV in StandardRock and
Air, the largest relative discrepancy in the restricted transport path is
`Julia/C - 1 = +20.4871%`, for Air at 100000 GeV (C `1.3235424379098786e16`,
Julia `1.5946979515782786e16`, in column-density units). Selected lower-energy
knots show:

| Material | Energy (GeV) | Transport path: Julia/C − 1 |
|---|---:|---:|
| StandardRock | 10 | +0.08569% |
| StandardRock | 100 | +1.80208% |
| Air | 10 | −2.62964% |
| Air | 100 | +0.06722% |

These sampled values are not a bound across the full scattering-energy domain.
The source differences above identify a concrete model limitation, but a full
term-by-term decomposition has not established what fraction of each observed
discrepancy they explain. Other differences may contribute; the opposite sign
at the 10 GeV Air knot alone prevents attributing the complete audit to a missing
positive radiative contribution.

The configured flat comparison disables scattering above 100 GeV, so its largest
high-energy table discrepancy is not directly a scattering contribution in that
configuration. Table errors also do not translate directly into equal flux errors.
Common-input flux comparisons test only their stated geometries, modes, energies
and sample uncertainties; they do not establish general PUMAS equivalence.

Fixed-path tomography production explicitly disables scattering and rejects
`scattering=true`, because it cannot retrace cell/topography boundaries. This
separate angular-moment limitation therefore does not block its energy-loss-only
calculation. Its calibration, quadrature, transport approximation and uncertainty
checks remain separate requirements.
