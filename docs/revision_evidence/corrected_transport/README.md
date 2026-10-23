# Corrected backward-muon transport evidence

The **final replay** is in [`final_precision/`](final_precision/), using core
commit 6498e86. Its complete launcher exits 0 and verifies identical source hashes
before/after all three-seed controls and the 102-point grid. The parent directory
retains the earlier 916ae68 comparison and its explicitly annotated cache-only
source change. Paper figures should consume the final replay.

This directory curates small, inspectable artifacts from the corrected revision
runs. Complete binaries, dumps, generated C wrappers, logs, and input copies
remain under `examples/data/revision_flat_corrected_20261023/` and
`examples/data/revision_physics_v2_final_20261023/`; their hashes and commands are
recorded here. Historical/pre-fix outputs are preserved separately.

The C reference is unmodified official PUMAS physics v1.2.3, tag v1.2 commit
529976ef51bd420f37ec63b1d3f615784d35b626. Its generated geometry wrapper supplies
explicit input/sample/seed/source-ceiling settings, B=0, and 17-digit output. Its
surface-flux wrapper aligns the muon mass constant with Julia. Generated wrapper
hashes and patches are retained with the complete reference. The C and Julia
serializations are different; each implementation creates its own physics from
the same pinned MDF and three muon stopping tables.

Both use SSR/SSR/DRSS, relative hard cutoff .05, elastic ratio .05, density 2650 kg/m³,
primary altitude 1000 m, GCCLY spectrum, charge ratio 1.2766, equal ±charge proposals,
and detector energies 1e−3..1e9 GeV. The corrected comparisons share a finite source
energy ceiling 1e9 GeV. Nominal transport switches from straggled/scattering below
100 GeV to restricted mixed/no scattering above 100 GeV. C and Julia use independent
RNG implementations; each seed resets across points, so grid residuals are
correlated and are not 102 independent tests. A 10k count does not establish high
precision: inspect the reported per-language standard errors.

The radiative integral previously included an erroneous extra q/(K+m); the
corrected dump also restores a first 1MeV input row that the parser skipped.
`physics-comparison.json` compares independent public C/Julia properties at five
energies in rock/air. CSDA stopping agrees exactly; restricted stopping/rates now
agree closely. CSDA proper-time tables and angular transport paths retain
quantified differences; this evidence does not establish full scattering,
forward-transport, or tau-transport equivalence. Fixed-path tomography forbids
scattering and follows the energy-loss-only scope described in
[the transport notes](../../mixed-transport.md).

`c_source_cutoff_control.csv` changes only the C source ceiling at three points,
with the exact same seed/sample count. Shared prefixes then divergent random
consumption make these estimates correlated. The ratio to the quadrature of
reported errors is descriptive, not a calibrated independent z-test.
`source_cutoff_summary.json` also derives a conservative incident GCCLY tail
bound of 4.3503e−13 m−2 s−1 sr−1 above 1e9 GeV. For fixed-path number-conserving energy
transport this bounds the omitted physical contribution, independently of the
noisy cutoff controls. It is not a bound on implementation bias.

For reproduction/build commands and licensing, see
[the C-reference instructions](../../fresh-c-reference.md). Publication claims
must use the completed full-grid summary and its source audit, rather than this
README as a passing assertion.
