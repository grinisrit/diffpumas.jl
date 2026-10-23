# Radiative DCS moment correction

The raw SSR bremsstrahlung, SSR pair-production and DRSS photonuclear functions
return differential cross sections per transferred energy and per atom.
Their macroscopic energy moment is

`M_j = N_A / (A * 10^-3) * integral[q^j * raw_DCS(K,q) dq]`.

The logarithmic quadrature already uses `dq = q d(log q)`. The previous Julia
integral multiplied by another `q` and divided by `K+m`, producing an unwanted
`q/(K+m)` factor. Official PUMAS's internal wrapper multiplies the raw model by
`(K+m) N_A/(A*10^-3)` before its integral divides by `K+m`; those two energy
factors cancel. The correction removes both the extra `q` and the final division.
The analytical ionisation branch is unchanged.

`Materials.RADIATIVE_DCS_TABLE_VERSION = 2` identifies corrected derived tables.
Imported CSDA arrays are unchanged. Newly generated mixed stopping powers,
hard-event cross sections, and empirical radiative stopping powers change.
Existing serialized tables do not acquire corrected values by loading them.
Historical dumps must remain preserved; stochastic validation requires newly
generated derived tables with recorded provenance.

## Independent checks

`test/data/pumas_radiative_moments.csv` contains 45 moments from official PUMAS
v1.2, commit `529976ef51bd420f37ec63b1d3f615784d35b626`, runtime library 1.2.3.
The fixture records library/source SHA256, material/particle constants, models,
energy bounds, and integration refinement. A separate composite Simpson rule in
log transferred energy integrates the C library's public raw-DCS functions;
it does not use either implementation's material-table integral.

For effective rock Z=11, A=22, the fixture covers muons at 10/100/1000 GeV over
transferred fractions 0.1–0.5, plus muons and taus at 100 GeV over the full hard
band from cutoff 0.05 to the physical endpoint. Reference refinement from
65536 to 131072 Simpson intervals changes moments by at most `7.86e-8` relative.
Interior-band checks use `1e-6` tolerance. The hard band includes model branch
transitions and physical endpoints: default 180-point Julia quadrature agrees
within `2.30e-4`, and 1800-point quadrature within `1.15e-5`. Tests enforce
`3e-4` and `2e-5` respectively. This finite quadrature error is separate from
the unit correction; no table-resolution change is included here. On the
interior reference bands, the previous cross section was only 0.156–0.237 of
the corrected one. These are raw-integral checks, not validation of complete
transport.

The focused test additionally checks the narrow-band differential limit and
positive-measure bounds for energy moments. All 75 checks passed on Julia
1.10.12 and 1.11.7. The fixture is used offline; tests need no C installation.

Reproduce the reference (supply the preserved official library and source):

```bash
python3 dev/reference_dcs_moments.py /path/to/libpumas.so.1.2.3 test/data/pumas_radiative_moments.csv /path/to/pumas/src/pumas.c
JULIA_NUM_THREADS=2 julia --project=. test/test_material_integrals.jl
```

Fresh material-property and transport comparisons against C remain separate
requirements after regenerating derived tables.
