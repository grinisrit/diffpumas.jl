# Muography And Tomography

DiffPumas.jl contains two levels of muography workflows:

- Flat and cubic synthetic studies for flux-vs-angle and aquifer detectability.
- Gran Sasso LVD workflows for measured topography, paper reproduction, 3D
  sensitivity volumes, and synthetic inversion.

The central quantity is transmitted muon flux. Absorption muography compares
observed flux to a forward model; tomography estimates a spatial field, usually
cell water fraction, from many directional flux measurements.

## Coordinate And Geometry Conventions

High-level examples use:

- Detector at the local origin.
- `z` positive upward.
- Zenith angle measured from vertical.
- Elevation angle equal to `90 - zenith`.
- Azimuth in degrees around the detector frame.
- Distances in metres.

For flat two-layer geometry, rock extends from the detector plane to
`rock_thickness`, and air extends above it.

For LVD workflows, `examples/lvd_muography.jl` reconstructs topography from the
Gran Sasso `nm_c.inc` slant-rock table and supporting geometry files in
`examples/data/lvd_conf/`.

## Adapting To Another Site

The transport and tomography APIs accept site-independent arrays and geometry.
The `lvd_*.jl` scripts additionally reconstruct a particular Gran Sasso site
from `examples/data/lvd_conf/`; changing a rock density or output directory does
not replace that site model. A new site needs these inputs:

| Input | Minimum information and API mapping |
| --- | --- |
| Detector and coordinates | Detector location/elevation, orientation, distance units, and zenith/azimuth conventions. Set `SiteConfig` and construct rays in the same local frame. |
| Terrain and overburden | A terrain surface or directional rock lengths, the inversion volume/grid, and each ray's cell lengths and external rock length. Construct `DirectionalPath` records; avoid counting the volume twice in external overburden. |
| Materials | Rock composition, densities, optional shallow layer, and a declared water-fraction convention. Set `MaterialConfig` and physics tables, recording the stopping-power source. |
| Incident spectrum | Surface muon spectrum, energy limits, charge convention, and primary sampling altitude consistent with the selected forward model. |
| Observations for inversion | Directional fluxes or event counts with detector acceptance, exposure, efficiency, angular binning, and uncertainties needed to compare data with predicted flux. Synthetic truth fields replace these only for simulations. |

`MaterialMixture` and the tomography helpers take **mass fractions directly**.
For the two-material rock–water case, `w` is the water mass fraction:

```text
rho(w) = 1 / ((1-w)/rho_rock + w/rho_water)
S(w)   = (1-w)*S_rock + w*S_water
```

The helpers do not convert an input volume fraction. If a water volume fraction
`phi` is supplied independently, the caller must first convert it within this
additive-volume model: `w = phi*rho_water / ((1-phi)*rho_rock + phi*rho_water)`.

This parameter is **not pore saturation at fixed porosity and rock skeleton**.
In a fixed-volume cell, increasing `w` reduces the rock mass and bulk density;
it describes a changing rock–water mixture. Filling existing air-filled pores
with water while holding the rock skeleton fixed would instead increase density
and requires a separate porosity/saturation model. An arithmetic mean of fitted
`w` values is a mixture-amplitude summary, not a measured water inventory.
Even a model-implied water mass `sum(w_i*rho(w_i)*V_i)` remains conditional on
these material and geometric assumptions.

Report the grid, priors, masks, regularization, random seeds, and solver settings
alongside site inputs for a reproducible inversion.

## Flat Muography

`examples/flat_muography.jl` runs synthetic studies in a flat/cubic geometry:

- Baseline flux versus zenith angle for different depths.
- Zenith-angle scattering spread.
- Flux versus zenith at fixed final energies.
- Water-fraction and aquifer-size sweeps.
- Moving aquifer studies.

Quick smoke run:

```bash
julia --project=. examples/flat_muography.jl \
  --output-dir examples/data/flat_smoke \
  --samples 20 \
  --part 1 \
  --no-scattering
```

Higher-statistics aquifer run:

```bash
julia --project=. examples/flat_muography.jl \
  --output-dir examples/data/flat_aquifer \
  --samples 1000 \
  --part 2
```

The script writes Plotly HTML files into the output directory.

## LVD Directional Muography

`examples/lvd_muography.jl` reproduces and visualizes Gran Sasso LVD directional
flux studies.

Parts:

- Part 1: directional flux on the paper-matching angular grid.
- Part 2: topography and representative trajectories.
- Part 3: checks against arXiv:0810.4635v1 benchmark curves.

Smoke run for Part 1:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_smoke \
  --part 1 \
  --samples 20
```

Topography/trajectory visualization:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_topography \
  --part 2
```

Paper-check run with lower energy-spectrum statistics:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_paper_check \
  --part 3 \
  --samples 200 \
  --paper-samples 1000
```

## Tomography Forward Model

The tomography module represents each angular direction as a `DirectionalPath`:

```julia
using DiffPumas
using DiffPumas.Tomography

path = DirectionalPath(
    30.0,                     # zenith, deg
    0.0,                      # azimuth, deg
    [CellSegment(1, 100.0)],  # one crossed cell, 100 m
    100.0,                    # distance in sensitivity volume
    100.0,                    # surface distance
    0.0,                      # remaining rock outside volume
    100.0 * cosd(30.0),       # surface exit height above detector
    true,                     # valid path
)
```

The material and site configuration are separated from the path:

```julia
matcfg = MaterialConfig(
    rock_idx,
    water_idx,
    air_idx,
    porous_idx,
    2650.0,  # rock density, kg/m^3
    1000.0,  # water density, kg/m^3
    2200.0,  # porous density, kg/m^3
    100.0,   # porous top-layer thickness, m
)

site = SiteConfig(
    963.0,    # detector elevation above sea level, m
    30_000.0, # primary flux altitude above detector, m
)
```

Each cell has a water **mass fraction** `w_i`. The helper functions map `w_i` directly to a
material mixture and density:

```julia
w = [0.3]
shallow_flags = falses(1)

rho = cell_density(w[1], shallow_flags[1], matcfg)
dedx = cell_stopping_power(physics, w[1], shallow_flags[1], matcfg, 10.0)

println((density = rho, stopping_power = dedx))
```

## Energy Samples

Tomography uses reusable `EnergySample` values. This keeps the forward model
deterministic for AD:

```julia
samples = sample_energy_set(
    16,
    1.0,     # energy min, GeV
    1.0e5;   # energy max, GeV
    seed = 42,
)
```

For real runs, choose sample counts based on the desired smoothness and runtime.
The LVD tomography example defaults to a small CSDA sample count for Jacobian
assembly and uses separate MC samples for validation.

## Directional Flux And Gradient

For a single path:

```julia
flux = compute_directional_flux_csda(
    physics,
    shallow_flags,
    matcfg,
    site,
    path,
    w,
    samples,
)

flux2, touched_cells, grad = directional_flux_and_grad_csda(
    physics,
    shallow_flags,
    matcfg,
    site,
    path,
    w,
    samples,
)

println("flux = $flux")
println("touched cells = $touched_cells")
println("gradient = $grad")
```

`grad[k]` corresponds to the derivative with respect to the `k`th touched cell in
`touched_cells`.

## Sparse Forward And Jacobian Assembly

For many paths:

```julia
paths = [path1, path2, path3]
w0 = zeros(number_of_cells)

flux0, J = assemble_forward_and_jacobian(
    physics,
    shallow_flags,
    matcfg,
    site,
    paths,
    w0,
    samples,
)
```

`flux0` is the nominal forward prediction and `J` is a sparse local Jacobian.
This is the main input to reconstruction methods.

For absorption deficits, a common linearization is:

```julia
observed = measured_flux
deficit = flux0 .- observed
A = -J
```

`A` and `deficit` can be passed to linearized solvers when both are nonnegative.

## Inverse Solvers

The inverse helpers work with a locally linearized system:

```julia
prior = SmoothnessPrior(neighbors, 1e-2)

w_sart, hist_sart = sart_reconstruct(
    A,
    deficit;
    n_iter = 100,
    relaxation = 0.2,
    prior = prior,
    smooth_strength = 0.1,
)

w_mlem, hist_mlem = mlem_reconstruct(
    A,
    deficit;
    n_iter = 100,
    prior = prior,
)
```

For nonlinear relinearized optimization, use `gradient_descent_reconstruct` or
`gauss_newton_reconstruct` through a model function that recomputes predictions
and Jacobians.

The solvers clamp water fractions to the configured box, usually `[0, 0.9]`.

## Reconstruction Metrics

Useful metrics include:

```julia
report = reconstruction_report(w_recon, w_truth)

println("RMSE = $(report.rmse)")
println("SNR = $(report.snr)")
println("CNR = $(report.cnr)")
```

Other exported helpers include `mse`, `rmse`, `snr_metric`, `recon_snr`, `cnr`,
`psnr`, and `ssim_metric`.

Resolution helpers include:

- `point_spread_recovery`.
- `resolution_vs_depth`.
- `resolution_map`.
- `radial_fwhm`.

## LVD Tomography Example

`examples/lvd_tomography.jl` is the comprehensive workflow. It builds a
sensitivity volume, traces angular paths, assembles CSDA sensitivities, validates
against full MC for selected cases, and runs synthetic reconstruction studies.

Fast grid smoke run:

```bash
julia --project=. examples/lvd_tomography.jl \
  --output-dir examples/data/lvd_tomo_smoke \
  --geometry grid \
  --grid-nz 3 \
  --zenith-max 10 \
  --azimuth-step 30 \
  --samples 2 \
  --mc-samples 4 \
  --validation-cases 1 \
  --reco-iters 5 \
  --gd-iters 2
```

Tetrahedral geometry run:

```bash
julia --project=. examples/lvd_tomography.jl \
  --output-dir examples/data/lvd_tomo_tetra \
  --geometry tetra \
  --mesh-half-km 1 \
  --surface-step-km 0.5 \
  --zenith-step 10 --azimuth-step 15 \
  --samples 32 --csda-quadrature midpoint \
  --mc-samples 64
```

This uses a central 1 km half-width to avoid thin peripheral slabs where the terrain
falls to the detector-plane floor. A 5 km tetrahedral domain can trigger extensive
TetGen quality refinement even with a large maximum cell volume; use `--geometry grid`
for that domain. In `auto` mode, floor-clamped sampled terrain selects the grid.
The geometry-only check is reproducible with
`python3 dev/run_bounded.py --cpus 2 --memory-gib 12 -- julia --project=. dev/check_tetra_geometry.jl`.


Paper-match focused run:

```bash
julia --project=. examples/lvd_tomography.jl \
  --output-dir examples/data/lvd_papermatch \
  --papermatch-only \
  --geometry auto \
  --samples 32 --csda-quadrature midpoint \
  --papermatch-mc-samples 64
```


CSDA uses deterministic log-energy midpoint quadrature by default (`--samples 32`).
Check flux and derivative convergence with more nodes before interpreting numerical
results. `--csda-quadrature random` reproduces the legacy seeded sampling path;
small random sample sets can have large flux bias. Monte Carlo transport retains
independent random energy and charge samples.

The full run can be expensive. Use coarse angular grids and low sample counts for
smoke tests, then scale up for production figures.

## Validation Strategy

A healthy tomography workflow usually checks:

- Geometry validity: paths exit the surface and touch plausible cells.
- CSDA AD self-consistency: AD gradients match direct CSDA finite differences.
- MC comparison: selected CSDA sensitivities are compared to common-random-number
  MC finite differences.
- Reconstruction sanity: synthetic truth fields recover the major anomaly at the
  expected depth and angular coverage.
- Residuals: paper-match or measured-data residuals are reviewed before
  interpreting water-fraction maps.

`examples/lvd_aquifer_validation.jl` is the smallest targeted validation script:

```bash
julia --project=. examples/lvd_aquifer_validation.jl 30 0
```

It slices one LVD line of sight into cells, places an aquifer in one cell, and
compares CSDA+AD with full-MC finite differences.

## Practical Guidance

Use direct CSDA for Jacobian assembly. It is smooth and fast enough for inverse
loops.

Use full MC for validation, uncertainty, and final physics comparisons. It
captures straggling and scattering but is not the main AD path.

Keep sample sets fixed when comparing models or computing finite differences.
Common random numbers reduce noise in gradient checks.

Use low-statistics command-line runs only to verify that the workflow executes.
Do not use smoke-run outputs as physics results.
