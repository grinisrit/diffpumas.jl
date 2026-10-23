# Examples Guide

The `examples/` directory contains both small API demonstrations and larger
research workflows. Run all commands from the repository root with
`julia --project=.`.

## Before Running Examples

Instantiate the environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Create a reusable physics dump:

```bash
julia --project=. examples/loader_example.jl \
  --mdf examples/data/materials.xml \
  --dump examples/data/materials.pumas
```

Most examples default to `examples/data/materials.pumas`; some LVD scripts use
their own LVD-specific default dump and MDF paths.

## Example Summary

`loader_example.jl` loads materials from an MDF file, builds physics tables, and
writes a binary dump.

`diff_flux.jl` computes integrated flux through a two-layer rock/air geometry,
including detailed transport uncertainty and direct-CSDA AD gradients.

`diff_flux_cube.jl` and `no_physics_cube.jl` are cube-oriented development
examples.

`flux_comparison.jl` compares flux parameterizations or transport settings.

`flat_muography.jl` runs flat/cubic muography studies with uncertainty bands and
aquifer scenarios.

`lvd_muography.jl` runs Gran Sasso LVD directional flux, topography, trajectory,
and paper-check workflows.

`lvd_aquifer_validation.jl` compares CSDA+AD sensitivities with full-MC finite
differences on one LVD line of sight.

`lvd_tomography.jl` runs the full LVD sensitivity-volume, Jacobian, validation,
and inversion workflow.

`export_lvd_topography.jl` exports topography artifacts used by figures or
external analysis.

## Loader Example

Show help:

```bash
julia --project=. examples/loader_example.jl --help
```

Build a muon dump:

```bash
julia --project=. examples/loader_example.jl \
  --mdf examples/data/materials.xml \
  --dump examples/data/materials.pumas \
  --particle muon
```

Build a tau dump:

```bash
julia --project=. examples/loader_example.jl \
  --mdf examples/data/materials.xml \
  --dump examples/data/materials_tau.pumas \
  --particle tau
```

Expected output includes a physics summary, material names, and CSDA ranges for
sample energies.

## Integrated Flux Example

Smoke run:

```bash
julia --project=. examples/diff_flux.jl \
  --dump examples/data/materials.pumas \
  --thickness 100 \
  --density 2650 \
  --zenith-max 45 \
  --n-angles 5 \
  --samples 10 \
  --output examples/data/diff_flux_smoke.html
```

Higher-statistics run:

```bash
julia --project=. examples/diff_flux.jl \
  --dump examples/data/materials.pumas \
  --thickness 200 \
  --density 2650 \
  --zenith-min 0 \
  --zenith-max 60 \
  --n-angles 100 \
  --samples 500 \
  --output examples/data/diff_flux_uncertainty.html
```

Disable stochastic loss modes for a deterministic CSDA comparison:

```bash
julia --project=. examples/diff_flux.jl \
  --dump examples/data/materials.pumas \
  --thickness 100 \
  --no-straggling \
  --no-scattering \
  --samples 100
```

The script reports:

- Detailed transport flux with MC and transport-systematic uncertainty.
- Finite-difference density gradient for the detailed path.
- Direct-CSDA flux and Zygote density gradient.
- Direct-CSDA finite-difference check.
- A Plotly uncertainty summary.

## Flat Muography

Run only Part 1 with low statistics:

```bash
julia --project=. examples/flat_muography.jl \
  --output-dir examples/data/flat_part1_smoke \
  --part 1 \
  --samples 20 \
  --no-scattering
```

Run aquifer scenarios only:

```bash
julia --project=. examples/flat_muography.jl \
  --output-dir examples/data/flat_aquifer \
  --part 2 \
  --samples 500 \
  --threshold 100
```

Run all parts:

```bash
julia --project=. examples/flat_muography.jl \
  --output-dir examples/data/flat_full \
  --samples 1000
```

Outputs include flux-vs-angle HTML plots, angular spread plots, fixed-energy
flux scans, and aquifer detection plots.

## LVD Muography

Part 1, directional flux grid:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_part1 \
  --part 1 \
  --samples 100
```

Part 2, topography and trajectories:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_part2 \
  --part 2
```

Part 3, paper reproduction checks:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_part3 \
  --part 3 \
  --samples 200 \
  --paper-samples 2000
```

Full run:

```bash
julia --project=. examples/lvd_muography.jl \
  --output-dir examples/data/lvd_full \
  --samples 1000 \
  --paper-samples 20000
```

The full run can take a long time. Start with a single part when checking a new
environment.

## LVD Aquifer Validation

Default direction:

```bash
julia --project=. examples/lvd_aquifer_validation.jl
```

Specific zenith/azimuth:

```bash
julia --project=. examples/lvd_aquifer_validation.jl 30 0
```

This script is useful after changing AD rules, CSDA stepping, material mixtures,
or MC transport. It checks:

- CSDA AD gradient versus CSDA finite difference.
- CSDA value versus full MC value.
- CSDA finite-difference sensitivity versus full-MC finite-difference
  sensitivity.

## LVD Tomography

Grid smoke run:

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

Tetrahedral sensitivity-volume run:

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


Use a simple single-box aquifer truth:

```bash
julia --project=. examples/lvd_tomography.jl \
  --output-dir examples/data/lvd_tomo_simple \
  --geometry grid \
  --simple-field \
  --aquifer-water-fraction 0.7 \
  --aquifer-center-z 700 \
  --samples 32 --csda-quadrature midpoint
```

Paper-match focused run:

```bash
julia --project=. examples/lvd_tomography.jl \
  --output-dir examples/data/lvd_papermatch \
  --papermatch-only \
  --samples 32 --csda-quadrature midpoint \
  --papermatch-mc-samples 64
```


CSDA uses deterministic log-energy midpoint quadrature by default (`--samples 32`).
Check flux and derivative convergence with more nodes before interpreting numerical
results. `--csda-quadrature random` reproduces the legacy seeded sampling path;
small random sample sets can have large flux bias. Monte Carlo transport retains
independent random energy and charge samples.

Useful controls:

- `--geometry auto|tetra|grid`: choose sensitivity volume type.
- `--zenith-max`, `--zenith-step`, `--azimuth-step`: angular grid size.
- `--samples`: CSDA energy samples per angular bin.
- `--mc-samples`: MC samples per validation case.
- `--validation-cases`: number of MC validation cases.
- `--reco-iters`, `--gd-iters`: inverse solver iterations.
- `--no-straggling`: disable stochastic energy-loss fluctuations in MC
  validation.

## Export LVD Topography

Use the export helper when a downstream plotting or manuscript workflow needs
topography data:

```bash
julia --project=. examples/export_lvd_topography.jl
```

Check the script help or source before relying on output paths, because this is
mainly a workflow utility.

## Choosing Sample Counts

For smoke tests:

- Use `--samples 2` to `--samples 20`.
- Use coarse angular steps, such as `--azimuth-step 30`.
- Run one part with `--part`.

For exploratory plots:

- Use `--samples 100` to `--samples 1000`.
- Keep angular grids moderate.

For paper-quality or validation results:

- Increase samples substantially.
- Keep seeds fixed for comparisons.
- Review MC uncertainty and transport-systematic bands.
- Prefer dedicated output directories so generated files are not overwritten.

## Sequential LVD Production Run

`dev/run_lvd_paper.sh` runs topography, the main pipeline, reconstruction sweep,
detection benchmark, and contrast sweep sequentially. It stops on the first
failed stage and returns its exit status. Defaults are 16 CPU cores, 36 GiB of
address space per process, one BLAS thread, and `examples/data/lvd_revision`.
The default physics dump is `examples/data/revision_physics/materials.pumas`;
tomography stages require it to exist. Pass `--dump PATH` to select another
verified physics source. See [setup](getting-started.md) for data acquisition.
The launcher uses the lower host/cgroup CPU and RAM budgets described in
`dev/run_bounded.py`; the address-space limit is not an aggregate job memory cap.
Together with the 4-CPU/12-GiB robustness launcher, the defaults total 20 CPUs
and 48 GiB; reserve 1 CPU/2 GiB for controls and pause other computations during
production. Julia startup files are disabled. Manifests record the selected
Julia binary/version/hash and transitive LVD helper source hashes.

Inspect the commands before starting a long run:

```bash
bash dev/run_lvd_paper.sh --dry-run
bash dev/run_lvd_paper.sh --threads 16 --memory-gib 36 \
  --dump examples/data/revision_physics/materials.pumas \
  --output-dir examples/data/lvd_revision --stages topo,main
```

The main stage allows up to six self-consistent calibration passes, stopping
when both field and coefficient changes meet relative tolerance 0.02. Change
these with `--recursions N --recur-tol X`; reaching the maximum is recorded as
termination by maximum passes and does not establish convergence.

Use `--stages sweep,detection,contrast` for the remaining stages. Every selected
stage runs in the supplied order and writes `log_<stage>.txt` under the chosen
output directory. Run the script inside tmux for unattended production work.

## Output Hygiene

Examples write HTML, CSV, text logs, and sometimes serialized intermediate data.
Use a unique output directory for each run:

```bash
OUT="examples/data/run_$(date +%Y%m%d_%H%M%S)"
julia --project=. examples/flat_muography.jl --output-dir "$OUT" --samples 100
```

Generated result directories under `examples/data/` can become large. Keep only
the outputs needed for the current analysis or paper reproduction.
