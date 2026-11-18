# Resolution map (Fig. S6) on the 2700-direction grid

Linearised point-spread map of the fixed dry-reference operator (Jacobian at w = 0, SART 120 iterations, relaxation 0.2, impulse 0.5, all 500 cells at or below 350 m), regenerated with only the ray set changed: zenith 0..58 step 2, azimuth 0..356 step 4 (2700 rays, 600-cell mesh, detector at the origin; the grid of the S7 protocol). The earlier map came from the 21600-ray run `material_main_recovery_20261024`.
- Generator: `examples/lvd_tomography.jl --resolution-only` (`run_resolution_only.sh OUTDIR 2 4`); with `1 1` it reproduces the resolution sections of the earlier `lvd_tomography_inverse.txt` byte for byte. Cost: 177 s CPU (one thread).
- `lvd_tomography_inverse.txt`: output; `fig_resolution_map.{pdf,png,json}`: figure and sidecar (fields as in the paper repo, with `source_sha256`); `summary.json`: sector counts; `make_psf_json.py`: sidecar builder (reuses `make_figures.py` of the paper repo).
- The figure is drawn by `make_figures.py --only fig_resolution_map` from the text file.
