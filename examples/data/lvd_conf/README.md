# LVD configuration and measured data

Static inputs for `lvd_muography.jl`, `lvd_tomography.jl`, and future event
reconstruction. Examples use `LVD_DATA_DIR = examples/data/lvd_conf`.

| Asset | Role |
|-------|------|
| `nm_c.inc`, `rr.for` | Slant rock thickness grid (nmap / `rr.for`) |
| `gran-sasso-lvd-raytracing-geometry.yaml` | Cross-section topography anchors |
| `rock_int.txt`, `lvd_single_muon_flux_2d.csv` | Measured 2D single-muon map |
| `lvd_paper_fig7_*.csv`, `lvd_paper_fig8_curve.csv` | Digitized paper benchmarks |
| `lvd_detector_geometry.json`, `lvd_*_counters.csv` | Detector layout (Fig. 1.6) |
| `lvd_muon_reconstruction_spec.md` | Reconstruction algorithm spec |

Run outputs go to `examples/data/lvd_results/` (gitignored).
