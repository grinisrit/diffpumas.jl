#!/usr/bin/env bash
# usage: run_resolution_only.sh OUTDIR ZSTEP ASTEP   (2700-direction grid: 2 4; withdrawn run: 1 1)
R=${R:-$(cd "$(dirname "$0")/../../.." && pwd)}
D=${D:-/home/roland/devspace/diffpumas.jl}   # checkout holding the (untracked) physics dump
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd $R && /usr/bin/time -v nice -n 19 julia --project=. -t 1 examples/lvd_tomography.jl --resolution-only \
 --dump $D/examples/data/revision_physics_v2_final_20261023/materials.pumas \
 --rock-density 2710 --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6 \
 --zenith-max 60 --zenith-step $2 --azimuth-step $3 --primary-altitude-km 30 --seed 42 \
 --samples 32 --csda-quadrature midpoint --energy-min 0.001 --energy-max 1e9 --threshold 100 \
 --correction-artifact $R/docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml \
 --solver-mode stable --output-dir $1
