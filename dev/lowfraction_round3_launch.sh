#!/usr/bin/env bash
set -euo pipefail
cd /home/roland/devspace/diffpumas.jl
export LOWFRACTION_CONVERGENCE_GATE=independently-validated-root-go
export JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python3 dev/run_revision_job.py \
  --output examples/data/lowfraction_round3_20261025 \
  --dump examples/data/revision_physics_v2_final_20261023/materials.pumas \
  --julia /home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia \
  --cpus 8 --cpu-start 8 --memory-gib 16 \
  --input dev/lowfraction_round3.jl \
  --input dev/lowfraction_round3_launch.sh \
  --input dev/validate_surrogate_v2.jl \
  --input dev/lowfraction_round3_summary.py \
  --input docs/revision_evidence/lowfraction_round3_20261025/plan.toml \
  --input docs/revision_evidence/surrogate_v2/production/material_calibration.toml \
  --input docs/revision_evidence/corrected_validation/calibration.toml \
  -- timeout --kill-after=30 5400 \
  /home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. dev/lowfraction_round3.jl pilot \
  examples/data/lowfraction_round3_20261025/results \
  examples/data/revision_physics_v2_final_20261023/materials.pumas
