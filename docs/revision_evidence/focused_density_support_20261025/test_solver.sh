#!/bin/bash
set -eu
export JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/roland/devspace/diffpumas.jl
exec /home/roland/.venv/bin/python3 /home/roland/devspace/diffpumas.jl/dev/run_revision_job.py --output /home/roland/devspace/diffpumas.jl/docs/revision_evidence/focused_density_support_20261025/solver_test --dump /home/roland/devspace/diffpumas.jl/examples/data/revision_physics_v2_final_20261023/materials.pumas --julia /home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia --cpus 8 --cpu-start 0 --memory-gib 16 --input /home/roland/devspace/diffpumas.jl/dev/focused_density_support_solver.jl --input /home/roland/devspace/diffpumas.jl/dev/test_focused_density_support_solver.jl -- /home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia --startup-file=no --project=. /home/roland/devspace/diffpumas.jl/dev/test_focused_density_support_solver.jl
