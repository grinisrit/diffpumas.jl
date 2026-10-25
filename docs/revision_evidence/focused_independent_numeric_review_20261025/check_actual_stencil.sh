#!/bin/bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/roland/devspace/diffpumas.jl
/home/roland/.venv/bin/python3 /home/roland/devspace/diffpumas.jl/dev/run_bounded.py --cpus 1 --memory-gib 2 -- taskset -c 19 /home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia --startup-file=no /home/roland/devspace/diffpumas.jl/docs/revision_evidence/focused_independent_numeric_review_20261025/check_actual_stencil.jl > /home/roland/devspace/diffpumas.jl/docs/revision_evidence/focused_independent_numeric_review_20261025/actual_stencil_unit.log 2>&1
