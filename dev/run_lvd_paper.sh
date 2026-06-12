#!/usr/bin/env bash
# Full LVD paper production run.
#
# Stages (sequential): topography export; main pipeline (12 validation cases,
# inverse demo with floor-aware resolution map, paper match with 2 self-consistent
# recursions and posterior/null-model/inventory statistics); reconstruction sweep;
# detection benchmark; detection contrast sweep.
#
# Per-stage logs land in examples/data/lvd_results/log_<stage>.txt.
# Launch detached:  tmux new-session -d -s lvd_paper \
#   'bash dev/run_lvd_paper.sh 2>&1 | tee examples/data/lvd_results/run_master.log'
set -u
cd "$(dirname "$0")/.."
OUT=examples/data/lvd_results
mkdir -p "$OUT"

run_stage() {
    local name=$1; shift
    echo "=== [$name] start $(date -u +%FT%TZ) ==="
    if julia --project=. -t auto "$@" > "$OUT/log_$name.txt" 2>&1; then
        echo "=== [$name] OK $(date -u +%FT%TZ) ==="
    else
        echo "=== [$name] FAILED exit=$? $(date -u +%FT%TZ) ==="
    fi
}

run_stage topo      examples/export_lvd_topography.jl
run_stage main      examples/lvd_tomography.jl --validation-cases 12 \
                    --papermatch --papermatch-recursions 2
run_stage sweep     examples/lvd_tomography.jl --reconstruction-sweep
run_stage detection examples/lvd_tomography.jl --detection-benchmark
run_stage contrast  examples/lvd_tomography.jl --detection-contrast-sweep
echo "ALL DONE $(date -u +%FT%TZ)"
