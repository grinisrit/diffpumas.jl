#!/usr/bin/env bash
# Full LVD paper production run.
#
# Stages (sequential): topography export; main pipeline (12 validation cases,
# inverse demo with floor-aware resolution map, paper match with 2 self-consistent
# recursions and posterior/null-model/inventory statistics); reconstruction sweep;
# detection benchmark; detection contrast sweep.
#
# Overburden rock density is pinned at 2710 kg/m^3 (rr.for Gran Sasso convention,
# consistent with the 1-D muography NMAP_ROCK_DENSITY) for every tomography stage.
#
# The three statistical stages (sweep / detection / contrast) are run on a coarse
# 2 deg x 4 deg angular grid (2,700 bins) instead of the 1 deg x 1 deg (21,600-bin)
# grid used for `main`: their figures are relative comparisons (solver-vs-solver
# RMSE/SSIM, Psi-vs-SART detection gain) that do not need full angular resolution,
# and the full grid relinearises the AD Jacobian thousands of times (~45 h). The
# coarse grid brings the whole run to ~6-9 h while preserving the relative orderings.
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

RHO=2710                       # overburden rock density (kg/m^3), rr.for Gran Sasso
GEO="--geometry grid"          # structured voxel volume (z from detector plane up): the
                               # overburden tapers to zero at the mountain edge, which a
                               # tetrahedral mesh cannot cleanly fill (inverted/degenerate
                               # cells where the inverted nm_c.inc map dips below the
                               # detector). The voxel grid starts at z=0 and masks sub-surface
                               # cells as air, so it carries no below-detector cells.
COARSE="--zenith-step 2 --azimuth-step 4"   # statistical stages: 30 x 90 = 2,700 bins

run_stage topo      examples/export_lvd_topography.jl
run_stage main      examples/lvd_tomography.jl --rock-density $RHO $GEO --validation-cases 12 \
                    --papermatch --papermatch-recursions 2
run_stage sweep     examples/lvd_tomography.jl --rock-density $RHO $GEO $COARSE \
                    --reconstruction-sweep --sweep-realizations 40 --gd-iters 30
run_stage detection examples/lvd_tomography.jl --rock-density $RHO $GEO $COARSE \
                    --detection-benchmark
run_stage contrast  examples/lvd_tomography.jl --rock-density $RHO $GEO $COARSE \
                    --detection-contrast-sweep
echo "ALL DONE $(date -u +%FT%TZ)"
