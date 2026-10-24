#!/usr/bin/env bash
# Sequential, bounded LVD production run. Every failed stage stops the run.
# Run inside tmux; use --dry-run to inspect exact commands before a long run.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=examples/data/lvd_revision
DUMP=examples/data/revision_physics/materials.pumas
THREADS=16
MEMORY_GIB=36
RECURSIONS=6
RECUR_TOL=0.02
SAMPLES=32
CALIBRATION_MC_SAMPLES=1024
INVERSE_MC_SAMPLES=256
MC_SAMPLES=1024
PAPERMATCH_MC_SAMPLES=512
STAGES=topo,main,sweep,detection,contrast
DRY_RUN=0
EXTRA_OPTIONS=()
JULIA_BIN=${JULIA_BIN:-julia}
ORIGINAL_ARGS=("$@")
usage() {
    cat <<'HELP'
Usage: bash dev/run_lvd_paper.sh [options]
  --output-dir PATH   Results and stage logs (default examples/data/lvd_revision)
  --dump PATH         Physics dump (default examples/data/revision_physics/materials.pumas)
  --threads N         CPU affinity and Julia threads (default 16)
  --memory-gib N      Per-process address-space limit (default 36)
  --recursions N      Maximum paper calibration passes (default 6)
  --recur-tol X       Relative field/coefficient tolerance (default 0.02)
  --samples N         CSDA midpoint energy nodes (default 32)
  --calibration-mc-samples N  Histories/bin for calibration (default 1024)
  --inverse-mc-samples N      Histories/bin for synthetic inverse data (default 256)
  --mc-samples N       Histories/validation case (default 1024)
  --papermatch-mc-samples N   Histories/bin for paper uncertainty grid (default 512)
  --stages LIST       Comma-separated sequential stages (default all five):
                      topo,main,sweep,detection,contrast
  --dry-run           Print commands without creating outputs or running Julia
  --correction-artifact PATH  Frozen dry/material correction artifact
  --solver-mode MODE          Synthetic GN: legacy or stable
  --papermatch-fit-mode MODE  Measured fit: legacy or stable
  --papermatch-initial-water X  Initial supported-cell fraction
  --papermatch-gn-iters N       Stable measured iteration ceiling
  --papermatch-gradient-tol X   Stable projected-gradient tolerance
  --papermatch-reg X            Relative graph regularization weight
  --papermatch-absolute-prior-weight X  Fixed absolute graph regularization weight
  --papermatch-log-scale X      Fixed pseudo-Huber log-residual scale
  --help              Show this help
Set JULIA_BIN to select a Julia executable. BLAS uses one thread. The bounded
launcher validates against the lower host/cgroup limits. Main defaults plus the
4-CPU/12-GiB robustness launcher total 20 CPUs/48 GiB; reserve 1 CPU/2 GiB for
controls and pause other computations during production. Stages stop at first error;
rerun selected stages explicitly after diagnosing the corresponding log.
HELP
}
while (($#)); do
    case "$1" in
        --correction-artifact|--solver-mode|--papermatch-fit-mode|--papermatch-initial-water|--papermatch-gn-iters|--papermatch-gradient-tol|--papermatch-log-scale|--papermatch-reg|--papermatch-absolute-prior-weight)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            EXTRA_OPTIONS+=("$1" "$2")
            shift 2 ;;
        --output-dir|--dump|--threads|--memory-gib|--recursions|--recur-tol|--stages|--samples|--calibration-mc-samples|--inverse-mc-samples|--mc-samples|--papermatch-mc-samples)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --output-dir) OUT=$2 ;;
                --dump) DUMP=$2 ;;
                --threads) THREADS=$2 ;;
                --memory-gib) MEMORY_GIB=$2 ;;
                --recursions) RECURSIONS=$2 ;;
                --recur-tol) RECUR_TOL=$2 ;;
                --stages) STAGES=$2 ;;
                --samples) SAMPLES=$2 ;;
                --calibration-mc-samples) CALIBRATION_MC_SAMPLES=$2 ;;
                --inverse-mc-samples) INVERSE_MC_SAMPLES=$2 ;;
                --mc-samples) MC_SAMPLES=$2 ;;
                --papermatch-mc-samples) PAPERMATCH_MC_SAMPLES=$2 ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'Threads must be a positive integer' >&2; exit 2; }
[[ "$RECURSIONS" =~ ^[1-9][0-9]*$ ]] || { echo 'Recursions must be a positive integer' >&2; exit 2; }
for count in "$SAMPLES" "$CALIBRATION_MC_SAMPLES" "$INVERSE_MC_SAMPLES" "$MC_SAMPLES" "$PAPERMATCH_MC_SAMPLES"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Sample counts must be positive integers" >&2; exit 2; }
done
[[ -n "$OUT" && -n "$STAGES" && "$STAGES" != ,* && "$STAGES" != *, && "$STAGES" != *,,* ]] || {
    echo 'Output directory and stages must be nonempty' >&2; exit 2;
}
IFS=, read -r -a selected_stages <<< "$STAGES"
for stage in "${selected_stages[@]}"; do
    case "$stage" in
        topo) ;;
        main|sweep|detection|contrast)
            if ((!DRY_RUN)) && [[ ! -f "$DUMP" ]]; then
                echo "Physics dump not found: $DUMP; generate the documented pinned-data dump or pass --dump PATH" >&2
                exit 2
            fi ;;
        *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
done
((DRY_RUN)) || mkdir -p "$OUT"
if ((!DRY_RUN)); then
    python3 dev/record_revision_run.py --julia "$JULIA_BIN" "$OUT" "$DUMP" bash dev/run_lvd_paper.sh "${ORIGINAL_ARGS[@]}"
fi

run_stage() {
    local name=$1; shift
    local -a command=(python3 dev/run_bounded.py --cpus "$THREADS" --memory-gib "$MEMORY_GIB"
                      -- "$JULIA_BIN" --startup-file=no --project=. "$@")
    if ((DRY_RUN)); then
        printf '[%s] ' "$name"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1\n' "$OUT/log_$name.txt"
        return
    fi
    echo "=== [$name] start $(date -u +%FT%TZ) ==="
    if "${command[@]}" > "$OUT/log_$name.txt" 2>&1; then
        echo "=== [$name] OK $(date -u +%FT%TZ) ==="
    else
        local status=$?
        echo "=== [$name] FAILED exit=$status $(date -u +%FT%TZ); see $OUT/log_$name.txt ===" >&2
        return "$status"
    fi
}

# rr.for Gran Sasso convention, consistent with NMAP_ROCK_DENSITY.
RHO=2710
# Structured grid avoids inverted cells where terrain dips below the detector.
COMMON=(examples/lvd_tomography.jl --dump "$DUMP" --rock-density "$RHO" --geometry grid --samples "$SAMPLES" --calibration-mc-samples "$CALIBRATION_MC_SAMPLES" --inverse-mc-samples "$INVERSE_MC_SAMPLES" --mc-samples "$MC_SAMPLES" --papermatch-mc-samples "$PAPERMATCH_MC_SAMPLES" --csda-quadrature midpoint --output-dir "$OUT")
COMMON+=("${EXTRA_OPTIONS[@]}")
# Statistical comparisons use 2 x 4 degree bins, rather than the main 1 x 1 grid.
COARSE=(--zenith-step 2 --azimuth-step 4)
for stage in "${selected_stages[@]}"; do
    case "$stage" in
        topo) run_stage topo examples/export_lvd_topography.jl --output-dir "$OUT" ;;
        main) run_stage main "${COMMON[@]}" --validation-cases 12 --papermatch --papermatch-recursions "$RECURSIONS" --papermatch-recur-tol "$RECUR_TOL" ;;
        sweep) run_stage sweep "${COMMON[@]}" "${COARSE[@]}" --reconstruction-sweep --sweep-realizations 40 --gd-iters 30 ;;
        detection) run_stage detection "${COMMON[@]}" "${COARSE[@]}" --detection-benchmark ;;
        contrast) run_stage contrast "${COMMON[@]}" "${COARSE[@]}" --detection-contrast-sweep ;;
    esac
done
if ((DRY_RUN)); then
    echo "DRY RUN COMPLETE"
else
    echo "ALL SELECTED STAGES DONE $(date -u +%FT%TZ)"
fi
