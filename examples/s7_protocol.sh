#!/usr/bin/env bash
# Order of execution of the reconstruction protocol (Q2.9, amended by the G0 ruling). Usage:
#   OUT=/path/to/out NPROC=16 examples/s7_protocol.sh <step>
# steps: gate | g0c | dry | templates | dev | test | diag | summary
#   gate       G1 report (truth only), crossed-cell list, coverage                   (one process)
#   g0c        gate G0c: same dry rays through the sweep and the validation drivers    (two processes in sequence)
#   dry        gate G0b: dry template (16384 histories, seed 44), s, dry_calibration.toml (one process)
#   templates  MC (16384 histories) + exact templates for fields D and T               (one process each, in parallel)
#   dev        phases 1-3 of the development stage with selection and edge rules, then selection.toml
#   test       the test stage on field T with the frozen selection.toml               (NPROC processes)
#   diag       oracle-weight diagnostic (GN-0, arm Mc, tau 1e4 and 1e8)
#   summary    test-stage summary (cell statistics, contrasts C1, C2, C3)
# Process budget: at most 16 worker processes, each single-threaded (Julia and BLAS threads = 1);
# keep at least 4 cores free. Realisations are shared out by `--shard i/n`; every process builds
# its own operator, and every solve is fixed by its seed, so the result does not depend on NPROC.
# Arms: X (exact-model control), Mc (MC data, dry-normalised operator; needs dry_calibration.toml)
# and M (MC data, operator as published; reduced grid, no GD).
set -euo pipefail
MAX_PROCESSES=16
R="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:?set OUT to the output directory}"
NPROC="${NPROC:-$MAX_PROCESSES}"
[[ "$NPROC" =~ ^[0-9]+$ ]] && (( NPROC >= 1 )) || { echo "NPROC must be a positive integer" >&2; exit 2; }
(( NPROC <= MAX_PROCESSES )) || { echo "NPROC=$NPROC exceeds the limit of $MAX_PROCESSES single-thread worker processes" >&2; exit 2; }
DATA="${DATA:-/home/roland/devspace/diffpumas.jl}"      # checkout holding the (untracked) physics dump
DUMP="$DATA/examples/data/revision_physics_v2_final_20261023/materials.pumas"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
COMMON=(--dump "$DUMP"
  --rock-density 2710 --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6
  --zenith-max 60 --zenith-step 2 --azimuth-step 4 --primary-altitude-km 30 --seed 42
  --samples 32 --csda-quadrature midpoint --energy-min 0.001 --energy-max 1e9 --threshold 100
  --correction-artifact "$R/docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml"
  --solver-mode stable --inverse-mc-samples "${HISTORIES:-16384}" --output-dir "$OUT"
  --reconstruction-sweep --sweep-data all)
jl() { nice -n 19 julia --project="$R" "$R/examples/lvd_tomography.jl" "${COMMON[@]}" "$@"; }
par() {   # par <args...>: run NPROC shards of the same command in parallel, fail if any fails
  local pids=() i rc=0
  for i in $(seq 1 "$NPROC"); do jl "$@" --shard "$i/$NPROC" > "$OUT/log_$(echo "$*" | tr -c 'a-z0-9' _)_$i.txt" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
mkdir -p "$OUT"
step="${1:?step}"
case "$step" in
  gate)      jl --protocol-stage gate --truth-field test ;;
  g0c)       jl --protocol-stage g0c
             nice -n 19 julia --project="$R" "$R/dev/g0c_validation_path.jl" "$OUT/g0c" "$DUMP" ;;
  dry)       jl --protocol-stage dry ;;
  templates) pids=(); rc=0
             for spec in "dev mc" "dev csda" "test mc" "test csda"; do
               read -r fld arm <<< "$spec"
               jl --protocol-stage templates --truth-field "$fld" --sweep-data "$arm" > "$OUT/log_templates_${fld}_${arm}.txt" 2>&1 & pids+=($!)
             done
             for p in "${pids[@]}"; do wait "$p" || rc=1; done
             exit "$rc" ;;
  dev)       for phase in 1 2 3; do
               for round in 1 2 3 4; do
                 par --protocol-stage dev --truth-field dev --protocol-phase "$phase"
                 rc=0; jl --protocol-stage select --truth-field dev --protocol-phase "$phase" || rc=$?
                 [ "$rc" = 0 ] && break
                 [ "$rc" = 10 ] || exit "$rc"          # 10: grids extended (edge rules, GN cap), rerun the phase
               done
               [ "$rc" = 0 ] || { echo "phase $phase still extending after 4 rounds"; exit 1; }
             done
             jl --protocol-stage finalize --truth-field dev ;;
  test)      par --protocol-stage test --truth-field test ;;
  diag)      par --protocol-stage diag --truth-field test ;;
  summary)   jl --protocol-stage summary --truth-field test ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
