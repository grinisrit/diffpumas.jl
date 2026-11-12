#!/usr/bin/env bash
# Order of execution of the reconstruction protocol (Q2.9). Usage:
#   OUT=/path/to/out NPROC=16 THREADS=2 examples/s7_protocol.sh <step>
# steps: gate | templates | dev | test | diag
#   gate       G1 report (truth only), crossed-cell list, coverage            (one process)
#   templates  MC (16384 histories) + exact templates for fields D and T       (one process each)
#   dev        phases 1-3 of the development stage with selection and edge rules, then selection.toml
#   test       the test stage on field T with the frozen selection.toml          (NPROC processes)
#   diag       oracle-weight diagnostic (GN-0, arm M, tau 1e4 and 1e8)
# Realisations are shared out by `--shard i/n`; every process builds its own operator.
set -euo pipefail
R="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:?set OUT to the output directory}"
NPROC="${NPROC:-8}"; THREADS="${THREADS:-2}"
DATA="${DATA:-/home/roland/devspace/diffpumas.jl}"      # checkout holding the (untracked) physics dump
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS="$THREADS"
COMMON=(--dump "$DATA/examples/data/revision_physics_v2_final_20261023/materials.pumas"
  --rock-density 2710 --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6
  --zenith-max 60 --zenith-step 2 --azimuth-step 4 --primary-altitude-km 30 --seed 42
  --samples 32 --csda-quadrature midpoint --energy-min 0.001 --energy-max 1e9 --threshold 100
  --correction-artifact "$R/docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml"
  --solver-mode stable --inverse-mc-samples "${HISTORIES:-16384}" --output-dir "$OUT"
  --reconstruction-sweep)
jl() { nice -n 19 julia --project="$R" "$R/examples/lvd_tomography.jl" "${COMMON[@]}" "$@"; }
par() {   # par <args...>: run NPROC shards of the same command in parallel, fail if any fails
  local pids=() i rc=0
  for i in $(seq 1 "$NPROC"); do jl "$@" --shard "$i/$NPROC" > "$OUT/log_$(echo "$*" | tr -c 'a-z0-9' _)_$i.txt" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
mkdir -p "$OUT"
case "${1:?step}" in
  gate)      jl --protocol-stage gate --truth-field test ;;
  templates) jl --protocol-stage templates --truth-field dev
             jl --protocol-stage templates --truth-field test ;;
  dev)       for phase in 1 2 3; do
               for round in 1 2 3; do
                 par --protocol-stage dev --truth-field dev --protocol-phase "$phase"
                 rc=0; jl --protocol-stage select --truth-field dev --protocol-phase "$phase" || rc=$?
                 [ "$rc" = 0 ] && break
                 [ "$rc" = 10 ] || exit "$rc"          # 10: grids extended, rerun the phase
               done
               [ "$rc" = 0 ] || { echo "phase $phase still extending after 3 rounds"; exit 1; }
             done
             jl --protocol-stage finalize --truth-field dev ;;
  test)      par --protocol-stage test --truth-field test ;;
  diag)      par --protocol-stage diag --truth-field test ;;
  *) echo "unknown step"; exit 2 ;;
esac
