#!/usr/bin/env bash
# Order of execution of the reconstruction protocol (Q2.9, amended by the G0 ruling). Usage:
#   OUT=/path/to/out [NPROC=34] examples/s7_protocol.sh <step>
# steps: gate | g0c | dry | templates | dev | test | diag | summary | run
#   gate       G1 report (truth only), crossed-cell list, coverage                    (one process)
#   g0c        gate G0c: same dry rays through the sweep and the validation drivers   (two processes in sequence)
#   dry        gate G0b (replaced): dry calibration, 2 x 8192 histories (seeds 441, 442), three checks, dry_calibration.toml
#   templates  MC (16384 histories, per-bin streams, one template per realisation: D 421-422, T 431-438; claimed in
#              chunks of 25 bins) + exact templates of fields D and T, then merge
#   dev        phases 1-3 of the development stage with selection and edge rules, then selection.toml
#   test       the 36 h trigger (plan step -> cuts.toml), then the test stage on field T with the frozen selection.toml
#   diag       oracle-weight diagnostic (GN-0, arm Mc, tau 1e4 and 1e8)
#   summary    test-stage summary (cell statistics, contrasts C1, C2, C3)
#   run        templates, dev, test, diag, summary in order, status file RUN_STATUS, marker RUN_DONE / RUN_FAILED
# Workers are single-threaded processes (Julia and BLAS threads = 1). Units (arm, exposure, realisation; or
# template chunks) are claimed by an atomic mkdir, so the number of workers can change at any time:
#   $OUT/WORKERS_WANTED  target number of workers (the monitor, examples/s7_monitor.sh, raises it up to NPROC)
#   $OUT/PAUSE           while this file exists no worker claims a new unit and no worker is started
# Every solve is fixed by its seed, so the result does not depend on the number of workers or on scheduling.
# Arms: X (exact-model control), Mc (MC data, dry-normalised operator; needs dry_calibration.toml)
# and M (MC data, operator as published; reduced grid, no GD). One log per stage: $OUT/log_<stage>.txt.
set -uo pipefail
MAX_PROCESSES=64
R="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:?set OUT to the output directory}"
NPROC="${NPROC:-34}"
[[ "$NPROC" =~ ^[0-9]+$ ]] && (( NPROC >= 1 )) || { echo "NPROC must be a positive integer" >&2; exit 2; }
(( NPROC <= MAX_PROCESSES )) || { echo "NPROC=$NPROC exceeds the limit of $MAX_PROCESSES single-thread worker processes" >&2; exit 2; }
DATA="${DATA:-/home/roland/devspace/diffpumas.jl}"      # checkout holding the (untracked) physics dump
DUMP="$DATA/examples/data/revision_physics_v2_final_20261023/materials.pumas"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 S7_NPROC="$NPROC"
COMMON=(--dump "$DUMP"
  --rock-density 2710 --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6
  --zenith-max 60 --zenith-step 2 --azimuth-step 4 --primary-altitude-km 30 --seed 42
  --samples 32 --csda-quadrature midpoint --energy-min 0.001 --energy-max 1e9 --threshold 100
  --correction-artifact "$R/docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml"
  --solver-mode stable --inverse-mc-samples "${HISTORIES:-16384}" --output-dir "$OUT"
  --reconstruction-sweep --sweep-data all)
WANTED_FILE="$OUT/WORKERS_WANTED"; PAUSE_FILE="$OUT/PAUSE"
mkdir -p "$OUT" "$OUT/claims"
TEMPLATE_MODE_FILE="$OUT/TEMPLATE_MODE"       # "replicates" (one MC template per realisation) or "single" (6 h fallback)
jl() { nice -n 19 julia --project="$R" "$R/examples/lvd_tomography.jl" "${COMMON[@]}" --template-mode "$(cat "$TEMPLATE_MODE_FILE" 2>/dev/null || echo replicates)" "$@"; }
decide_template_mode() {   # fallback rule: one template per field if the projected template stage exceeds 6 h wall
  local th mode proj
  th=$(awk '/thread_hours_per_16384_template/{print $3}' "$OUT/dry/dry_report.toml" 2>/dev/null)
  if [ -z "$th" ]; then mode=replicates; proj="unknown"; else
    proj=$(awk -v th="$th" -v n="$NPROC" 'BEGIN{printf "%.2f", 10*th/n}')           # 10 templates (D: 2, T: 8) of 16384 histories
    mode=$(awk -v p="$proj" 'BEGIN{print (p > 6 ? "single" : "replicates")}')
  fi
  echo "$mode" > "$TEMPLATE_MODE_FILE"
  LOGSTAGE=templates log "template mode = $mode (projected template stage $proj h at $NPROC workers from $th thread-hours per template; limit 6 h)"
}

STAGE="-"; STAGE_DONE=0; STAGE_TOTAL=0; STAGE_T0=$(date +%s); WORKERS_ALIVE=0
log() { echo "[orch $(date -u +%FT%TZ)] $*" >> "$OUT/log_${LOGSTAGE:-run}.txt"; }
status_line() {   # one line: stage, units done/total, start time, ETA, workers, run CPU share and RSS from the monitor
  local now start eta res cpu rss
  now=$(date +%s); start=$(cat "$OUT/RUN_START" 2>/dev/null || echo "$now")
  eta="n/a"
  if (( STAGE_DONE > 0 && STAGE_TOTAL > STAGE_DONE )); then
    eta=$(date -u -d "@$(( now + (now - STAGE_T0) * (STAGE_TOTAL - STAGE_DONE) / STAGE_DONE ))" +%FT%TZ)   # stage ETA
  fi
  res=$(tail -n 1 "$OUT/resources.log" 2>/dev/null || true); cpu=$(awk '{print $4}' <<< "$res"); rss=$(awk '{print $5}' <<< "$res")
  echo "stage=$STAGE units=$STAGE_DONE/$STAGE_TOTAL start=$(date -u -d "@$start" +%FT%TZ) stage_eta=$eta workers=$WORKERS_ALIVE run_cpu=${cpu:-n/a} run_rss_gb=${rss:-n/a} updated=$(date -u +%FT%TZ)" > "$OUT/RUN_STATUS.tmp" &&
    mv "$OUT/RUN_STATUS.tmp" "$OUT/RUN_STATUS"
}
fail() {          # fail <reason>: stop the workers this run started, leave the marker
  log "FAILED: $*"
  echo "stage=$STAGE reason=$*" > "$OUT/RUN_FAILED"; status_line
  # the run is started with setsid, so its process group is exactly this orchestrator and its workers
  if [ "$(ps -o pgid= -p $$ | tr -d ' ')" = "$$" ]; then trap '' TERM; kill -TERM -- "-$$" 2>/dev/null; fi
  exit 1
}

# pool <stage-name> <jl args...>: run workers that claim the units of one stage until all are done.
pool() {
  local name="$1"; shift
  local claimdir="$OUT/claims/$name" pass pids p alive rc total claimed done_n want wid=0 d n
  LOGSTAGE="${name%%_*}"; STAGE="$name"
  for pass in 1 2 3; do
    mkdir -p "$claimdir"
    for d in "$claimdir"/*/; do [ -d "$d" ] || continue; n=$(basename "$d"); [ -e "$claimdir/$n.done" ] || rmdir "$d" 2>/dev/null; done   # claims of dead workers
    pids=(); STAGE_T0=$(date +%s)
    while :; do
      alive=()
      for p in "${pids[@]}"; do
        if kill -0 "$p" 2>/dev/null; then alive+=("$p"); else wait "$p"; rc=$?; (( rc == 0 )) || fail "worker $p of stage $name exited with $rc (see $OUT/log_${LOGSTAGE}.txt)"; fi
      done
      pids=("${alive[@]}"); WORKERS_ALIVE=${#pids[@]}
      total=$(cat "$claimdir/TOTAL" 2>/dev/null || echo 0)
      claimed=$(find "$claimdir" -mindepth 1 -maxdepth 1 -type d | wc -l)
      done_n=$(find "$claimdir" -mindepth 1 -maxdepth 1 -name '*.done' | wc -l)
      STAGE_DONE=$done_n; STAGE_TOTAL=$total
      want=$(cat "$WANTED_FILE" 2>/dev/null || echo 4); (( want > NPROC )) && want=$NPROC
      if [ ! -e "$PAUSE_FILE" ]; then
        if (( total == 0 )); then
          (( ${#pids[@]} == 0 )) && { wid=$((wid+1)); launch_worker "$name" "$wid" "$claimdir" "$@"; pids+=($!); }
        elif (( claimed < total && ${#pids[@]} < want )); then
          wid=$((wid+1)); launch_worker "$name" "$wid" "$claimdir" "$@"; pids+=($!)       # one per loop: staggers the operator builds
        fi
      fi
      status_line
      (( total > 0 && done_n >= total && ${#pids[@]} == 0 )) && break
      (( total > 0 && ${#pids[@]} == 0 && claimed >= total )) && break                       # all claimed, workers gone: next pass cleans stale claims
      (( total == 0 && ${#pids[@]} == 0 )) && fail "first worker of stage $name left no unit list"
      sleep 10
    done
    (( total > 0 && done_n >= total )) && { log "stage $name complete: $done_n/$total units"; return 0; }
    log "stage $name pass $pass ended with $done_n/$total units done; retrying unfinished units"
  done
  fail "stage $name did not finish all units"
}
launch_worker() {   # launch_worker <name> <id> <claimdir> <jl args...> (the pid is $!)
  local name="$1" wid="$2" claimdir="$3"; shift 3
  jl "$@" --claim-dir "$claimdir" --pause-file "$PAUSE_FILE" --worker "$wid" >> "$OUT/log_${LOGSTAGE}.txt" 2>&1 &
}

run_step() {
  case "$1" in
    templates) pool templates --protocol-stage templates
               LOGSTAGE=templates; jl --protocol-stage merge >> "$OUT/log_templates.txt" 2>&1 || fail "template merge failed" ;;
    dev)       local phase round rc
               for phase in 1 2 3; do
                 for round in 1 2 3 4; do
                   pool "dev_p${phase}_r${round}" --protocol-stage dev --truth-field dev --protocol-phase "$phase"
                   LOGSTAGE=dev; rc=0; jl --protocol-stage select --truth-field dev --protocol-phase "$phase" >> "$OUT/log_dev.txt" 2>&1 || rc=$?
                   [ "$rc" = 0 ] && break
                   [ "$rc" = 10 ] || fail "selection of phase $phase failed ($rc)"          # 10: grids extended (edge rules, GN cap), rerun the phase
                 done
                 [ "$rc" = 0 ] || fail "phase $phase still extending after 4 rounds"
               done
               LOGSTAGE=dev; jl --protocol-stage finalize --truth-field dev >> "$OUT/log_dev.txt" 2>&1 || fail "finalize failed" ;;
    test)      local el; LOGSTAGE=test
               el=$(awk -v n="$(date +%s)" -v s="$(cat "$OUT/RUN_START" 2>/dev/null || date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600}')
               jl --protocol-stage plan --truth-field dev --elapsed-hours "$el" --ref-processes "$NPROC" >> "$OUT/log_test.txt" 2>&1 || fail "36 h plan failed"
               pool test --protocol-stage test --truth-field test ;;
    diag)      pool diag --protocol-stage diag --truth-field test ;;
    summary)   LOGSTAGE=summary; jl --protocol-stage summary --truth-field test >> "$OUT/log_summary.txt" 2>&1 || fail "summary failed" ;;
  esac
}

step="${1:?step}"
case "$step" in
  gate)  jl --protocol-stage gate --truth-field test ;;
  g0c)   jl --protocol-stage g0c
         nice -n 19 julia --project="$R" "$R/dev/g0c_validation_path.jl" "$OUT/g0c" "$DUMP" ;;
  dry)   jl --protocol-stage dry ;;
  templates|dev|test|diag|summary) [ -e "$WANTED_FILE" ] || echo "$NPROC" > "$WANTED_FILE"; run_step "$step" ;;
  run)   [ -e "$OUT/RUN_START" ] || date +%s > "$OUT/RUN_START"
         [ -e "$WANTED_FILE" ] || echo $(( NPROC < 8 ? NPROC : 8 )) > "$WANTED_FILE"
         rm -f "$OUT/RUN_FAILED" "$OUT/RUN_DONE"
         [ -s "$OUT/dry_calibration.toml" ] || fail "no dry_calibration.toml: gate G0b has not passed"
         [ -e "$TEMPLATE_MODE_FILE" ] || decide_template_mode
         for s in templates dev test diag summary; do run_step "$s"; done
         STAGE=done; status_line; echo "all stages complete $(date -u +%FT%TZ)" > "$OUT/RUN_DONE"; log "RUN_DONE" ;;
  *) echo "unknown step" >&2; exit 2 ;;
esac
