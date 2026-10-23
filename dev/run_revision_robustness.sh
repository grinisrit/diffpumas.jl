#!/usr/bin/env bash
# Sequential nonlinear refits; each Julia child inherits explicit resource bounds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision_root="$(realpath -m "${1:-examples/data/revision_robustness_20261023}")"
mkdir -p "$revision_root"
revision_dump="${REVISION_PHYSICS_DUMP:-examples/data/revision_physics/materials.pumas}"
revision_julia="${JULIA_BIN:-julia}"
[[ -f "$revision_dump" ]] || { printf 'Missing pinned physics dump: %s\n' "$revision_dump" >&2; exit 1; }
revision_nodes="${REVISION_CSDA_SAMPLES:-32}"
revision_calibration_mc="${REVISION_CALIBRATION_MC_SAMPLES:-1024}"
for count in "$revision_nodes" "$revision_calibration_mc"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Sample counts must be positive integers" >&2; exit 2; }
done
revision_common=(--startup-file=no --project=. examples/lvd_tomography.jl --dump "$revision_dump" --papermatch-fit-only
    --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6
    --zenith-step 10 --azimuth-step 15 --samples "$revision_nodes" --csda-quadrature midpoint
    --calibration-bins 24 --calibration-mc-samples "$revision_calibration_mc"
    --papermatch-recursions 3 --rock-density 2710 --seed 20261022)
revision_run() {
    local scenario="$1"
    shift
    local run_dir="$revision_root/$scenario"
    local -a scenario_command=("$revision_julia" "${revision_common[@]}" --output-dir "$run_dir" "$@")
    local action
    action="$(python3 dev/robustness_run_identity.py prepare --cpus 4 --memory-gib 12 \
        "$run_dir" "$revision_dump" -- "${scenario_command[@]}")"
    if [[ "$action" == skip ]]; then
        printf 'Using completed scenario: %s\n' "$scenario"
        return
    fi
    [[ "$action" == run ]] || { printf 'Unexpected resume decision: %s\n' "$action" >&2; return 2; }
    python3 - "$run_dir" "$scenario" "${scenario_command[@]}" <<'PY'
import datetime,json,pathlib,sys
pathlib.Path(sys.argv[1],'command.json').write_text(json.dumps(dict(scenario=sys.argv[2],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=sys.argv[3:],cpus=4,address_space_gib=12),indent=2)+'\n')
PY
    printf 'Starting nonlinear scenario: %s\n' "$scenario"
    if python3 dev/run_bounded.py --cpus 4 --memory-gib 12 -- \
        "${scenario_command[@]}" > "$run_dir/run.log" 2>&1; then
        python3 dev/robustness_run_identity.py complete "$run_dir"
    else
        local exit_code="$?"
        python3 - "$run_dir" "$exit_code" <<'PY'
import datetime,json,pathlib,sys
pathlib.Path(sys.argv[1],'failed.json').write_text(json.dumps(dict(failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=int(sys.argv[2])))+'\n')
PY
        python3 dev/summarize_revision_robustness.py "$revision_root"
        return "$exit_code"
    fi
    python3 dev/summarize_revision_robustness.py "$revision_root"
}
revision_run baseline
# One alternate seed is a descriptive MC-noise comparison, not a calibrated robustness test.
revision_run baseline_seed2 --seed 20261023
revision_run cutoff350 --papermatch-prior-cutoff 350
revision_run cutoff650 --papermatch-prior-cutoff 650
revision_run maskInf --papermatch-mask-ratio Inf
revision_run rho2660 --rock-density 2660
revision_run rho2760 --rock-density 2760
revision_run initialwater03 --papermatch-calibration-water 0.3
revision_scale="$(python3 - "$revision_root/baseline/lvd_fit_predictions.csv" <<'PY'
import csv,sys
with open(sys.argv[1]) as f:
    rows=list(csv.DictReader(f))
s=float(rows[0]['normalization'])
assert s>0 and all(float(r['normalization'])==s for r in rows)
print(repr(s))
PY
)"
revision_scale_low="$(python3 -c 'import sys;print(repr(float(sys.argv[1])*0.95))' "$revision_scale")"
revision_scale_high="$(python3 -c 'import sys;print(repr(float(sys.argv[1])*1.05))' "$revision_scale")"
revision_run scale_minus5pct --papermatch-abs-scale "$revision_scale_low"
revision_run scale_plus5pct --papermatch-abs-scale "$revision_scale_high"
python3 dev/summarize_revision_robustness.py "$revision_root"
printf 'Nonlinear robustness comparisons: %s/comparison.csv\n' "$revision_root"
