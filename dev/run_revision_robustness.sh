#!/usr/bin/env bash
# Sequential nonlinear refits; each Julia child inherits explicit resource bounds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision_root="$(realpath -m "${1:-examples/data/revision_robustness_20261023}")"
mkdir -p "$revision_root"
revision_dump="${REVISION_PHYSICS_DUMP:-examples/data/revision_physics/materials.pumas}"
[[ -f "$revision_dump" ]] || { printf 'Missing pinned physics dump: %s\n' "$revision_dump" >&2; exit 1; }
revision_common=(--project=. examples/lvd_tomography.jl --dump "$revision_dump" --papermatch-fit-only
    --geometry grid --mesh-half-km 5 --surface-step-km 1 --grid-nz 6
    --zenith-step 10 --azimuth-step 15 --samples 32 --csda-quadrature midpoint
    --calibration-bins 24 --calibration-mc-samples 128
    --papermatch-recursions 3 --rock-density 2710 --seed 20261022)
revision_run() {
    local scenario="$1"
    shift
    local run_dir="$revision_root/$scenario"
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/completed.json" ]]; then
        printf 'Using completed scenario: %s\n' "$scenario"
        return
    fi
    python3 - "$run_dir" "$scenario" "${revision_common[@]}" --output-dir "$run_dir" "$@" <<'PY'
import datetime,json,pathlib,sys
pathlib.Path(sys.argv[1],'command.json').write_text(json.dumps(dict(scenario=sys.argv[2],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=['julia']+sys.argv[3:],cpus=4,address_space_gib=12),indent=2)+'\n')
PY
    printf 'Starting nonlinear scenario: %s\n' "$scenario"
    if python3 dev/run_bounded.py --cpus 4 --memory-gib 12 -- \
        julia "${revision_common[@]}" --output-dir "$run_dir" "$@" > "$run_dir/run.log" 2>&1; then
        python3 - "$run_dir" <<'PY'
import datetime,json,pathlib,sys
pathlib.Path(sys.argv[1],'completed.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
PY
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
