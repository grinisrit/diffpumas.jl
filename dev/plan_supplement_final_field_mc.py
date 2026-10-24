#!/usr/bin/env python3
"""Prepare the disjoint 60-ray closure supplement; --execute requires reviewed plan."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'examples/data/material_production_20261024/stages/main/execution.json'
BASELINE = ROOT / 'examples/data/material_main_recovery_20261024/data'
HELPER = ROOT / 'dev/supplement_final_field_mc_closure.jl'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    command = json.loads(ORIGINAL.read_text())['command']
    position = next(i for i, x in enumerate(command) if x.endswith('/examples/lvd_tomography.jl'))
    flags = command[position + 1:]
    flags[flags.index('--output-dir') + 1] = str(output / 'job/data')
    julia = command[0]
    inputs = [ORIGINAL, HELPER, Path(__file__).resolve(), ROOT/'dev/validate_surrogate_v2.jl', ROOT/'examples/revision_suite.jl']
    inputs += [BASELINE / n for n in ('lvd_fit_field.csv', 'lvd_fit_predictions.csv', 'lvd_fit_mask.csv', 'lvd_solver_configuration.toml', 'lvd_mc_coverage.csv', 'lvd_mc_common_subset.csv')]
    artifact = ROOT / 'docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml'
    inputs += [artifact]
    wrapped = [sys.executable, str(ROOT/'dev/run_revision_job.py'), '--output', str(output/'job'), '--dump', str(ROOT/'examples/data/revision_physics_v2_final_20261023/materials.pumas'), '--julia', julia, '--cpus', '4', '--cpu-start', '16', '--memory-gib', '8']
    for path in inputs:
        wrapped += ['--input', str(path)]
    wrapped += ['--', julia, '--startup-file=no', '--project=.', str(HELPER), *flags, '--baseline-results', str(BASELINE)]
    record = {'scope': '60 complementary fitted directions, frozen recovered field, 8192 fixed-path histories/ray; original630 outputs untouched. No optimization or scattering.', 'energy_seed': 819, 'transport_seed': '16000000+bin_idx, disjoint from original900042+covered_position', 'command': wrapped, 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    output.mkdir(parents=True, exist_ok=True)
    plan = output/'preregistered_plan.json'
    if plan.exists() and json.loads(plan.read_text()) != record:
        raise SystemExit('plan or inputs changed; select a fresh output root')
    plan.write_text(json.dumps(record, indent=2)+'\n')
    if args.execute:
        raise SystemExit(subprocess.call(wrapped, cwd=ROOT))
    print(plan)

if __name__ == '__main__':
    main()
