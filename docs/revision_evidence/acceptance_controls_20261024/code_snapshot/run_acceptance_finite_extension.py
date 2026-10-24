#!/usr/bin/env python3
"""Wait for clean first-order experiment, bind exact inputs, run paired extension.

Never interrupts or changes the original job. Run this launcher in durable tmux
only after review. --prepare-only waits/audits/prepares but does not launch.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from robustness_run_identity import julia_runtime

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstorder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = args.firstorder.resolve()
    if args.output.exists():
        parser.error('Use a fresh extension output directory')
    args.output.mkdir(parents=True)
    status_path = args.output / 'queue_state.json'
    status_path.write_text(json.dumps(dict(status='waiting_for_clean_firstorder', source=str(source)), indent=2)+'\n')
    while not (source / 'end_status.json').is_file():
        time.sleep(30)
    end = json.loads((source / 'end_status.json').read_text())
    if not (end.get('execution_completed') and end.get('returncode') == 0 and
            end.get('source_unchanged') and end.get('runtime_unchanged')):
        raise ValueError('First-order experiment failed; no extension launched')
    checked = {}
    for relative, expected in end['output_sha256'].items():
        path = source / relative
        if digest(path) != expected:
            raise ValueError('First-order output changed: ' + str(path))
        checked[str(path)] = expected
    execution = json.loads((source / 'execution.json').read_text())
    for name, expected in execution['source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if digest(path) != expected:
            raise ValueError('Original scientific/generator input changed: ' + str(path))
    if julia_runtime(args.julia) != execution['julia_runtime']:
        raise ValueError('Serialization/operator Julia runtime changed')
    checked[str(source / 'end_status.json')] = digest(source / 'end_status.json')
    checked[str(source / 'execution.json')] = digest(source / 'execution.json')
    results = source / 'results'
    helper = ROOT / 'dev/acceptance_poisson_draws.py'
    checked[str(helper)] = digest(helper)
    # Explicit old config ensures first three factors and stream numbering match.
    import tomllib
    original = tomllib.loads((results / 'config.toml').read_text())
    if original['pilot_exposure_factors'] != [.5, 1., 2.] or not original['poisson_pilot']:
        raise ValueError('Original exposure grid or pilot differs')
    if any(original[key] != value for key, value in (
            ('poisson_calibration_null', 3000), ('poisson_test_null', 3000),
            ('poisson_signal', 600), ('seed', 20261024))):
        raise ValueError('Original population counts or seed differ')
    config = dict(output=str((args.output / 'job/results').resolve()),
                  firstorder_results=str(results),
                  arrays=str(results / 'frozen_arrays.bin'),
                  directional_templates=str(results / 'directional_templates.csv'),
                  exposure_shares=str(results / 'exposure_shares.csv'),
                  poisson_helper=str(helper), seed=20261024,
                  exposure_factors=[.5, 1., 2., 4., 8., 16.])
    config_path = args.output / 'config.toml'
    text = '\n'.join(f'{key} = {json.dumps(value)}' for key, value in config.items())
    text += '\n\n[input_sha256]\n' + '\n'.join(
        f'{json.dumps(key)} = {json.dumps(value)}' for key, value in checked.items()) + '\n'
    config_path.write_text(text)
    proof = dict(firstorder_end_sha256=digest(source / 'end_status.json'),
                 original_runtime=execution['julia_runtime'],
                 verified_firstorder_output_count=len(end['output_sha256']),
                 input_sha256=checked, reuse='Original first-order factors .5/1/2 nulls and linear statistics; new independent finite signals. Higher factors fresh paired populations.',
                 scope='Finite-CSDA alternative remains a conditional biased-surrogate experiment, not stochastic transport truth.')
    proof_path = args.output / 'reuse_certificate.json'
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    status_path.write_text(json.dumps(dict(status='prepared' if args.prepare_only else 'launching', config=str(config_path)), indent=2)+'\n')
    if args.prepare_only:
        return
    dump = ROOT / 'examples/data/revision_physics_v2_final_20261023/materials.pumas'
    helper_julia = ROOT / 'dev/detection_acceptance_finite_extension.jl'
    extra_inputs = [Path(__file__).resolve(), helper_julia, helper, config_path, proof_path,
                    *map(Path, checked)]
    for phase in ('load_smoke', 'job'):
        command = [sys.executable, str(ROOT / 'dev/run_revision_job.py'),
                   '--output', str(args.output / phase), '--dump', str(dump),
                   '--julia', args.julia, '--cpus', '8', '--cpu-start', '8',
                   '--memory-gib', '16']
        for path in extra_inputs:
            command += ['--input', str(path)]
        command += ['--', args.julia, '--startup-file=no', '--project=.',
                    str(helper_julia), '--load-only' if phase == 'load_smoke' else str(config_path)]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            break
    status_path.write_text(json.dumps(dict(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode), indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
