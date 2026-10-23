#!/usr/bin/env python3
"""Bind robustness resume/completion to exact inputs, code, command and runtime."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUTS = ('lvd_fit_field.csv', 'lvd_fit_predictions.csv',
                    'lvd_recursion_status.txt', 'lvd_recursion_history.csv')


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def source_inputs(root, dump):
    paths = list((root / 'src').glob('*.jl'))
    paths += [root / name for name in (
        'Project.toml', 'Manifest.toml', 'examples/lvd_tomography.jl',
        'examples/lvd_muography.jl', 'dev/run_revision_robustness.sh',
        'dev/run_bounded.py', 'dev/robustness_run_identity.py',
        'dev/summarize_revision_robustness.py')]
    paths += [p for p in (root / 'examples/data/lvd_conf').rglob('*') if p.is_file()]
    paths.append(dump.resolve())
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError('Missing identity input: ' + ', '.join(missing))
    return {str(p.relative_to(root) if p.is_relative_to(root) else p): digest_file(p)
            for p in sorted(set(paths))}


def julia_runtime(executable):
    resolved = shutil.which(executable)
    if resolved is None:
        raise ValueError('Julia executable not found: ' + executable)
    # No packages or research calculation: identify the actual binary behind
    # juliaup/symlinks using one Julia thread and one BLAS thread.
    env = dict(os.environ, JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    query = 'println(VERSION); println(joinpath(Sys.BINDIR, Base.julia_exename()))'
    result = subprocess.run([resolved, '--startup-file=no', '--threads=1', '-e', query],
                            check=True, capture_output=True, text=True, env=env, timeout=30)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2:
        raise ValueError('Unexpected Julia runtime identity output')
    binary = Path(lines[1]).resolve(strict=True)
    return dict(version=lines[0], binary=str(binary), sha256=digest_file(binary))


def build_identity(root, dump, command, cpus, memory_gib, runtime=None):
    if not command:
        raise ValueError('A Julia command is required')
    return dict(schema_version=1, dump=str(dump.resolve()), sha256=source_inputs(root, dump), command=command,
                cpus=cpus, address_space_gib=memory_gib,
                runtime=julia_runtime(command[0]) if runtime is None else runtime,
                environment={k: os.environ.get(k) for k in (
                    'JULIA_DEPOT_PATH', 'JULIA_LOAD_PATH', 'JULIA_PROJECT',
                    'JULIA_CPU_TARGET', 'PUMAS_MATERIALS_PATH', 'LD_LIBRARY_PATH')})


def read_record(directory):
    record = json.loads((directory / 'run_identity.json').read_text())
    if record['sha256'] != canonical_digest(record['identity']):
        raise ValueError('Stored robustness identity is corrupt')
    return record


def prepare(directory, identity):
    fingerprint = canonical_digest(identity)
    path = directory / 'run_identity.json'
    if path.exists():
        record = read_record(directory)
        if record['sha256'] != fingerprint:
            raise ValueError('Robustness inputs/code/command/runtime changed; use a NEW output root')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Existing scenario has no verified identity; use a NEW output root')
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sha256=fingerprint, identity=identity), indent=2) + '\n')
    completed = directory / 'completed.json'
    if completed.exists():
        marker = json.loads(completed.read_text())
        if marker.get('run_identity_sha256') != fingerprint:
            raise ValueError('Completion marker lacks matching provenance; use a NEW output root')
        expected = marker.get('outputs_sha256', {})
        if set(expected) != set(REQUIRED_OUTPUTS):
            raise ValueError('Completion marker lacks verified output identities')
        for name, sha in expected.items():
            output = directory / name
            if not output.is_file() or digest_file(output) != sha:
                raise ValueError('Completed robustness output changed or missing: ' + name)
        return 'skip'
    return 'run'


def complete(directory, current_identity=None):
    record = read_record(directory)
    identity = record['identity']
    if current_identity is None:
        current_identity = build_identity(ROOT, Path(identity['dump']), identity['command'],
                                          identity['cpus'], identity['address_space_gib'])
    if canonical_digest(current_identity) != record['sha256']:
        raise ValueError('Inputs/code/runtime changed during the run; completion not recorded')
    if (directory / 'completed.json').exists():
        raise ValueError('Completion marker already exists; refusing to overwrite')
    outputs = {}
    for name in REQUIRED_OUTPUTS:
        output = directory / name
        if not output.is_file() or not output.stat().st_size:
            raise ValueError('Cannot complete scenario: missing/empty ' + name)
        outputs[name] = digest_file(output)
    marker = dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  run_identity_sha256=record['sha256'], outputs_sha256=outputs)
    (directory / 'completed.json').write_text(json.dumps(marker, indent=2) + '\n')
    (directory / 'failed.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--memory-gib', type=float, default=12)
    p.add_argument('directory', type=Path)
    p.add_argument('dump', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('complete')
    p.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        if args.operation == 'prepare':
            command = args.command[1:] if args.command[:1] == ['--'] else args.command
            print(prepare(args.directory, build_identity(ROOT, args.dump, command,
                                                        args.cpus, args.memory_gib)))
        else:
            complete(args.directory)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f'Robustness provenance check failed: {error}\n')


if __name__ == '__main__':
    main()
