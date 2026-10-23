#!/usr/bin/env python3
"""Record transitive LVD code/input identities and the selected Julia runtime."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from robustness_run_identity import julia_runtime

ROOT = Path(__file__).resolve().parents[1]


def record_sources(root, dump):
    paths = list((root / 'src').glob('*.jl'))
    paths += [root / name for name in (
        'Project.toml', 'Manifest.toml', 'examples/lvd_tomography.jl',
        'examples/lvd_muography.jl', 'examples/export_lvd_topography.jl',
        'dev/run_lvd_paper.sh', 'dev/run_bounded.py', 'dev/record_revision_run.py',
        'dev/robustness_run_identity.py')]
    paths += list((root / 'examples/data/lvd_conf').rglob('*'))
    paths.append(dump.resolve())
    return {str(f.relative_to(root) if f.is_relative_to(root) else f):
            hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(set(paths)) if f.is_file()}


def build_record(root, dump, command, julia):
    dump = dump.resolve()
    return dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host=socket.gethostname(), command=command, julia_runtime=julia_runtime(julia),
        physics_dump=dict(path=str(dump), exists=dump.is_file(),
            sha256=hashlib.sha256(dump.read_bytes()).hexdigest() if dump.is_file() else None),
        environment={k: os.environ.get(k) for k in (
            'JULIA_DEPOT_PATH', 'JULIA_LOAD_PATH', 'JULIA_PROJECT',
            'JULIA_CPU_TARGET', 'PUMAS_MATERIALS_PATH', 'LD_LIBRARY_PATH')},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                           cwd=root, text=True),
        sha256=record_sources(root, dump))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--julia', default=os.environ.get('JULIA_BIN', 'julia'))
    parser.add_argument('output', type=Path)
    parser.add_argument('dump', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args=parser.parse_args()
    record=build_record(ROOT, args.dump, args.command, args.julia)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    # Existing evidence is never overwritten, even on a timestamp collision.
    with (args.output / f'run_manifest_{stamp}.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
