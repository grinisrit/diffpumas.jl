#!/usr/bin/env python3
"""Run a fresh, bounded revision job and audit source/runtime/output identity.

Invoke inside tmux. The coordinator allocates disjoint CPU pools and ensures the
sum of concurrent job memory allocations stays within the overall 30% budget.
The address-space limit is inherited per process, not an aggregate cgroup cap.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from record_revision_run import record_sources
from robustness_run_identity import julia_runtime
from run_bounded import cgroup_limits, effective_budget

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dump', type=Path, required=True)
    p.add_argument('--julia', required=True)
    p.add_argument('--cpus', type=int, required=True)
    p.add_argument('--cpu-start', type=int, default=0)
    p.add_argument('--memory-gib', type=float, required=True)
    p.add_argument('--input', action='append', type=Path, default=[])
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('command required after --')
    available = sorted(os.sched_getaffinity(0))
    budget = effective_budget(available, os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'), cgroup_limits())
    pool = list(range(a.cpu_start, a.cpu_start + a.cpus))
    memory = int(a.memory_gib * 1024 ** 3)
    if not 1 <= a.cpus <= budget['max_cpus'] or not set(pool) <= set(available):
        p.error('CPU allocation exceeds effective budget or available affinity')
    if not 0 < memory <= budget['max_address_space_bytes']:
        p.error('memory allocation exceeds effective budget')
    output = a.output.resolve()
    if output.exists():
        p.error('output exists; use a fresh directory to avoid mixed provenance')
    dump = a.dump.resolve(strict=True)
    extras = [Path(__file__).resolve(), ROOT / 'dev/record_revision_run.py',
              ROOT / 'dev/robustness_run_identity.py', ROOT / 'dev/run_bounded.py']
    extras += [f.resolve(strict=True) for f in a.input]
    # A script passed directly to the command is also an identity input.
    extras += [Path(arg).resolve() for arg in command if Path(arg).is_file()]
    expected = record_sources(ROOT, dump)
    expected.update({str(f): digest(f) for f in extras})
    runtime = julia_runtime(a.julia)
    wrapped = [sys.executable, str(ROOT / 'dev/run_bounded.py'), '--cpus', str(a.cpus),
               '--memory-gib', str(a.memory_gib), '--', 'taskset', '-c',
               ','.join(map(str, pool)), *command]
    output.mkdir(parents=True)
    record = dict(started_utc=utc(), command=command, wrapped_command=wrapped,
                  cpu_pool=pool, address_space_bytes=memory, effective_budget=budget,
                  julia_runtime=runtime, source_sha256=expected,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    with (output / 'run.log').open('wb') as stream:
        child = subprocess.Popen(wrapped, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        (output / 'pid.json').write_text(json.dumps(dict(pid=child.pid, launcher_pid=os.getpid(), started_utc=utc())) + '\n')
        returncode = child.wait()
    changed = []
    for name, sha in expected.items():
        f = Path(name) if Path(name).is_absolute() else ROOT / name
        if not f.is_file() or digest(f) != sha:
            changed.append(name)
    runtime_unchanged = julia_runtime(a.julia) == runtime
    outputs = {str(f.relative_to(output)): digest(f) for f in sorted(output.rglob('*'))
               if f.is_file() and f.name not in ('end_status.json', 'pid.json')}
    completed = returncode == 0 and not changed and runtime_unchanged
    status = dict(ended_utc=utc(), returncode=returncode, execution_completed=completed,
                  source_unchanged=not changed, source_changes=changed,
                  runtime_unchanged=runtime_unchanged, output_sha256=outputs)
    (output / 'end_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
