#!/usr/bin/env python3
"""Resource-only continuation of a stopped, provenance-verified robustness root.

Never stops processes. --dry-run performs validation but creates no files.
Incomplete attempts require explicit --preserve-incomplete and are moved intact.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time

import robustness_run_identity as identity

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [('baseline', []), ('baseline_seed2', ['--seed', '20261023']),
             ('cutoff350', ['--papermatch-prior-cutoff', '350']),
             ('cutoff650', ['--papermatch-prior-cutoff', '650']),
             ('maskInf', ['--papermatch-mask-ratio', 'Inf']),
             ('rho2660', ['--rock-density', '2660']),
             ('rho2760', ['--rock-density', '2760']),
             ('initialwater03', ['--papermatch-calibration-water', '0.3']),
             ('scale_minus5pct', None), ('scale_plus5pct', None)]


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def invariant(value):
    value = dict(value)
    value.pop('cpus')
    value.pop('address_space_gib')
    value['command'] = value['command'][1:]
    return identity.canonical_digest(value)


def live_processes(output, proc=Path('/proc')):
    """Inspect all owners: a driver and its case command must not race continuation."""
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            # Match argv values, not a grep/shell string containing a root name.
            owns = any(a == str(output) or a.startswith(str(output) + '/') for a in command)
            relevant = any('julia' in Path(a).name.lower() or
                           Path(a).name in ('run_revision_robustness.sh',
                                            'continue_revision_robustness.py', 'run_lvd_paper.sh') for a in command if a)
            if owns and relevant:
                found.append(dict(pid=int(entry.name), command=[a for a in command if a]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise ValueError('Cannot establish stopped state for process ' + entry.name) from error
    return found


def require_stopped(output):
    alive = live_processes(output)
    if alive:
        raise ValueError('Robustness driver/cases still alive; parent must stop them safely first: ' +
                         json.dumps(alive))


MAIN_REPORTS = ('lvd_topography_grid.csv', 'lvd_single_muon_angular_distribution.txt',
                'lvd_reconstruction_sweep.txt', 'lvd_detection_benchmark.txt',
                'lvd_detection_contrast_sweep.txt')


def main_outer_processes(wrapper_paths, proc=Path('/proc')):
    alive = []
    for entry in proc.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if 'main' in command and any(p in command for p in wrapper_paths):
                alive.append(dict(pid=int(entry.name), command=[a for a in command if a]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return alive


def validate_main(output):
    output = output.absolute()
    if output.resolve() != output or output.is_symlink() or not output.is_dir():
        raise ValueError('Main run must be an existing canonical directory')
    require_stopped(output)
    execution = json.loads((output / 'execution.json').read_text())
    if execution.get('mode') != 'main' or execution.get('output') != str(output):
        raise ValueError('Main root lacks matching execution provenance')
    # The outer wrapper argv contains only its mode, so also check it directly;
    # this closes the gap between sequential stages with no Julia child alive.
    wrapper_paths = {str(Path(p).resolve()) for p in execution['expected_source_sha256']
                     if Path(p).name.startswith('launch_recorded_production')}
    alive = main_outer_processes(wrapper_paths)
    if alive:
        raise ValueError('Main outer driver remains alive: ' + json.dumps(alive))
    end = json.loads((output / 'end_status.json').read_text())
    if (end.get('returncode') != 0 or end.get('execution_completed') is not True or
        end.get('source_unchanged') is not True or end.get('source_changes') or
        any(end.get('stage_reports_present', {}).get(p) is not True for p in MAIN_REPORTS)):
        raise ValueError('Main has not completed all stages with a clean source audit')
    for name, digest in execution['expected_source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or identity.digest_file(path) != digest:
            raise ValueError('Main recorded source/input changed: ' + name)
    text = (output / 'launcher.log').read_text()
    if ('ALL SELECTED STAGES DONE ' not in text or
        any('=== [' + stage + '] OK ' not in text for stage in ('topo','main','sweep','detection','contrast'))):
        raise ValueError('Main launcher lacks all five successful stage markers')
    paths = [output / p for p in MAIN_REPORTS + ('end_status.json','execution.json','launcher.log')]
    if any(not p.is_file() or not p.stat().st_size for p in paths):
        raise ValueError('Main stage report is missing or empty')
    return dict(root=str(output), artifacts_sha256={str(p):identity.digest_file(p) for p in paths})


def verified_record(directory, expected):
    record = identity.read_record(directory)
    if record['identity'] != expected:
        verify_previous_continuation(directory, record, expected)
    if (directory / 'completed.json').exists():
        if identity.prepare(directory, record['identity']) != 'skip':
            raise ValueError('Invalid completion marker')
        return 'skip', record
    return 'preserve', record


def verify_previous_continuation(directory, record, expected):
    if not (directory / 'continuation_audit.json').is_file():
        raise ValueError('Scenario numerical inputs/source/runtime do not match baseline plan')
    audit = json.loads((directory / 'continuation_audit.json').read_text())
    manifest_path = Path(audit['root_manifest']).resolve(strict=True)
    if not manifest_path.is_relative_to(directory.parent / 'continuations'):
        raise ValueError('Continuation manifest escapes owned root')
    if identity.digest_file(manifest_path) != audit['root_manifest_sha256']:
        raise ValueError('Continuation manifest changed')
    manifest = json.loads(manifest_path.read_text())
    actual = record['identity']
    if (audit['new_identity_sha256'] != record['sha256'] or
        audit['prior_identity_sha256'] != identity.canonical_digest(audit['original_identity']) or
        audit['invariant_sha256'] != invariant(actual) or invariant(actual) != invariant(expected) or
        invariant(audit['original_identity']) != invariant(expected)):
        raise ValueError('Scenario numerical inputs/source/runtime do not match baseline plan')
    shim = Path(manifest['shim']).resolve(strict=True)
    if (not shim.is_relative_to(directory.parent / 'continuations') or
        identity.digest_file(shim) != manifest['shim_sha256'] or
        actual['command'][0] != str(shim) or actual['cpus'] != manifest['resources']['cpus'] or
        actual['address_space_gib'] != manifest['resources']['address_space_gib'] or
        manifest['helper_sha256'] != identity.digest_file(Path(__file__).resolve())):
        raise ValueError('Previous resource continuation is not verified')


def baseline_scale(output):
    directory = output / 'baseline'
    record = identity.read_record(directory)
    if identity.prepare(directory, record['identity']) != 'skip':
        raise ValueError('Baseline must complete before scale variants can be derived')
    with (directory / 'lvd_fit_predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    scales = [float(row['normalization']) for row in rows]
    if not scales or not all(math.isfinite(x) and x > 0 and x == scales[0] for x in scales):
        raise ValueError('Baseline normalization is not finite, positive and constant')
    return scales[0]


def base_configuration(output, runtime):
    baseline = identity.read_record(output / 'baseline')['identity']
    prior_audit = output / 'baseline/continuation_audit.json'
    if prior_audit.exists():
        audit = json.loads(prior_audit.read_text())
        manifest_path = Path(audit['root_manifest']).resolve(strict=True)
        if not manifest_path.is_relative_to(output / 'continuations') or identity.digest_file(manifest_path) != audit['root_manifest_sha256']:
            raise ValueError('Baseline continuation manifest is not verified')
        baseline = json.loads(manifest_path.read_text())['baseline_identity']
    command = baseline['command']
    if command.count('--output-dir') != 1:
        raise ValueError('Baseline must contain exactly one output directory')
    pos = command.index('--output-dir')
    if command[pos + 1] != str(output / 'baseline') or pos + 2 != len(command):
        raise ValueError('Unexpected recorded baseline command/output tail')
    if '--papermatch-fit-only' not in command or '--startup-file=no' not in command:
        raise ValueError('Recorded command is not the fit-only robustness configuration')
    current = identity.build_identity(ROOT, Path(baseline['dump']), command,
                                      baseline['cpus'], baseline['address_space_gib'], runtime=runtime)
    if current != baseline:
        raise ValueError('Current source/dump/runtime/environment differ from recorded baseline')
    return baseline, command[:pos]


def expected_case(baseline, common, output, name, extra):
    value = dict(baseline)
    value['command'] = common + ['--output-dir', str(output / name)] + extra
    return value


def validate_root(output):
    if not output.is_dir() or output.is_symlink() or output.stat().st_uid != os.getuid():
        raise ValueError('Require an existing non-symlink root owned by this user')
    execution = json.loads((output / 'execution.json').read_text())
    if execution.get('mode') != 'robustness' or execution.get('output') != str(output):
        raise ValueError('Root lacks matching original robustness execution provenance')
    for name, digest in execution['expected_source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or identity.digest_file(path) != digest:
            raise ValueError('Original root source/wrapper/input changed: ' + name)
    return execution


def run(args):
    output = args.root.absolute()
    if output.resolve() != output:
        raise ValueError('Root must use its canonical path without symlink components')
    require_stopped(output)
    original = validate_root(output)
    main_evidence = validate_main(args.main_run)
    recorded = identity.read_record(output / 'baseline')['identity']
    old_runtime = identity.julia_runtime(recorded['command'][0])
    actual_runtime = identity.julia_runtime(args.julia or old_runtime['binary'])
    if actual_runtime != old_runtime or actual_runtime != recorded['runtime']:
        raise ValueError('Continuation requires the identical verified actual Julia runtime')
    baseline, common = base_configuration(output, actual_runtime)
    pending = []
    scale = None
    if (output / 'baseline/completed.json').exists():
        scale = baseline_scale(output)
    for name, extra in SCENARIOS:
        if extra is None and scale is None:
            pending.append(dict(scenario=name, action='deferred_until_verified_baseline'))
            continue
        if extra is None:
            extra = ['--papermatch-abs-scale', repr(scale * (0.95 if 'minus' in name else 1.05))]
        expected = expected_case(baseline, common, output, name, extra)
        directory = output / name
        if directory.is_symlink():
            raise ValueError('Scenario is a symlink: ' + name)
        if directory.exists():
            if directory.stat().st_uid != os.getuid():
                raise ValueError('Scenario is not owned by this user: ' + name)
            action, _ = verified_record(directory, expected)
            if action == 'preserve' and not args.preserve_incomplete:
                raise ValueError('Incomplete existing case; use explicit --preserve-incomplete after safe stop: ' + name)
        else:
            action = 'run'
        new_command = ['<continuation shim>'] + expected['command'][1:]
        pending.append(dict(scenario=name, action=action, expected_old_identity_sha256=identity.canonical_digest(expected),
                            invariant_sha256=invariant(expected), numerical_command_tail=new_command[1:]))
    if args.dry_run:
        print(json.dumps(dict(root=str(output), cpus=args.cpus, address_space_gib=args.memory_gib,
                              requested_affinity=args.affinity, runtime=actual_runtime, cases=pending), indent=2))
        return
    require_stopped(output)  # Recheck immediately before mutations; never kill a process.
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output / 'continuations' / stamp
    archive.mkdir(parents=True)
    affinity = parse_affinity(args.affinity)
    if len(affinity) != args.cpus:
        raise ValueError('Affinity size must equal requested CPU count')
    shim = archive / 'julia.sh'
    shim.write_text('#!/bin/sh\nexec taskset -c ' + shlex.quote(args.affinity) + ' ' +
                    shlex.quote(actual_runtime['binary']) + ' "$@"\n')
    shim.chmod(0o755)
    manifest = dict(schema_version=1, started_utc=stamp, root=str(output),
                    original_execution_sha256=identity.digest_file(output / 'execution.json'), main_run=main_evidence,
                    baseline_identity_sha256=identity.canonical_digest(baseline), baseline_identity=baseline,
                    original_resources=dict(cpus=baseline['cpus'], address_space_gib=baseline['address_space_gib']),
                    resources=dict(cpus=args.cpus, address_space_gib=args.memory_gib, requested_affinity=affinity),
                    runtime=actual_runtime, source_sha256=baseline['sha256'], cases=pending,
                    helper_sha256=identity.digest_file(Path(__file__).resolve()),
                    identity_helper_sha256=identity.digest_file(Path(identity.__file__).resolve()),
                    shim=str(shim), shim_sha256=identity.digest_file(shim), argv=sys.argv,
                    scope='Resource-only continuation; execution completion does not establish scientific convergence.')
    manifest_path = archive / 'manifest.json'
    write_json(manifest_path, manifest)
    status = dict(returncode=1, completed_cases=[], skipped_cases=[], failed_case=None)
    try:
        for name, extra in SCENARIOS:
            require_stopped(output)
            if validate_main(args.main_run) != main_evidence:
                raise ValueError('Verified main evidence changed')
            base_configuration(output, actual_runtime)  # Recheck frozen inputs before each new case.
            if identity.digest_file(Path(__file__).resolve()) != manifest['helper_sha256'] or identity.digest_file(shim) != manifest['shim_sha256']:
                raise ValueError('Continuation helper/shim changed during execution')
            if extra is None:
                scale = baseline_scale(output)
                extra = ['--papermatch-abs-scale', repr(scale * (0.95 if 'minus' in name else 1.05))]
            old = expected_case(baseline, common, output, name, extra)
            directory = output / name
            prior = old
            if directory.exists():
                action, record = verified_record(directory, old)
                prior = record['identity']
                if action == 'skip':
                    status['skipped_cases'].append(name)
                    continue
                if not args.preserve_incomplete:
                    raise ValueError('Incomplete case appeared after preflight: ' + name)
                preserved = archive / 'preserved_incomplete'
                preserved.mkdir(exist_ok=True)
                directory.rename(preserved / name)
            new = dict(old, cpus=args.cpus, address_space_gib=args.memory_gib,
                       command=[str(shim)] + old['command'][1:])
            if invariant(old) != invariant(new):
                raise ValueError('Numerical invariant changed')
            identity.prepare(directory, new)
            write_json(directory / 'continuation_audit.json', dict(schema_version=1,
                       root_manifest=str(manifest_path), root_manifest_sha256=identity.digest_file(manifest_path),
                       prior_identity_sha256=identity.canonical_digest(prior),
                       new_identity_sha256=identity.canonical_digest(new),
                       invariant_sha256=invariant(new), original_identity=prior,
                       changed_fields=['cpus', 'address_space_gib', 'command[0]']))
            write_json(directory / 'command.json', dict(scenario=name, command=new['command'],
                       cpus=args.cpus, address_space_gib=args.memory_gib, continuation_manifest=str(manifest_path)))
            command = [sys.executable, str(ROOT / 'dev/run_bounded.py'), '--cpus', str(args.cpus),
                       '--memory-gib', str(args.memory_gib), '--'] + new['command']
            print('Starting continuation scenario:', name, flush=True)
            with (directory / 'run.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                observation = observe_child(child, actual_runtime['binary'], affinity)
                write_json(directory / 'runtime_observation.json', observation)
                # Child exits before the next scenario; its bounded launcher logs actual limits.
                code = child.wait()
            if code:
                status['failed_case'] = name
                status['returncode'] = code
                raise ValueError('Scenario failed with exit ' + str(code) + ': ' + name)
            if observation.get('affinity_matches') is not True:
                status['failed_case'] = name
                raise ValueError('Actual worker affinity was not verified; no next case launched')
            identity.complete(directory)
            status['completed_cases'].append(name)
            subprocess.run([sys.executable, str(ROOT / 'dev/summarize_revision_robustness.py'), str(output)],
                           cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'dev/summarize_revision_robustness.py'), str(output)],
                       cwd=ROOT, check=True)  # Also refresh an all-skipped verified resume.
        status['returncode'] = 0
    finally:
        changed = []
        for name, digest in manifest['source_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if not path.is_file() or identity.digest_file(path) != digest:
                changed.append(name)
        for path, digest in [(Path(__file__).resolve(), manifest['helper_sha256']),
                             (shim, manifest['shim_sha256'])]:
            if not path.is_file() or identity.digest_file(path) != digest:
                changed.append(str(path))
        status.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_changes=changed,
                      runtime_unchanged=check_runtime(actual_runtime),
                      manifest_sha256=identity.digest_file(manifest_path))
        observations = [json.loads((output / name / 'runtime_observation.json').read_text())
                        for name in status['completed_cases']]
        status['affinity_verified'] = all(x.get('affinity_matches') is True for x in observations)
        status['verified_case_identities'] = {}
        for name in status['skipped_cases'] + status['completed_cases']:
            directory = output / name
            record = identity.read_record(directory)
            marker = json.loads((directory / 'completed.json').read_text())
            audit = directory / 'continuation_audit.json'
            status['verified_case_identities'][name] = dict(identity_sha256=record['sha256'],
                continuation_audit_sha256=identity.digest_file(audit) if audit.exists() else None,
                outputs_sha256=marker['outputs_sha256'],
                runtime_observation_sha256=identity.digest_file(directory / 'runtime_observation.json')
                    if (directory / 'runtime_observation.json').exists() else None)
        status['execution_completed'] = status['returncode'] == 0 and not changed and status['runtime_unchanged'] and status['affinity_verified']
        write_json(archive / 'end_status.json', status)
    if not status['execution_completed']:
        raise ValueError('Continuation final audit failed')


def check_runtime(runtime):
    try:
        return identity.julia_runtime(runtime['binary']) == runtime
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def observe_child(child, binary, expected_affinity):
    # No signals or process mutations. The bounded wrapper and exec shim create
    # one worker; sample its real Linux affinity after exec, not the launch shell.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and child.poll() is None:
        descendants = {child.pid}
        processes = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                status = (entry / 'status').read_text()
                parent = int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
                processes.append((int(entry.name), parent, entry))
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                continue
        for _ in range(4):
            descendants.update(pid for pid, parent, _ in processes if parent in descendants)
        for pid, _, entry in processes:
            if pid not in descendants:
                continue
            try:
                if (entry / 'exe').resolve() != Path(binary).resolve():
                    continue
                actual = sorted(os.sched_getaffinity(pid))
                if actual != sorted(expected_affinity):
                    # Never kill a process automatically; retain observation and
                    # let the parent stop a misconfigured worker safely.
                    return dict(pid=pid, actual_affinity=actual, affinity_matches=False)
                return dict(pid=pid, actual_affinity=actual, affinity_matches=True)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        time.sleep(0.05)
    return dict(pid=None, actual_affinity=None, affinity_matches=None,
                observation='Worker exited before observation or runtime exec not observed')


def parse_affinity(value):
    result = []
    for part in value.split(','):
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', part):
            raise ValueError('Malformed CPU affinity')
        ends = part.split('-')
        if int(ends[-1]) < int(ends[0]):
            raise ValueError('Reversed CPU affinity range')
        result.extend(range(int(ends[0]), int(ends[-1]) + 1))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise ValueError('Invalid/duplicate CPU affinity')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--main-run', type=Path, required=True, help='Completed five-stage main root; successful audits and no live driver required')
    parser.add_argument('--cpus', type=int, default=20)
    parser.add_argument('--memory-gib', type=float, default=36)
    parser.add_argument('--affinity', default='0-19')
    parser.add_argument('--julia', help='Must resolve to the identical actual recorded runtime')
    parser.add_argument('--preserve-incomplete', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if not 1 <= args.cpus <= 20 or not 0 < args.memory_gib <= 36 or not math.isfinite(args.memory_gib):
            raise ValueError('Continuation limits are 1–20 CPUs and >0–36 GiB')
        if len(parse_affinity(args.affinity)) != args.cpus:
            raise ValueError('Affinity size must equal requested CPU count')
        run(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, 'Continuation refused: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
