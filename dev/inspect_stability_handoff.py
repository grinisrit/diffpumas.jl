#!/usr/bin/env python3
"""Read-only queue-boundary inspection. Never signals, waits for, or launches jobs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(pid):
    base = Path('/proc') / str(pid)
    try:
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        command = [x.decode(errors='replace') for x in (base/'cmdline').read_bytes().split(b'\0') if x]
        threads = []
        for task in sorted((base/'task').iterdir()):
            try:
                status = dict(line.split(':', 1) for line in (task/'status').read_text().splitlines() if ':' in line)
                threads.append(dict(tid=int(task.name), cpu_affinity=status.get('Cpus_allowed_list', '').strip()))
            except (FileNotFoundError, ProcessLookupError):
                pass
        # Recheck identity after collecting the thread snapshot.
        after = (base/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[19] != after[19]:
            return None
        return dict(pid=pid, ppid=int(stat[1]), start_ticks=int(stat[19]), state=after[0],
                    command=command, threads=threads)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def audit_completed(folder):
    endpath = folder/'end_status.json'
    end = json.loads(endpath.read_text())
    okay = all(end.get(key) is True for key in ('execution_completed', 'source_unchanged', 'runtime_unchanged'))
    changed = [name for name, digest in end['output_sha256'].items() if sha(folder/name) != digest]
    return dict(path=str(folder), clean_execution=okay and not changed, end_sha256=sha(endpath),
                outputs_verified=len(end['output_sha256']), changed_outputs=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.old_root.resolve(strict=True)
    state_path = root/'queue_state.json'
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    plan = json.loads((root/'queue_plan.json').read_text())
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = process(int(path.name))
            if info:
                processes[info['pid']] = info
    controllers = [p for p in processes.values() if any(Path(s).name == 'queue_stability_v2_controls.py' for s in p['command'])
                   and str(root) in p['command']]
    controller_ids = {p['pid'] for p in controllers}
    descendants = set(controller_ids)
    while True:
        found = {p['pid'] for p in processes.values() if p['ppid'] in descendants}
        if found <= descendants:
            break
        descendants |= found
    current = root / state['case']
    completed = {}
    for case in state.get('completed_cases', []):
        folder = Path(plan['dry_reference']) if case == 'dry/start0' else root/case
        completed[case] = audit_completed(folder)
    relevant = {'queue_stability_v2_controls.py', 'run_revision_job.py', 'run_bounded.py',
                'build_surrogate_stability_report.py', 'julia'}
    old_root_processes = [p for p in processes.values() if any(str(root) in s for s in p['command'])
                          and any(Path(s).name in relevant for s in p['command'])]
    record = dict(schema_version=1, read_only=True, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  queue_root=str(root), state=state, queue_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                  queue_state_unchanged_during_inspection=state_bytes == state_path.read_bytes(),
                  controllers=controllers, controller_descendants=[processes[p] for p in sorted(descendants-controller_ids)],
                  old_root_processes=old_root_processes, completed_cases=completed,
                  current_child_path=str(current), current_wrapper=processes.get(state.get('wrapper_pid')),
                  current_end_exists=(current/'end_status.json').is_file(),
                  inspection_source_sha256=sha(Path(__file__)),
                  boundary_plan=[
                      'This is an observation, not authorization or a durable PID handle; refresh at the actual boundary.',
                      'After certificate approval, verify controller boot/start ticks and preferably hold a pidfd.',
                      'Coordinator may stop only that controller; never signal its process group or numerical descendants.',
                      'Let the current wrapper finish and independently verify its clean end/source/runtime/output audit.',
                      'Persist the last completed case and all original provenance before retiring the controller.',
                      'Verify controller is absent and no old-root numerical/report child survives before starting successor.',
                      'Use the actual completed-case map at handoff; do not assume exactly three cases completed.'])
    if record['current_end_exists']:
        record['current_end_audit'] = audit_completed(current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), controllers=[(p['pid'], p['start_ticks']) for p in controllers],
                          current_case=state['case'], completed=list(completed),
                          stable_snapshot=record['queue_state_unchanged_during_inspection'])))


if __name__ == '__main__':
    main()
