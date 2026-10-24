#!/usr/bin/env python3
"""External two-pool coordinator; frozen scientific commands are never rewritten.
Only the exact original Python queue parent is stopped/retired. Its existing
numerical child is observed, never signaled. Once external launch is committed,
failures preserve the original parent paused to prevent duplicate original cases.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

import resume_terrain_stress as common

ROOT=common.ROOT
SELF=Path(__file__).resolve()
TEST=SELF.with_name('test_parallel_resume_terrain_stress.py')
SESSION='terrain_parallel_resume_20261025'
WATCH_SESSION='terrain_parallel_review_20261025'

def write(path,value):common.write(path,value)
def digest(path):return common.digest(path)
def read(path):return common.read(path)
def utc():return common.utc()

def rollback_action(external_launch_committed):
    return 'preserve_original_parent_paused' if external_launch_committed else 'resume_original_parent'

def claim_case(cases,ownership,cpu_start):
    for case in cases:
        if ownership[case['name']]['phase']!='pending':continue
        if cpu_start==8 and case['screen']!='all':continue
        ownership[case['name']]=dict(phase='claimed',cpu_start=cpu_start)
        return case
    return None

def validate_schedule(schedule,original):
    names=[case['name'] for case in original['cases']]
    completed=set(schedule['completed_before_resume'])
    foreign=schedule.get('foreign_case')
    excluded=completed|({foreign} if foreign else set())
    if not excluded<=set(names) or foreign in completed:raise ValueError('invalid completed/foreign names')
    expected=[case for case in original['cases'] if case['name'] not in excluded]
    if [case['name'] for case in schedule['cases']]!=[case['name'] for case in expected]:
        raise ValueError('schedule duplicates, omissions or changed order')
    for case,source in zip(schedule['cases'],expected):
        allowed=[0,8] if source['screen']=='all' else [0]
        if case['original']!=source or case['allowed_cpu_starts']!=allowed:
            raise ValueError('scientific case or pool assignment changed')
        command=source['command']
        if command[command.index('--cpus')+1]!='8' or command[command.index('--memory-gib')+1]!='16':
            raise ValueError('original per-job budget changed')
        variants={str(start):common.resource_command(command,start) for start in allowed}
        if case['resource_only_commands']!=variants:raise ValueError('resource/numerical command variant changed')
    return expected

def foreign_snapshot(folder,parent,timeout=120.):
    deadline=time.monotonic()+timeout
    while not (folder/'pid.json').exists() and not (folder/'end_status.json').exists():
        if time.monotonic()>=deadline:raise ValueError('foreign PID/end manifest timeout')
        time.sleep(.2)
    if (folder/'end_status.json').exists():return dict(already_completed=True,proof=common.audit_case(folder))
    pids=common.read_retry(folder/'pid.json',max(0.,deadline-time.monotonic()))
    wrapper=common.process_identity(pids['launcher_pid'])
    if wrapper['ppid']!=parent['pid'] or wrapper['state'] in ('Z','X','x'):
        raise ValueError('foreign wrapper is not a live child of the verified parent')
    try:child=common.process_identity(pids['pid'])
    except FileNotFoundError:child=dict(pid=pids['pid'],already_exited_before_snapshot=True)
    return dict(already_completed=False,case=folder.name,wrapper=wrapper,child=child)

def wait_foreign(folder,snapshot,poll=1.):
    if snapshot['already_completed']:return common.audit_case(folder)
    wrapper=snapshot['wrapper'];child=snapshot['child']
    while True:
        if (folder/'end_status.json').exists():
            proof=common.audit_case(folder)
            if not child.get('already_exited_before_snapshot'):
                try:now=common.process_identity(child['pid'])
                except FileNotFoundError:now=None
                if now and now['start_ticks']==child['start_ticks'] and now['state'] not in ('Z','X','x'):
                    raise ValueError('foreign Julia process still alive after its end record')
            return proof
        try:now=common.process_identity(wrapper['pid'])
        except FileNotFoundError:now=None
        if now is None or now['start_ticks']!=wrapper['start_ticks'] or now['state'] in ('Z','X','x'):
            # Recheck a just-written end after observing process exit.
            if (folder/'end_status.json').exists():continue
            raise ValueError('foreign wrapper died or PID changed without an end record')
        time.sleep(poll)

def prepare(queue_root,output):
    if output.exists():raise ValueError('fresh preparation required')
    original=read(queue_root/'preregistered_plan.json');common.verify_context(original)
    parent=common.process_identity(int(common.tmux('display-message','-p','-t',common.PANE,'#{pane_pid}')))
    common.verify_parent(parent,queue_root)
    completed,pending,active=common.pending_cases(queue_root,original)
    if len(active)>1:raise ValueError('original queue is not serial')
    dependencies=[SELF,TEST,Path(common.__file__).resolve(),ROOT/'dev/run_revision_job.py',ROOT/'dev/watch_terrain_stress.py',ROOT/'dev/summarize_terrain_stress.py']
    output.mkdir(parents=True)
    write(output/'transition_plan.json',dict(scope='Orchestration-only two-pool parallel resume; numerical commands/starts/budgets and within-job ordering remain exact.',
        prepared_utc=utc(),queue_root=str(queue_root),original_parent=parent,
        original_plan_sha256=digest(queue_root/'preregistered_plan.json'),
        host_affinity=sorted(os.sched_getaffinity(0)),external_session=SESSION,watch_session=WATCH_SESSION,
        context_sha256={str(path):digest(path) for path in dependencies},
        completed_at_preparation=completed,active_at_preparation=active,pending_at_preparation=[case['name'] for case in pending],
        pool_cpu_starts=[0,8],cpus_per_job=8,address_space_gib_per_job=16,
        failure_rule='Before any external launch attempt: CONT verified original parent. After external launch commitment: NEVER CONT original serial parent; preserve it paused and finish already-running numerical children, then recover from clean end records with a reviewed new coordinator.',
        original_watcher_status=str(ROOT/'examples/data/terrain_stress_review_candidates_20261025/watcher_status.json')))
    print(output/'transition_plan.json')

def verify(output):
    plan=read(output/'transition_plan.json');queue_root=Path(plan['queue_root'])
    for name,value in plan['context_sha256'].items():
        if not Path(name).is_file() or digest(Path(name))!=value:raise ValueError('parallel helper/context changed: '+name)
    if digest(queue_root/'preregistered_plan.json')!=plan['original_plan_sha256']:raise ValueError('original plan changed')
    original=read(queue_root/'preregistered_plan.json');common.verify_context(original)
    return plan,queue_root,original

def verify_scientific_execution(folder,case,cpu_start):
    proof=common.audit_case(folder)
    if proof is None:raise ValueError('missing clean end: '+str(folder))
    actual=read(folder/'execution.json');command=case['command'];inner=command[command.index('--')+1:]
    if actual['command']!=inner or actual['cpu_pool']!=list(range(cpu_start,cpu_start+8)) or actual['address_space_bytes']!=16*1024**3:
        raise ValueError('scientific command or allocated resource changed: '+str(folder))
    return proof

def run(output):
    plan,queue_root,original=verify(output)
    if (queue_root/'queue_complete.json').exists():raise ValueError('original already complete; no transition needed')
    if int(common.tmux('display-message','-p','-t',SESSION+':0.0','#{pane_pid}'))!=os.getpid():
        raise ValueError('run must own the declared external tmux session')
    parent=common.verify_parent(plan['original_parent'],queue_root)
    self_identity=common.process_identity(os.getpid());marker=queue_root/'parallel_resume_owner.json'
    with marker.open('x') as stream:
        json.dump(dict(controller=self_identity,transition_plan=str(output/'transition_plan.json'),transition_plan_sha256=digest(output/'transition_plan.json')),stream,indent=2)
    mutex=threading.RLock();failed=threading.Event();external_committed=False;stopped=False;retired=False
    ownership={};schedule_path=output/'execution_schedule.json'
    def event(phase,**values):
        with mutex:
            with (output/'events.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),phase=phase,**values))+'\n')
            write(output/'coordinator_status.json',dict(updated_utc=utc(),phase=phase,original_parent_paused=stopped and not retired,
                external_launch_committed=external_committed,controller=self_identity,ownership=ownership,**values))
    try:
        common.verify_parent(parent,queue_root);os.kill(parent['pid'],signal.SIGSTOP);stopped=True
        while common.verify_parent(parent,queue_root)['state'] not in ('T','t'):time.sleep(.1)
        completed,pending,active=common.pending_cases(queue_root,original)
        if len(active)>1:raise ValueError('multiple foreign active cases')
        foreign=active[0] if active else None
        snapshot=foreign_snapshot(queue_root/foreign,parent) if foreign else None
        if snapshot and snapshot['already_completed']:
            completed,pending,active=common.pending_cases(queue_root,original);foreign=None;snapshot=None
        schedule=dict(created_utc=utc(),completed_before_resume=completed,foreign_case=foreign,foreign_identity=snapshot,cases=[])
        for source in pending:
            starts=[0,8] if source['screen']=='all' else [0]
            schedule['cases'].append(dict(name=source['name'],original=source,allowed_cpu_starts=starts,
                resource_only_commands={str(start):common.resource_command(source['command'],start) for start in starts}))
        validate_schedule(schedule,original);write(schedule_path,schedule);schedule_hash=digest(schedule_path)
        (output/'original_queue.log').write_bytes((queue_root/'queue.log').read_bytes())
        ownership.update({case['name']:dict(phase='pending') for case in pending})
        watcher_output=output/'review_candidates'
        watcher=[sys.executable,str(ROOT/'dev/watch_terrain_stress.py'),str(queue_root),'--output',str(watcher_output),'--queue-session',SESSION]
        common.tmux('new-session','-d','-s',WATCH_SESSION,'exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(watcher))
        event('original_parent_paused_external_coordinator_ready',foreign_case=foreign,new_watcher=str(watcher_output))
        scheduled={case['name']:case for case in schedule['cases']}
        def worker(cpu_start):
            nonlocal external_committed
            try:
                if cpu_start==0 and foreign:
                    proof=wait_foreign(queue_root/foreign,snapshot)
                    event('foreign_current_completed',case=foreign,proof=proof)
                while not failed.is_set():
                    with mutex:
                        source=claim_case(pending,ownership,cpu_start)
                        if source is None:return
                    with common.POSTPROCESS_LOCK.open('a+') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);verify(output)
                        if digest(schedule_path)!=schedule_hash:raise ValueError('immutable schedule changed')
                        validate_schedule(read(schedule_path),original)
                        if common.verify_parent(parent,queue_root)['state'] not in ('T','t'):raise ValueError('original serial parent is no longer paused')
                    folder=queue_root/source['name']
                    if folder.exists():raise ValueError('refusing duplicate/existing output: '+str(folder))
                    command=list(scheduled[source['name']]['resource_only_commands'][str(cpu_start)])
                    separator=command.index('--');extras=[]
                    for path in (SELF,Path(common.__file__).resolve(),output/'transition_plan.json',schedule_path):extras+=['--input',str(path)]
                    command[separator:separator]=extras
                    with mutex:
                        if failed.is_set():
                            ownership[source['name']]=dict(phase='pending');return
                        external_committed=True
                    event('external_launch_committed',case=source['name'],cpu_start=cpu_start,actual_command=command)
                    launched=['taskset','-c',','.join(map(str,plan['host_affinity'])),*command]
                    result=subprocess.run(launched,cwd=ROOT,stdout=subprocess.DEVNULL)
                    if result.returncode:raise ValueError('case wrapper failed: '+source['name'])
                    proof=verify_scientific_execution(folder,source,cpu_start)
                    with mutex:ownership[source['name']]=dict(phase='completed',cpu_start=cpu_start,proof=proof)
                    event('case_completed',case=source['name'])
            except Exception as error:
                failed.set()
                with mutex:
                    write(output/'failure.json',dict(execution_completed=False,utc=utc(),error=str(error),
                        action=rollback_action(external_committed),original_parent_paused=True,
                        original_parent=parent,controller=self_identity,ownership=ownership,
                        recovery='No new cases will be launched. Already-running children finish. After external commitment, do not CONT the old serial parent; preserve outputs and recover with a reviewed coordinator adopting clean end records.'))
                event('worker_failed',cpu_start=cpu_start,error=str(error));raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,start) for start in (0,8)]
            for future in futures:future.result()
        verify(output);all_proofs={}
        for source in original['cases']:
            owner=ownership.get(source['name']);start=owner['cpu_start'] if owner else 0
            all_proofs[source['name']]=verify_scientific_execution(queue_root/source['name'],source,start)
        certificate=dict(execution_completed=True,ended_utc=utc(),original_controller_completed=False,
            controller=self_identity,transition_plan_sha256=digest(output/'transition_plan.json'),
            schedule_sha256=schedule_hash,cases=all_proofs,ownership=ownership)
        write(output/'completion.json',certificate)
        write(queue_root/'queue_complete.json',dict(execution_completed=True,original_controller_completed=False,
            controller='audited_external_parallel_resume',external_completion=str(output/'completion.json'),
            external_completion_sha256=digest(output/'completion.json'),cases=list(all_proofs)))
        event('all_cases_certified_waiting_watcher_ack')
        # Both watchers observe the honest completion pointer before original
        # session retirement, avoiding a false 'controller vanished' transition.
        statuses=[Path(plan['original_watcher_status']),watcher_output/'watcher_status.json']
        deadline=time.monotonic()+120
        def acknowledged(path):
            try:return read(path).get('queue_complete') is True
            except (FileNotFoundError,json.JSONDecodeError):return False
        while not all(acknowledged(path) for path in statuses):
            if time.monotonic()>deadline:
                event('complete_retirement_pending_watcher_ack',recovery='After watcher completion acknowledgement, retire only the recorded original PID/start identity.');return
            time.sleep(1)
        common.verify_parent(parent,queue_root);os.kill(parent['pid'],signal.SIGKILL);retired=True
        event('complete_original_parent_retired')
    except Exception as error:
        action=rollback_action(external_committed)
        if stopped and not external_committed:
            common.verify_parent(parent,queue_root);os.kill(parent['pid'],signal.SIGCONT);stopped=False
        failure=dict(execution_completed=False,utc=utc(),error=str(error),action=action,
            original_parent_paused=stopped and not retired,original_parent=parent,controller=self_identity,ownership=ownership,
            recovery='Do not CONT the original serial controller after external launch commitment. Preserve all case directories. Verify clean end records and review a new coordinator that adopts completed/owned cases; never overwrite or blindly rerun an existing case.')
        write(output/'failure.json',failure);event('failed',failure=failure)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--queue-root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if args.mode=='prepare':prepare(args.queue_root.resolve(),output)
    else:
        os.sched_setaffinity(0,{20}) # coordinator/identity work only, no inherited hard AS limit
        run(output)
if __name__=='__main__':main()
