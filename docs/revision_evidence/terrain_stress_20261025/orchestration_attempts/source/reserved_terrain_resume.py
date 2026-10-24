#!/usr/bin/env python3
"""No-signal handoff using an atomic output reservation and one output relocation.
The original launcher refuses the reserved next output before Popen. The same
scientific case runs under that unchanged launcher at a fresh alternate path.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import threading
import time

import resume_terrain_stress as common
import parallel_resume_terrain_stress as observed

ROOT=common.ROOT;SELF=Path(__file__).resolve();TEST=SELF.with_name('test_reserved_terrain_resume.py')
SESSION='terrain_reserved_resume_20261025';WATCH_SESSION='terrain_reserved_review_20261025'

def read(path):return common.read(path)
def write(path,value):common.write(path,value)
def digest(path):return common.digest(path)
def utc():return common.utc()

def relocate_command(command,alternate):
    result=common.resource_command(command,8);separator=result.index('--')
    outer=result.index('--output');inner=result.index('--output-dir',separator)
    result[outer+1]=str(alternate);result[inner+1]=str(alternate/'data')
    expected=list(command);expected[expected.index('--cpu-start')+1]='8'
    expected[outer+1]=str(alternate);expected[inner+1]=str(alternate/'data')
    if result!=expected:raise ValueError('unapproved command difference')
    return result

def reserve(folder,marker_payload):
    folder.mkdir() # atomic winner; an existing scientific output is never touched
    marker=folder/'orchestration_guard.json';marker.write_text(json.dumps(marker_payload,indent=2)+'\n')
    info=folder.stat()
    return dict(path=str(folder),device=info.st_dev,inode=info.st_ino,marker_sha256=digest(marker),marker_payload=marker_payload)

def verify_guard(identity):
    folder=Path(identity['path']);info=folder.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_dev!=identity['device'] or info.st_ino!=identity['inode']:
        raise ValueError('reserved directory identity changed')
    if sorted(item.name for item in folder.iterdir())!=['orchestration_guard.json']:
        raise ValueError('guard contains unknown/scientific files; preserve everything')
    marker=folder/'orchestration_guard.json'
    if not stat.S_ISREG(marker.lstat().st_mode) or digest(marker)!=identity['marker_sha256']:
        raise ValueError('owned guard marker changed')

def release_and_alias(identity,alternate):
    verify_guard(identity);folder=Path(identity['path'])
    (folder/'orchestration_guard.json').unlink();folder.rmdir() # exact own files only
    relative=os.path.relpath(alternate,folder.parent);folder.symlink_to(relative,target_is_directory=True)
    return dict(original_case_path=str(folder),relative_target=relative,actual_case_path=str(alternate),
        end_status_sha256=digest(alternate/'end_status.json'))

def wait_parent_exit(identity,timeout=120.):
    deadline=time.monotonic()+timeout
    while True:
        try:now=common.process_identity(identity['pid'])
        except FileNotFoundError:return
        if now['start_ticks']!=identity['start_ticks'] or now['state'] in ('Z','X','x'):return
        if time.monotonic()>=deadline:raise ValueError('original controller did not exit after reserved-output refusal')
        time.sleep(.2)

def verify_science(folder,source,actual_command,start):
    proof=common.audit_case(folder)
    if proof is None:raise ValueError('missing clean scientific end')
    execution=read(folder/'execution.json')
    if execution['command']!=actual_command[actual_command.index('--')+1:] or execution['cpu_pool']!=list(range(start,start+8)) or execution['address_space_bytes']!=16*1024**3:
        raise ValueError('actual science/resource command differs from reviewed command')
    return proof

def prepare(queue_root,output):
    if output.exists():raise ValueError('fresh preparation required')
    original=read(queue_root/'preregistered_plan.json');common.verify_context(original)
    parent=common.process_identity(int(common.tmux('display-message','-p','-t',common.PANE,'#{pane_pid}')))
    common.verify_parent(parent,queue_root)
    completed,pending,active=common.pending_cases(queue_root,original)
    if len(active)>1 or not pending or pending[0]['forward_only']:raise ValueError('no next unstarted refit worth relocating')
    guard=pending[0];alternate=output/'relocated_case'
    dependencies=[SELF,TEST,Path(common.__file__).resolve(),Path(observed.__file__).resolve(),ROOT/'dev/run_revision_job.py',ROOT/'dev/watch_terrain_stress.py',ROOT/'dev/summarize_terrain_stress.py']
    output.mkdir(parents=True)
    write(output/'transition_plan.json',dict(prepared_utc=utc(),scope='No-signal output-reservation handoff; only CPU pool and output paths change for one case.',
        queue_root=str(queue_root),original_parent=parent,original_plan_sha256=digest(queue_root/'preregistered_plan.json'),
        context_sha256={str(path):digest(path) for path in dependencies},host_affinity=sorted(os.sched_getaffinity(0)),
        completed_at_preparation=completed,active_at_preparation=active,guard_case=guard,
        alternate_case_path=str(alternate),relocated_command=relocate_command(guard['command'],alternate),
        external_session=SESSION,watch_session=WATCH_SESSION))
    print(output/'transition_plan.json')

def verify(output):
    plan=read(output/'transition_plan.json');root=Path(plan['queue_root'])
    for name,value in plan['context_sha256'].items():
        if not Path(name).is_file() or digest(Path(name))!=value:raise ValueError('handoff helper/context changed: '+name)
    if digest(root/'preregistered_plan.json')!=plan['original_plan_sha256']:raise ValueError('original scientific plan changed')
    original=read(root/'preregistered_plan.json');common.verify_context(original)
    source=next(case for case in original['cases'] if case['name']==plan['guard_case']['name'])
    if source!=plan['guard_case'] or plan['relocated_command']!=relocate_command(source['command'],Path(plan['alternate_case_path'])):
        raise ValueError('case or permitted relocation command changed')
    return plan,root,original

def run(output):
    plan,root,original=verify(output);parent=common.verify_parent(plan['original_parent'],root)
    if int(common.tmux('display-message','-p','-t',SESSION+':0.0','#{pane_pid}'))!=os.getpid():raise ValueError('wrong coordinator session')
    completed,pending,active=common.pending_cases(root,original)
    if len(active)>1 or not pending or pending[0]!=plan['guard_case']:raise ValueError('queue advanced during review; no guard/action taken')
    foreign=active[0] if active else None
    snapshot=observed.foreign_snapshot(root/foreign,parent) if foreign else None
    guard_case=plan['guard_case'];guard_folder=root/guard_case['name'];alternate=Path(plan['alternate_case_path'])
    self_identity=common.process_identity(os.getpid());mutex=threading.RLock();failed=threading.Event();guard_identity=None
    def event(phase,**fields):
        with mutex:
            with (output/'events.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),phase=phase,**fields))+'\n')
            write(output/'coordinator_status.json',dict(updated_utc=utc(),phase=phase,controller=self_identity,**fields))
    def record_failure(error):
        write(output/'failure.json',dict(execution_completed=False,utc=utc(),error=str(error),guard=guard_identity,
            original_parent=parent,no_process_signals_sent=True,
            recovery='Preserve all case/alternate directories and the owned guard. Adopt only clean end records. Remove only the unchanged marker/empty directory after original parent exit; never overwrite a scientific output or rerun an already-owned case.'))
    def launch(source,command,start):
        with common.POSTPROCESS_LOCK.open('a+') as context_lock:
            import fcntl
            fcntl.flock(context_lock,fcntl.LOCK_EX);verify(output)
        separator=command.index('--');actual=list(command);inputs=[]
        for path in (SELF,Path(common.__file__).resolve(),Path(observed.__file__).resolve(),output/'transition_plan.json',output/'guard_identity.json'):
            inputs+=['--input',str(path)]
        actual[separator:separator]=inputs
        event('launching',case=source['name'],cpu_start=start,actual_command=actual)
        result=subprocess.run(['taskset','-c',','.join(map(str,plan['host_affinity'])),*actual],cwd=ROOT,stdout=subprocess.DEVNULL)
        if result.returncode:raise ValueError('case wrapper failed: '+source['name'])
        folder=Path(actual[actual.index('--output')+1]);proof=verify_science(folder,source,actual,start)
        event('case_completed',case=source['name'],proof=proof);return proof
    try:
        # Existence check and atomic mkdir in the ORIGINAL unchanged launcher are
        # both before child Popen. This directory remains present until old exit.
        guard_identity=reserve(guard_folder,dict(owner=self_identity,transition_plan_sha256=digest(output/'transition_plan.json'),
            scope='Owned orchestration-only reservation; no numerical data.'))
        write(output/'guard_identity.json',guard_identity)
        remaining=[case for case in pending if case['name']!=guard_case['name']]
        schedule=dict(completed_before_handoff=completed,foreign_case=foreign,foreign_identity=snapshot,
            relocated=guard_case,remaining=remaining,original_parent=parent)
        write(output/'execution_schedule.json',schedule)
        watcher_output=output/'review_candidates'
        watcher=[sys.executable,str(ROOT/'dev/watch_terrain_stress.py'),str(root),'--output',str(watcher_output),'--queue-session',SESSION]
        common.tmux('new-session','-d','-s',WATCH_SESSION,'exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(watcher))
        event('owned_output_reserved',guard=guard_identity,foreign_case=foreign)
        def secondary():
            try:return launch(guard_case,list(plan['relocated_command']),8)
            except Exception as error:failed.set();record_failure(error);event('secondary_failed',error=str(error));raise
        def primary():
            try:
                if foreign:observed.wait_foreign(root/foreign,snapshot)
                wait_parent_exit(parent)
                verify_guard(guard_identity)
                (output/'original_queue.log').write_bytes((root/'queue.log').read_bytes())
                event('original_controller_exited_guard_intact')
                for source in remaining:
                    if failed.is_set():return
                    if (root/source['name']).exists():raise ValueError('refusing duplicate remaining output')
                    launch(source,list(source['command']),0)
            except Exception as error:failed.set();record_failure(error);event('primary_failed',error=str(error));raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            second=pool.submit(secondary);first=pool.submit(primary)
            first.result();second.result()
        verify(output);verify_guard(guard_identity);wait_parent_exit(parent,timeout=0.)
        alias=release_and_alias(guard_identity,alternate);write(output/'output_relocation.json',alias)
        proofs={}
        for source in original['cases']:
            folder=root/source['name']
            if source['name']==guard_case['name']:proofs[source['name']]=verify_science(folder,source,plan['relocated_command'],8)
            else:proofs[source['name']]=verify_science(folder,source,source['command'],0)
        certificate=dict(execution_completed=True,ended_utc=utc(),original_controller_completed=False,
            scope='Reviewed output-path and CPU-pool relocation only; original controller expectedly refused its owned reservation.',
            transition_plan_sha256=digest(output/'transition_plan.json'),output_relocation=alias,cases=proofs)
        write(output/'completion.json',certificate)
        write(root/'queue_complete.json',dict(execution_completed=True,original_controller_completed=False,
            controller='audited_reserved_output_resume',external_completion=str(output/'completion.json'),external_completion_sha256=digest(output/'completion.json'),cases=list(proofs)))
        event('all_cases_certified',completion=str(output/'completion.json'))
    except Exception as error:
        record_failure(error)
        event('failed',error=str(error));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--queue-root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if args.mode=='prepare':prepare(args.queue_root.resolve(),output)
    else:os.sched_setaffinity(0,{20});run(output)
if __name__=='__main__':main()
