#!/usr/bin/env python3
"""Reviewed orchestration-only handoff for the frozen terrain queue.
Prepare is read-only toward the active queue. Transition stops only the verified
original Python parent, waits for its existing child to finish cleanly, then
reuses the same tmux pane/session for a two-pool resume controller.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

from robustness_run_identity import julia_runtime

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
TEST=SELF.with_name('test_resume_terrain_stress.py')
POSTPROCESS_LOCK=Path('/mnt/big_data/agent-builds/surrogate-stability-20261024/cpu20-python-postprocess.lock')
SESSION='terrain_stress_20261025'
PANE=SESSION+':0.0'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def read_retry(path,timeout=5.):
    deadline=time.monotonic()+timeout
    while True:
        try:return read(path)
        except (json.JSONDecodeError,FileNotFoundError):
            if time.monotonic()>=deadline:raise
            time.sleep(.1)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def tmux(*args):return subprocess.check_output(['tmux',*args],text=True).strip()

def process_identity(pid):
    base=Path('/proc')/str(pid)
    fields=(base/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],ppid=int(fields[1]),
        command=[x.decode() for x in (base/'cmdline').read_bytes().split(b'\0') if x],cwd=str((base/'cwd').resolve()))

def verify_parent(expected,queue_root):
    current=process_identity(expected['pid'])
    if current['start_ticks']!=expected['start_ticks'] or current['command']!=expected['command']:
        raise ValueError('queue parent PID identity changed')
    command=current['command'];original=str(ROOT/'dev/run_terrain_stress.py')
    if len(command)!=5 or command[1]!=original or '--execute' not in command or '--output' not in command:
        raise ValueError('target is not the original terrain queue parent')
    if Path(command[command.index('--output')+1]).resolve()!=queue_root.resolve():
        raise ValueError('queue parent targets another output')
    return current

def verify_context(plan):
    for name,expected in plan['source_and_input_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if not path.is_file() or digest(path)!=expected:raise ValueError('frozen context changed: '+str(path))
    executable=plan['cases'][0]['command'][plan['cases'][0]['command'].index('--julia')+1]
    if julia_runtime(executable)!=plan['julia_runtime']:raise ValueError('frozen Julia runtime changed')

def audit_case(folder):
    path=folder/'end_status.json'
    if not path.is_file():return None
    end=read_retry(path)
    if end.get('returncode')!=0 or any(end.get(key) is not True for key in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise ValueError('unclean case: '+str(folder))
    for name,expected in end['output_sha256'].items():
        item=folder/name
        if not item.is_file() or digest(item)!=expected:raise ValueError('completed case output changed: '+str(item))
    return dict(end_status_sha256=digest(path),output_sha256=end['output_sha256'])

def pending_cases(queue_root,plan):
    completed={};pending=[];active=[]
    for case in plan['cases']:
        folder=queue_root/case['name'];proof=audit_case(folder)
        if proof is not None:completed[case['name']]=proof
        elif folder.exists():active.append(case['name'])
        else:pending.append(case)
    return completed,pending,active

def resource_command(command,cpu_start):
    # Change only the outer audit launcher's CPU pool; numerical Julia argv is exact.
    result=list(command);separator=result.index('--')
    position=result.index('--cpu-start')
    if position>=separator or result[position+1]!='0' or cpu_start not in (0,8):
        raise ValueError('unexpected original resource command')
    result[position+1]=str(cpu_start)
    if result[separator:]!=command[separator:]:raise AssertionError('numerical command changed')
    return result

def wait_current_case(folder,parent,output,manifest_timeout=120.):
    deadline=time.monotonic()+manifest_timeout
    while not (folder/'pid.json').exists() and not (folder/'end_status.json').exists():
        if time.monotonic()>=deadline:raise ValueError('current wrapper produced no PID/end manifest before timeout')
        time.sleep(.2)
    if (folder/'end_status.json').exists():return audit_case(folder)
    pid_record=read_retry(folder/'pid.json',timeout=max(0.,deadline-time.monotonic()))
    wrapper=process_identity(pid_record['launcher_pid'])
    if wrapper['ppid']!=parent['pid']:raise ValueError('in-flight wrapper is not a child of verified parent')
    try:child=process_identity(pid_record['pid'])
    except FileNotFoundError:child=dict(pid=pid_record['pid'],already_exited_before_snapshot=True)
    write(output/'inflight_identity.json',dict(case=folder.name,wrapper=wrapper,child=child))
    while True:
        if (folder/'end_status.json').exists():return audit_case(folder)
        try:current=process_identity(wrapper['pid'])
        except FileNotFoundError:raise ValueError('current wrapper disappeared without an end record')
        if current['start_ticks']!=wrapper['start_ticks'] or current['state'] in ('Z','X','x'):
            raise ValueError('current wrapper died or PID changed without an end record')
        time.sleep(1)


def validate_schedule(schedule,plan,queue_root):
    originals={case['name']:case for case in plan['cases']}
    if len(originals)!=len(plan['cases']):raise ValueError('duplicate original case names')
    completed=schedule['completed_before_resume']
    if not set(completed)<=set(originals):raise ValueError('unknown completed case name')
    for name,proof in completed.items():
        if audit_case(queue_root/name)!=proof:raise ValueError('pre-resume completed case binding changed')
    expected=[case for case in plan['cases'] if case['name'] not in completed]
    actual=schedule['cases']
    if [case['name'] for case in actual]!=[case['name'] for case in expected]:
        raise ValueError('schedule names/order differ from unique pending original cases')
    for index,(case,original) in enumerate(zip(actual,expected)):
        start=(0,8)[index%2]
        command=original['command'];separator=command.index('--')
        if command[command.index('--cpus')+1]!='8' or command[command.index('--memory-gib')+1]!='16':
            raise ValueError('original case resource budget differs')
        if case['cpu_start']!=start or case['original_command']!=command or case['resource_only_command']!=resource_command(command,start):
            raise ValueError('schedule contains unapproved command/resource changes')
        if case['resource_only_command'][separator:]!=command[separator:]:
            raise ValueError('numerical Julia command changed')
    return actual


def prepare(queue_root,output):
    if output.exists():raise ValueError('fresh transition output required')
    plan=read(queue_root/'preregistered_plan.json');verify_context(plan)
    parent=process_identity(int(tmux('display-message','-p','-t',PANE,'#{pane_pid}')))
    verify_parent(parent,queue_root)
    completed,pending,active=pending_cases(queue_root,plan)
    if len(active)>1:raise ValueError('original serial queue has multiple active cases')
    output.mkdir(parents=True)
    transition=dict(scope='Resource/orchestration-only resume; original scientific cases, starts, budgets and within-job ordering unchanged.',
        prepared_utc=utc(),queue_root=str(queue_root),original_plan_sha256=digest(queue_root/'preregistered_plan.json'),
        original_parent=parent,original_session=SESSION,original_pane=PANE,
        original_remain_on_exit=tmux('show-options','-A','-p','-v','-t',PANE,'remain-on-exit') or 'off',
        host_affinity=sorted(os.sched_getaffinity(0)),
        completed_at_preparation=completed,pending_at_preparation=[case['name'] for case in pending],active_at_preparation=active,
        pool_cpu_starts=[0,8],cpus_per_job=8,address_space_gib_per_job=16,
        context_sha256={str(p):digest(p) for p in (SELF,TEST,ROOT/'dev/run_revision_job.py')},
        completion='External resume completion certificate plus an explicit compatibility pointer; original controller is not represented as having completed.')
    write(output/'transition_plan.json',transition)
    print(output/'transition_plan.json')

def verify_transition(output):
    transition=read(output/'transition_plan.json');queue_root=Path(transition['queue_root'])
    for name,value in transition['context_sha256'].items():
        if not Path(name).is_file() or digest(Path(name))!=value:raise ValueError('transition helper/context changed: '+name)
    if digest(queue_root/'preregistered_plan.json')!=transition['original_plan_sha256']:raise ValueError('original plan changed')
    plan=read(queue_root/'preregistered_plan.json');verify_context(plan)
    return transition,queue_root,plan

def transition(output):
    record,queue_root,plan=verify_transition(output)
    if (queue_root/'queue_complete.json').exists():raise ValueError('original queue already complete; no handoff needed')
    parent=verify_parent(record['original_parent'],queue_root)
    stopped=False;retired=False
    try:
        # No other process is signaled. start_ticks and argv prevent PID reuse.
        os.kill(parent['pid'],signal.SIGSTOP);stopped=True
        while verify_parent(parent,queue_root)['state'] not in ('T','t'):time.sleep(.1)
        completed,pending,active=pending_cases(queue_root,plan)
        write(output/'transition_state.json',dict(phase='parent_stopped_current_child_running',updated_utc=utc(),
            stopped_parent=parent,completed=completed,active=active))
        if len(active)>1:raise ValueError('multiple in-flight cases after stopping serial parent')
        if active:
            wait_current_case(queue_root/active[0],parent,output)
        completed,pending,active=pending_cases(queue_root,plan)
        if active:raise ValueError('in-flight case remained after its end audit')
        verify_context(plan);verify_parent(parent,queue_root)
        if not pending:
            os.kill(parent['pid'],signal.SIGCONT);stopped=False
            write(output/'transition_state.json',dict(phase='original_queue_allowed_to_finish',updated_utc=utc()))
            return
        schedule=[]
        for index,case in enumerate(pending):
            start=record['pool_cpu_starts'][index%2]
            schedule.append(dict(name=case['name'],cpu_start=start,original_command=case['command'],
                resource_only_command=resource_command(case['command'],start)))
        write(output/'execution_schedule.json',dict(created_utc=utc(),completed_before_resume=completed,cases=schedule))
        (output/'original_queue.log').write_bytes((queue_root/'queue.log').read_bytes())
        # keep the same tmux session alive even after the retired parent exits
        tmux('set-option','-p','-t',PANE,'remain-on-exit','on')
        verify_parent(parent,queue_root);os.kill(parent['pid'],signal.SIGKILL);retired=True
        command=[sys.executable,str(SELF),'resume','--output',str(output)]
        pane_command='exec '+shlex.join(command)+' > '+shlex.quote(str(output/'resume.log'))+' 2>&1'
        deadline=time.monotonic()+10
        while tmux('display-message','-p','-t',PANE,'#{pane_dead}')!='1':
            if time.monotonic()>deadline:raise ValueError('retired pane did not become dead; refusing forced respawn')
            time.sleep(.1)
        # No -k: refuse to kill any unexpected replacement process in this pane.
        tmux('respawn-pane','-t',PANE,pane_command)
        write(output/'transition_state.json',dict(phase='handed_off_to_resume',updated_utc=utc(),
            retired_parent=parent,resume_pane_pid=int(tmux('display-message','-p','-t',PANE,'#{pane_pid}')),
            preserved_original_plan_sha256=record['original_plan_sha256']))
    except Exception as error:
        rolled_back=False
        if stopped and not retired:
            verify_parent(parent,queue_root);os.kill(parent['pid'],signal.SIGCONT);rolled_back=True
        write(output/'transition_failure.json',dict(error=str(error),utc=utc(),
            original_parent_retired=retired,original_parent_resumed=rolled_back))
        raise

def resume(output):
    record,queue_root,plan=verify_transition(output);schedule_path=output/'execution_schedule.json'
    schedule=read(schedule_path);validate_schedule(schedule,plan,queue_root);schedule_sha256=digest(schedule_path)
    events=output/'resume_events.jsonl';lock=threading.Lock();failed=threading.Event()
    def event(value):
        with lock:
            with events.open('a') as stream:stream.write(json.dumps(dict(utc=utc(),**value))+'\n')
    def worker(cpu_start):
        for case in schedule['cases']:
            if case['cpu_start']!=cpu_start or failed.is_set():continue
            try:
                with POSTPROCESS_LOCK.open('a+') as context_lock:
                    fcntl.flock(context_lock,fcntl.LOCK_EX)
                    verify_transition(output)
                    if digest(schedule_path)!=schedule_sha256:raise ValueError('frozen execution schedule changed')
                    validate_schedule(read(schedule_path),plan,queue_root)
                folder=queue_root/case['name']
                if folder.exists():raise ValueError('pending output already exists; refusing duplicate: '+str(folder))
                command=list(case['resource_only_command']);separator=command.index('--')
                additions=[]
                for path in (SELF,output/'transition_plan.json',output/'execution_schedule.json'):
                    additions+=['--input',str(path)]
                command[separator:separator]=additions
                event(dict(phase='starting',case=case['name'],cpu_start=cpu_start,actual_command=command))
                # The lightweight wrapper must see the original host affinity to
                # apply its 30% budget check; it then pins Julia to the exact pool.
                launch=['taskset','-c',','.join(map(str,record['host_affinity'])),*command]
                result=subprocess.run(launch,cwd=ROOT,stdout=subprocess.DEVNULL)
                if result.returncode:raise ValueError('case wrapper failed: '+case['name'])
                proof=audit_case(folder)
                event(dict(phase='completed',case=case['name'],proof=proof))
            except Exception as error:
                failed.set();event(dict(phase='failed',case=case['name'],error=str(error)));raise
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,start) for start in record['pool_cpu_starts']]
            for future in futures:future.result()
        verify_transition(output);completed,pending,active=pending_cases(queue_root,plan)
        if pending or active or len(completed)!=len(plan['cases']):raise ValueError('incomplete resumed queue')
        certificate=dict(execution_completed=True,ended_utc=utc(),scope='Audited orchestration-only two-pool resume',
            original_controller_completed=False,original_plan_sha256=record['original_plan_sha256'],
            transition_plan_sha256=digest(output/'transition_plan.json'),schedule_sha256=digest(output/'execution_schedule.json'),cases=completed)
        write(output/'completion.json',certificate)
        write(queue_root/'queue_complete.json',dict(execution_completed=True,original_controller_completed=False,
            controller='audited_resource_only_resume',external_completion=str(output/'completion.json'),
            external_completion_sha256=digest(output/'completion.json'),cases=list(completed)))
        event(dict(phase='queue_complete',certificate=str(output/'completion.json')))
    except Exception as error:
        write(output/'failure.json',dict(execution_completed=False,ended_utc=utc(),error=str(error)))
        raise
    finally:
        tmux('set-option','-p','-t',PANE,'remain-on-exit',record['original_remain_on_exit'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','transition','resume'))
    parser.add_argument('--queue-root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if args.mode=='prepare':prepare(args.queue_root.resolve(),output)
    else:
        os.sched_setaffinity(0,{20}) # light coordination/identity queries share CPU20
        if args.mode=='transition':transition(output)
        else:resume(output)
if __name__=='__main__':main()
