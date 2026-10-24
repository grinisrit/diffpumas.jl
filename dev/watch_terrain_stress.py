#!/usr/bin/env python3
"""Observe terrain end records and prepare fresh, hash-verified review candidates.
No manuscript installation or automatic scientific conclusion. CPU20 and a shared
Python postprocess lock bound rendering; source/core calculations are untouched.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/'dev/summarize_terrain_stress.py'
LOCK=Path('/mnt/big_data/agent-builds/surrogate-stability-20261024/cpu20-python-postprocess.lock')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--queue-session',default='terrain_stress_20261025')
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    if output.exists():parser.error('fresh watcher output required')
    output.mkdir(parents=True);os.sched_setaffinity(0,{20})
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    expected={str(path):digest(path) for path in (Path(__file__).resolve(),SUMMARY,root/'preregistered_plan.json')}
    plan=json.loads((root/'preregistered_plan.json').read_text());seen=None;sequence=0
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    while True:
        changed=[name for name,value in expected.items() if not Path(name).is_file() or digest(Path(name))!=value]
        if changed:
            save(output/'watcher_status.json',dict(phase='failed_source_change',ended_utc=utc(),changed=changed));return 1
        snapshot={case['name']:digest(root/case['name']/'end_status.json')
            for case in plan['cases'] if (root/case['name']/'end_status.json').is_file()}
        complete=(root/'queue_complete.json').is_file()
        failed=[name for name in snapshot if json.loads((root/name/'end_status.json').read_text()).get('execution_completed') is not True]
        controller_alive=subprocess.run(['tmux','has-session','-t',args.queue_session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if not controller_alive and not complete:
            failed.append('queue controller exited without completion record')
        if snapshot and (snapshot!=seen or complete or failed):
            with LOCK.open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                destination=output/f'candidate_{sequence:03d}'
                command=[sys.executable,str(SUMMARY),str(root),'--allow-partial','--output-dir',str(destination),'--figures']
                with (output/f'candidate_{sequence:03d}.log').open('wb') as stream:
                    result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                status=dict(phase='candidate_ready' if result.returncode==0 else 'failed_postprocess',updated_utc=utc(),
                    candidate=str(destination),source_sha256=expected,end_record_sha256=snapshot,
                    queue_complete=complete,failed_cases=failed,returncode=result.returncode,
                    cpu_pool=[20],address_space_gib=2,shared_lock=str(LOCK))
                if result.returncode==0:
                    status['candidate_sha256']={p.name:digest(p) for p in destination.iterdir() if p.is_file()}
                save(output/'watcher_status.json',status)
                if result.returncode:return 1
            sequence+=1;seen=snapshot
        if complete or failed:
            return 0 if complete and not failed else 1
        time.sleep(30)
if __name__=='__main__':raise SystemExit(main())
