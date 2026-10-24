#!/usr/bin/env python3
"""Wait for clean recovered main, then audit its unchanged final-field q domain."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
from robustness_run_identity import julia_runtime
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(folder,current=False):
    end=json.loads((folder/'end_status.json').read_text())
    if end.get('returncode')!=0 or not all(end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise ValueError('main recovery did not finish cleanly')
    for name,digest in end['output_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('changed main output: '+name)
    if current:
        execution=json.loads((folder/'execution.json').read_text())
        for name,digest in execution['source_sha256'].items():
            path=Path(name);path=path if path.is_absolute() else ROOT/path
            if sha(path)!=digest:raise ValueError('main input changed: '+name)
    return end

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('main','output','artifact','dump'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--julia',required=True);p.add_argument('--poll-seconds',type=float,default=30)
    a=p.parse_args();main=a.main.resolve();out=a.output.resolve();artifact=a.artifact.resolve(strict=True);dump=a.dump.resolve(strict=True)
    full_affinity=set(os.sched_getaffinity(0));original_limit=resource.getrlimit(resource.RLIMIT_AS)
    os.sched_setaffinity(0,{0})
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,original_limit[1]))
    out.parent.mkdir(parents=True,exist_ok=True)
    statepath=out.parent/(out.name+'_watcher.json')
    lock=open(out.parent/(out.name+'_watcher.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if out.exists() or statepath.exists():p.error('fresh output/state required')
    inputs=[Path(__file__).resolve(),ROOT/'dev/audit_material_column_domain.jl',ROOT/'dev/validate_surrogate_v2.jl',ROOT/'examples/revision_suite.jl',ROOT/'dev/run_revision_job.py',ROOT/'dev/robustness_run_identity.py',artifact,dump,ROOT/'Project.toml',ROOT/'Manifest.toml',*sorted((ROOT/'src').glob('*.jl'))]
    frozen={str(path):sha(path) for path in inputs}
    state=dict(schema_version=1,main=str(main),output=str(out),phase='waiting_clean_main',source_sha256=frozen,revision_complete=False,watcher_cpu=[0],watcher_address_space_bytes=256*1024**2,child_cpu=[0],child_address_space_bytes=4*1024**3)
    def write():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath)
    write()
    try:
        while not (main/'end_status.json').exists():time.sleep(a.poll_seconds)
        end=audit(main,True)
        for path,digest in frozen.items():
            if sha(path)!=digest:raise ValueError('column watcher input changed: '+path)
        provenance=json.loads((main/'recovery_provenance.json').read_text())
        failed=Path(provenance['original_failed_stage'])
        if sha(failed/'end_status.json')!=provenance['original_end_status_sha256']:raise ValueError('original failed end changed')
        original=json.loads((failed/'end_status.json').read_text())
        for name,digest in original['output_sha256'].items():
            if sha(failed/name)!=digest:raise ValueError('original failed output changed')
            if name.startswith('data/') and end['output_sha256'].get(name)!=digest:raise ValueError('recovery changed completed field/data')
        cfg=json.loads((failed/'execution.json').read_text())['source_sha256']
        for physical in (artifact,dump):
            key=str(physical.relative_to(ROOT)) if physical.is_relative_to(ROOT) else str(physical)
            if cfg.get(str(physical),cfg.get(key))!=sha(physical):raise ValueError('physical input does not match original science')
        # Julia's runtime query maps shared libraries; use the authorized child
        # address-space cap briefly after the main allocation has been released.
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,original_limit[1]))
        try:runtime=julia_runtime(a.julia)
        finally:resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,original_limit[1]))
        if runtime!=json.loads((main/'execution.json').read_text())['julia_runtime'] or runtime!=json.loads((failed/'execution.json').read_text())['julia_runtime']:
            raise ValueError('column Julia runtime differs from original/recovery')
        field=main/'data/lvd_fit_field.csv';rays=main/'data/lvd_fit_mask.csv'
        cmd=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(out),'--dump',str(dump),'--julia',a.julia,'--cpus','1','--cpu-start','0','--memory-gib','4']
        bound=[*inputs,field,rays,main/'end_status.json',main/'execution.json',main/'recovery_provenance.json',failed/'end_status.json']
        for path in bound:cmd+=['--input',str(path)]
        cmd+=['--',a.julia,'--startup-file=no','--project=.',str(ROOT/'dev/audit_material_column_domain.jl'),'--fieldcsv',str(field),'--raycsv',str(rays),'--artifact',str(artifact),'--dump',str(dump),'--output-dir',str(out)]
        state.update(phase='running_column_audit',command=cmd,main_end_sha256=sha(main/'end_status.json'));write()
        with (out.parent/(out.name+'_launcher.log')).open('xb') as stream:
            status=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                preexec_fn=lambda:(os.sched_setaffinity(0,full_affinity),resource.setrlimit(resource.RLIMIT_AS,original_limit))).returncode
        if status:raise ValueError('column audit execution failed: '+str(status))
        audit(out);audit(main)
        state.update(phase='completed_column_audit',end_status_sha256=sha(out/'end_status.json'),review_required=True);write()
    except Exception as exc:
        state.update(phase='failed',error=str(exc));write();raise
if __name__=='__main__':main()
