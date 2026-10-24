#!/usr/bin/env python3
"""Recover only the unexecuted report/MC tail of an audited failed main stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from robustness_run_identity import julia_runtime
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'dev/recover_material_main.jl'
EXAMPLE=ROOT/'examples/lvd_tomography.jl'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit_failed(folder):
    execution=json.loads((folder/'execution.json').read_text())
    end=json.loads((folder/'end_status.json').read_text())
    if end.get('returncode')!=1 or end.get('execution_completed') is not False or not all(end.get(k) is True for k in ('source_unchanged','runtime_unchanged')):
        raise ValueError('expected source/runtime-clean failed original stage')
    if 'UndefVarError: `robust_model` not defined' not in (folder/'run.log').read_text():
        raise ValueError('unexpected original failure')
    for key,digest in end['output_sha256'].items():
        if sha(folder/key)!=digest:raise ValueError('changed failed output: '+key)
    for key,digest in execution['source_sha256'].items():
        path=Path(key);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=digest:raise ValueError('changed original scientific source/input: '+key)
    return execution,end

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed-stage',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--julia',required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--parity-only',action='store_true',help='stop before unchanged report/MC tail')
    a=p.parse_args();failed=a.failed_stage.resolve(strict=True);output=a.output_dir.resolve()
    execution,end=audit_failed(failed)
    if julia_runtime(a.julia)!=execution['julia_runtime']:
        raise ValueError('recovery Julia runtime differs from original stage')
    if output.exists():p.error('recovery data directory must be fresh')
    command=execution['command'];index=next(i for i,item in enumerate(command) if Path(item).name=='lvd_tomography.jl')
    args=command[index+1:]
    if args.count('--output-dir')!=1:p.error('ambiguous original output directory')
    args[args.index('--output-dir')+1]=str(output)
    source=EXAMPLE.read_text();tail=source[source.index('\n    open(joinpath(args.output_dir,"lvd_recursion_status.txt"),"w") do io'):]
    record=dict(schema_version=1,recovery_kind='restore_completed_fit_execute_unchanged_report_mc_tail',
        original_failed_stage=str(failed),original_execution_sha256=sha(failed/'execution.json'),
        original_end_status_sha256=sha(failed/'end_status.json'),original_run_log_sha256=sha(failed/'run.log'),original_returncode=1,
        original_outputs_sha256=end['output_sha256'],original_source_sha256=execution['source_sha256'],
        original_example_sha256=sha(EXAMPLE),recovery_helper_sha256=sha(HELPER),
        recovery_runner_sha256=sha(__file__),unchanged_tail_sha256=hashlib.sha256(tail.encode()).hexdigest(),
        synthetic_solver_recomputed=False,measured_optimizer_recomputed=False,
        mc_seed_protocol='Original tail unchanged: energy samples seed+777; per covered-position transport seed+900000+k; both common model and field unchanged',
        original_julia_runtime=execution['julia_runtime'],command=[a.julia,'--startup-file=no','--project='+str(ROOT),str(HELPER),*args],
        prepare_only=a.prepare_only,parity_only=a.parity_only,scientific_review_required=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    provenance=output.parent/'recovery_provenance.json'
    if provenance.exists():p.error('fresh recovery parent required')
    provenance.write_text(json.dumps(record,indent=2)+'\n')
    if a.prepare_only:return 0
    output.mkdir()
    shutil.copyfile(failed/'run.log',output.parent/'original_failed_run.log')
    for name,digest in end['output_sha256'].items():
        path=Path(name)
        if path.parts[0]=='data':
            target=output/Path(*path.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(failed/name,target)
    env=os.environ.copy();env['DIFFPUMAS_RECOVERY_FAILED_DATA']=str(failed/'data')
    env['DIFFPUMAS_RECOVERY_PARITY_ONLY']='1' if a.parity_only else '0'
    status=subprocess.run(record['command'],cwd=ROOT,env=env).returncode
    audit_failed(failed)
    for name,digest in end['output_sha256'].items():
        path=Path(name)
        if path.parts[0]=='data' and sha(output/Path(*path.parts[1:]))!=digest:
            raise ValueError('recovery changed completed original data: '+name)
    if sha(output.parent/'original_failed_run.log')!=record['original_run_log_sha256']:
        raise ValueError('original run log copy changed')
    if sha(HELPER)!=record['recovery_helper_sha256'] or sha(EXAMPLE)!=record['original_example_sha256']:
        raise ValueError('recovery implementation changed')
    return status
if __name__=='__main__':raise SystemExit(main())
