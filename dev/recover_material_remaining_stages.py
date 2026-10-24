#!/usr/bin/env python3
"""Replay independent remaining production stages in a fresh, audited root.

No numerical source is transformed. Original failed evidence is read-only. The
recorded detection/contrast/sweep commands change only their output directory;
resource allocation is an explicit execution variant. Default is plan-only.
This root has no completed main stage and cannot establish full paper completion.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from robustness_run_identity import julia_runtime

ROOT=Path(__file__).resolve().parents[1]
STAGES=('detection','contrast','sweep')
REQUIRED={'detection':'lvd_detection_benchmark.txt',
          'contrast':'lvd_detection_contrast_sweep.txt',
          'sweep':'lvd_reconstruction_sweep.txt'}
FLAGS={'detection':'--detection-benchmark','contrast':'--detection-contrast-sweep',
       'sweep':'--reconstruction-sweep'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def failed_parent(parent):
    execution=read(parent/'execution.json');end=read(parent/'stages/main/end_status.json')
    if end.get('execution_completed') is not False or end.get('returncode')==0:
        raise ValueError('parent main must have an explicit failed execution audit')
    if end.get('source_unchanged') is not True or end.get('runtime_unchanged') is not True or end.get('source_changes')!=[]:
        raise ValueError('failed parent changed source/runtime; cannot replay its scientific plan')
    if not end.get('output_sha256') or 'execution.json' not in end['output_sha256']:
        raise ValueError('failed parent output manifest missing')
    for name,digest in end['output_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(parent/'stages/main'/name)!=digest:
            raise ValueError('parent failed-stage evidence changed: '+name)
    example=ROOT/'examples/lvd_tomography.jl';source=example.read_text()
    # Each selected branch returns before the normal inverse/measured path.
    main_start=source.index('function main()')
    for name,condition in [('sweep','if args.reconstruction_sweep'),('contrast','if args.detection_contrast_sweep'),('detection','if args.detection_benchmark')]:
        start=source.index(condition,main_start);finish=source.index('\n    end',start)
        if 'return 0' not in source[start:finish] or start>source.index('    if args.papermatch_only',main_start):
            raise ValueError('independent stage early-return contract changed: '+name)
    return execution,end


def plan_commands(execution,output):
    plan={}
    for stage in STAGES:
        command=list(execution['commands'][stage])
        if command.count('--output-dir')!=1 or command.count(FLAGS[stage])!=1:
            raise ValueError('unexpected frozen scientific command: '+stage)
        if any(flag in command for flag in ('--papermatch','--papermatch-fit-only')):
            raise ValueError('remaining stage unexpectedly invokes measured fit')
        if any(flag in command for other,flag in FLAGS.items() if other!=stage):
            raise ValueError('ambiguous independent stage mode')
        command[command.index('--output-dir')+1]=str(output/'stages'/stage/'data')
        plan[stage]=command
    return plan


def audit_success(folder):
    end=read(folder/'end_status.json')
    if end.get('returncode')!=0 or not all(end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise RuntimeError('unclean recovered stage: '+str(folder))
    for name,digest in end['output_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(folder/name)!=digest:
            raise RuntimeError('recovered stage output changed: '+name)
    return end


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-start',type=int,default=4)
    p.add_argument('--cpus',type=int,default=12)
    p.add_argument('--memory-gib',type=float,default=28)
    p.add_argument('--figure-helper',type=Path,default=ROOT/'diffmuontomoLVD/make_figures.py')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();parent=a.failed_root.resolve(strict=True);output=a.output.resolve()
    if output.exists():p.error('fresh recovery root required; prior evidence is never overwritten')
    original,failed=failed_parent(parent);plan=plan_commands(original,output)
    expected=dict(original['source_sha256'])
    inputs=[Path(__file__).resolve(),parent/'execution.json',parent/'stages/main/end_status.json',
        ROOT/'dev/run_revision_job.py',Path(original['correction_artifact'])]
    expected.update({str(path):sha(path) for path in inputs})
    def unchanged():
        changed=[name for name,digest in expected.items()
            if not (Path(name) if Path(name).is_absolute() else ROOT/name).is_file()
            or sha(Path(name) if Path(name).is_absolute() else ROOT/name)!=digest]
        if changed:raise RuntimeError('frozen source/input changed: '+', '.join(changed))
    unchanged()
    runtime=original['julia_runtime'];julia=plan['detection'][0]
    if julia_runtime(julia)!=runtime:raise RuntimeError('parent Julia runtime identity changed')
    available=set(os.sched_getaffinity(0));pool=set(range(a.cpu_start,a.cpu_start+a.cpus))
    if a.cpus<1 or not pool<=available or not 0<a.memory_gib<=36:p.error('invalid allocated resource pool')
    identity=dict(started_utc=utc(),orchestrator_pid=os.getpid(),commands=plan,
        physics_dump=original['physics_dump'],correction_artifact=original['correction_artifact'],
        correction_metadata=original['correction_metadata'],configuration=original['configuration'],
        source_sha256=expected,julia_runtime=runtime,
        recovery=dict(parent_root=str(parent),parent_execution_sha256=sha(parent/'execution.json'),
            failed_main_end_sha256=sha(parent/'stages/main/end_status.json'),
            failed_main_execution_completed=False,scientific_command_changes=['output directory only'],
            independence='Dedicated early-return branches execute before run_paper_match; robust_model path unreachable',
            timing_scope='Resource variant: independent stages on explicitly recorded pool; no timing equivalence claim'),
        cpu_pool=sorted(pool),memory_gib=a.memory_gib,
        scope='Independent remaining stages only; failed main is not promoted or represented as complete')
    if not a.execute:print(json.dumps(identity,indent=2));return 0
    output.mkdir(parents=True);write(output/'execution.json',identity)
    state=dict(started_utc=utc(),current_stage=None,completed_stages=[],postprocessing={},execution_completed=False,
        missing_required_main=True,review_required=True)
    write(output/'status.json',state)
    os.sched_setaffinity(0,pool)
    failure=None
    try:
        for stage in STAGES:
            unchanged();state.update(current_stage=stage,stage_started_utc=utc());write(output/'status.json',state)
            folder=output/'stages'/stage
            command=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(folder),
                '--dump',original['physics_dump'],'--julia',julia,'--cpus',str(a.cpus),
                '--cpu-start',str(a.cpu_start),'--memory-gib',str(a.memory_gib)]
            for path in inputs:command+=['--input',str(path)]
            command+=['--',*plan[stage]]
            with (output/(stage+'_launcher.log')).open('wb') as stream:
                code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda:os.sched_setaffinity(0,available)).returncode
            if code:raise RuntimeError('stage wrapper failed: '+stage)
            audit_success(folder);unchanged()
            required=folder/'data'/REQUIRED[stage]
            if not required.is_file() or not required.stat().st_size:raise RuntimeError('required stage report absent: '+stage)
            if stage=='sweep' and '4 realizations/point;' not in required.read_text():
                raise RuntimeError('sweep realization count differs from original plan')
            state['completed_stages'].append(stage);write(output/'status.json',state)
            post=[sys.executable,str(ROOT/'dev/run_bounded.py'),'--cpus','1','--memory-gib','2','--',
                'taskset','-c','20',sys.executable,str(ROOT/'dev/postprocess_material_production.py'),
                '--root',str(output),'--stage',stage,'--figure-helper',str(a.figure_helper.resolve())]
            with (output/(stage+'_postprocess.log')).open('wb') as stream:
                code=subprocess.run(post,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda:os.sched_setaffinity(0,available)).returncode
            state['postprocessing'][stage]=dict(returncode=code,completed_utc=utc());write(output/'status.json',state)
        state['execution_completed']=True
    except Exception as error:failure=str(error)
    finally:
        try:unchanged();same=True
        except Exception as error:same=False;failure=failure or str(error)
        runtime_same=julia_runtime(julia)==runtime
        state.update(ended_utc=utc(),failure=failure,source_unchanged=same,runtime_unchanged=runtime_same,
            execution_completed=state['execution_completed'] and same and runtime_same,
            candidate_exports_complete=len(state['postprocessing'])==len(STAGES) and all(v['returncode']==0 for v in state['postprocessing'].values()))
        write(output/'status.json',state);write(output/'end_status.json',state)
    print(json.dumps(state),flush=True)
    return 0 if state['execution_completed'] and state['candidate_exports_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
