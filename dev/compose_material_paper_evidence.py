#!/usr/bin/env python3
"""Compose real audited stage directories for paper exports; never simulate.

This is a derived evidence view, not a numerical production run. It preserves
original failure records and symlinks stage directories without changing their
execution/end records. Plan-only unless --execute is explicitly requested.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import resource

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'diffmuontomoLVD'
BOOT_HASHES={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__).resolve(),PAPER/'revision/stage_result_parsers.py',PAPER/'revision/build_surrogate_stability_report.py')}
sys.path.insert(0,str(PAPER/'revision'))
from stage_result_parsers import completed_stage
from build_surrogate_stability_report import audit
if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in BOOT_HASHES.items()):
    raise RuntimeError('composition helper changed during import')
STAGES=('topo','main','detection','contrast','sweep')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def composition_plan(original,main,remaining):
    sources={'topo':original/'stages/topo','main':main,
             **{stage:remaining/'stages'/stage for stage in STAGES[2:]}}
    original_record=read(original/'execution.json');remaining_record=read(remaining/'execution.json')
    main_record=read(main/'execution.json') # actual wrapper command, before tail ends
    records={stage:(read(path/'execution.json') if (path/'execution.json').exists() else remaining_record)
             for stage,path in sources.items()}
    commands={stage:(record['commands'][stage] if 'commands' in record else record['command'])
              for stage,record in records.items()}
    runtime=original_record['julia_runtime']
    identity=lambda record:{key:val for key,val in record['source_sha256'].items()
        if key.startswith('src/') or key in ('Project.toml','Manifest.toml','examples/lvd_tomography.jl')}
    for stage,record in records.items():
        if record['julia_runtime']!=runtime or identity(record)!=identity(original_record):
            raise ValueError('composed stages differ in scientific core/runtime: '+stage)
    artifact=original_record['correction_artifact']
    artifact_digest=original_record['source_sha256'].get(artifact)
    if not artifact_digest:raise ValueError('original correction artifact lacks source binding')
    recovery=read(main/'recovery_provenance.json')
    for stage,record in records.items():
        if stage=='topo':continue
        digest=record['source_sha256'].get(artifact)
        if stage=='main' and digest is None:digest=recovery['original_source_sha256'].get(artifact)
        if digest!=artifact_digest:raise ValueError('composed stage correction identities differ')
    if remaining_record.get('correction_artifact')!=artifact:raise ValueError('remaining artifact path differs')
    bound_roots={str(path):sha(path) for path in (original/'execution.json',main/'execution.json',remaining/'execution.json')}
    manifest=dict(schema_version=1,scope='paper_evidence_composition_not_numerical_run',
        original_root=str(original),main_recovery=str(main),remaining_root=str(remaining),
        source_root_execution_sha256=bound_roots,
        stage_sources={stage:dict(path=str(path),actual_command=commands[stage]) for stage,path in sources.items()},
        original_failed_main_end_sha256=sha(original/'stages/main/end_status.json'),
        original_failed_main_execution_completed=False,
        recovery_provenance_path=str(main/'recovery_provenance.json'),recovery_provenance_sha256=sha(main/'recovery_provenance.json'),
        helper_sha256=BOOT_HASHES,review_required=True)
    return manifest,sources,commands,original_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--main-recovery',type=Path,required=True)
    p.add_argument('--remaining-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--column-domain',type=Path)
    p.add_argument('--quadrature',type=Path)
    p.add_argument('--figure-helper',type=Path,default=PAPER/'make_figures.py')
    p.add_argument('--poll-seconds',type=float,default=30)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();output=a.output.resolve()
    available=set(os.sched_getaffinity(0))
    if 20 not in available:p.error('CPU20 must be available to the controller')
    os.sched_setaffinity(0,{20})
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    if output.exists():p.error('fresh derived composition root required')
    if a.poll_seconds<1:p.error('poll interval must be at least one second')
    original=a.original_root.resolve(strict=True);main_root=a.main_recovery.resolve(strict=True);remaining=a.remaining_root.resolve(strict=True)
    manifest,sources,commands,original_record=composition_plan(original,main_root,remaining)
    if not a.execute:print(json.dumps(manifest,indent=2));return 0
    output.mkdir(parents=True);(output/'stages').mkdir();(output/'controls').mkdir()
    write(output/'composition_manifest.json',manifest)
    for stage,path in sources.items():(output/'stages'/stage).symlink_to(path,target_is_directory=True)
    for name,path in [('column_domain',a.column_domain),('final_field_quadrature',a.quadrature)]:
        if path:(output/'controls'/name).symlink_to(path.resolve(),target_is_directory=True)
    record=dict(started_utc=utc(),scope=manifest['scope'],orchestrator_pid=os.getpid(),commands=commands,
        source_sha256=original_record['source_sha256'],julia_runtime=original_record['julia_runtime'],
        physics_dump=original_record['physics_dump'],correction_artifact=original_record['correction_artifact'],
        configuration=original_record['configuration'],composition_manifest_sha256=sha(output/'composition_manifest.json'),
        numerical_run=False,review_required=True)
    write(output/'execution.json',record)
    state=dict(started_utc=utc(),scope=manifest['scope'],completed_stages=[],stage_end_sha256={},
        failed_stages={},postprocessing={},execution_completed=False,numerical_run=False,review_required=True)
    write(output/'status.json',state)
    try:
        while len(state['completed_stages'])+len(state['failed_stages'])<len(STAGES):
            if sha(manifest['recovery_provenance_path'])!=manifest['recovery_provenance_sha256']:raise RuntimeError('recovery provenance changed')
            for name,digest in manifest['helper_sha256'].items():
                if sha(name)!=digest:raise RuntimeError('composition parser/auditor/helper changed: '+name)
            for name,digest in manifest['source_root_execution_sha256'].items():
                if sha(name)!=digest:raise RuntimeError('composition source plan changed: '+name)
            for stage,path in sources.items():
                if stage in state['completed_stages'] or stage in state['failed_stages'] or not (path/'end_status.json').exists():continue
                try:
                    checked=audit(path)
                    execution=read(path/'execution.json')
                    if execution['command']!=commands[stage] or execution['julia_runtime']!=record['julia_runtime']:
                        raise ValueError('actual completed wrapper differs from composed plan')
                    parsed=completed_stage(path,stage)
                    if stage=='main' and not parsed.get('recovery'):raise ValueError('composed main lacks audited recovery provenance')
                except Exception as error:
                    state['failed_stages'][stage]=dict(error=str(error),ended_utc=utc());write(output/'status.json',state);continue
                state['completed_stages'].append(stage);state['stage_end_sha256'][stage]=checked['end_status_sha256']
                write(output/'status.json',state)
                command=[sys.executable,str(ROOT/'dev/run_bounded.py'),'--cpus','1','--memory-gib','2','--','taskset','-c','20',
                    sys.executable,str(ROOT/'dev/postprocess_material_production.py'),'--root',str(output),'--stage',stage,
                    '--figure-helper',str(a.figure_helper.resolve())]
                with (output/(stage+'_postprocess.log')).open('wb') as log:
                    code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:os.sched_setaffinity(0,available)).returncode
                state['postprocessing'][stage]=dict(returncode=code,ended_utc=utc());write(output/'status.json',state)
            if len(state['completed_stages'])+len(state['failed_stages'])<len(STAGES):time.sleep(a.poll_seconds)
    except Exception as error:
        state['controller_failure']=str(error)
    state.update(ended_utc=utc(),execution_completed=len(state['completed_stages'])==len(STAGES) and not state['failed_stages'] and not state.get('controller_failure'),
        candidate_exports_complete=not state.get('controller_failure') and not state['failed_stages'] and len(state['postprocessing'])==len(STAGES) and all(item['returncode']==0 for item in state['postprocessing'].values()))
    write(output/'status.json',state);write(output/'end_status.json',state)
    return 0 if state['candidate_exports_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
