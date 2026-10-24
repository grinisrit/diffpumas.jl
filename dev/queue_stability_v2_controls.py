#!/usr/bin/env python3
"""Durable sequential controls after an externally audited dry reference completes.

Does not modify runtime source. Nonstationary fits are retained and the queue
continues; execution, source, runtime or input-identity failures stop the queue.
Run inside tmux on the coordinator's allocated CPU pool.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from collect_stability_v2 import collect
from record_revision_run import record_sources

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def readjson(path):
    return json.loads(path.read_text())


def audit(folder):
    end=readjson(folder/'end_status.json')
    if not all(end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')) or end.get('returncode')!=0:
        raise RuntimeError(f'{folder}: unsuccessful execution or source/runtime audit')
    for name,digest in end['output_sha256'].items():
        if sha(folder/name)!=digest:
            raise RuntimeError(f'{folder}: output digest mismatch {name}')
    return end


def scale(folder):
    with (folder/'results/lvd_fit_predictions.csv').open() as stream:
        values=[float(row['normalization']) for row in csv.DictReader(stream)]
    if not values or not math.isfinite(values[0]) or values[0]<=0 or len(set(values))!=1:
        raise ValueError('baseline normalization must be constant, positive and finite')
    return values[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-reference',type=Path,required=True)
    p.add_argument('--corrected-artifact',type=Path,required=True)
    p.add_argument('--absolute-prior-weight',type=float,required=True)
    p.add_argument('--cpu-start',type=int,default=16)
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--memory-gib',type=float,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--report-builder',type=Path,default=ROOT/'diffmuontomoLVD/revision/build_surrogate_stability_report.py')
    a=p.parse_args()
    output=a.output.resolve();reference=a.dry_reference.resolve();corrected=a.corrected_artifact.resolve(strict=True)
    if output.exists():p.error('output exists; use a fresh queue root')
    if not math.isfinite(a.absolute_prior_weight) or a.absolute_prior_weight<0:p.error('absolute prior weight must be finite and nonnegative')
    execution=readjson(reference/'execution.json');base=execution['command']
    dry=Path(base[base.index('--correction-artifact')+1]).resolve(strict=True)
    dump=Path(base[base.index('--dump')+1]).resolve(strict=True)
    if tomllib.loads(dry.read_text())['csda']!=tomllib.loads(corrected.read_text())['csda']:
        p.error('dry and corrected artifacts must share exactly the dry correction')
    full_affinity=set(os.sched_getaffinity(0))
    control_affinity=set(range(a.cpu_start,a.cpu_start+a.cpus))
    if not control_affinity <= full_affinity:p.error('control pool unavailable')
    os.sched_setaffinity(0,control_affinity)
    resource.setrlimit(resource.RLIMIT_AS,(int(a.memory_gib*1024**3),int(a.memory_gib*1024**3)))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    output.mkdir(parents=True)
    inputs={str(path):sha(path) for path in (dry,corrected,Path(__file__).resolve(),
        ROOT/'dev/collect_stability_v2.py',ROOT/'dev/run_stability_v2.py')}
    frozen=record_sources(ROOT,dump);frozen.update(inputs)
    cases={'dry/start0':reference}
    plan=dict(created_utc=utc(),dry_reference=str(reference),corrected_artifact=str(corrected),
        absolute_prior_weight=a.absolute_prior_weight,cpus=list(range(a.cpu_start,a.cpu_start+a.cpus)),
        memory_gib=a.memory_gib,seed=a.seed,seed_scope='Midpoint quadrature and frozen calibration make these fit-only controls deterministic; seed differs from the pre-existing dry reference and is recorded, not treated as an independent replicate.',
        source_and_input_sha256=frozen,
        scenarios=['dry/start015','dry/start03','dry/scale_minus5pct','dry/scale_plus5pct','dry/maskInf',
            'corrected/start0','corrected/start015','corrected/start03','corrected/scale_minus5pct','corrected/scale_plus5pct','corrected/maskInf'])
    (output/'queue_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def event(**data):
        with (output/'queue_events.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),**data))+'\n')
    def state(**data):(output/'queue_state.json').write_text(json.dumps(dict(updated_utc=utc(),**data),indent=2)+'\n')
    def unchanged():
        changed=[key for key,digest in frozen.items() if sha(Path(key) if Path(key).is_absolute() else ROOT/key)!=digest]
        if changed:raise RuntimeError(f'Frozen suite source/input changed: {changed}')
    def report(index):
        reportfile=output/f'collector_{index:02d}.json'
        reportfile.write_text(json.dumps(collect(cases),indent=2,allow_nan=False)+'\n')
        state(phase='reporting',collector=str(reportfile),completed_cases=list(cases))
        if a.report_builder.is_file():
            reportdir=output/'reports'/f'{index:02d}'
            command=[sys.executable,str(a.report_builder),'--stability',str(reportfile),'--output',str(reportdir)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            (output/f'report_{index:02d}.log').write_text(result.stdout+result.stderr)
            event(kind='report',returncode=result.returncode,collector=str(reportfile),report_output=str(reportdir),report_builder_sha256=sha(a.report_builder))
        return reportfile
    try:
        state(phase='waiting_reference',reference=str(reference));event(kind='waiting_reference')
        while not (reference/'end_status.json').exists():
            unchanged();time.sleep(15)
        audit(reference);unchanged()
        config=tomllib.loads((reference/'results/lvd_solver_configuration.toml').read_text())
        if config['absolute_prior_weight']!=a.absolute_prior_weight or config['adjoint_backend']!='forwarddiff':
            raise RuntimeError('reference prior weight/backend differs from fixed control plan')
        reference_status=dict(line.split('=',1) for line in (reference/'results/lvd_solver_status.txt').read_text().splitlines() if '=' in line)
        event(kind='reference_completed',kkt_converged=reference_status.get('kkt_converged'))
        report(0)
        for number,name in enumerate(plan['scenarios'],1):
            unchanged();group,scenario=name.split('/')
            artifact=dry if group=='dry' else corrected
            folder=output/group/scenario
            command=base.copy()
            command[command.index('--output-dir')+1]=str(folder/'results')
            command[command.index('--correction-artifact')+1]=str(artifact)
            command[command.index('--seed')+1]=str(a.seed)
            command[command.index('--papermatch-initial-water')+1]={'start015':'0.15','start03':'0.3'}.get(scenario,'0')
            command+=['--papermatch-absolute-prior-weight',repr(a.absolute_prior_weight)]
            if scenario.startswith('scale_'):
                baseline=cases[group+'/start0'];audit(baseline)
                command+=['--papermatch-abs-scale',repr(scale(baseline)*(.95 if 'minus' in scenario else 1.05))]
            elif scenario=='maskInf':command+=['--papermatch-mask-ratio','Inf']
            wrapper=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(folder),'--dump',str(dump),
                '--julia',base[0],'--cpus',str(a.cpus),'--cpu-start',str(a.cpu_start),'--memory-gib',str(a.memory_gib)]
            for path in inputs:wrapper+=['--input',path]
            wrapper+=['--',*command]
            event(kind='launch',case=name,command=command,wrapped_command=wrapper)
            # The audited wrapper computes its 30% host budget before applying
            # the explicit scientific pool. Only that launcher sees full affinity;
            # this queue and all report processes stay on the control pool.
            child=subprocess.Popen(wrapper,cwd=ROOT,preexec_fn=lambda:os.sched_setaffinity(0,full_affinity))
            state(phase='running',case=name,wrapper_pid=child.pid,completed_cases=list(cases))
            returncode=child.wait();event(kind='child_ended',case=name,returncode=returncode)
            if returncode!=0:raise RuntimeError(f'Control execution/provenance failure: {name}')
            audit(folder);unchanged();cases[name]=folder
            configuration=tomllib.loads((folder/'results/lvd_solver_configuration.toml').read_text())
            if configuration['absolute_prior_weight']!=a.absolute_prior_weight or configuration['adjoint_backend']!='forwarddiff':
                raise RuntimeError(f'{name}: actual prior weight/backend changed')
            report(number)
        state(phase='complete',completed_cases=list(cases),collector=str(output/f'collector_{len(plan["scenarios"]):02d}.json'))
        event(kind='complete',cases=len(cases));return 0
    except Exception as error:
        state(phase='failed',error=str(error),completed_cases=list(cases));event(kind='failed',error=str(error));raise


if __name__=='__main__':
    raise SystemExit(main())
