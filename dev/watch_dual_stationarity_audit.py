#!/usr/bin/env python3
"""Wait for a completed dual field, then audit one coordinate under CPU20 flock.

Keeps the raw audited Julia output immutable and separately binds derived
one-sided differences. This check does not certify global stationarity,
nondifferentiability, a local minimum, or uniqueness.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();reference=a.reference.resolve();output=a.output.resolve()
    if output.exists():p.error('output exists; choose a fresh audit root')
    statefile=output.with_name(output.name+'_watcher.json')
    def state(**data):statefile.write_text(json.dumps(dict(updated_utc=utc(),**data),indent=2)+'\n')
    state(phase='waiting_reference',reference=str(reference))
    while not (reference/'end_status.json').exists():time.sleep(15)
    end=json.loads((reference/'end_status.json').read_text())
    if not all(end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):
        state(phase='failed',reason='reference execution or provenance failed');return 1
    for name,digest in end['output_sha256'].items():
        if sha(reference/name)!=digest:raise RuntimeError('reference output digest mismatch: '+name)
    config=tomllib.loads((reference/'results/lvd_solver_configuration.toml').read_text())
    if config['adjoint_backend']!='forwarddiff':raise RuntimeError('expected a forwarddiff reference')
    execution=json.loads((reference/'execution.json').read_text());command=execution['command'].copy()
    example=str(ROOT/'examples/lvd_tomography.jl')
    command[command.index(example)]=str(ROOT/'dev/audit_stability_stationarity.jl')
    command[command.index('--output-dir')+1]=str(output/'results')
    command+=['--baseline-results',str(reference/'results')]
    wrapper=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(output),
        '--dump',command[command.index('--dump')+1],'--julia',command[0],
        '--cpus','1','--cpu-start','20','--memory-gib','5']
    for path in [Path(__file__).resolve(),ROOT/'examples/lvd_tomography.jl',
            Path(command[command.index('--correction-artifact')+1]),
            *[reference/'results'/name for name in ('lvd_fit_field.csv','lvd_fit_predictions.csv','lvd_solver_status.txt','lvd_solver_configuration.toml')]]:
        wrapper+=['--input',str(path)]
    wrapper+=['--',*command]
    original=set(os.sched_getaffinity(0));os.sched_setaffinity(0,{20})
    resource.setrlimit(resource.RLIMIT_AS,(5*1024**3,5*1024**3))
    a.lock.parent.mkdir(parents=True,exist_ok=True)
    state(phase='waiting_or_holding_cpu20_lock',lock=str(a.lock),wrapped_command=wrapper)
    completed=subprocess.run(['flock',str(a.lock),*wrapper],cwd=ROOT,
        preexec_fn=lambda:os.sched_setaffinity(0,original)).returncode
    if completed:
        state(phase='failed',reason='stationarity wrapper failed',returncode=completed);return completed
    raw=output/'results/stationarity_audit.toml';audit=tomllib.loads(raw.read_text())
    value=audit['water_fraction'];f0=audit['objective'];checks=[]
    for row in audit['directional_checks']:
        checks.append(dict(step=row['step'],coordinate=value,ad_gradient=row['ad_gradient'],
            central_or_box_secant=row['finite_difference'],
            left_one_sided=(f0-row['objective_lo'])/(value-row['lo']) if value>row['lo'] else None,
            right_one_sided=(row['objective_hi']-f0)/(row['hi']-value) if row['hi']>value else None))
    derived=dict(cell=audit['cell'],reference_backend=config['adjoint_backend'],checks=checks,
        interpretation='Finite-step checks in one selected coordinate; no global stationarity, nondifferentiability, local-minimum or uniqueness certification.')
    result=output/'one_sided_objective_checks.json';result.write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
    manifest=dict(created_utc=utc(),postprocessor_sha256=sha(Path(__file__).resolve()),
        raw_audit_sha256=sha(raw),execution_end_sha256=sha(output/'end_status.json'),derived_output_sha256=sha(result))
    (output/'postprocess_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    state(phase='complete',output=str(output),cpu20_released=True);return 0


if __name__=='__main__':raise SystemExit(main())
