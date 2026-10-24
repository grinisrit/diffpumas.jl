#!/usr/bin/env python3
"""Prepare or run predeclared terrain stresses. Default only writes a plan.
Run the reviewed --execute command in tmux after the coordinator approves it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from record_revision_run import record_sources
from robustness_run_identity import julia_runtime

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'examples/data/material_main_recovery_20261024/data'
EXECUTION=ROOT/'examples/data/material_production_20261024/stages/main/execution.json'
ARTIFACT=ROOT/'docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml'
DUMP=ROOT/'examples/data/revision_physics_v2_final_20261023/materials.pumas'
HELPER=ROOT/'dev/refit_terrain_stress.jl'
INPUTS=[HELPER,ROOT/'dev/terrain_stress_geometry.jl',ROOT/'dev/prototype_stability_guarded_ray_cache.jl',
        ROOT/'dev/generated_stability_cached_flux_factory.jl',Path(__file__).resolve(),EXECUTION,ARTIFACT,
        *(BASELINE/n for n in ['lvd_solver_configuration.toml','lvd_fit_field.csv','lvd_fit_predictions.csv','lvd_solver_status.txt'])]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def plan(output):
    original=json.loads(EXECUTION.read_text())['command']
    pos=next(i for i,x in enumerate(original) if x.endswith('/examples/lvd_tomography.jl'))
    flags=original[pos+1:]
    cases=[]
    # Declared before outcomes: six fits, then four cheap ±1% diagnostics.
    for screen in ('fixed','all'):
        for amp in (0.0,-.05,.05,-.01,.01):
            forward=abs(amp)==.01
            name=f'{screen}_{"zero" if amp==0 else ("minus" if amp<0 else "plus")+str(round(abs(amp)*100))}'
            folder=output/name
            args=list(flags);args[args.index('--output-dir')+1]=str(folder/'data')
            cmd=[original[0],'--startup-file=no','--project=.',str(HELPER),*args,
                 '--baseline-results',str(BASELINE),'--terrain-amplitude',str(amp),
                 '--terrain-screen',screen,'--terrain-forward-only',str(forward).lower()]
            wrapper=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(folder),
                '--dump',str(DUMP),'--julia',original[0],'--cpus','8','--cpu-start','0','--memory-gib','16']
            for p in INPUTS:wrapper+=['--input',str(p)]
            wrapper+=['--',*cmd]
            cases.append(dict(name=name,screen=screen,amplitude=amp,forward_only=forward,command=wrapper))
    context=record_sources(ROOT,DUMP)
    context.update({str(p):sha(p) for p in INPUTS})
    return dict(julia_runtime=julia_runtime(original[0]),source_and_input_sha256=context,scope='Hypothetical localized terrain stress; not measured DEM errors, calibrated uncertainty, or residual cause identification.',
        geometry='Original elevation grid heights relative detector multiplied by 1+a*Hann(az135..180)*Hann(zen40..55); original voxel grid, support, observations and fixed artifact/lambda retained.',
        comparison='Each ±5% refit compared to matching zero continuation; old main was nonstationary. ±1% forward only predeclared independently of outcomes.',
        cases=cases,input_sha256={str(p):sha(p) for p in INPUTS})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    record=plan(output);target=output/'preregistered_plan.json'
    if target.exists() and json.loads(target.read_text())!=record:raise SystemExit('plan/input changed; use a fresh directory')
    target.write_text(json.dumps(record,indent=2)+'\n')
    if not a.execute:
        print(f'Prepared {len(record["cases"])} cases; no numerical production launched: {target}');return
    for case in record['cases']:
        for name,digest in record['source_and_input_sha256'].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if not path.is_file() or sha(path)!=digest:raise SystemExit(f'pre-case source/input changed: {path}')
        if julia_runtime(case['command'][case['command'].index('--julia')+1])!=record['julia_runtime']:raise SystemExit('pre-case Julia runtime changed')
        print('Starting',case['name'],flush=True)
        result=subprocess.run(case['command'],cwd=ROOT,stdout=subprocess.DEVNULL)
        if result.returncode:raise SystemExit(f'case failed: {case["name"]}; inspect run.log and end_status.json')
    (output/'queue_complete.json').write_text(json.dumps({'execution_completed':True,'cases':[c['name'] for c in record['cases']]},indent=2)+'\n')
if __name__=='__main__':main()
