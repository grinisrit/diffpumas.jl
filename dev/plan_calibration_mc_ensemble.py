#!/usr/bin/env python3
"""Prepare the preregistered eight-replicate calibration-MC ensemble."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    julia='/home/roland/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia'
    dump=ROOT/'examples/data/revision_physics_v2_final_20261023/materials.pumas'
    artifact=ROOT/'docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml'
    baseline=ROOT/'examples/data/material_main_recovery_20261024/data'
    coverage=ROOT/'docs/revision_evidence/final_field_mc_coverage_20261024'
    training=ROOT/'docs/revision_evidence/surrogate_v2/production/training.csv'
    helper=ROOT/'dev/calibration_mc_ensemble.jl'
    inputs=[helper,Path(__file__).resolve(),ROOT/'dev/validate_surrogate_v2.jl',ROOT/'examples/revision_suite.jl',artifact,training]
    inputs += [baseline/n for n in ('lvd_fit_field.csv','lvd_fit_predictions.csv','lvd_solver_configuration.toml','lvd_mc_coverage.csv')]
    inputs += [coverage/n for n in ('final_field_coverage.csv','final_jacobian_sparse.csv','coverage_summary.toml','end_status.json')]
    coverage_end=json.loads((coverage/'end_status.json').read_text())
    if not all(coverage_end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise SystemExit('coverage run lacks clean source/runtime completion')
    for name in ('final_field_coverage.csv','final_jacobian_sparse.csv','coverage_summary.toml'):
        if hashlib.sha256((coverage/name).read_bytes()).hexdigest()!=coverage_end['output_sha256']['data/'+name]:
            raise SystemExit('coverage output binding mismatch: '+name)
    command=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(out/'job'),'--dump',str(dump),'--julia',julia,'--cpus','4','--cpu-start','16','--memory-gib','8']
    for p in inputs:command+=['--input',str(p)]
    command+=['--',julia,'--startup-file=no','--project=.',str(helper),'--output',str(out/'job/data'),'--dump',str(dump),'--artifact',str(artifact),'--baseline',str(baseline),'--coverage',str(coverage),'--training',str(training)]
    record={'replicates':8,'training_rays':12,'fields':[0,.2,.6],'samples':8192,'total_histories':2359296,'scope':'Finite-MC material-coefficient calibration dispersion only; fixed dry correction. No structural error coverage, field posterior or calibrated intervals. Eight replicates chosen before timing/results. First replicate writes checkpoint before the rest.','input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'command':command}
    out.mkdir(parents=True,exist_ok=True);plan=out/'preregistered_plan.json'
    if plan.exists() and json.loads(plan.read_text())!=record:raise SystemExit('changed source/input; select fresh output root')
    plan.write_text(json.dumps(record,indent=2)+'\n')
    if args.execute:raise SystemExit(subprocess.call(command,cwd=ROOT))
    print(plan)

if __name__=='__main__':main()
