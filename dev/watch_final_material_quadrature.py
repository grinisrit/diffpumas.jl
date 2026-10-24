#!/usr/bin/env python3
"""Run the authorized conditional final-field quadrature after clean main completion.

Separate durable watcher; never edits or holds the production orchestrator.
Julia audits serialize with the agreed CPU20 lock. No fit or MC is performed.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path('/mnt/big_data/agent-builds/surrogate-stability-20261024/cpu20-julia-audit.lock')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def audit(folder, verify_current_sources=False):
    end = json.loads((folder/'end_status.json').read_text())
    if not all(end.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise ValueError(f'unclean numerical stage: {folder}')
    for name, expected in end['output_sha256'].items():
        if sha(folder/name) != expected:
            raise ValueError(f'changed output: {folder/name}')
    if verify_current_sources:
        execution=json.loads((folder/'execution.json').read_text())
        for name, expected in execution['source_sha256'].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path)!=expected:raise ValueError('main producer input changed: '+str(path))
    return end


def summarize(folder, main, selection):
    with (folder/'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result = dict(scope='Deterministic 512-midpoint convergence comparator, not physical truth or an exact-error certificate',
                  selected_primary_field=selection, groups=[])
    for corrected in ('false','true'):
        for field in ('dry','final_fit'):
            for nodes in (32,64,128,256,512):
                subset = [r for r in rows if r['corrected']==corrected and r['field']==field and int(r['nodes'])==nodes]
                flux = [abs(float(r['relative_flux_error'])) for r in subset if r['relative_flux_defined']=='true']
                gradient = [abs(float(r['relative_gradient_error'])) for r in subset if r['relative_gradient_defined']=='true']
                result['groups'].append(dict(corrected=corrected=='true',field=field,nodes=nodes,
                    valid_flux_comparisons=len(flux),valid_gradient_comparisons=len(gradient),
                    maximum_absolute_relative_flux_error=max(flux) if flux else None,
                    median_absolute_relative_flux_error=statistics.median(flux) if flux else None,
                    maximum_absolute_relative_gradient_error=max(gradient) if gradient else None,
                    median_absolute_relative_gradient_error=statistics.median(gradient) if gradient else None))
    (folder/'derived_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    provenance=dict(created_utc=utc(),main_end_status_sha256=sha(main/'end_status.json'),
        main_field_sha256=sha(main/'data/lvd_fit_field.csv'),quadrature_end_status_sha256=sha(folder/'end_status.json'),
        comparison_sha256=sha(folder/'comparison.csv'),summary_sha256=sha(folder/'derived_summary.json'),
        watcher_sha256=sha(Path(__file__).resolve()))
    (folder/'derived_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--dump',type=Path,required=True)
    ap.add_argument('--artifact',type=Path,required=True)
    ap.add_argument('--julia',required=True)
    args=ap.parse_args()
    run=args.run_root.resolve(strict=True);dump=args.dump.resolve(strict=True);artifact=args.artifact.resolve(strict=True)
    field=run/'stages/main/data/lvd_fit_field.csv';stage=run/'stages/main'
    output=run/'controls/final_field_quadrature';statefile=run/'controls/final_field_quadrature_watch.json'
    if output.exists() or statefile.exists():ap.error('fresh watcher/output required')
    statefile.parent.mkdir(parents=True,exist_ok=True)
    helper=ROOT/'dev/validate_material_quadrature.jl'
    expected={str(p):sha(p) for p in (Path(__file__).resolve(),helper,artifact,dump)}
    state=dict(started_utc=utc(),phase='waiting_main',output=str(output),lock=str(LOCK),input_sha256=expected,
        scope='Authorized preselected corrected start0 output field; remains conditional even if main is not KKT stationary')
    def save(**changes):
        state.update(updated_utc=utc(),**changes)
        temporary=statefile.with_suffix('.pending');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(statefile)
    save()
    try:
        while not (stage/'end_status.json').exists():time.sleep(15)
        audit(stage)
        # Let the main stage's own export finish first; the stage queue then
        # advances independently while this separate audit runs.
        while 'main' not in json.loads((run/'status.json').read_text()).get('postprocessing',{}):time.sleep(5)
        save(phase='waiting_julia_audit_lock')
        LOCK.parent.mkdir(parents=True,exist_ok=True)
        with LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            audit(stage,verify_current_sources=True)
            for name, expected_hash in expected.items():
                if sha(Path(name))!=expected_hash:raise ValueError('watcher input changed: '+name)
            solver=dict(line.split('=',1) for line in (stage/'data/lvd_solver_status.txt').read_text().splitlines() if '=' in line)
            selection=dict(main_kkt_converged=solver.get('kkt_converged')=='true',primary='corrected_start0',
                conditional_field=True,field_sha256=sha(field),absolute_prior_weight=solver.get('absolute_prior_weight'))
            command=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(output),'--dump',str(dump),
                '--julia',args.julia,'--cpus','1','--cpu-start','20','--memory-gib','5']
            for p in (Path(__file__).resolve(),ROOT/'dev/energy_quadrature_probe.jl',ROOT/'examples/revision_suite.jl',artifact,field,stage/'end_status.json'):
                command+=['--input',str(p)]
            command+=['--',args.julia,'--startup-file=no','--project=.',str(helper),'--fieldcsv',str(field),
                '--correctionartifact',str(artifact),'--output',str(output),'--dump',str(dump)]
            save(phase='running',command=command,selection=selection)
            with (run/'controls/final_field_quadrature_launcher.log').open('wb') as stream:
                code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('quadrature wrapper failed: '+str(code))
            audit(output);summarize(output,stage,selection)
            for name, expected_hash in expected.items():
                if sha(Path(name))!=expected_hash:raise ValueError('watcher input changed during audit: '+name)
            save(phase='complete',ended_utc=utc(),execution_completed=True,
                 end_status_sha256=sha(output/'end_status.json'),derived_provenance_sha256=sha(output/'derived_provenance.json'))
        return 0
    except Exception as error:
        save(phase='failed',ended_utc=utc(),execution_completed=False,error=str(error))
        raise


if __name__=='__main__':raise SystemExit(main())
