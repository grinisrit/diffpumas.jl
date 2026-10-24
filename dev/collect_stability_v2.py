#!/usr/bin/env python3
"""Collect named immutable stability runs without altering their evidence.

Example: --case dry/start0=examples/data/stability_v2_pilot_20261024/dry/start0
         --case dry/start015=examples/data/stability_v2_dry_start015_20261024
Names use GROUP/SCENARIO; each GROUP/start0 is the matched comparison reference.
The output must be fresh. Missing or active runs are reported, never discarded.
"""
import argparse
import datetime
import json
import math
from pathlib import Path

from run_stability_v2 import compare, hash_values, result, rows, sha


def elapsed_seconds(start, end):
    return (datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds()


def collect(cases):
    parsed={}
    report={'scope':'Fixed-objective matched controls; KKT is conditional first-order stationarity, not uniqueness or statistical coverage.',
            'runs':{}, 'comparisons':{}}
    for name,folder in cases.items():
        entry=report['runs'][name]={'path':str(folder)}
        try:
            execution=json.loads((folder/'execution.json').read_text())
            entry.update(started_utc=execution['started_utc'],command=execution['command'],
                         cpu_pool=execution['cpu_pool'],address_space_bytes=execution['address_space_bytes'],
                         source_sha256=execution['source_sha256'],runtime=execution['julia_runtime'])
            r=result(folder)
            end=r['execution']
            history=rows(folder/'results/lvd_solver_history.csv')
            objectives=[float(x['objective']) for x in history]
            pg=[float(x['projected_gradient']) for x in history]
            final=r['solver']
            iterations=max(int(x['iteration']) for x in history)
            elapsed=elapsed_seconds(execution['started_utc'],end['ended_utc'])
            fractions=[float(x['w']) for x in r['field'] if x['aquifer_mask'].lower()=='true']
            artifact=Path(execution['command'][execution['command'].index('--correction-artifact')+1])
            entry.update(execution_completed=end['execution_completed'],source_unchanged=end['source_unchanged'],
                runtime_unchanged=end['runtime_unchanged'],ended_utc=end['ended_utc'],wall_seconds=elapsed,
                iterations=iterations,wall_seconds_per_iteration_including_setup=elapsed/max(iterations,1),
                kkt_converged=final.get('kkt_converged')=='true',failed_stationarity=final.get('kkt_converged')!='true',
                initial_objective=objectives[0],final_objective=objectives[-1],final_projected_gradient=pg[-1],
                objective_monotone=all(b<=a+1e-12*max(1,abs(a)) for a,b in zip(objectives,objectives[1:])),
                history_matches_status=math.isclose(objectives[-1],float(final['objective']),rel_tol=1e-12,abs_tol=1e-12)
                    and math.isclose(pg[-1],float(final['projected_gradient']),rel_tol=1e-12,abs_tol=1e-12),
                cells=len(r['field']),support_cells=len(fractions),covered_bins=len(r['pred']),
                fit_bins=sum(float(x['fit_mask'])>0 for x in r['pred'].values()),
                support_mean_water=sum(fractions)/len(fractions),support_inventory=sum(fractions),
                support_dry_cells=sum(x<=1e-3 for x in fractions),support_ceiling_cells=sum(x>=.899 for x in fractions),
                expected_full_grid=len(r['field'])==600 and len(r['pred'])==1003,
                support_sha256=hash_values([(x['cell_idx'],x['aquifer_mask']) for x in r['field']]),
                screen_sha256=hash_values([(k,r['pred'][k]['fit_mask']) for k in sorted(r['pred'])]),
                correction_artifact_sha256=sha(artifact),solver=final,
                evidence_sha256={file.name:sha(file) for file in [folder/'execution.json',folder/'end_status.json',
                    folder/'results/lvd_solver_history.csv',folder/'results/lvd_solver_status.txt',
                    folder/'results/lvd_fit_field.csv',folder/'results/lvd_fit_predictions.csv']})
            parsed[name]=r
        except (OSError,KeyError,ValueError) as error:
            entry.update(error=str(error),execution_completed=False,failed_stationarity=True)
    for name,candidate in parsed.items():
        baseline=name.split('/')[0]+'/start0'
        if baseline not in parsed:
            continue
        entry=compare(parsed[baseline],candidate)
        ref,cand=report['runs'][baseline],report['runs'][name]
        entry['objective_difference']=cand['final_objective']-ref['final_objective']
        entry['objective_difference_comparable']=entry['same_screen'] and '/scale_' not in name
        entry['same_correction_artifact']=ref['correction_artifact_sha256']==cand['correction_artifact_sha256']
        entry['same_runtime']=ref['runtime']==cand['runtime']
        def core_sources(run):
            return {key:value for key,value in run['source_sha256'].items()
                if key.startswith('src/') or key.startswith('examples/') or key in ('Project.toml','Manifest.toml')}
        entry['same_core_sources']=core_sources(ref)==core_sources(cand)
        common=set(ref['source_sha256']) & set(cand['source_sha256'])
        entry['shared_audited_inputs_match']=all(ref['source_sha256'][k]==cand['source_sha256'][k] for k in common)
        entry['audited_inputs_only_in_reference']=sorted(set(ref['source_sha256'])-set(cand['source_sha256']))
        entry['audited_inputs_only_in_candidate']=sorted(set(cand['source_sha256'])-set(ref['source_sha256']))
        report['comparisons'][name]=entry
    if 'dry/start0' in parsed and 'corrected/start0' in parsed:
        report['comparisons']['corrected_vs_dry']=compare(parsed['dry/start0'],parsed['corrected/start0'])
    report['failed_stationarity']=[name for name,r in report['runs'].items() if r.get('failed_stationarity')]
    report['incomplete_execution']=[name for name,r in report['runs'].items() if not r.get('execution_completed')]
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',default=[],required=True,metavar='GROUP/SCENARIO=PATH')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous report and choose a fresh filename')
    cases={}
    for specification in args.case:
        name,path=specification.split('=',1)
        if name in cases or '/' not in name:
            parser.error('case names must be unique GROUP/SCENARIO')
        cases[name]=Path(path).resolve()
    report=collect(cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output),'incomplete_execution':report['incomplete_execution'],
                      'failed_stationarity':report['failed_stationarity']}))


if __name__=='__main__':
    main()
