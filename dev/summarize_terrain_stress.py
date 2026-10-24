#!/usr/bin/env python3
"""Summarize predeclared terrain stresses against the matching zero continuation."""
import argparse,csv,json,math,tomllib
from pathlib import Path

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def norm(v):return math.sqrt(sum(x*x for x in v))
def logmetrics(values):
    return dict(count=len(values),mean_log_residual=sum(values)/len(values) if values else None,
                rms_log_residual=norm(values)/math.sqrt(len(values)) if values else None)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    plan=json.loads((root/'preregistered_plan.json').read_text());out=[]
    for case in plan['cases']:
        folder=root/case['name'];end=json.loads((folder/'end_status.json').read_text())
        if end.get('execution_completed') is not True:raise SystemExit(f'unclean case {folder}')
        report=tomllib.loads((folder/'data/report.toml').read_text());pred=rows(folder/'data/predictions.csv')
        reference=root/(case['screen']+'_zero')/'data'
        zero=rows(reference/'predictions.csv');field=rows(folder/'data/fit_field.csv');zf=rows(reference/'fit_field.csv')
        if [r['bin_idx'] for r in pred]!=[r['bin_idx'] for r in zero]:raise SystemExit('direction order mismatch')
        entry=dict(case=case['name'],screen=case['screen'],amplitude=case['amplitude'],forward_only=case['forward_only'],
            objective=report['objective'],projected_gradient_inf=None if case['forward_only'] else report['projected_gradient_inf'],
            stationary=report['stationary_at_requested_tolerance'],max_abs_slant_change_m=report['max_abs_slant_change_m'],
            max_abs_map_height_change_m=report['max_abs_map_height_change_m'],
            off_sector_changed_paths=report['off_sector_changed_paths'],changed_supported_cell_intersections=report['changed_supported_cell_intersections'])
        entry['field_relative_to_matching_zero']=None if case['forward_only'] else norm([float(r['water'])-float(z['water']) for r,z in zip(field,zf)])/max(norm([float(z['water']) for z in zf]),1e-30)
        entry['fixed_field_flux_max_relative_to_zero']=max(abs(float(r['fixed_field_flux'])/float(z['fixed_field_flux'])-1) for r,z in zip(pred,zero))
        for label,selected in [('all',pred),('sector',[r for r in pred if r['sector']=='true']),('outside_sector',[r for r in pred if r['sector']!='true']),('fit',[r for r in pred if float(r['fit_mask'])>0])]:
            entry[label+'_fixed_field']=logmetrics([float(r['fixed_log_residual']) for r in selected])
            entry[label+'_refit']=logmetrics([float(r['refit_log_residual']) for r in selected])
        out.append(entry)
    payload=dict(scope=plan['scope'],interpretation='Pairs use matching-screen zero continuations from the same nonstationary old baseline. If projected-gradient tolerance fails, field differences are finite-budget solver outcomes, not converged terrain sensitivity or uncertainty bounds.',cases=out)
    (root/'terrain_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(root/'terrain_summary.json')
if __name__=='__main__':main()
