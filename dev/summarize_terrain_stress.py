#!/usr/bin/env python3
"""Hash-verify terrain outcomes and pair them with matching zero continuations.
Outputs are review candidates; no manuscript prose is generated or installed.
"""
import argparse
import csv
import hashlib
import json
import math
import tomllib
from pathlib import Path

REQUIRED=('data/report.toml','data/predictions.csv','data/fit_field.csv')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def norm(values):
    return math.sqrt(sum(x*x for x in values))

def logmetrics(values):
    return dict(count=len(values),mean_log_residual=sum(values)/len(values) if values else None,
                rms_log_residual=norm(values)/math.sqrt(len(values)) if values else None)

def verified_case(folder):
    endpath=folder/'end_status.json'
    if not endpath.exists():
        return None
    end=json.loads(endpath.read_text())
    if any(end.get(key) is not True for key in ('execution_completed','source_unchanged','runtime_unchanged')):
        raise ValueError(f'unclean case end record: {folder}')
    adopted={}
    for name in REQUIRED:
        path=folder/name
        expected=end.get('output_sha256',{}).get(name)
        if not expected or not path.is_file() or digest(path)!=expected:
            raise ValueError(f'adopted output hash mismatch: {path}')
        adopted[name]=expected
    return dict(end_status_sha256=digest(endpath),adopted_output_sha256=adopted)

def summarize(root,allow_partial=False):
    planpath=root/'preregistered_plan.json'
    plan=json.loads(planpath.read_text())
    outcomes=[];pending=[];failed=[]
    for case in plan['cases']:
        folder=root/case['name'];reference=root/(case['screen']+'_zero')
        try:
            proof=verified_case(folder)
            if proof is None:
                pending.append(case['name']);continue
            zero_proof=verified_case(reference)
            if zero_proof is None:
                pending.append(case['name']+' (matching zero pending)');continue
        except ValueError as error:
            if not allow_partial:raise
            failed.append(dict(case=case['name'],reason=str(error)));continue
        report=tomllib.loads((folder/'data/report.toml').read_text())
        pred=rows(folder/'data/predictions.csv');zero=rows(reference/'data/predictions.csv')
        field=rows(folder/'data/fit_field.csv');zero_field=rows(reference/'data/fit_field.csv')
        if [r['bin_idx'] for r in pred]!=[r['bin_idx'] for r in zero]:
            raise ValueError(f'direction order mismatch: {folder}')
        if [(r['cell_idx'],r['support']) for r in field]!=[(r['cell_idx'],r['support']) for r in zero_field]:
            raise ValueError(f'field support/order mismatch: {folder}')
        entry=dict(case=case['name'],screen=case['screen'],amplitude=case['amplitude'],forward_only=case['forward_only'],
            objective=report['objective'],projected_gradient_inf=None if case['forward_only'] else report['projected_gradient_inf'],
            stationary=report['stationary_at_requested_tolerance'],max_abs_slant_change_m=report['max_abs_slant_change_m'],
            max_abs_map_height_change_m=report['max_abs_map_height_change_m'],
            off_sector_changed_paths=report['off_sector_changed_paths'],
            changed_supported_cell_intersections=report['changed_supported_cell_intersections'],
            case_provenance=proof,matching_zero_provenance=zero_proof)
        entry['field_relative_to_matching_zero']=None if case['forward_only'] else norm([
            float(r['water'])-float(z['water']) for r,z in zip(field,zero_field)])/max(norm([float(z['water']) for z in zero_field]),1e-30)
        entry['fixed_field_flux_max_relative_to_zero']=max(abs(float(r['fixed_field_flux'])/float(z['fixed_field_flux'])-1) for r,z in zip(pred,zero))
        zero_report=tomllib.loads((reference/'data/report.toml').read_text())
        entry['objective_difference_from_matching_zero']=None if case['forward_only'] else report['objective']-zero_report['objective']
        for label,predicate in [('all',lambda r:True),('sector',lambda r:r['sector']=='true'),
                ('outside_sector',lambda r:r['sector']!='true'),('fit',lambda r:float(r['fit_mask'])>0)]:
            selected=[r for r in pred if predicate(r)];zero_selected=[r for r in zero if predicate(r)]
            for stage,column in [('fixed_field','fixed_log_residual'),('refit','refit_log_residual')]:
                metric=logmetrics([float(r[column]) for r in selected])
                reference_metric=logmetrics([float(r[column]) for r in zero_selected])
                metric['matching_zero_rms_log_residual']=reference_metric['rms_log_residual']
                metric['rms_change_from_matching_zero']=(metric['rms_log_residual']-reference_metric['rms_log_residual']) if selected else None
                entry[label+'_'+stage]=metric
        outcomes.append(entry)
    if pending and not allow_partial:
        raise ValueError('pending cases: '+', '.join(pending))
    return dict(scope=plan['scope'],plan_sha256=digest(planpath),complete=not pending and not failed,
        interpretation='Pairs use matching-screen zero continuations from the same nonstationary old baseline. If projected-gradient tolerance fails, field differences are finite-budget solver outcomes, not converged terrain sensitivity or uncertainty bounds.',
        cases=outcomes,pending=pending,failed=failed)

def figures(payload,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for screen,label,color in [('fixed','Original screen','tab:blue'),('all','All covered directions','tab:orange')]:
        all_rows=sorted((r for r in payload['cases'] if r['screen']==screen),key=lambda r:r['amplitude'])
        fits=[r for r in all_rows if not r['forward_only']]
        for ax,data,fn in [(axes[0,0],all_rows,lambda r:r['sector_fixed_field']['rms_log_residual']),
                (axes[0,1],fits,lambda r:r['sector_refit']['rms_log_residual']),
                (axes[1,0],fits,lambda r:100*r['field_relative_to_matching_zero']),
                (axes[1,1],fits,lambda r:r['projected_gradient_inf'])]:
            ax.plot([100*r['amplitude'] for r in data],[fn(r) for r in data],color=color,linewidth=1,label=label)
            for r in data:
                ax.scatter(100*r['amplitude'],fn(r),color=color,marker='o' if r['stationary'] else 'x',s=45)
    axes[0,0].set_title('Sector residual at fixed original field')
    axes[0,1].set_title('Sector residual after finite-budget refit')
    axes[1,0].set_title('Field change relative to matching zero')
    axes[1,1].set_title('Refit stationarity diagnostic')
    for ax in axes.flat:
        ax.set_xlabel('Assumed peak terrain-height stress (%)');ax.set_xlim(-5.5,5.5);ax.grid(alpha=.2)
    axes[0,0].set_ylabel('RMS log residual');axes[0,1].set_ylabel('RMS log residual')
    axes[1,0].set_ylabel('Relative field norm change (%)')
    axes[1,1].set_ylabel('Projected-gradient infinity norm');axes[1,1].set_yscale('log')
    axes[1,1].axhline(1e-6,color='grey',linestyle=':',label='Requested tolerance')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Conditional terrain stress — review candidate'+(' (partial)' if not payload['complete'] else '')+
        '\nSector: azimuth 135–180°, zenith 40–55°; × denotes no stationarity certificate',fontsize=12)
    fig.savefig(output/'terrain_stress_candidate.pdf')
    fig.savefig(output/'terrain_stress_candidate.png',dpi=160)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--figures',action='store_true')
    args=parser.parse_args();payload=summarize(args.root,args.allow_partial)
    output=args.output_dir or args.root;output.mkdir(parents=True,exist_ok=True)
    (output/'terrain_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    flat_keys=('case','screen','amplitude','forward_only','objective','objective_difference_from_matching_zero',
        'projected_gradient_inf','stationary','field_relative_to_matching_zero','max_abs_slant_change_m','max_abs_map_height_change_m','off_sector_changed_paths')
    with (output/'terrain_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=flat_keys);writer.writeheader()
        writer.writerows({key:r[key] for key in flat_keys} for r in payload['cases'])
    if args.figures:figures(payload,output)
    print(output/'terrain_summary.json')
if __name__=='__main__':main()
