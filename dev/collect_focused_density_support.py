#!/usr/bin/env python3
"""Compare completed finite-budget arms; never rank different-model minima."""
import argparse,csv,hashlib,json,math,tomllib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadcsv(p):return list(csv.DictReader(p.open()))
def vector(rows,k):return [float(r[k]) for r in rows]
def norm(x):return math.sqrt(sum(v*v for v in x))
def change(a,b):
    d=[x-y for x,y in zip(a,b)]
    return dict(count=len(d),l2=norm(d),infinity=max(map(abs,d),default=0.),relative_l2=norm(d)/max(norm(b),1e-300))
def fractions(fields,indices):
    result=dict(count=len(indices),weighting='Unweighted cell mean of replacement mass fraction; not pore saturation, water volume or inventory.')
    for key in ['baseline_water','initial_water','final_water']:
        values=[float(fields[i][key]) for i in indices]
        result[key]=dict(mean=sum(values)/len(values) if values else None,minimum=min(values,default=None),maximum=max(values,default=None))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('output',type=Path);p.add_argument('--runs-subdir',default='arms_v2');a=p.parse_args()
    runs={};raw={};inputs={}
    for arm in ('baseline','rho2660','rho2760','cut400','cut800'):
        root=a.evidence/a.runs_subdir/arm
        end=json.loads((root/'end_status.json').read_text());inputs[str(root/'end_status.json')]=sha(root/'end_status.json')
        passed=end['execution_completed'] and end['source_unchanged'] and end['runtime_unchanged']
        for file,digest in end['output_sha256'].items():
            if sha(root/file)!=digest:raise ValueError(f'Output changed: {arm}/{file}')
        if not passed:
            checkpoints=sorted((root/'results').glob('accepted_*.toml'))
            runs[arm]=dict(completed=False,last_full_confirmed_checkpoint=str(checkpoints[-1]) if checkpoints else None)
            continue
        r=tomllib.load((root/'results/report.toml').open('rb'));effects=tomllib.load((root/'results/initial_effects.toml').open('rb'))
        fields=loadcsv(root/'results/field.csv');predictions=loadcsv(root/'results/predictions.csv')
        assert len(fields)==600 and len(predictions)==1003 and sum(float(x['fit_mask'])>0 for x in predictions)==690
        assert r['accepted_updates']<=10 and r['outer_attempts']<=10 and not r['generalized_stationarity_certified']
        trials=loadcsv(root/'results/trials.csv')
        assert sum(x['accepted']=='true' for x in trials)==r['accepted_updates']
        assert all(float(x['confirmed_objective'])<float(x['objective_before']) for x in trials if x['accepted']=='true')
        support=[x['support']=='true' for x in fields];original=[x['original_support']=='true' for x in fields];crossed=[x['fit_crossed']=='true' for x in fields]
        regions=dict(whole_grid=list(range(600)),current_support=[i for i in range(600) if support[i]],original_support=[i for i in range(600) if original[i]],common_with_original=[i for i in range(600) if support[i] and original[i]],union_with_original=[i for i in range(600) if support[i] or original[i]],added=[i for i in range(600) if support[i] and not original[i]],removed=[i for i in range(600) if original[i] and not support[i]],fit_crossed_current=[i for i in range(600) if support[i] and crossed[i]],prior_only_current=[i for i in range(600) if support[i] and not crossed[i]])
        planes=sorted({float(x['z_m']) for x in fields});assert len(planes)==6
        profiles=[dict(z_m=z,regions={name:fractions(fields,[i for i in indices if float(fields[i]['z_m'])==z]) for name,indices in regions.items()}) for z in planes]
        runs[arm]=dict(completed=True,report=r,initial_effects=effects,regional_fraction_summaries={name:fractions(fields,indices) for name,indices in regions.items()},fixed_grid_z_plane_profiles=profiles,trial_count=len(trials),rejection_counts={reason:sum(x['reason']==reason for x in trials) for reason in sorted({x['reason'] for x in trials})})
        raw[arm]=(fields,predictions)
    comparisons={}
    if 'baseline' in raw:
        basefields,basepred=raw['baseline'];basew=vector(basefields,'final_water')
        for arm,(fields,pred) in raw.items():
            assert [x['bin_idx'] for x in pred]==[x['bin_idx'] for x in basepred]
            assert [x['fit_mask'] for x in pred]==[x['fit_mask'] for x in basepred]
            w=vector(fields,'final_water');support=[x['support']=='true' for x in fields];basesupport=[x['support']=='true' for x in basefields]
            common=[i for i in range(600) if support[i] and basesupport[i]];union=[i for i in range(600) if support[i] or basesupport[i]];added=[i for i in range(600) if support[i] and not basesupport[i]];removed=[i for i in range(600) if basesupport[i] and not support[i]]
            compare=dict(common_support_field_change=change([w[i] for i in common],[basew[i] for i in common]),union_support_field_change=change([w[i] for i in union],[basew[i] for i in union]),common_support_means=dict(count=len(common),case=sum(w[i] for i in common)/len(common),baseline_continuation=sum(basew[i] for i in common)/len(common)),union_support_means=dict(count=len(union),case=sum(w[i] for i in union)/len(union),baseline_continuation=sum(basew[i] for i in union)/len(union)),new_region=dict(count=len(added),mean_fraction=sum(w[i] for i in added)/len(added) if added else None,maximum=max((w[i] for i in added),default=None)),removed_region=dict(count=len(removed),baseline_continuation_mean=sum(basew[i] for i in removed)/len(removed) if removed else None),full_field_change=change(w,basew))
            for label,indices in [('covered',list(range(1003))),('fit',[i for i in range(1003) if float(pred[i]['fit_mask'])>0])]:
                unscaled=change([float(pred[i]['final_flux']) for i in indices],[float(basepred[i]['final_flux']) for i in indices])
                logs=[float(pred[i]['final_log_residual']) for i in indices];baselogs=[float(basepred[i]['final_log_residual']) for i in indices]
                compare[label]=dict(unscaled_flux_change=unscaled,profiled_log_residual_rms=norm(logs)/math.sqrt(len(logs)),baseline_profiled_log_residual_rms=norm(baselogs)/math.sqrt(len(logs)),profiled_log_residual_change=change(logs,baselogs))
            comparisons[arm]=compare
    result=dict(run_subdirectory=a.runs_subdir,scope='Matched finite-budget sensitivity with fixed absolute prior/screen; not equilibrated minima, calibrated geological uncertainty or a cross-model objective ranking.',runs=runs,comparisons_with_baseline_continuation=comparisons,input_sha256=inputs,collector_sha256=sha(Path(__file__)))
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False)
if __name__=='__main__':main()
