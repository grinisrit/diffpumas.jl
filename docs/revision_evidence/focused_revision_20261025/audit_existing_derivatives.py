#!/usr/bin/env python3
"""Postprocess existing evidence only; no transport, new draws, or reselection."""
import csv,json,hashlib,math,statistics as st
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];EV=ROOT/'docs/revision_evidence';OUT=Path(__file__).resolve().parent
inputs=[]
def read(name):
 p=EV/name;inputs.append(p);return list(csv.DictReader(p.open()))
def dist(v):
 a=np.asarray(list(v),dtype=float)
 assert np.isfinite(a).all()
 if not len(a):return dict(n=0)
 return dict(n=len(a),minimum=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
def err(rows):
 delta=[float(r['ad'])-float(r['fd']) for r in rows];rel=[abs(v)/max(abs(float(r['ad'])),abs(float(r['fd'])),1e-30) for v,r in zip(delta,rows)]
 return dict(signed_error=dist(delta),absolute_error=dist(map(abs,delta)),relative_error=dist(rel),ad_magnitude=dist(abs(float(r['ad'])) for r in rows),negative_ad=sum(float(r['ad'])<0 for r in rows),zero_ad=sum(float(r['ad'])==0 for r in rows),over_1e4=sum(x>=1e-4 for x in rel))
pop=read('corrected_validation/selection_population_audit.csv');selected=[r for r in pop if int(r['selected_case'])>0]
features=['zenith_deg','slant_m','cell_z_m','gradient'];coverage={name:dict(population=dist(abs(float(r[name])) if name=='gradient' else float(r[name]) for r in pop),selected=dist(abs(float(r[name])) if name=='gradient' else float(r[name]) for r in selected)) for name in features}
strata=lambda rows:{tuple(r[k] for k in ['zenith_quartile','slant_quartile','height_quartile','sensitivity_quartile']) for r in rows}
quartiles={q:dict(population=sum(int(r['sensitivity_quartile'])==q for r in pop),selected=sum(int(r['sensitivity_quartile'])==q for r in selected)) for q in range(1,5)}
ad=read('surrogate_v2/forwarddiff_replay/ad_fd_replay.csv');step_summaries={p:{str(h):err([r for r in ad if r['population']==p and float(r['step'])==h]) for h in [1e-4,1e-5,1e-6,1e-7,1e-8]} for p in ['historical','new']}
pilotad=read('lowfraction_round3_20261025/pilot/results/ad_fd_steps.csv')
step_summaries.update({f'lowfraction_{a}':{str(h):err([r for r in pilotad if r['arm']==a and float(r['step'])==h]) for h in [1e-4,1e-5,1e-6]} for a in ['control','candidate']})
mcsets=[]
history=read('surrogate_v2/historical/historical_derivatives.csv');new=read('surrogate_v2/production/heldout_derivatives.csv');low=read('lowfraction_round3_20261025/pilot/results/heldout_derivatives.csv')
for name,rows,keys,target,reference in [('historical_before',history,['bin','cell','fraction'],'before','mc_secant'),('historical_after',history,['bin','cell','fraction'],'after','mc_secant'),('new_before',new,['bin','cell','field'],'before','mc_secant'),('new_after',new,['bin','cell','field'],'after','mc_secant'),('lowfraction_control',[r for r in low if r['arm']=='control'],['bin','cell','field'],'csda_secant','mc_secant'),('lowfraction_candidate',[r for r in low if r['arm']=='candidate'],['bin','cell','field'],'csda_secant','mc_secant')]:
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 summaries=[]
 for key,rr in groups.items():
  assert len(rr)==8
  ref=np.array([float(r[reference]) for r in rr]);model=np.array([float(r[target]) for r in rr]);bias=model-ref
  mu=float(ref.mean());refse=float(ref.std(ddof=1)/math.sqrt(8));b=float(bias.mean());bse=float(bias.std(ddof=1)/math.sqrt(8));resolved=abs(mu)>5*refse
  summaries.append(dict(key=key,mc_mean=mu,mc_SE=refse,signed_bias=b,bias_SE=bse,absolute_bias=abs(b),relative_bias=b/abs(mu) if mu else None,reference_resolved=resolved,negative_reference=mu<0,zero_reference=mu==0,over_2SE=abs(b)>2*bse))
 relative=[s['relative_bias'] for s in summaries if s['relative_bias'] is not None];resolved=[s['relative_bias'] for s in summaries if s['reference_resolved']]
 mcsets.append(dict(population=name,groups=len(groups),signed_absolute_units_bias=dist(s['signed_bias'] for s in summaries),absolute_bias=dist(s['absolute_bias'] for s in summaries),point_relative_bias=dist(relative),point_absolute_relative_bias=dist(map(abs,relative)),resolved_reference_relative_bias=dist(resolved),resolved_reference_absolute_relative_bias=dist(map(abs,resolved)),reference_precision_unresolved=sum(not s['reference_resolved'] for s in summaries),negative_reference_groups=sum(s['negative_reference'] for s in summaries),zero_reference_groups=sum(s['zero_reference'] for s in summaries),over_2_paired_SE=sum(s['over_2SE'] for s in summaries),cases=summaries))
J=read('final_field_mc_coverage_20261024/final_jacobian_sparse.csv');cover=read('final_field_mc_coverage_20261024/final_field_coverage.csv');supported=[float(r['derivative']) for r in J if r['in_support']=='true'];fitids={int(r['bin_idx']) for r in cover if r['fit_mask']=='true'};fitJ=[float(r['derivative']) for r in J if r['in_support']=='true' and int(r['bin_idx']) in fitids]
def matrix_summary(v,slots):return dict(slots=slots,stored_entries=len(v),nonzero=sum(x!=0 for x in v),negative=sum(x<0 for x in v),zero_including_unstored=slots-sum(x!=0 for x in v),absolute_nonzero_distribution=dist(abs(x) for x in v if x!=0))
result=dict(scope='Existing-output descriptive audit only. Geometric selection coverage is not full-domain validation. MC comparisons are finite contrasts, not local-AD truth; eight-stream SEs are conditional and are not confidence coverage.',selection=dict(nonzero_population=len(pop),selected_pairs=len(selected),fraction_states=[0,.3,.8],historical_case_count=72,populated_joint_strata=len(strata(pop)),selected_joint_strata=len(strata(selected)),distinct_selected_rays=len({r['bin'] for r in selected}),coverage=coverage,sensitivity_quartiles=quartiles,selection_excludes_zero=True,new48_selection='Largest absolute eligible interior sensitivity per ray/field; not random or sensitivity-balanced.',lowfraction48_selection='Middle sorted crossed cell, fixed geometrically; new low-fraction states; original/candidate arms do not double independent cases.'),deterministic_step_distributions=step_summaries,mc_bias_distributions=mcsets,current_field_support_jacobian=dict(all_covered=matrix_summary(supported,1003*147),fitted=matrix_summary(fitJ,690*147)),input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'existing_derivative_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
compact=[]
for p,steps in step_summaries.items():
 for h,d in steps.items():compact.append(dict(population=p,step=h,cases=d['relative_error']['n'],median_relative_error=d['relative_error']['median'],p95_relative_error=d['relative_error']['p95'],max_relative_error=d['relative_error']['maximum'],cases_over_1e4=d['over_1e4'],median_absolute_error=d['absolute_error']['median'],max_absolute_error=d['absolute_error']['maximum']))
with (OUT/'deterministic_step_distributions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
print(json.dumps(dict(selection=result['selection'],deterministic_at_1e6={p:s['1e-06'] for p,s in step_summaries.items()},mc=[{k:v for k,v in r.items() if k not in ['cases']} for r in mcsets],current_field_support_jacobian=result['current_field_support_jacobian']),indent=2))
