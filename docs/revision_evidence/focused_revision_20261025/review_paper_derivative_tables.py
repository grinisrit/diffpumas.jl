#!/usr/bin/env python3
from pathlib import Path
import json,re,csv,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent;TEX=ROOT/'diffmuontomoLVD/revision/focused_derivative_audits.tex';text=TEX.read_text();r=json.loads((P/'existing_derivative_audit.json').read_text());checked=0

def table(label):
 start=text.index('\\label{'+label+'}');a=text.index('\\begin{tabular}',start);b=text.index('\\end{tabular}',a);body=text[a:b].split('\\midrule',1)[1];result={}
 for line in body.splitlines():
  if '&' not in line:continue
  values=line.replace('\\\\','').split('&');name=values.pop(0).strip();clean=[]
  for v in values:
   v=re.sub(r'\\texttt\{([^}]*)\}',r'\1',v).strip();v=v.replace('\\bottomrule','').strip();clean.append(None if v=='--' else float(v))
  result[name]=clean
 return result

def check(label,expected):
 global checked
 actual=table(label);assert set(actual)==set(expected),(label,set(actual),set(expected))
 for key,vals in expected.items():
  assert len(actual[key])==len(vals)
  for a,e in zip(actual[key],vals):
   target=None if e is None else float(f'{e:.3g}')
   assert a==target,(label,key,a,target,e)
   checked+=1

sel=r['selection']['coverage'];check('tab:focused-selection',{'Zenith [deg]':[sel['zenith_deg'][p][k] for p in ['population','selected'] for k in ['minimum','maximum']],'Rock slant length [m]':[sel['slant_m'][p][k] for p in ['population','selected'] for k in ['minimum','maximum']],'Cell height above detector [m]':[sel['cell_z_m'][p][k] for p in ['population','selected'] for k in ['minimum','maximum']],'$|J_{mi}|$ [flux/unit $w$]':[sel['gradient'][p][k] for p in ['population','selected'] for k in ['minimum','maximum']]})
expected={}
for label,pop in [('Historical','historical'),('Held-out','new'),('Low-$q$ control','lowfraction_control'),('Low-$q$ candidate','lowfraction_candidate')]:
 s=r['deterministic_step_distributions'][pop]['1e-06'];expected[label]=[s['absolute_error']['n']]+[s[k][q] for k in ['absolute_error','relative_error'] for q in ['median','p95','maximum']]
check('tab:focused-local-errors',expected)
read=lambda p:list(csv.DictReader(p.open()))
tail=read(P/'current_grid_tail_run_v2/results/tail_fd_checks.csv');expected={}
for label,band in [('Weak','weak'),('Middle','middle'),('High','high'),('Structural zero','structural_zero')]:
 rows=[x for x in tail if x['band']==band and float(x['step'])==1e-6];a=[float(x['absolute_error']) for x in rows];d=[float(x['relative_error']) for x in rows];expected[label]=[12,float(np.median(a)),float(np.quantile(a,.95)),max(a)]+([None]*3 if band=='structural_zero' else [float(np.median(d)),float(np.quantile(d,.95)),max(d)])
check('tab:focused-tail-errors',expected)
expected={}
for h in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]:expected['\\texttt{'+format(h,'g')+'}']=[max(float(x['relative_error']) for x in tail if x['band']==b and float(x['step'])==h) for b in ['weak','middle','high']]
check('tab:focused-tail-ladder',expected)
window=read(P/'finite_contrast_run_v2/results/finite_contrast_vs_local_ad.csv');expected={}
for label,pop in [('Historical','historical'),('Held-out','new'),('Low-$q$ control','lowfraction_control')]:
 rows=[x for x in window if x['population']==pop];a=[float(x['absolute_difference']) for x in rows];d=[float(x['relative_difference']) for x in rows];expected[label]=[len(rows),float(np.median(a)),float(np.quantile(a,.95)),max(a),float(np.median(d)),float(np.quantile(d,.95)),max(d)]
check('tab:focused-finite-window',expected)
absolute={};relative={}
for s in r['mc_bias_distributions']:
 label={'historical_before':'Historical base','historical_after':'Historical corrected','new_before':'Held-out base','new_after':'Held-out corrected','lowfraction_control':'Low-$q$ control','lowfraction_candidate':'Low-$q$ candidate'}[s['population']];a=s['absolute_bias'];d=s['resolved_reference_absolute_relative_bias'];absolute[label]=[s['groups'],s['reference_precision_unresolved'],s['negative_reference_groups'],a['median'],a['p95'],a['maximum']];relative[label]=[d['n']]+[100*d[k] for k in ['median','p95','maximum']]
check('tab:focused-mc-absolute',absolute);check('tab:focused-mc-relative',relative)
out=dict(passed=True,numeric_entries_checked=checked,tables_checked=7,tex_sha256=hashlib.sha256(TEX.read_bytes()).hexdigest(),scope='Independent exact3-significant-digit numerical check against authoritative raw v2 and retainedexistingdata, not semantic or physical-model certification.')
(P/'paper_derivative_table_review.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
