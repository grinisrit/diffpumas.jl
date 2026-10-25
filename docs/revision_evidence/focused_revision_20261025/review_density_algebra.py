#!/usr/bin/env python3
import csv,json,math,tomllib,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'docs/revision_evidence/focused_density_support_20261025';arms=P/'arms_v2';base=ROOT/'docs/revision_evidence/material_main_recovery_20261024/completed/main/data'
read=lambda p:list(csv.DictReader(p.open()))
fields=read(base/'lvd_fit_field.csv');xyz=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in fields]);axes=[sorted(set(xyz[:,i])) for i in range(3)];lookup={tuple(v):i for i,v in enumerate(xyz)};edges=[]
for i,coord in enumerate(xyz):
 for d in range(3):
  rank=axes[d].index(coord[d])
  if rank+1<len(axes[d]):
   nxt=coord.copy();nxt[d]=axes[d][rank+1];j=lookup[tuple(nxt)];edges.append((i,j))
assert len(edges)==1580
observations=read(ROOT/'examples/data/lvd_conf/lvd_single_muon_flux_2d.csv');lam=3.954930197858131
results=[]
for arm in ['baseline','rho2660','rho2760','cut400','cut800']:
 p=arms/arm/'results';r=tomllib.loads((p/'report.toml').read_text());fr=read(p/'field.csv');pr=read(p/'predictions.csv');support=np.array([x['support']=='true' for x in fr]);y=np.array([float(x['observation']) for x in pr]);theta=np.array([float(observations[int(x['bin_idx'])-1]['zenith_center_deg']) for x in pr]);W=np.array([float(x['fit_mask']) for x in pr])/(1+(.15*np.maximum(1/np.cos(np.deg2rad(theta))-1,0))**2)
 def evaluate(w,f):
  z=np.log(f)-np.log(y);lo=-z.max();hi=-z.min()
  for _ in range(80):
   a=(lo+hi)/2;t=z+a;s=np.sum(W*t/np.sqrt(1+(t/.3)**2));hi=a if s>0 else hi;lo=lo if s>0 else a
  a=(lo+hi)/2;t=z+a;data=float(np.sum(W*t*t/(np.sqrt(1+(t/.3)**2)+1)));prior=lam*sum((w[i]-w[j])**2 for i,j in edges)
  return data+prior,data,prior,math.exp(a)
 checkpoints=sorted(p.glob('accepted_*.toml'));assert len(checkpoints)==r['accepted_updates']+1;objectives=[];maxdiff=0
 for n,f in enumerate(checkpoints):
  s=tomllib.loads(f.read_text());assert s['accepted_index']==n;w=np.array(s['field']);flux=np.array(s['predictions']);assert len(w)==600 and len(flux)==1003 and np.all((w>=0)&(w<=.9)) and np.all(w[~support]==0)
  actual=evaluate(w,flux);expected=[s['objective'],s['data_objective'],s['prior_objective'],s['normalization']]
  for a,b in zip(actual,expected):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
  maxdiff=max(maxdiff,abs(actual[0]-s['objective']));objectives.append(s['objective'])
 assert all(a>b for a,b in zip(objectives,objectives[1:]))
 tr=read(p/'trials.csv');taken=[x for x in tr if x['accepted']=='true'];assert len(taken)==r['accepted_updates']<=10 and r['outer_attempts']<=10
 for x in taken:
  f0=float(x['objective_before']);f1=float(x['confirmed_objective']);d=float(x['directional']);assert d<0 and f1<f0 and f1<=f0+1e-4*d
 assert objectives[-1]==r['final']['objective']
 results.append(dict(arm=arm,checkpoints=len(checkpoints),trials=len(tr),accepted=len(taken),maximum_objective_recompute_discrepancy=maxdiff,initial_objective=r['initial']['objective'],final_objective=r['final']['objective'],branch_pg=r['final']['selected_branch_pg'],termination=r['termination'],elapsed_seconds=r['elapsed_seconds'],support_cells=r['support_cells']))
OUT=Path(__file__).resolve().parent/'density_algebra_review.json';OUT.write_text(json.dumps(dict(passed=True,independent_graph_edges=1580,arms=results,scope='Independent robust profile and graph-prior algebra on all full-confirmed checkpoints; Armijo/box/support/budget audit, not independent forward replay.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(results)
