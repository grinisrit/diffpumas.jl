#!/usr/bin/env python3
import csv,json,math,sys,tomllib
from pathlib import Path
p=Path(sys.argv[1]); baseline=Path(sys.argv[2]);final=tomllib.loads((p/'final.toml').read_text());s=tomllib.loads((p/'polish_state.toml').read_text())
fields=list(csv.DictReader((baseline/'lvd_fit_field.csv').open()));active=[i for i,r in enumerate(fields) if r['aquifer_mask']=='true'];initial=[float(r['w']) for r in fields];harm=tomllib.loads((p/'harmonic_reduction.toml').read_text());w=harm['field'][:];active=[c-1 for c in harm['reduced_cells']]
def feasible(v):
 assert len(v)==600 and all(math.isfinite(x) and 0<=x<=.9 for x in v)
 assert all(v[i]==initial[i] for i in range(600) if fields[i]['aquifer_mask']!='true')
assert s['objective_evaluations']==len(s['attempts'])<=14
assert s['gradient_assemblies']<=45
f=harm['objective_after'];accepted=0
for a in s['attempts']:
 assert a['objective_before']==f
 assert a['objective_change']==a['objective_trial']-f
 assert a['accepted']==(a['objective_change'] < -s['noise_floor'])
 if a['accepted']:
  accepted+=1;v=tomllib.loads((p/f'accepted_{accepted}.toml').read_text());feasible(v['field']);assert v['objective']==a['objective_trial'];w=v['field'];f=v['objective']
assert accepted==s['accepted']==final['accepted_updates']
assert w==s['field']==final['field'];feasible(w)
qp=[]
for rec in s['sample_records']:
 base=rec['field_before'];feasible(base);alpha=rec['alpha'];G=rec['gradients'];assert len(alpha)==len(G)==21
 assert abs(sum(alpha)-1)<1e-10 and min(alpha)>=0
 for x in rec['fields']:
  feasible(x);assert max(abs(a-b) for a,b in zip(x,base))<=rec['radius']+1e-15
 q=[sum(alpha[j]*G[j][i] for j in range(21)) for i in range(len(active))]
 r=[min(x,0.) if base[c]==0 else max(x,0.) if base[c]==.9 else x for x,c in zip(q,active)]
 residual=math.sqrt(sum(x*x for x in r));assert math.isclose(residual,rec['sampled_residual_l2'],rel_tol=1e-10,abs_tol=1e-12)
 d=[0.]*600
 for c,x in zip(active,r):d[c]=-x
 dn=max(map(abs,d))
 if dn>0:d=[x/dn for x in d]
 assert max(abs(a-b) for a,b in zip(d,rec['direction']))<1e-10
 gradient=[sum(g[i]*r[i] for i in range(len(r))) for g in G];gap=max(sum(a*b for a,b in zip(alpha,gradient))-min(gradient),0.);assert math.isclose(gap,rec['simplex_frank_wolfe_gap'],rel_tol=1e-5,abs_tol=1e-13);assert rec['simplex_converged']==(gap<=1e-14+1e-8*residual**2/2)
 qp.append(dict(radius=rec['radius'],residual=residual,converged=rec['simplex_converged']))
assert final['generalized_stationarity_certified'] is False
print(json.dumps(dict(passed=True,accepted=accepted,objective_improvement=final['objective_improvement'],selected_branch_pg=final['selected_branch_projected_gradient'],evaluations=len(s['attempts']),gradient_assemblies=s['gradient_assemblies'],qp=qp,scope='Structural and algebraic audit; independent full numerical endpoint check is separate.'),indent=2))
