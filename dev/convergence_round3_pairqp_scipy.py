#!/usr/bin/env python3
"""Independent constrained finite-gradient convex hull check; not stationarity."""
import csv, json, sys, tomllib
import numpy as np
from scipy.optimize import minimize
state=tomllib.load(open(sys.argv[1],'rb'))
fields=list(csv.DictReader(open(sys.argv[2])))
active=np.array([i for i,r in enumerate(fields) if r['aquifer_mask']=='true'])
refined=tomllib.load(open(sys.argv[3],'rb'))['results']
results=[]
for sample,julia in zip(state['sample_records'],refined):
    G=np.array(sample['gradients'],dtype=float).T
    w=np.array(sample['field_before'])[active]
    lower=w==0;upper=w==.9
    def parts(alpha):
        q=G@alpha;r=q.copy();r[lower]=np.minimum(r[lower],0);r[upper]=np.maximum(r[upper],0)
        return .5*r@r,G.T@r
    # Rescale only the tiny auxiliary QP for numerical conditioning.
    factor=1e6
    opt=minimize(lambda a:factor*parts(a)[0],np.full(G.shape[1],1/G.shape[1]),
        jac=lambda a:factor*parts(a)[1],bounds=[(0,1)]*G.shape[1],
        constraints=[dict(type='eq',fun=lambda a:np.sum(a)-1,jac=lambda a:np.ones_like(a))],
        method='SLSQP',options=dict(ftol=1e-13,maxiter=2000))
    psi,gradient=parts(opt.x);gap=float(opt.x@gradient-gradient.min())
    comparison=abs(psi-julia['objective'])
    passed=(np.min(opt.x)>=-1e-10 and abs(sum(opt.x)-1)<1e-9 and comparison<=1e-10 and gap<=1e-9)
    results.append(dict(radius=sample['radius'],passed=bool(passed),scipy_success=bool(opt.success),
        scipy_message=opt.message,objective=float(psi),julia_objective=julia['objective'],
        objective_difference=float(comparison),frank_wolfe_gap=gap,alpha=opt.x.tolist()))
with open(sys.argv[4],'x') as stream:json.dump(dict(results=results,passed=all(x['passed'] for x in results)),stream,indent=2)
assert all(x['passed'] for x in results),results
print('Independent scipy finite-QP comparison: PASS',flush=True)
