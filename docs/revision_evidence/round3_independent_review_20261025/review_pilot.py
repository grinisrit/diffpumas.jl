#!/usr/bin/env python3
"""Independent vectorized locked-estimand audit. No fitting or additional draws."""
import csv,json,math,sys,tomllib,hashlib
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);root=Path(__file__).resolve().parents[3]
def read(n):return list(csv.DictReader((p/n).open()))
F=read('heldout_flux.csv');D=read('heldout_derivatives.csv');A=read('ad_fd_steps.csv');TR=read('training.csv')
assert len(F)==len(D)==768 and len(A)==288 and len(TR)==36
keys=[(b,k) for b in range(1,13) for k in range(1,5)];arms=['control','candidate']
def group(rows):
 result={}
 for r in rows:
  key=(int(r['bin']),int(r['field']),r['arm'],int(r['replicate']));assert key not in result;result[key]=r
 assert set(result)=={(b,k,a,i) for b,k in keys for a in arms for i in range(1,9)}
 return result
fg,dg=group(F),group(D)
mc=np.empty((48,8));dry=np.empty_like(mc);md=np.empty_like(mc);cs=np.empty((2,48));cd=np.empty_like(cs);ds=np.empty((2,48,8));q=np.empty(48)
for j,(b,k) in enumerate(keys):
 for a,arm in enumerate(arms):
  for i in range(8):
   f=fg[b,k,arm,i+1];d=dg[b,k,arm,i+1];ref=fg[b,k,'control',i+1];rd=dg[b,k,'control',i+1]
   assert int(f['seed'])==int(d['seed'])==2046102501+100000*b+1000*k+10*(i+1)
   assert all(f[x]==ref[x] for x in ['mc','mc_dry','column_water_fraction','dry_column_water_fraction'])
   assert all(d[x]==rd[x] for x in ['mc_secant','mc_lo','mc_hi','cell','lo','hi','column_fraction_lo','column_fraction_hi'])
   assert float(f['dry_column_water_fraction'])==0 and 0<=float(f['column_water_fraction'])<=.15
   assert all(f[x]==fg[b,k,arm,1][x] for x in ['csda','csda_dry'])
   assert math.isclose(float(d['mc_secant']),(float(d['mc_hi'])-float(d['mc_lo']))/(float(d['hi'])-float(d['lo'])),rel_tol=1e-12,abs_tol=1e-20)
   mc[j,i]=float(f['mc']);dry[j,i]=float(f['mc_dry']);md[j,i]=float(d['mc_secant']);ds[a,j,i]=float(d['csda_secant'])
  cs[a,j]=float(f['csda']);cd[a,j]=float(f['csda_dry']);q[j]=float(f['column_water_fraction'])
assert np.isfinite(mc).all() and np.isfinite(dry).all() and np.isfinite(md).all() and (mc>0).all() and (dry>0).all()
assert all(np.isfinite(v).all() for v in [cs,cd,ds,q])
assert np.array_equal(cd[0],cd[1])
assert {(int(r['bin']),int(r['field'])) for r in TR}=={(b,k) for b in range(1,13) for k in range(1,4)}
for r in TR:assert int(r['seed'])==2036102501+10000*int(r['bin']) and float(r['dry'])>0 and float(r['wet'])>0
control=tomllib.loads((root/'docs/revision_evidence/surrogate_v2/production/material_calibration.toml').read_text());candidate=tomllib.loads((p/'candidate_frozen_before_validation.toml').read_text())['material']
predicted=np.exp((candidate['linear']-control['linear'])*q+(candidate['quadratic']-control['quadratic'])*q*q)
assert np.allclose(cs[1]/cs[0],predicted,rtol=1e-12,atol=1e-15)
def residual(idx):
 wet=mc[:,idx].mean(axis=1);d=dry[:,idx].mean(axis=1)
 return np.stack([cs/cd/(wet/d)-1,cs/wet-1,ds[:,:,idx].mean(axis=2)-md[:,idx].mean(axis=1)])
full=residual(np.arange(8));loo=np.stack([residual(np.array([j for j in range(8) if j!=i])) for i in range(8)])
def se(v):return np.sqrt(7/8*np.sum((v-v.mean(axis=0))**2,axis=0))
change=np.abs(full[:,1])-np.abs(full[:,0]);lochange=np.abs(loo[:,:,1])-np.abs(loo[:,:,0]);chgse=se(lochange)
gain=1-np.median(abs(full[0,1]))/np.median(abs(full[0,0]));lgain=1-np.median(abs(loo[:,0,1]),axis=1)/np.median(abs(loo[:,0,0]),axis=1);gse=float(se(lgain))
dgain=np.median(abs(full[2,0]))-np.median(abs(full[2,1]));ldgain=np.median(abs(loo[:,2,0]),axis=1)-np.median(abs(loo[:,2,1]),axis=1);dgse=float(se(ldgain))
refs_unresolved=any(np.any(abs(v.mean(axis=1))<=5*v.std(axis=1,ddof=1)/np.sqrt(8)) for v in [mc,dry,md])
margin_unresolved=any(np.any(abs(change[i])<=2*chgse[i]) for i in [1,2]);worsening=any(np.any(change[i]>2*chgse[i]) for i in [1,2])
aderr=max(abs(float(r['ad'])-float(r['fd']))/max(abs(float(r['ad'])),abs(float(r['fd'])),1e-30) for r in A)
lad={}
for r in A:lad.setdefault((int(r['bin']),int(r['field']),r['arm']),[]).append(r)
assert set(lad)=={(b,k,a) for b,k in keys for a in arms}
for rows in lad.values():assert sorted(float(r['step']) for r in rows)==[1e-6,1e-5,1e-4]
laderr=max((max(float(r['fd']) for r in rows)-min(float(r['fd']) for r in rows))/max(max(abs(float(r['fd'])) for r in rows),1e-30) for rows in lad.values())
decision='reject' if gain+2*gse<.3 or worsening else 'unresolved' if refs_unresolved or margin_unresolved or aderr>=1e-4 or laderr>=1e-4 or gain-.3<=2*gse or dgain<=2*dgse else 'promising screening only; independent confirmation required'
tq=np.array([float(r['column_water_fraction']) for r in TR]);domain=tomllib.loads((p/'column_fraction_domain.toml').read_text())
for name,value in [('training_q_min',tq.min()),('training_q_max',tq.max()),('heldout_q_min',q.min()),('heldout_q_max',q.max())]:assert domain[name]==value
assert domain['heldout_below_training_min_count']==np.count_nonzero(q<tq.min()) and domain['heldout_above_training_max_count']==np.count_nonzero(q>tq.max())
report=dict(passed=True,decision=decision,gain=float(gain),gain_SE=gse,derivative_gain=float(dgain),derivative_gain_SE=dgse,max_ad_fd_relative_error=aderr,ladder_error=laderr,reference_precision_unresolved=bool(refs_unresolved),paired_margin_unresolved=bool(margin_unresolved),q_domain=domain,aggregates={a:{label:dict(median_absolute_bias=float(np.median(abs(full[i,j]))),maximum_absolute_bias=float(np.max(abs(full[i,j])))) for i,label in enumerate(['ratio','absolute','derivative'])} for j,a in enumerate(arms)},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Independent locked arithmetic and raw pairing/q-domain audit; no uncertainty-coverage or physical-accuracy certification.')
if len(sys.argv)>2:
 locked=json.loads(Path(sys.argv[2]).read_text());assert locked['decision']==decision
 for x,y in [(gain,locked['gain']['mean']),(gse,locked['gain']['SE']),(dgain,locked['derivative_median_gain']['mean']),(dgse,locked['derivative_median_gain']['SE'])]:assert math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-12)
 case_errors=[];stderr=se(loo)
 for j,(b,k) in enumerate(keys):
  case=next(x for x in locked['cases'] if (x['bin'],x['field'])==(b,k))
  for i,label in enumerate(['ratio','absolute','derivative']):
   for a,arm in enumerate(arms):
    for v,expected in [(full[i,a,j],case['arms'][arm][label]['mean']),(stderr[i,a,j],case['arms'][arm][label]['SE'])]:
     assert math.isclose(v,expected,rel_tol=1e-9,abs_tol=1e-12);case_errors.append(abs(float(v)-expected))
   for v,expected in [(change[i,j],case['paired_absolute_change'][label]['mean']),(chgse[i,j],case['paired_absolute_change'][label]['SE'])]:
    assert math.isclose(v,expected,rel_tol=1e-9,abs_tol=1e-12);case_errors.append(abs(float(v)-expected))
 report['case_statistics_checked']=len(case_errors);report['max_case_statistic_absolute_discrepancy']=max(case_errors)

print(json.dumps(report,indent=2))
