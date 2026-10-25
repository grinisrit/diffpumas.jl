#!/usr/bin/env python3
"""Synthetic summary checks; no physics, training, or MC execution."""
import csv,importlib.util,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('summary',Path(__file__).with_name('lowfraction_round3_summary.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def dataset(path,candidate=1.,dry=1.):
    flux=[];der=[];ad=[]
    for b,k in m.KEYS:
        for arm,value in [('control',1.3),('candidate',candidate)]:
            for rep in range(1,9):
                seed=2046102501+100000*b+1000*k+10*rep
                flux.append(dict(bin=b,field=k,arm=arm,replicate=rep,seed=seed,csda=value,csda_dry=1.,mc=1.,mc_dry=dry))
                der.append(dict(bin=b,field=k,arm=arm,replicate=rep,seed=seed,cell=1,lo=.01,hi=.03,csda_secant=value,mc_secant=1.,mc_lo=.99,mc_hi=1.01))
            for step in (1e-4,1e-5,1e-6):ad.append(dict(bin=b,field=k,arm=arm,step=step,ad=1.,fd=1.))
    for name,rows in [('heldout_flux.csv',flux),('heldout_derivatives.csv',der),('ad_fd_steps.csv',ad)]:
        with (path/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
with tempfile.TemporaryDirectory(prefix='lowfraction-summary-') as tmp:
    p=Path(tmp);dataset(p)
    r=m.summarize(p)
    assert r['decision'].startswith('promising'),r['decision']
    assert r['residual_sign_diagnostics'] and not r['unresolved_endpoints']
    ladder=list(csv.DictReader((p/'ad_fd_steps.csv').open()))
    ladder[0]['fd']='1.01'
    with (p/'ad_fd_steps.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=ladder[0]);w.writeheader();w.writerows(ladder)
    r=m.summarize(p)
    assert r['decision']=='unresolved'
    assert any(x['tag']=='larger-step crossing candidate' for x in r['local_derivative_checks'])
    ladder[0]['fd']='1.';ladder[2]['fd']='1.01'
    with (p/'ad_fd_steps.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=ladder[0]);w.writeheader();w.writerows(ladder)
    r=m.summarize(p)
    assert r['decision']=='unresolved'
    assert any(x['tag']=='smallest-step unresolved' for x in r['local_derivative_checks'])
    dataset(p,candidate=1.3);r=m.summarize(p)
    assert r['decision']=='reject' # material gain is exactly zero, below30%
    assert any(x[-1]=='nonworsening unresolved' for x in r['unresolved_endpoints'])
    dataset(p,dry=0.)
    try:m.summarize(p)
    except ValueError as e:assert 'nonpositive' in str(e)
    else:raise AssertionError('zero denominator accepted')
print('synthetic zero-bias, paired-identity, zero-denominator and FD interpretation checks passed')
