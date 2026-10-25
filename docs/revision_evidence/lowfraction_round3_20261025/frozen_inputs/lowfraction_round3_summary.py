#!/usr/bin/env python3
"""Frozen paired jackknife diagnostics; no automatic adoption."""
import csv,json,math,statistics as st,sys
from pathlib import Path
ARMS=('control','candidate')
KEYS=[(b,k) for b in range(1,13) for k in range(1,5)]
def jack(full,loo):
    return {'mean':full,'SE':math.sqrt(7/8*sum((x-st.mean(loo))**2 for x in loo))}
def summarize(folder):
    folder=Path(folder)
    def load(name):
        rows=list(csv.DictReader((folder/name).open()))
        for r in rows:
            for k,v in r.items():
                if k!='arm' and not math.isfinite(float(v)):raise ValueError('nonfinite raw data')
        return rows
    fr=load('heldout_flux.csv');dr=load('heldout_derivatives.csv');ad=load('ad_fd_steps.csv')
    def group(rows):
        out={}
        for r in rows:
            k=(int(r['bin']),int(r['field']),r['arm']);out.setdefault(k,[]).append(r)
        if set(out)!={(b,k,a) for b,k in KEYS for a in ARMS}:raise ValueError('unexpected/missing cases')
        for rows in out.values():
            rows.sort(key=lambda r:int(r['replicate']))
            if [int(r['replicate']) for r in rows]!=list(range(1,9)):raise ValueError('replicate IDs mismatch')
        return out
    F,D=group(fr),group(dr);uncertain=[];sign_uncertain=[];cases=[];stats={};loos={}
    def stat(f,d,keep):
        mean=lambda name,rows:st.mean(float(rows[i][name]) for i in keep)
        wet,dry=mean('mc',f),mean('mc_dry',f)
        if min(wet,dry)<=0:raise ValueError('nonpositive flux reference')
        return {'ratio':(float(f[0]['csda'])/float(f[0]['csda_dry']))/(wet/dry)-1,'absolute':float(f[0]['csda'])/wet-1,'derivative':mean('csda_secant',d)-mean('mc_secant',d)}
    for b,k in KEYS:
        for i in range(8):
            fc,fn=F[b,k,'control'][i],F[b,k,'candidate'][i]
            dc,dn=D[b,k,'control'][i],D[b,k,'candidate'][i]
            if any(fc[x]!=fn[x] for x in ('seed','mc','mc_dry')) or any(dc[x]!=dn[x] for x in ('seed','mc_secant','mc_lo','mc_hi','cell','lo','hi')) or fc['seed']!=dc['seed']:raise ValueError('paired references mismatch')
        case={'bin':b,'field':k,'arms':{},'paired_absolute_change':{}}
        for arm in ARMS:
            f,d=F[b,k,arm],D[b,k,arm]
            if any(int(r['seed'])!=2046102501+100000*b+1000*k+10*(i+1) for i,r in enumerate(f)):raise ValueError('unplanned seed')
            if any(r[x]!=f[0][x] for r in f for x in ('csda','csda_dry')):raise ValueError('model not invariant across replicates')
            full=stat(f,d,range(8));loo=[stat(f,d,[j for j in range(8) if j!=i]) for i in range(8)]
            stats[b,k,arm]=full;loos[b,k,arm]=loo
            case['arms'][arm]={x:jack(full[x],[s[x] for s in loo]) for x in full}
            for x in full:
                if abs(full[x])<=2*case['arms'][arm][x]['SE']:sign_uncertain.append([b,k,arm,x,'bias sign unresolved; diagnostic only'])
            for name,rows in [('mc',f),('mc_dry',f),('mc_secant',d)]:
                vals=[float(r[name]) for r in rows];se=st.stdev(vals)/math.sqrt(8)
                if abs(st.mean(vals))<=5*se:uncertain.append([b,k,arm,name,'reference precision unresolved'])
        for x in ('ratio','absolute','derivative'):
            full=abs(stats[b,k,'candidate'][x])-abs(stats[b,k,'control'][x])
            loo=[abs(loos[b,k,'candidate'][i][x])-abs(loos[b,k,'control'][i][x]) for i in range(8)]
            case['paired_absolute_change'][x]=jack(full,loo)
            if x in ('absolute','derivative') and abs(full)<=2*case['paired_absolute_change'][x]['SE']:uncertain.append([b,k,x,'nonworsening unresolved'])
        cases.append(case)
    def gain(omitted=None):
        med={}
        for arm in ARMS:
            vals=[abs((stats[b,k,arm] if omitted is None else loos[b,k,arm][omitted])['ratio']) for b,k in KEYS]
            med[arm]=st.median(vals)
        if med['control']<=1e-30:raise ValueError('zero control median; unresolved')
        return 1-med['candidate']/med['control']
    gs=jack(gain(),[gain(i) for i in range(8)])
    if len(ad)!=48*2*3:raise ValueError('AD ladder incomplete')
    ladder={}
    for r in ad:ladder.setdefault((int(r['bin']),int(r['field']),r['arm']),[]).append(r)
    if set(ladder)!=set(F):raise ValueError('AD cases mismatch')
    for rows in ladder.values():
        if sorted(float(r['step']) for r in rows)!=[1e-6,1e-5,1e-4]:raise ValueError('AD ladder steps mismatch')
    maxad=max(abs(float(r['ad'])-float(r['fd']))/max(abs(float(r['ad'])),abs(float(r['fd'])),1e-30) for r in ad)
    laddererr=max(max(abs(float(r['fd'])-float(s['fd'])) for r in rows for s in rows)/max(max(abs(float(r['fd'])) for r in rows),1e-30) for rows in ladder.values())
    worsening=any(c['paired_absolute_change'][x]['mean']>2*c['paired_absolute_change'][x]['SE'] for c in cases for x in ('absolute','derivative'))
    def derivative_gain(omitted=None):
        return st.median(abs((stats[b,k,'control'] if omitted is None else loos[b,k,'control'][omitted])['derivative']) for b,k in KEYS)-st.median(abs((stats[b,k,'candidate'] if omitted is None else loos[b,k,'candidate'][omitted])['derivative']) for b,k in KEYS)
    dg=jack(derivative_gain(),[derivative_gain(i) for i in range(8)])
    localchecks=[]
    for key,rows in sorted(ladder.items()):
        errors=[abs(float(r['ad'])-float(r['fd']))/max(abs(float(r['ad'])),abs(float(r['fd'])),1e-30) for r in rows]
        scale=max(max(abs(float(r['fd'])) for r in rows),1e-30)
        err=max(abs(float(r['fd'])-float(s['fd'])) for r in rows for s in rows)/scale
        finest=min(rows,key=lambda r:float(r['step']))
        finesterr=abs(float(finest['ad'])-float(finest['fd']))/max(abs(float(finest['ad'])),abs(float(finest['fd'])),1e-30)
        tag='stable ladder' if max(errors)<1e-4 and err<1e-4 else ('larger-step crossing candidate' if finesterr<1e-4 else 'smallest-step unresolved')
        localchecks.append(dict(bin=key[0],field=key[1],arm=key[2],per_step=[dict(step=float(r['step']),relative_error=e) for r,e in zip(rows,errors)],maximum_error=max(errors),finest_error=finesterr,ladder_error=err,stable_ladder=err<1e-4,tag=tag,status='resolved' if tag=='stable ladder' else 'unresolved'))
    if gs['mean']+2*gs['SE']<.3 or worsening:decision='reject'
    elif uncertain or maxad>=1e-4 or laddererr>=1e-4 or gs['mean']-.3<=2*gs['SE'] or dg['mean']<=2*dg['SE']:decision='unresolved'
    else:decision='promising screening only; independent confirmation required'
    aggregates={a:{x:{'median_absolute_bias':st.median(abs(stats[b,k,a][x]) for b,k in KEYS),'maximum_absolute_bias':max(abs(stats[b,k,a][x]) for b,k in KEYS)} for x in ('ratio','absolute','derivative')} for a in ARMS}
    return dict(local_derivative_checks=localchecks,aggregates=aggregates,decision=decision,scope='descriptive conditional jackknife SE; no coverage guarantee, no adoption',gain=gs,derivative_median_gain=dg,ladder_error=laddererr,max_ad_fd_relative_error=maxad,unresolved_endpoints=uncertain,residual_sign_diagnostics=sign_uncertain,cases=cases)
if __name__=='__main__':
    try:result=summarize(sys.argv[1])
    except (ValueError,ZeroDivisionError) as exc:result={'decision':'unresolved','reason':str(exc)}
    print(json.dumps(result,indent=2))
