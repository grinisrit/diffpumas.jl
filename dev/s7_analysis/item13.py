from common import *
D=load('dev/rows_phase*_shard*.tsv')
print(len(D))
# group rows
G=collections.defaultdict(lambda: collections.defaultdict(dict))  # (solver,arm,exp)-> (cap,param,iter)-> rep -> rmse
dup=0
for r in D:
    if r['weights']!='obs': continue
    k=(r['solver'],r['arm'],int(r['exp_index']))
    kk=(r['cap'],float(r['param']),r['iter'])
    rep=int(r['rep'])
    if rep in G[k][kk]: dup+=1
    G[k][kk][rep]=float(r['rmse_s'])
print('duplicates',dup)
def ik(l): return 10**9 if l=='final' else int(l.split('=')[-1])
rows=[]; chk=0
for key,v in sel['selection'].items():
    s=v['solver'];a=v['arm'];e=v['exp_index']
    cap=str(v.get('cap','')) 
    c={ (p,i):reps for (cp,p,i),reps in G[(s,a,e)].items() if cp==cap and len(reps)==2}
    means={k:sum(r.values())/2 for k,r in c.items()}
    order=sorted(means,key=lambda k:(means[k],-k[0],ik(k[1])))
    best=order[0]
    sp=(v['param'],v['iter'])
    assert abs(means[sp]-v['dev_rmse_s_mean'])<1e-12 and best==sp,(key,best,sp)
    ru=order[1]
    ru_dp=next((k for k in order[1:] if k[0]!=best[0]),None)
    def paired(k2):
        d=[c[sp][r]-c[k2][r] for r in (1,2)]
        return means[k2]-means[sp], d
    m1,d1=paired(ru)
    row=dict(key=key,solver=s,arm=a,exp=e,sel=f'{best[0]:g}/{best[1]}',ru=f'{ru[0]:g}/{ru[1]}',ncand=len(order),selmean=means[sp],margin=m1,nlow=sum(1 for x in d1 if x<0))
    if ru_dp:
        m2,d2=paired(ru_dp); row.update(ru2=f'{ru_dp[0]:g}/{ru_dp[1]}',margin2=m2,nlow2=sum(1 for x in d2 if x<0))
    else: row.update(ru2='-',margin2=float('nan'),nlow2=0)
    rows.append(row)
import pickle; pickle.dump(rows,open(WORK+'item13.pkl','wb'))
print(len(rows),'selections checked: selected = lowest dev mean among complete candidates, dev_rmse_s_mean reproduced')
