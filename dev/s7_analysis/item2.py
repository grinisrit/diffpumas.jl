from load_vec import *
T=load('test/rows_shard*.tsv')
sel_=sel['selection']
rows={}
for r in T:
    if r['rec'] and r['solver'] in ('GN-0','MLEM->GN') and r['weights']=='obs':
        rows[(r['arm'],int(r['exp_index']),int(r['rep']),r['solver'],r['iter'])]=r
M=None
# M = number of bins from chi2m: need count; take from template length
import pickle
def getM():
    # template length via python? use obs count unknown -> read from dump of template not available; use 2700 minus invalid? print later
    return None
print('## Item 2: GN-0 against MLEM->GN at equal c (test rows, reps 1-2, saved iterates)\n')
print('Cells where selected c of GN-0 and MLEM->GN are equal (selection.toml): ')
eq=[]
for arm in ['X','Mc','M']:
    for e in [1,3,4,5,6,7]:
        a=sel_.get(f'GN-0|{arm}|{e}'); b=sel_.get(f'MLEM->GN|{arm}|{e}')
        if a and b and a['param']==b['param'] and any(k[0]==arm and k[1]==e for k in rows): eq.append((arm,e,a['param'],a['iter'],b['iter']))
print('| arm | tau | c | K GN-0 | K MLEM->GN |'); print('|---|---|---|---|---|')
for x in eq: print(f'| {x[0]} | {tl(x[1])} | {x[2]:g} | {x[3]} | {x[4]} |')
print()
print('Quantities per realisation (rep 1, 2): d_sel = max|w_GN0(K_sel) - w_MLEMGN(K_sel)| over the 600 cells, each solver at its own selected K; d_fin = same at `final`; d_K1, d_K2 = at equal K = 1, 2; d_free = d_sel over the 76 crossed cells only. rmse_s of both at selected K, and the data part of the objective, 0.5 M chi2/M, from the logged `chi2m` (M = 2700 valid bins = length of the saved templates), for the selected-K iterate. The prior part of the objective is NOT in any row or rec (update_log is not serialised) and was not computed; GN objective values are therefore not available in full.\n')
print('| arm | tau | c | rep | K sel (GN-0 / M->GN) | d_sel | d_free | d_fin | d_K1 | d_K2 | rmse_s GN-0 | rmse_s M->GN | 0.5 M chi2/M GN-0 | 0.5 M chi2/M M->GN | stop GN-0 / M->GN |'); print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
out=[]
for arm,e,c,ka,kb in eq:
    for rep in (1,2):
        ra=rows.get((arm,e,rep,'GN-0',ka)); rb=rows.get((arm,e,rep,'MLEM->GN',kb))
        if ra is None or rb is None: print(f'| {arm} | {tl(e)} | {c:g} | {rep} | missing | | | | | | | | | | |'); continue
        va=V[ra['rec']]; vb=V[rb['rec']]
        d=lambda x,y: float(np.max(np.abs(x-y)))
        def get(solver,lbl):
            r=rows.get((arm,e,rep,solver,lbl)); return V[r['rec']] if r else None
        fa,fb=get('GN-0','final'),get('MLEM->GN','final')
        k1=[get('GN-0','K=1'),get('MLEM->GN','K=1')]; k2=[get('GN-0','K=2'),get('MLEM->GN','K=2')]
        dfin=d(fa,fb) if fa is not None and fb is not None else float('nan')
        dk1=d(*k1) if None not in [x is not None and 1 for x in k1] and all(x is not None for x in k1) else float('nan')
        dk2=d(*k2) if all(x is not None for x in k2) else float('nan')
        fr=cr==1
        print(f"| {arm} | {tl(e)} | {c:g} | {rep} | {ka.replace('K=','')} / {kb.replace('K=','')} | {d(va,vb):.4f} | {float(np.max(np.abs(va-vb)[fr])):.4f} | {dfin:.4f} | {dk1:.4f} | {dk2:.4f} | {float(ra['rmse_s']):.4f} | {float(rb['rmse_s']):.4f} | {1350*float(ra['chi2m']):.1f} | {1350*float(rb['chi2m']):.1f} | {ra['stop']} / {rb['stop']} |")
print()
print('Per cell, maximum over reps 1-2 of the maximum absolute difference (the quantity of the protocol attribution check, Q2.8: GN-0 against MLEM->GN at K = final) and at the selected K:\n')
print('| arm | tau | c | max_abs_diff at final (over reps 1-2) | max_abs_diff at selected K (over reps 1-2) |'); print('|---|---|---|---|---|')
for arm,e,c,ka,kb in eq:
    dl=[];ds=[]
    for rep in (1,2):
        ra=rows.get((arm,e,rep,'GN-0',ka)); rb=rows.get((arm,e,rep,'MLEM->GN',kb))
        fa=rows.get((arm,e,rep,'GN-0','final')); fb=rows.get((arm,e,rep,'MLEM->GN','final'))
        if fa and fb: dl.append(float(np.max(np.abs(V[fa['rec']]-V[fb['rec']]))))
        if ra and rb: ds.append(float(np.max(np.abs(V[ra['rec']]-V[rb['rec']]))))
    print(f'| {arm} | {tl(e)} | {c:g} | {max(dl):.4f} | {max(ds):.4f} |')
print('\nNote: GN-0 starts at 0 and MLEM->GN starts at the selected MLEM field; at equal c the two minimise the same objective (same W, prior, free cells, nonlinear prediction) from different starts. The `final` iterate is the iterate at the solver stop (stop reasons in the table above), not necessarily a converged minimum (stop = max_updates in some runs). Cells where c differs (not listed): X 3.16e4 (1 vs 10), Mc 1e4 (10 vs 100), Mc 3.16e4 (1 vs 10), Mc 1e8 (0.1 vs 0.01), M 1e4 (10 vs 100), M 1e8 (0.1 vs 0.01).')
