from load_vec import *
import statistics as st
T=load('test/rows_shard*.tsv')
S=[r for r in T if r['selected']=='1' and r['weights']=='obs' and r['solver']!='ZERO']
SO=['SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0']
zl=np.round(z,1); layers=sorted(set(zl))[1:]
cell=collections.defaultdict(list)
for r in S: cell[(r['arm'],int(r['exp_index']),r['solver'])].append(r)
def ms(v):
    v=[x for x in v if not math.isnan(x)]
    return (st.mean(v), st.stdev(v) if len(v)>1 else float('nan')) if v else (float('nan'),float('nan'))
P=print
P('## Item 8: spread of the secondary metrics and metrics by height band (test, selected rows, 8 realisations; mean +- sd)\n')
P('Source: test/rows_shard*.tsv columns anom_rmse, bg_rmse, corr, s1_mean..s3_mean, relres, chi2m; height-band RMSE from the saved selected reconstruction vectors (640 rec files) and the truth field T in gate/cells.tsv. The saved vectors reproduce the logged rmse_s of all 640 selected rows to 5.6e-17.\n')
P('### 8a. Spread of the secondary metrics (sd over the 8 realisations)\n')
P('Cell = mean +- sd. corr = Pearson over D_s; anom = anomaly RMSE over D_s cells with w_true > 0 (n = 23); bg = background RMSE over D_s cells with w_true = 0; s1,s2,s3 = mean recovered w in structures a,b,c inside D_s (truth values 0.4, 0.6, 0.8 per the earlier facts file).\n')
cols=[('corr','corr'),('anom_rmse','anom'),('bg_rmse','bg'),('s1_mean','s1'),('s2_mean','s2'),('s3_mean','s3')]
for sv in SO:
    P(f'#### {sv}\n')
    P('| arm | tau | '+' | '.join(c[1] for c in cols)+' |'); P('|---|---|'+'---|'*len(cols))
    for a in ['X','Mc','M']:
        for e in range(1,8):
            rr=cell.get((a,e,sv))
            if not rr: continue
            out=[]
            for c,_ in cols:
                m,s=ms([float(r[c]) for r in rr]); out.append(f'{m:.3f} +- {s:.3f}')
            P(f'| {a} | {tl(e)} | '+' | '.join(out)+' |')
    P('')
# bands
P('### 8b. RMSE by height band (cell layer; 5 evaluated layers, centroid heights in m above the detector plane, as in gate/cells.tsv column z)\n')
P('Layer layout: 100 cells per layer; the lowest layer (z = 159.8 m) is outside the 500-cell evaluation mask. Per layer: number of crossed cells in D_s (the 72 cells of rmse_s) and number of truth-nonzero cells in D_s. rmse_Ds(layer) = RMS over the D_s cells of that layer; rmse_eval(layer) = RMS over all 100 evaluated cells of that layer (uncrossed cells are fixed at 0 by every solver, truth-nonzero uncrossed cells count as error).\n')
P('| layer | z m | D_s cells | truth nonzero in D_s | truth nonzero in layer (all 100 cells) |'); P('|---|---|---|---|---|')
for i,zz in enumerate(layers,1):
    m=zl==zz; P(f'| L{i} | {zz:.1f} | {int((m&Ds).sum())} | {int((m&Ds&(wt>0)).sum())} | {int((m&(wt>0)).sum())} |')
P('')
def bandvals(w):
    out=[]
    for zz in layers:
        m=(zl==zz)&Ds; m2=(zl==zz)&(ev==1)
        out.append((rmse(w,m),rmse(w,m2)))
    return out
zero=np.zeros(len(wt)); zb=bandvals(zero)
tsv=open(WORK+'item8_bands.tsv','w'); tsv.write('arm\ttau\tsolver\tlayer\tz_m\trmse_Ds_mean\trmse_Ds_sd\trmse_eval_mean\trmse_eval_sd\tn\n')
res={}
for key,rr in cell.items():
    per=[bandvals(V[r['rec']]) for r in rr]
    for li in range(len(layers)):
        a=[p[li][0] for p in per]; b=[p[li][1] for p in per]
        ma,sa=ms(a); mb,sb=ms(b); res[(key,li)]=(ma,sa,mb,sb)
        tsv.write(f"{key[0]}\t{tl(key[1])}\t{key[2]}\t{li+1}\t{layers[li]:.1f}\t{ma:.5f}\t{sa:.5f}\t{mb:.5f}\t{sb:.5f}\t{len(rr)}\n")
tsv.close()
P('Zero-field RMSE per layer (rmse_Ds / rmse_eval): '+'; '.join(f'L{i+1} {zb[i][0]:.3f} / {zb[i][1]:.3f}' for i in range(len(layers)))+'\n')
P('Full table of all (arm, tau, solver, layer) means and sds: `item8_bands.tsv` (in the work directory) (columns rmse_Ds_mean/sd and rmse_eval_mean/sd). Below: rmse_Ds by layer, mean +- sd, for SART, MLEM, GN-lin, GN-0, MLEM->GN, GD-0 at tau 1e4, 1e6, 1e8 (arms X and Mc), and M.\n')
for a in ['X','Mc','M']:
    for e in [3,6,7]:
        if not any((a,e,s) in cell for s in SO): continue
        P(f'#### arm {a}, tau {tl(e)}: rmse_Ds by layer\n')
        P('| solver | '+' | '.join(f'L{i+1} ({layers[i]:.0f} m)' for i in range(len(layers)))+' | all D_s |'); P('|---|'+'---|'*(len(layers)+1))
        for s in SO:
            if (a,e,s) not in cell: continue
            tot=ms([float(r['rmse_s']) for r in cell[(a,e,s)]])
            P(f'| {s} | '+' | '.join(f'{res[((a,e,s),li)][0]:.3f} +- {res[((a,e,s),li)][1]:.3f}' for li in range(len(layers)))+f' | {tot[0]:.3f} +- {tot[1]:.3f} |')
        P('')
