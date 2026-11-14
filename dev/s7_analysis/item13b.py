import pickle,statistics as st,math
from common import WORK
rows=pickle.load(open(WORK+'item13.pkl','rb'))
SO=['SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0','MLEM->GD']
rows.sort(key=lambda r:(SO.index(r['solver']),['X','Mc','M'].index(r['arm']),r['exp']))
TL={1:'1e3',2:'3.16e3',3:'1e4',4:'3.16e4',5:'1e5',6:'1e6',7:'1e8'}
print('## Item 13: dev-stage margins (selected setting vs runner-up), per cell\n')
print('Source: dev/rows_phase*_shard*.tsv (weights obs, 2 dev realisations, all candidates with both reps; cap matched to the selection). Candidate = (param, iter label); for GN-family and GD each K is a candidate at the same c. Re-derived selection = lowest mean RMSE_s with the protocol tie rule (Tomography_protocol.jl:606-611); it equals selection.toml in all 113 cells and reproduces dev_rmse_s_mean exactly. margin = mean RMSE_s(runner-up) - mean RMSE_s(selected) (>= 0 by construction); n_low = number of the 2 dev realisations (paired by rep, same observations) in which the selected setting is lower than the runner-up. "other-param runner-up" = best candidate with a different prior weight / q / c_M (relevant for GN where the runner-up is often the neighbouring K at the same c). With 2 realisations no interval is given.\n')
print('| solver | arm | tau | selected (param/iter) | dev mean | runner-up | margin | n_low | other-param runner-up | margin | n_low | n candidates |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['solver']} | {r['arm']} | {TL[r['exp']]} | {r['sel']} | {r['selmean']:.4f} | {r['ru']} | {r['margin']:.5f} | {r['nlow']}/2 | {r['ru2']} | {'-' if math.isnan(r['margin2']) else format(r['margin2'],'.5f')} | {r['nlow2']}/2 | {r['ncand']} |")
print()
print('Distribution of margin by solver (margin to runner-up / to other-param runner-up): median, min, max, number of cells with margin < 0.002, number with n_low < 2.\n')
print('| solver | cells | median margin | min | max | margin<0.002 | n_low<2 | median other-param margin | min | other-param margin<0.002 |')
print('|---|---|---|---|---|---|---|---|---|---|')
for s in SO:
    rr=[r for r in rows if r['solver']==s]
    m=[r['margin'] for r in rr]; m2=[r['margin2'] for r in rr if not math.isnan(r['margin2'])] or [float('nan')]
    print(f"| {s} | {len(rr)} | {st.median(m):.5f} | {min(m):.5f} | {max(m):.5f} | {sum(1 for x in m if x<0.002)} | {sum(1 for r in rr if r['nlow']<2)} | {st.median(m2):.4f} | {min(m2):.4f} | {sum(1 for x in m2 if x<0.002)} |")
