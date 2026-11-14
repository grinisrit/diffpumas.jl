from common import *
import numpy as np
T=load('test/rows_shard*.tsv')
S=[r for r in T if r['selected']=='1' and r['weights']=='obs']
print('## Item 4: solves, forward and Jacobian evaluations, run times (from test rows; no timing run)\n')
print('Source: columns `secs`, `fwd`, `jac`, `updates` of test/rows_shard*.tsv (one value per solver run, repeated on every K row of that run; counters in `EvalCounter`, Tomography_protocol.jl:299-302 and `run_gn`/`run_gd`: `fwd` = flux-only predictions counted in line searches, `jac` = full model (prediction + Jacobian) evaluations, both inside the solver; `secs` = wall time of the solver call only (time() around the solve), measured while up to 34 processes ran concurrently (resources.log), so it is a loaded time, not a benchmark). SART and MLEM use matrix products on stored J0 and report fwd = jac = 0. The rows of one run are identical in these columns. Runs of reps 1-2 go to the cap (GN 20, or 32 once; GD 40); reps 3-8 stop at the selected K (cut 0), so reps 3-8 counts are truncated and are given separately for GN/GD. Run-level total (all test rows, one value per run): see the last table.\n')
def runs(rows):
    d={}
    for r in rows:
        k=(r['arm'],r['solver'],r['exp_index'],r['rep'],r['cap'],r['param'])
        d[k]=r
    return d
allr=[r for r in T if r['weights']=='obs' and r['solver']!='ZERO']
R=runs(allr)
print(f'Number of distinct solver runs in the test stage (excluding ZERO): {len(R)}.\n')
rows=[]
SO=['SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0']
print('Median (and min-max) per run, by solver and arm, over exposures and realisations. Split reps 1-2 (to cap) / reps 3-8 (stop at selected K).\n')
print('| solver | arm | reps | runs | secs median [min, max] | fwd median [min,max] | jac median [min,max] | updates/iterations median [min,max] |'); print('|---|---|---|---|---|---|---|---|')
for s in SO:
  for a in ['X','Mc','M']:
    for lab,rr in (('1-2',(1,2)),('3-8',tuple(range(3,9)))):
        v=[x for k,x in R.items() if k[0]==a and k[1]==s and int(k[3]) in rr]
        if not v: continue
        def q(c):
            z=np.array([float(x[c]) for x in v]); return f'{np.median(z):.4g} [{z.min():.4g}, {z.max():.4g}]'
        print(f'| {s} | {a} | {lab} | {len(v)} | {q("secs")} | {q("fwd")} | {q("jac")} | {q("updates")} |')
print()
print('Sum over all runs of the test stage (secs are loaded wall times of the solver calls; the sum adds the solver calls of all workers, not the elapsed 8.90 h of the test stage):\n')
print('| solver | runs | total secs | total h | total fwd | total jac |'); print('|---|---|---|---|---|---|')
for s in SO:
    v=[x for k,x in R.items() if k[1]==s]
    print(f'| {s} | {len(v)} | {sum(float(x["secs"]) for x in v):.0f} | {sum(float(x["secs"]) for x in v)/3600:.2f} | {sum(float(x["fwd"]) for x in v):.0f} | {sum(float(x["jac"]) for x in v):.0f} |')
print()
print('Per exposure, reps 1-2 (full runs), median secs / jac / fwd, arm Mc and X, GN-0 and MLEM->GN and GN-lin:\n')
print('| arm | solver | '+' | '.join(tl(e) for e in (1,3,4,5,6,7))+' |'); print('|---|---|'+'---|'*6)
for a in ['X','Mc']:
    for s in ['GN-lin','GN-0','MLEM->GN','GD-0','MLEM']:
        cells=[]
        for e in (1,3,4,5,6,7):
            v=[x for k,x in R.items() if k[0]==a and k[1]==s and int(k[2])==e and int(k[3]) in (1,2)]
            cells.append('-' if not v else f'{np.median([float(x["secs"]) for x in v]):.1f}s / {np.median([float(x["jac"]) for x in v]):.0f} / {np.median([float(x["fwd"]) for x in v]):.0f}')
        print(f'| {a} | {s} | '+' | '.join(cells)+' |')
print()
# dev stage counts
D=load('dev/rows_phase*_shard*.tsv')
dr={}
for r in D:
    if r['solver']!='ZERO': dr[(r['arm'],r['solver'],r['exp_index'],r['rep'],r['cap'],r['param'],r['stage'])]=r
tot={}
for k,r in dr.items(): tot.setdefault(k[1],[0,0.0,0,0]); t=tot[k[1]]; t[0]+=1; t[1]+=float(r['secs']); t[2]+=float(r['fwd']); t[3]+=float(r['jac'])
print('Dev stage (all candidates logged; one entry per (arm, solver, exposure, rep, cap, param) run):\n')
print('| solver | runs | total secs | total h | total fwd | total jac |'); print('|---|---|---|---|---|---|')
for s,t in sorted(tot.items()): print(f'| {s} | {t[0]} | {t[1]:.0f} | {t[1]/3600:.2f} | {t[2]:.0f} | {t[3]:.0f} |')
print('\nStage wall times (log_*.txt [orch] lines; see s7_results_facts.md section 1): templates 4.32 h, dev 25.13 h, test 8.90 h, diag 1.00 h. No per-solve timing outside the rows; no separate timing run was made (the timing directory of the out dir is empty; timing_t2.log, dry*.log are earlier runs).')
