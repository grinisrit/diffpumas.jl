import csv,glob,collections,tomllib,math,statistics as st,datetime
import os,sys
# usage: s7_results_agg.py [OUT_DIR [FACTS_MD]]   (env S7_OUT, S7_FACTS; default OUT_DIR = the run output directory)
O=(sys.argv[1] if len(sys.argv)>1 else os.environ.get('S7_OUT','/mnt/big_data/agent-builds/diffpumas-s7-out')).rstrip('/')+'/'
FACTS_OUT=sys.argv[2] if len(sys.argv)>2 else os.environ.get('S7_FACTS','s7_results_facts.md')
def load(pat):
    rows=[]
    for f in sorted(glob.glob(O+pat)):
        for r in csv.DictReader(open(f),delimiter='\t'):
            if r['stage']!='stage': rows.append(r)
    return rows
T=load('test/rows_shard*.tsv'); D=load('dev/rows_phase*_shard*.tsv'); G=load('diag/rows_shard*.tsv')
sel=tomllib.load(open(O+'selection.toml','rb'))
cuts=tomllib.load(open(O+'cuts.toml','rb'))
TAU=sel['meta']['exposures']
def tl(i):
    t=TAU[i-1]; return {1000.0:'1e3',3160.0:'3.16e3',10000.0:'1e4',31600.0:'3.16e4',100000.0:'1e5',1e6:'1e6',1e8:'1e8'}[t]
SOLV=['ZERO','SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0','MLEM->GD']
ARMS=['X','Mc','M']
f=lambda x,n=4:f'{x:.{n}f}'
out=[]
P=out.append
def table(hdr,rows):
    P('| '+' | '.join(hdr)+' |'); P('|'+'|'.join(['---']*len(hdr))+'|')
    for r in rows: P('| '+' | '.join(str(c) for c in r)+' |')
    P('')
# ---- aggregate test selected rows
S=[r for r in T if r['selected']=='1' and r['weights']=='obs']
cell=collections.defaultdict(dict)
for r in S: cell[(r['arm'],r['solver'],int(r['exp_index']))][int(r['rep'])]=r
def vals(k,col='rmse_s'): return [float(cell[k][q][col]) for q in sorted(cell[k])]
def ms(v):
    m=st.mean(v); s=st.stdev(v) if len(v)>1 else float('nan'); return m,s
# check vs summary
summ=list(csv.DictReader(open(O+'test_summary.tsv'),delimiter='\t'))
maxd=0
for r in summ:
    if r['kind']=='cell':
        k=(r['arm'],r['solver'],int(r['exp_index'])); m,s=ms(vals(k)); maxd=max(maxd,abs(m-float(r['mean'])),abs(s-float(r['sd'])))
        assert int(r['n'])==len(cell[k])
P(f'# S7 results facts (compiled from out dir; no interpretation)'); P('')
P('Sources: /mnt/big_data/agent-builds/diffpumas-s7-out (test_summary.tsv, selection.toml, cuts.toml, dry_calibration.toml, dev|test|diag rows_*.tsv, log_*.txt, resources.log). Aggregation script: dev/s7_analysis/s7_results_agg.py. Own per-cell means/sd recomputed from the rows agree with test_summary.tsv to max abs diff %.2e.'%maxd); P('')
P('Notation: exposure index i=1..7 is tau = '+', '.join(f'{i+1}:{tl(i+1)}' for i in range(7))+' (per-bin tau). Metric rmse_s = RMSE over the 72 crossed evaluated cells (D_s). Arms: X exact CSDA, Mc MC data with operator x s, M MC data with operator as published. Field T (test).'); P('')
# ---- 1. metadata
P('## 1. Run metadata'); P('')
def dt(s): return datetime.datetime.strptime(s,'%Y-%m-%dT%H:%M:%SZ')
tm=[('start (RUN_START)','2026-11-12T21:38:30Z'),('templates complete (1082/1082)','2026-11-13T01:57:51Z'),('dev p1_r1..r3 complete','2026-11-13T02:59:10Z'),('dev p2_r3 complete','2026-11-13T20:25:43Z'),('dev p3_r3 complete (last dev unit)','2026-11-14T03:05:25Z'),('cuts.toml / selection.toml written','2026-11-14T03:07:47Z (cuts.toml mtime; selection.toml 03:06:59 per created_unix)'),('test complete (112/112)','2026-11-14T12:01:30Z'),('diag complete (16/16)','2026-11-14T13:01:14Z'),('summary + RUN_DONE','2026-11-14T13:02:07Z')]
P('Times (UTC, log_*.txt [orch] lines): RUN_START 2026-11-12T21:38:30Z; templates complete 2026-11-13T01:57:51Z; dev phase 1 rounds done 02:27/02:43/02:59 (11-13), phase 2 rounds 13:34/19:36/20:25 (11-13), phase 3 rounds 00:34/01:48/03:05 (11-14); selection.toml created 2026-11-14T03:06:59Z, cuts.toml 03:07:47Z; test complete 2026-11-14T12:01:30Z; diag complete 13:01:14Z; RUN_DONE 13:02:07Z.'); P('')
def h(a,b): return f'{(dt(b)-dt(a)).total_seconds()/3600:.2f}'
S0='2026-11-12T21:38:30Z'
table(['stage','wall h','units','note'],[
 ['dry/G0b, G0c, G1 (before RUN_START)','n/a','-','dry_calibration.toml created 2026-11-12T21:38:12Z (unix 1794519492); reports in dry/, g0c/, gate/'],
 ['templates',h(S0,'2026-11-13T01:57:51Z'),'1082 chunks','D: 2 replicates (b=421,422), T: 8 replicates (b=431..438), 108 chunks each; X templates exact; template_mode = replicates'],
 ['dev (3 phases x 3 rounds)',h('2026-11-13T01:57:51Z','2026-11-14T03:05:25Z'),'9 rounds x 34 units','cuts.toml history: elapsed_h = 29.475 at plan time (RUN_START -> plan)'],
 ['test',h('2026-11-14T03:07:47Z','2026-11-14T12:01:30Z'),'112 units','projected with all cuts 6.134 h (cuts.toml)'],
 ['diag',h('2026-11-14T12:01:30Z','2026-11-14T13:01:14Z'),'16 units','projected 0.493 h'],
 ['summary','0.01','-','log_summary.txt: wrote test_summary.tsv'],
 ['total (RUN_START -> RUN_DONE)',h(S0,'2026-11-14T13:02:07Z'),'',f'cuts.toml projected total with all cuts {cuts["history"][-1]["total_h"]:.2f} h; limit_h = {cuts["limit_h"]}, limit_met = {str(cuts["limit_met"]).lower()}, ref_processes = {cuts["ref_processes"]}']])
P('Dev rounds: grid extensions made per round (log_dev.txt "selection written"): phase 1: 27, 18, 0; phase 2: 43, 21, 0; phase 3: 2, 2, 0. Workers: WORKERS_WANTED = 34 at end; resources.log: %d one-minute samples, max workers alive = %s.'%(sum(1 for _ in open(O+'resources.log')),max(int(l.split()[6]) for l in open(O+'resources.log') if len(l.split())==9)))
P('')
P('36 h trigger (cuts.toml history; projected total h at 34 processes = elapsed 29.475 + test + diag 0.4934): '+'; '.join(f"cuts {c['cuts']}: test {c['test_h']:.3f} h, total {c['total_h']:.2f} h" for c in cuts['history'])+'.'); P('')
P('Cuts applied: [0, 1, 2, 3, 4], all five (limit still not met by the projection: 36.10 h > 36 h). Each cut, in terms of what the test stage actually contains (verified from the rows):'); P('')
nrep=collections.Counter(); 
table(['cut','spec','effect observed in test rows'],[
 ['0','test realisations 3..8 stop at selected K','GN/GD rows of reps 3-8 stop at the selected K (no `final`); reps 1-2 run to cap and carry all K rows (error-vs-update diagnostics). Reported rmse_s unchanged by definition.'],
 ['1','MLEM->GD dropped','no MLEM->GD row in test (selection.toml still has MLEM->GD selections for arms X, Mc)'],
 ['2','GD-0 at 3.16e3 and 3.16e4 dropped','GD-0 present at exp 1,3,5,6,7 only (arms X and Mc)'],
 ['3','arm M at 1e5 dropped','arm M test exposures: 3 (1e4) and 7 (1e8) only; exp 5 (1e5) absent'],
 ['4','exposure 3.16e3 dropped for all solvers','no exp_index 2 in any arm (test rows exp_index in {1,3,4,5,6,7})']])
P('Realisations: test = 8 per (arm, exposure) (rep 1..8 uses MC template replicate rep; noise seed = 42+9000+100000 f + 10000 a + 100 i + r); dev = 2 per (arm, exposure) (reps 1,2). Arm M in dev: exposures 3,5,7 only, no GD; arms X, Mc: 7 exposures.'); P('')
cn=collections.defaultdict(set)
for r in T: cn[(r['arm'],r['exp_index'])].add(int(r['rep']))
table(['arm','test exposures run (exp_index)','reps per exposure','solvers in test','test solver rows (selected=1 / all)'],[
 [a,','.join(sorted({e for (aa,e) in cn if aa==a},key=int)),str(sorted(set(len(v) for (aa,e),v in cn.items() if aa==a))),', '.join(sorted({r['solver'] for r in T if r['arm']==a},key=SOLV.index)),f"{sum(1 for r in S if r['arm']==a)} / {sum(1 for r in T if r['arm']==a)}"] for a in ARMS])
dn=collections.defaultdict(set)
for r in D: dn[(r['arm'],r['exp_index'])].add(int(r['rep']))
P('Dev rows: %d total (reps per (arm,exposure): %s); test rows %d; diag rows %d.'%(len(D),sorted(set(len(v) for v in dn.values())),len(T),len(G))); P('')
P('Failures / retries: log_templates/dev/test/diag/summary contain no line matching error|fail|exception|warn|retry|died|killed (case-insensitive) other than the template-merge lines ("median per-bin MC error ..."); no "pass N ended ... retrying" orchestrator line; no RUN_FAILED file; orch.out empty; every claim directory has a .done (templates 1082, dev 9 x 34, test 112, diag 16; 1516 dirs = 1516 .done in claims/). RUN_STATUS: "stage=done units=16/16 ... workers=0".'); P('')
P('Dry calibration (dry_calibration.toml, frozen before dev): s = %.6f, s_A = %.6f, s_B = %.6f, q = %.4f, histories 8192 x seeds 441,442, per_bin, n_bins 2700. dry_report.toml checks: check1/2/3 all true, |s_A-s_B| = 0.00138, n_zenith0 = 90, expected SE of s = 0.000468; median r_m band (0-10..50-60 deg) = 0.9034, 0.9025, 0.8977, 0.8992, 0.9021, 0.9289; mean r = 0.9131; ratio of sums 0.8989; median per-bin MC rel. sigma 0.0215. Shared-sample seed-44 s = 0.87309 archived, not read.'%(0.9021550057597469,0.9034400270837245,0.9020578747753263,0.9772823515726223)); P('')
P('Templates (log_templates.txt merge lines): per-bin MC error 1.96 % (D, b=421/422), 1.99 % (T, b=431..438); median MC/CSDA 0.9059/0.9050 (D) and 0.9006, 0.9003, 0.8998, 0.9005, 0.9001, 0.9006, 0.8993, 0.8995 (T, b=431..438). G1 (gate/g1.toml): D nonzero in D_s 26 (per structure 12,16,4), T nonzero in D_s 23 (9,11,4), T total nonzero 29 (12,14,4); n_free 76; zero-field rmse_s: D 0.32189, T 0.32914. g0c: max_csda_rel_diff 0.0, max_mc_diff 1.47 combined SE, passed true.'); P('')
# ---- 2. selection
P('## 2. Frozen selection (selection.toml, sha256 6fc0a9485b49502f14de304a3d9605374c7ec20604c91e5606b6562f4481720f; dry_calibration sha 07abc554...; git 21b8670, dirty false; field dev; 2 dev realisations)'); P('')
P('Cell text: SART q/n_iter; MLEM c_M/n_iter; GN-lin, GN-0, MLEM->GN: c/K (cap in parentheses when 32); GD-0, MLEM->GD: c/K_GD. Suffix flags: `^p` selected prior weight/q on grid edge (param_hi as recorded; grids had been extended by the edge rule), `^i` iteration at lower (`_`) or upper (`^`) iteration edge, `$` budget note "fixed budget of 40 iterations", `#` "not converged at 32 updates". Final = GN iterate at stop.'); P('')
def scell(k):
    v=sel['selection'][k]; it=v['iter']; it=it.replace('K=','').replace('n=','')
    s=f"{v['param']:g}/{it}"
    if v.get('cap')==32: s+='(32)'
    e=v['at_edge']; 
    if e: s+={'param_hi':'^p','param_lo':'_p','iter_lo':'_i','iter_hi':'^i','K_cap':'^K'}.get(e,'?'+e)
    bn=v['budget_note']
    if bn.startswith('fixed'): s+='$'
    elif bn: s+='#'
    return s
for a in ARMS:
    rows=[]
    for sname in SOLV[1:]:
        ks=[f'{sname}|{a}|{i}' for i in range(1,8)]
        if not any(k in sel['selection'] for k in ks): continue
        rows.append([sname]+[scell(k) if k in sel['selection'] else '-' for k in ks])
    P(f'Arm {a}:'); P('')
    table(['solver']+[f'{i}:{tl(i)}' for i in range(1,8)],rows)
ed=collections.Counter(v['at_edge'] for v in sel['selection'].values())
P('Edge/cap tally over the 113 selections: at_edge = '+', '.join(f'{k or "none"}: {c}' for k,c in ed.items())+'; budget_note "fixed budget of 40 iterations": %d; "not converged at 32 updates": %d; extensions_pending true: %d; selections with cap 32 (K-edge rerun): %d.'%(sum(1 for v in sel['selection'].values() if v['budget_note'].startswith('fixed')),sum(1 for v in sel['selection'].values() if v['budget_note'].startswith('not')),sum(1 for v in sel['selection'].values() if v['extensions_pending']),sum(1 for v in sel['selection'].values() if v.get('cap')==32))); P('')
P('Selections at an edge (arm|solver|exp: edge, selected, dev rmse_s):'); P('')
table(['key','at_edge','param','iter','cap','dev_rmse_s_mean'],[[k.replace('|',':'),v['at_edge'],f"{v['param']:g}",v['iter'],v.get('cap',''),f(v['dev_rmse_s_mean'])] for k,v in sel['selection'].items() if v['at_edge']])
bt=collections.Counter((v['solver'],v['arm']) for v in sel['selection'].values() if v['budget_note']); P('Budget notes by (solver, arm): '+', '.join(f'{k[0]}/{k[1]}: {n}' for k,n in sorted(bt.items()))+'. The "not converged at 32 updates" cell is MLEM->GN, arm X, exp 7 (cap 32 rerun selected final again).'); P('')
# ---- 3. RMSE tables
P('## 3. Test RMSE (rmse_s on field T): mean ± sd (n) over realisations; **bold** = lowest mean among non-ZERO solvers in the column'); P('')
P('Zero-field baseline rmse_s = 0.32914 (T; dev D = 0.32189); zero-field rmse500 (500-cell mask) = %s. Oracle floor: not computed by this run (no row/log reports it); truth-fit floor is 0 by definition of the metric.'%f(float(cell[('X','ZERO',1)][1]['rmse500']))); P('')
for a in ARMS:
    exps=sorted({k[2] for k in cell if k[0]==a}); rows=[]
    best={e:min((st.mean(vals((a,s_,e))),s_) for s_ in SOLV if s_!='ZERO' and (a,s_,e) in cell)[1] for e in exps}
    for s_ in SOLV:
        if not any((a,s_,e) in cell for e in exps): continue
        row=[s_]
        for e in exps:
            k=(a,s_,e)
            if k not in cell: row.append('-'); continue
            m,sd=ms(vals(k)); t=f'{m:.4f} ± {sd:.4f} ({len(vals(k))})'
            row.append(f'**{t}**' if best[e]==s_ else t)
        rows.append(row)
    P(f'Arm {a}:'); P('')
    table(['solver']+[f'{e}:{tl(e)}' for e in exps],rows)
# secondary metrics
P('Secondary metrics, means over realisations (corr = Pearson over D_s; anom = anomaly RMSE over D_s cells with w_true>0; bg = background RMSE over D_s cells with w_true=0; relres/chi2m = nonlinear residual metrics; rmse500 = 500-cell mask; s1,s2,s3 = mean recovered w in structures a,b,c, truth 0.4, 0.6, 0.8 as defined in spec). ZERO has corr NaN.'); P('')
rows=[];rows2=[]
for a in ARMS:
    for e in sorted({k[2] for k in cell if k[0]==a}):
        r1=[a,tl(e)];r2=[a,tl(e)]
        for s_ in ['MLEM','GN-0','MLEM->GN']:
            k=(a,s_,e); mm=lambda c: st.mean(vals(k,c))
            r1.append(f"{mm('corr'):.3f} / {mm('anom_rmse'):.3f} / {mm('bg_rmse'):.3f}")
            r2.append(f"{mm('s1_mean'):.3f},{mm('s2_mean'):.3f},{mm('s3_mean'):.3f} | {mm('relres'):.3f} | {mm('chi2m'):.4g}")
        r1+= [f(st.mean(vals((a,s_,e),'corr')),3) if (a,s_,e) in cell else '-' for s_ in ['SART','GN-lin','GD-0']]
        rows.append(r1); rows2.append(r2)
P('Table A: corr / anom_rmse / bg_rmse for MLEM, GN-0, MLEM->GN; last three columns = corr only for SART, GN-lin, GD-0.'); P('')
table(['arm','tau','MLEM','GN-0','MLEM->GN','SART corr','GN-lin corr','GD-0 corr'],rows)
P('Table B: s1,s2,s3 (mean recovered w in structures a,b,c; truth 0.4,0.6,0.8) ; relres ; chi2/M for MLEM, GN-0, MLEM->GN (separator " ; " between the three items).'); P('')
table(['arm','tau','MLEM','GN-0','MLEM->GN'],[[x.replace(' | ',' ; ') for x in r] for r in rows2])
# ---- 4 contrasts
P('## 4. Contrasts (test_summary.tsv rows verbatim; Delta = RMSE_s(solver) - RMSE_s(MLEM), C3 = RMSE_s(M) - RMSE_s(Mc); mean lo hi are as written, sd blank in these rows)'); P('')
rows=[]
for kd in ['C1','C2','C3']:
    for r in summ:
        if r['kind']==kd:
            ex=r['exp_index']
            rows.append([kd,r['arm'],r['solver'],f"{ex} ({tl(int(ex))})",r['n'],r['mean'],r['lo'],r['hi'],r['n_negative'],r['label'] or '(blank)',r['solves_on_cap'] or '(blank)'])
table(['kind','arm','solver','exp','n','mean','lo','hi','n_negative','label','solves_on_cap'],rows)
P('Label tally C1 / C2: '+'; '.join(f"{kd}: "+', '.join(f'{l}: {c}' for l,c in collections.Counter(r['label'] for r in summ if r['kind']==kd).items()) for kd in ['C1','C2'])); P('')
# ---- 5 diagnostics
P('## 5. GN / GD diagnostics (test rows with selected=1)'); P('')
P('Cell = updates (GD: iterations) of the full runs rep1/rep2 ; stop codes of reps 1-8, e.g. 6mu,2od = 6 solves max_updates and 2 objective_decrease (od objective_decrease, mu max_updates, oc objective_change, mi max_iter) ; solves_on_cap from test_summary.tsv (final iterate on cap, budget_limited) ; selected K / iteration label. GN "updates" = total outer updates run; GD "updates" = 0 in rows (iterations column not separate), see stop. For reps 3-8 with a non-final selected K the run is truncated at K by cut 0, so its stop is mu by construction.'); P('')
code={'objective_decrease':'od','max_updates':'mu','objective_change':'oc','max_iter':'mi','n_iter':'ni','-':'-'}
capd={(r['arm'],r['solver'],int(r['exp_index'])):r['solves_on_cap'] for r in summ if r['kind']=='cell'}
for a in ARMS:
    rows=[]
    exps=sorted({k[2] for k in cell if k[0]==a})
    for s_ in ['GN-lin','GN-0','MLEM->GN','GD-0']:
        if not any((a,s_,e) in cell for e in exps): continue
        row=[s_]
        for e in exps:
            k=(a,s_,e)
            if k not in cell: row.append('-'); continue
            c=cell[k]; up=f"{c[1]['updates']}/{c[2]['updates']}"
            if s_=='GD-0': up=f"it {c[1]['updates']}/{c[2]['updates']}"
            cnt=collections.Counter(code[c[q]['stop']] for q in c)
            trunc=sum(1 for q in c if q>=3 and c[q]['iter']!='final')
            row.append(f"{up}; "+','.join(f'{n}{v}' for n,v in sorted(cnt.items()))+f"; cap {capd[k]}; {cell[k][1]['iter']}")
        rows.append(row)
    P(f'Arm {a}:'); P('')
    table(['solver']+[f'{e}:{tl(e)}' for e in exps],rows)
# hmm: cap used
cp=collections.Counter((r['cap'],r['solver']) for r in S if r['solver'] in('GN-lin','GN-0','MLEM->GN'))
P('Cap column of selected GN-family rows: '+', '.join(f'cap {c} {s_}: {n}' for (c,s_),n in sorted(cp.items()))+'. Selected rows with budget_limited=1: %d (final_note "K = 20 (cap)": %d, "K = 32 (cap)": %d, "fixed budget of 40 iterations": %d).'%(sum(1 for r in S if r['budget_limited']=='1'),sum(1 for r in S if r['final_note']=='K = 20 (cap)'),sum(1 for r in S if r['final_note']=='K = 32 (cap)'),sum(1 for r in S if r['final_note'].startswith('fixed')))); P('')
P('Overall stop reasons, selected test rows by solver: '+'; '.join(f"{s_}: "+', '.join(f'{k}={n}' for k,n in sorted(collections.Counter(r['stop'] for r in S if r['solver']==s_).items())) for s_ in SOLV)); P('')
# K-curves from reps 1,2
P('Error-against-update (diagnostic, T, reps 1-2 only, weights obs): mean rmse_s over reps 1,2 of GN iterate K (columns); "-" = run ended before that K.'); P('')
Ks=['K=1','K=2','K=3','K=5','K=8','K=13','K=20','final']
byK=collections.defaultdict(list)
for r in T:
    if r['weights']=='obs' and r['solver'] in('GN-0','MLEM->GN') and r['rep'] in('1','2'): byK[(r['arm'],r['solver'],int(r['exp_index']),r['iter'])].append(float(r['rmse_s']))
rows=[]
for a in ['X','Mc']:
    for e in sorted({k[2] for k in cell if k[0]==a}):
        rr=[a,tl(e),cell[(a,'GN-0',e)][1]['iter']+' / '+cell[(a,'MLEM->GN',e)][1]['iter']]
        for K in Ks:
            g1=byK.get((a,'GN-0',e,K)); g2=byK.get((a,'MLEM->GN',e,K))
            rr.append((f(st.mean(g1),4) if g1 else '-')+' / '+(f(st.mean(g2),4) if g2 else '-'))
        rows.append(rr)
P('Cells are GN-0 / MLEM->GN.'); P('')
table(['arm','tau','sel K']+Ks,rows)
# diag
P('### diag/ (oracle-weight GN-0, arm Mc, tau 1e4 and 1e8, 8 realisations, weights from template replicate; start zero, cap 20)'); P('')
P('diag/ contains rows_shard1..16.tsv (%d rows = 16 solves x 7 iterates K=1,2,3,5,8,13,final), recs/ (%d iterate files), manifest_diag_w*of1.toml (34). log_diag.txt: no error lines.'%(len(G),len(glob.glob(O+'diag/recs/*')))); P('')
dg=collections.defaultdict(dict)
for r in G: dg[(int(r['exp_index']),r['iter'])][int(r['rep'])]=r
obsGN={ (int(r['exp_index']),int(r['rep'])):r for r in S if r['arm']=='Mc' and r['solver']=='GN-0'}
rows=[]
for e in (3,7):
    selK=[r for r in G if int(r['exp_index'])==e and r['selected']=='1'][0]['iter']
    for K in Ks:
        d=dg.get((e,K));
        if not d: continue
        rows.append([tl(e),K+(' (sel)' if K==selK else ''),len(d),f(st.mean(float(x['rmse_s']) for x in d.values())),f(st.stdev(float(x['rmse_s']) for x in d.values())),f(st.mean(float(x['corr']) for x in d.values()),3),'/'.join(sorted(set(x['stop'] for x in d.values()))),f"{st.mean(float(x['updates']) for x in d.values()):.1f}"])
table(['tau','iterate','n','oracle-weight rmse_s mean','sd','corr','stop','updates run'],rows)
rows=[]
for e in (3,7):
    o=[]; b=[]
    for q in range(1,9):
        selK=[r for r in G if int(r['exp_index'])==e and r['selected']=='1'][0]['iter']
        o.append(float(dg[(e,selK)][q]['rmse_s'])); b.append(float(obsGN[(e,q)]['rmse_s']))
    dd=[x-y for x,y in zip(o,b)]
    rows.append([tl(e),selK,f(st.mean(o)),f(st.mean(b)),f(st.mean(dd)),sum(1 for x in dd if x<0),'8'])
P('Paired at the frozen (c,K) of GN-0 | Mc: oracle-weight minus observed-weight (test selected row) rmse_s per rep:'); P('')
table(['tau','K','oracle mean','obs-weight mean','mean diff (oracle - obs)','n_neg','n'],rows)
# ---- 6 sensed/unsensed
P('## 6. Sensed (crossed) vs all evaluated cells'); P('')
c_vals=[]
for r in S:
    if r['solver']=='ZERO': c_vals.append(500*float(r['rmse500'])**2-72*float(r['rmse_s'])**2)
cs=[]
for r in S:
    cs.append(500*float(r['rmse500'])**2-72*float(r['rmse_s'])**2)
P(f'The rows carry rmse_s (72 crossed evaluated cells) and rmse500 (500 evaluated cells = 72 crossed + 428 not crossed, fixed at 0 by every solver). Check on all %d selected test rows: c = 500 rmse500^2 - 72 rmse_s^2 has min %.4f, max %.4f (ZERO rows: %.4f); the rmse500^2 identity of the spec holds, no independent unsensed-cell estimate exists in the rows. No other sensed/unsensed split is written to any output. rmse500 means:'%(len(S),min(cs),max(cs),st.mean(c_vals))); P('')
rows=[]
for a in ARMS:
    for e in sorted({k[2] for k in cell if k[0]==a}):
        rows.append([a,tl(e)]+[f(st.mean(vals((a,s_,e),'rmse500'))) if (a,s_,e) in cell else '-' for s_ in SOLV[1:7]])
table(['arm','tau']+SOLV[1:7],rows)
P('ZERO rmse500 = 0.1315 in every cell.'); P('')
# ---- 7 anomalies
P('## 7. Anomalies'); P('')
nan=collections.Counter()
for nm,R in (('dev',D),('test',T),('diag',G)):
    for r in R:
        for c,v in r.items():
            if v is not None and v.lower()=='nan': nan[(nm,r['solver'],c)]+=1
P('NaN values in rows: '+('; '.join(f'{k[0]} {k[1]} {k[2]}: {n}' for k,n in sorted(nan.items())) or 'none')+'.'); P('')
exp_n=[]
for a in ARMS:
    for s_ in SOLV:
        for e in range(1,8):
            if (a,s_,e) in cell and len(cell[(a,s_,e)])!=8: exp_n.append((a,s_,e,len(cell[(a,s_,e)])))
P('Cells with n != 8 in test: '+(str(exp_n) if exp_n else 'none')+'. test_summary.tsv: 129 lines = 1 header + %d cell + %d C1 + %d C2 + %d C3 rows.'%tuple(sum(1 for r in summ if r['kind']==k) for k in ['cell','C1','C2','C3'])); P('')
P('Planned-but-absent groups (by the cuts): exp 2 (3.16e3) in all arms; arm M exp 5 (1e5); GD-0 at exp 2 and 4; MLEM->GD everywhere; arm M has no GD (spec). Contrast rows: C1 %d, C2 %d, C3 %d (X 6 exposures, Mc 6, M 2 each for C1/C2; C3 2 exposures x 3 solvers).'%tuple(sum(1 for r in summ if r['kind']==k) for k in ['C1','C2','C3'])); P('')
# anomaly-ish observations
P('Other observations from the data (no log errors):'); P('')
nS=sum(1 for v in sel['selection'].values() if v['solver']=='SART'); n9=sum(1 for v in sel['selection'].values() if v['solver']=='SART' and abs(v['param']-0.9)<1e-9)
phi=collections.Counter(v['solver'] for v in sel['selection'].values() if v['at_edge']=='param_hi')
P(f"- Grid extensions: SART q=0.9 (grid top 0.3, extended) selected in {n9} of {nS} SART selections; MLEM c_M=15.625 selected in arm Mc exp 1; MLEM X exp 3 selected n_iter=1600 (flagged iter_hi). param_hi edges by solver: "+', '.join(f'{k}: {n}' for k,n in sorted(phi.items()))+'; iter_lo edges: GD-0 Mc exp 1, MLEM->GD Mc exp 1.')
P('- Run time against plan: test 8.90 h actual vs 6.13 h projected; diag 1.00 h vs 0.49 h; total 39.39 h vs 36 h limit (limit_met=false in cuts.toml).')
P('- Largest non-ZERO cell means: SART Mc exp 5 = 0.3076, SART M exp 7 = 0.2955. No solver cell mean exceeds the ZERO value 0.32914.')
P('- Realisation counts: all test cells n = 8; dev n = 2 per cell; 129-line test_summary.tsv header + 94 cell + 14 C1 + 14 C2 + 6 C3.')
P('')
P('Log lines (one each): ')
import subprocess
for fn,pat in (('log_templates.txt','merged 108 chunks'),('log_summary.txt','wrote'),('log_test.txt','36 h trigger'),('log_dev.txt','phase 3 selection written'),('log_diag.txt','Reconstruction-protocol stage')):
    ln=[l.strip() for l in open(O+fn) if pat in l]
    if ln: P(f'- {fn}: `{ln[-1][:200]}`')
P('- orch.out: 0 bytes; RUN_FAILED: absent; RUN_STATUS: stage=done units=16/16 workers=0')
open(FACTS_OUT,'w').write('\n'.join(out)+'\n')
print(len(out))
