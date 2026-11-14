import tomllib,collections
from common import *
g=tomllib.load(open(O+'dev/grids.toml','rb'))
DEF={'SART':([0.0,0.03,0.1,0.3],[1,2,5,10,25,50,100,200]),'MLEM':([0.0,0.05,0.16,0.5,1.6,5.0],[10,25,50,100,200,400]),'GN':([0.001,0.01,0.1,1.0],[1,2,3,5,8,13]),'GD':([],[5,10,20])}
def key(s): return 'GN' if s in('GN-lin','GN-0','MLEM->GN') else ('GD' if 'GD' in s else s)
fmt=lambda l:'['+', '.join(f'{x:g}' for x in sorted(l))+']'
P=print
P('## Item 9: grids actually run after the edge-rule extensions (dev/grids.toml, final state) and the SART q stop\n')
P('Defaults: SART q [0, 0.03, 0.1, 0.3], n_iter [1, 2, 5, 10, 25, 50, 100, 200]; MLEM c_M [0, 0.05, 0.16, 0.5, 1.6, 5], n_iter [10, 25, 50, 100, 200, 400]; GN-family c [0.001, 0.01, 0.1, 1], K [1, 2, 3, 5, 8, 13 (+20 after the cap-32 rerun)]; GD K [5, 10, 20] (c taken from the GN selection). Extension counters are the `ext` fields (param_lo, param_hi, iter_lo, iter_hi, cap).\n')
for s in ['SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0','MLEM->GD']:
    P(f'### {s}\n')
    P('| arm | exp | param grid (final) | iter/K grid (final) | cap | ext lo/hi param | ext lo/hi iter | cap ext | selected (param/iter) |'); P('|---|---|---|---|---|---|---|---|---|')
    dp,di=DEF[key(s)]
    for a in ['X','Mc','M']:
        for e in range(1,8):
            k=f'{s}|{a}|{e}'
            if k not in g: continue
            v=g[k]; ex=v['ext']
            changed = set(v['param'])!=set(dp) or (set(v['iter'])!=set(di) and not (key(s)=='GN' and set(v['iter'])==set(di)|{20}))
            sl=sel['selection'].get(k)
            selt=f"{sl['param']:g}/{sl['iter']}" if sl else '-'
            if changed or v.get('cap') not in (None,20,40):
                P(f"| {a} | {tl(e)} | {fmt(v['param']) if v['param'] else '-'} | {fmt(v['iter'])} | {v.get('cap','')} | {ex['param_lo']}/{ex['param_hi']} | {ex['iter_lo']}/{ex['iter_hi']} | {ex.get('cap',0)} | {selt} |")
    P('')
P('Groups with a default grid are not listed (only groups whose grid differs from the default are shown). Counts of groups per solver: '+', '.join(f'{s} {sum(1 for k in g if k.startswith(s+"|"))}' for s in ['SART','MLEM','GN-lin','GN-0','MLEM->GN','GD-0','MLEM->GD'])+' (arm M has no GD).\n')
P('### Extremes\n')
P('| solver | largest param in any final grid | largest n_iter / K in any grid | number of groups at the second extension (ext param_hi = 2) | selected value at the largest param? |'); P('|---|---|---|---|---|')
for s in ['SART','MLEM','GN-lin','GN-0','MLEM->GN']:
    gg=[(k,v) for k,v in g.items() if k.split('|')[0]==s]
    mp=max(max(v['param']) for k,v in gg); mi=max(max(v['iter']) for k,v in gg)
    n2=[k for k,v in gg if v['ext']['param_hi']==2]
    atmax=[k for k,v in gg if max(v['param'])==mp and k in sel['selection'] and sel['selection'][k]['param']==mp]
    P(f'| {s} | {mp:g} | {mi} | {len(n2)} | {len(atmax)} |')
P('')
P('MLEM c_M beyond 15.625: grid value 48.828125 (= 15.625 * 15.625/5, the second extension) was run in one group only, MLEM|Mc|1 (arm Mc, tau = 1e3); there c_M = 15.625 was selected (not 48.828125, so the grid there ends one value above the selected one). 15.625 was run in 6 groups in total (MLEM|Mc|1, MLEM|Mc|2, MLEM|Mc|3, MLEM|Mc|5, MLEM|Mc|6, MLEM|X|2; group index = exposure index); it was selected only in MLEM|Mc|1. MLEM n_iter beyond 1600: none; 1600 (= 400 -> 800 -> 1600, two extensions) was run in two groups, MLEM|X|3 (selected n = 1600, flagged iter_hi, at the grid edge after both extensions) and MLEM|M|7 (selected n = 800). SART n_iter 800 was run in SART|Mc|5 only (selected 400 with q = 0.3, whose q grid stopped at 0.9 after one extension). MLEM->GN|X|7 has the one lower extension of the c grid (0.0001) and the cap-32 K grid [1, 2, 3, 5, 8, 13, 20].\n')
P('### Why SART q stopped at 1\n')
P('- Rule in code: `extend_value(vals, :hi; cap)` (src/Tomography_protocol.jl:620-633) returns `nothing` when the largest value is already >= `cap`, else `min(v_end * v_end/v_(end-1), cap)`; in `select_phase!` the cap is `q_cap` for SART (line 661: `qcap = solver == \"SART\" ? q_cap : Inf`) and `q_cap` defaults to 1.0 (line 643). The docstring of `select_phase!` (line 636-642) lists `q > 1` among the hard bounds that are not extended.')
P('- Unit test of the rule (test/test_s7_protocol.jl:692-694): [0, 0.03, 0.1, 0.3] extends to 0.9; [..., 0.3, 0.9] extends to 1.0; [..., 0.9, 1.0] extends to `nothing`. The unextended ratio rule would give 0.3 -> 0.9 -> 2.7; the second step is clamped to 1.0 by `min(new, cap)`.')
P('- Meaning of the bound: SART smoothing step is `out_k = (1 - q) w_k + q * mean(neighbours)` (src/Tomography_inverse.jl:66-78, applied at line 125); q = 1 replaces the cell by the neighbour mean, q > 1 would put a negative weight on the cell itself. The protocol file s7_decision.md Q2.7 states the generic edge rule ("extend by one step of the same ratio, at most twice") and does not mention the q <= 1 bound; the bound was added in code (git history of src/Tomography_protocol.jl: commits 896ad44, 01a63cb in the worktree; the commit message does not give the reason, so the stated meaning is read from the code, not from a recorded rationale).')
P('- Observed: 2 SART selections ended at param_hi = 1 (SART:Mc:4 and SART:X:2, both q = 1, n = 50) and were reported as at_edge = param_hi in selection.toml; SART q = 0.9 was selected in 12 of 17 selections (0.9 was the first extension, 0.3 -> 0.9).')
