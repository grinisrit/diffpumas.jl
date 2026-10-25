#!/usr/bin/env python3
"""Assemble run evidence without turning finite diagnostics into certificates."""
import hashlib,json,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/revision_evidence/convergence_round3_20261025'
runs=['diagnostic_run','polish_run','trace375_run','v2_tests_run','polish_v2_run']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
identity={};sourcecache={}
for name in runs:
    path=E/name;end=json.loads((path/'end_status.json').read_text());execution=json.loads((path/'execution.json').read_text())
    outputs={f:sha(path/f)==h for f,h in end['output_sha256'].items()}
    sources={}
    for f,h in execution['source_sha256'].items():
        p=Path(f) if Path(f).is_absolute() else ROOT/f
        if p not in sourcecache:sourcecache[p]=sha(p)
        sources[f]=sourcecache[p]==h
    assert end['execution_completed'] and end['source_unchanged'] and end['runtime_unchanged'] and all(outputs.values()) and all(sources.values())
    identity[name]=dict(execution_sha256=sha(path/'execution.json'),end_status_sha256=sha(path/'end_status.json'),output_hashes_verified=len(outputs),source_hashes_verified=len(sources),passed=True)
a=tomllib.load((E/'polish_run/results/final.toml').open('rb'));b=tomllib.load((E/'polish_v2_run/results/final.toml').open('rb'))
h=tomllib.load((E/'polish_v2_run/results/harmonic_reduction.toml').open('rb'));state=tomllib.load((E/'polish_v2_run/results/polish_state.toml').open('rb'))
summary=dict(run_identity=identity,baseline_objective=a['initial_objective'],baseline_selected_branch_pg=a['initial_projected_gradient'],first_polish={k:v for k,v in a.items() if k not in ('field','predictions')},revised_polish={k:v for k,v in b.items() if k not in ('field','predictions')},harmonic={k:v for k,v in h.items() if k not in ('field','reduced_tangent','components','unseen_cells','reduced_cells')},sampled_qp=[{k:v for k,v in s.items() if k not in ('fields','gradients','direction','field_before','alpha','six_gradient_alpha')} for s in state['sample_records']],generalized_stationarity_certified=False,report_source_sha256=sha(Path(__file__)))
(E/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for label,s in [('Frozen main',dict(objective=a['initial_objective'],selected_branch_projected_gradient=a['initial_projected_gradient'])),('First polish',a),('Revised polish',b)]:rows.append(f"| {label} | {s['objective']:.17g} | {s['selected_branch_projected_gradient']:.10g} |")
text='''# Convergence investigation, round 3

The unchanged objective admits small validated decreases. This round does not establish generalized stationarity. The final selected-branch tolerance 1e-6 status is recorded explicitly in final_summary.json. Original baseline sources and fitted fields were preserved; new results and manuscript additions are supplemental evidence.

| Endpoint | Objective | Selected-branch projected gradient infinity norm |
|---|---:|---:|
'''+ '\n'.join(rows)+f'''

The diagnostic checked 43 finite feasible directions at six step sizes (258 exact objective trials), including coordinate, coupled, damped GN and seeded random directions. Twenty-two trials decrease the objective above the numerical repeatability screen. These finite secants are not a generalized stationarity test. The independent value-only baseline repeated identically; the numerical screen is 1.4210854715202004e-12, not a rigorous error bound.

The first polisher accepted {a['accepted_updates']} exact decreases in {a['objective_evaluations']} trials using {a['gradient_assemblies']} full gradient assemblies. Its full endpoint objective and gradient were independently reproduced by the reviewer. One six-gradient auxiliary QP remained materially suboptimal despite a small weight update; a Frank-Wolfe gap exposed this limitation. The revised method uses exact pairwise simplex transfers and records an explicit finite-QP optimality gap.

The revised pass eliminated {h['unseen_count']} supported cells absent from all retained fitted rays by exact conditional graph-prior minimization, leaving {h['reduced_count']} supported cells crossed by fitted rays. All eight unseen components were anchored. Fitted fluxes and profile normalization remain exactly unchanged for this harmonic-only step. Its objective change is {h['objective_change']:.17g}; unseen prior-gradient infinity residual is {h['prior_gradient_unseen_inf']:.8g}. The explicit reduced derivative is Tᵀg, checked against smooth toy finite differences. Every physical sample checks full-field radius and harmonic feasibility.

Two revised sampling radii use 21 gradients each (base, 12 random reduced neighbors and both signs of four dominant coordinates), comparing six-gradient and enriched finite hulls at the same center. The revised pass accepted {b['accepted_updates']} exact decreases in {b['objective_evaluations']} line trials using {b['gradient_assemblies']} full assemblies. Total objective decrease from the frozen endpoint is {b['objective_improvement']:.17g}. Termination is the preregistered finite budget. Finite sampled hull residuals and their auxiliary QP gaps do not certify the Clarke hull or constrained stationarity.

Tracing the largest baseline branch-gradient coordinate, cell 375, localized the numerical corner. At perturbations ±1e-8, only bin 831, energy node 16, final air step 1177 changes an interpolation interval; adaptive step counts, limiter regimes and air floors do not change. The actually used Air table (material index 3) has knots 25000, 30000 and 35000 GeV at indices 118:120. Its predictor/next-energy interpolation crosses the 30 TeV knot. The profiled robust envelope-score calculation attributes 99.9529283% of the finite 1e-8 objective slope jump to this ray. This is finite-scale mechanism localization, not a derivative-limit proof or a convergence certificate. Raw trace, source-parity checks, score algebra and input hashes are retained in trace_mechanism_assessment.json.

Tests passed: 11 first-direction checks; 13 exact pair-QP cases, 12 actual saved-gradient QP checks and 8 harmonic/explicit reduced-derivative checks; independent SciPy comparisons of all three saved convex subproblems. Every traced flux matches its production counterpart within relative 1e-13. Final forward/J evaluations reproduce incrementally cached objectives and predictions. Source/runtime/output hash verification for all five bounded jobs is recorded in final_summary.json. Independent reviewer evidence is separate under round3_independent_review_20261025.

All physical jobs ran sequentially in tmux on research_vm_roland, CPU0-7, at most 16 GiB address space, eight Julia threads and BLAS1. Raw execution records contain exact argv, Julia binary identity, source/input hashes and status; no tolerance was relabeled, no objective was smoothed, and no core transport source or primary fitted result was changed. Manuscript additions are supplementary only. No further polishing is authorized in this round. Remaining stationarity limitations must accompany any downstream calibration pilot.
'''
(E/'README.md').write_text(text)
(E/'report_manifest.json').write_text(json.dumps(dict(report_source_sha256=sha(Path(__file__)),outputs={f:sha(E/f) for f in ['README.md','final_summary.json','trace_mechanism_assessment.json']},run_identity=identity),indent=2)+'\n')
print('Convergence round 3 report and all run hashes: PASS')
