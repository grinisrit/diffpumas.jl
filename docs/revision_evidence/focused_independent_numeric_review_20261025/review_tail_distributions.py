#!/usr/bin/env python3
"""Independent actual-stencil/raw-statistics review; no physics or producer import."""
import csv,hashlib,json,math,os,resource,statistics,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];E=ROOT/'docs/revision_evidence';A=E/'focused_revision_20261025';OUT=Path(__file__).resolve().parent
inputs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):inputs[str(p.relative_to(ROOT))]=sha(p);return list(csv.DictReader(p.open()))
def js(p):inputs[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def close(a,b):return math.isclose(float(a),float(b),rel_tol=3e-12,abs_tol=1e-18)
def quant(v,p):
    a=sorted(v);t=(len(a)-1)*p;i=int(t);return a[i]+(t-i)*(a[min(i+1,len(a)-1)]-a[i])
def dist(v):
    a=list(v);assert all(math.isfinite(x) for x in a)
    return dict(n=0) if not a else dict(n=len(a),minimum=min(a),p05=quant(a,.05),median=statistics.median(a),p90=quant(a,.9),p95=quant(a,.95),maximum=max(a))
def compare_dist(claim,actual):
    assert set(claim)==set(actual)
    for k,v in actual.items():assert close(claim[k],v),(k,claim[k],v)
run=A/'current_grid_tail_run_v2';end=js(run/'end_status.json');exe=js(run/'execution.json')
assert end['execution_completed'] and end['source_unchanged'] and end['runtime_unchanged']
for f,h in end['output_sha256'].items():assert sha(run/f)==h
for f,h in exe['source_sha256'].items():assert sha(Path(f) if Path(f).is_absolute() else ROOT/f)==h,(f,'source changed')
selection=csvread(A/'current_grid_tail_selection.csv');plan=js(A/'current_grid_tail_selection.json')
for f,h in plan['input_sha256'].items():assert sha(ROOT/f)==h
assert sha(run/'results/selection_before_evaluation.csv')==sha(A/'current_grid_tail_selection.csv')
rows=csvread(run/'results/tail_fd_checks.csv');assert len(rows)==48*6 and len(selection)==48
casekeys={int(r['case']):r for r in selection};assert len(casekeys)==48
groups={};all_ad=True
for r in rows:
    source=casekeys[int(r['case'])];h=float(r['step']);x=float(r['local_fraction']);f0=float(r['flux']);fa=float(r['flux_a']);fb=float(r['flux_b']);ad=float(r['ad'])
    assert r['band']==source['band'] and int(r['bin'])==int(source['bin']) and int(r['cell'])==int(source['cell'])
    assert close(x,float(source['local_fraction'])) and close(r['stored_ad'],source['stored_ad']) and math.isclose(ad,float(r['stored_ad']),rel_tol=1e-10,abs_tol=1e-25)
    stencil='central' if x-h>=0 and x+h<=.9 else 'forward2' if x+2*h<=.9 else 'backward2';assert stencil==r['stencil']
    value=(fb-fa)/(2*h) if stencil=='central' else (-3*f0+4*fa-fb)/(2*h) if stencil=='forward2' else (3*f0-4*fa+fb)/(2*h)
    assert close(value,r['fd']);error=value-ad
    assert close(error,r['signed_error']) and close(abs(error),r['absolute_error'])
    floor=max(10*abs(float(r['repeat_flux'])-f0),100*math.ulp(abs(f0)));assert close(floor,r['numerical_floor'])
    resolved=max(abs(fa-f0),abs(fb-f0))>floor;zero=r['band']=='structural_zero'
    assert (r['finite_change_resolved']=='true')==resolved and (r['relative_error_interpretable']=='true')==(resolved and not zero)
    if zero:
        assert ad==0 and struct.pack('d',fa)==struct.pack('d',f0)==struct.pack('d',fb) and r['zero_perturbations_bitwise_equal']=='true'
    groups.setdefault((r['band'],h),[]).append(r)
claimed=js(A/'current_grid_tail_v2_summary.json');assert claimed['authoritative_run']=='current_grid_tail_run_v2'
tail={}
for (band,h),rr in groups.items():
    assert len(rr)==12
    absolute=[float(r['absolute_error']) for r in rr];relative=[float(r['relative_error']) for r in rr if r['relative_error_interpretable']=='true']
    claim=claimed['groups'][band][str(h)]
    assert claim['cases']==12 and claim['resolved']==sum(r['finite_change_resolved']=='true' for r in rr)
    assert close(claim['max_absolute'],max(absolute))
    if relative:
        assert close(claim['median'],statistics.median(relative)) and close(claim['p95'],quant(relative,.95)) and close(claim['maximum'],max(relative))
    else:assert all(claim[k] is None for k in ['median','p95','maximum'])
    tail.setdefault(band,{})[str(h)]=dict(absolute_error=dist(absolute),resolved_relative_error=dist(relative),resolved=sum(r['finite_change_resolved']=='true' for r in rr))
# Independently reproduce the pre-existing deterministic distributions and MC denominator audit.
existing=js(A/'existing_derivative_audit.json')
for f,h in existing['input_sha256'].items():assert sha(ROOT/f)==h
adrows=csvread(E/'surrogate_v2/forwarddiff_replay/ad_fd_replay.csv');pilotad=csvread(E/'lowfraction_round3_20261025/pilot/results/ad_fd_steps.csv')
for population,steps in existing['deterministic_step_distributions'].items():
    rr=[r for r in pilotad if r['arm']==population.split('_',1)[1]] if population.startswith('lowfraction_') else [r for r in adrows if r['population']==population]
    for step,claim in steps.items():
        selected=[r for r in rr if float(r['step'])==float(step)];d=[float(r['ad'])-float(r['fd']) for r in selected];rel=[abs(v)/max(abs(float(r['ad'])),abs(float(r['fd'])),1e-30) for r,v in zip(selected,d)]
        compare_dist(claim['signed_error'],dist(d));compare_dist(claim['absolute_error'],dist(map(abs,d)));compare_dist(claim['relative_error'],dist(rel));assert claim['over_1e4']==sum(v>=1e-4 for v in rel)
mc_sources={'historical':csvread(E/'surrogate_v2/historical/historical_derivatives.csv'),'new':csvread(E/'surrogate_v2/production/heldout_derivatives.csv'),'lowfraction':csvread(E/'lowfraction_round3_20261025/pilot/results/heldout_derivatives.csv')}
mc=[]
for claim in existing['mc_bias_distributions']:
    name=claim['population'];population,arm=name.split('_',1);rr=mc_sources[population]
    if population=='lowfraction':rr=[r for r in rr if r['arm']==arm]
    target='csda_secant' if population=='lowfraction' else arm;group={}
    for r in rr:group.setdefault((r['bin'],r['cell'],r['fraction'] if population=='historical' else r['field']),[]).append(r)
    calculated=[]
    for key,values in group.items():
        assert len(values)==8;ref=[float(r['mc_secant']) for r in values];delta=[float(r[target])-v for r,v in zip(values,ref)]
        mean=statistics.mean(ref);se=statistics.stdev(ref)/math.sqrt(8);bias=statistics.mean(delta);bse=statistics.stdev(delta)/math.sqrt(8)
        calculated.append(dict(key=list(key),mc_mean=mean,mc_SE=se,signed_bias=bias,bias_SE=bse,absolute_bias=abs(bias),relative_bias=bias/abs(mean) if mean else None,reference_resolved=abs(mean)>5*se,negative_reference=mean<0,zero_reference=mean==0,over_2SE=abs(bias)>2*bse))
    bykey={tuple(v['key']):v for v in claim['cases']}
    for v in calculated:
        c=bykey[tuple(v['key'])]
        for k in ['mc_mean','mc_SE','signed_bias','bias_SE','absolute_bias']:assert close(c[k],v[k]),(name,k)
        for k in ['reference_resolved','negative_reference','zero_reference','over_2SE']:assert c[k]==v[k]
    relative=[v['relative_bias'] for v in calculated if v['relative_bias'] is not None];resolved=[v['relative_bias'] for v in calculated if v['reference_resolved']]
    for k,v in [('signed_absolute_units_bias',[x['signed_bias'] for x in calculated]),('absolute_bias',[x['absolute_bias'] for x in calculated]),('point_relative_bias',relative),('point_absolute_relative_bias',list(map(abs,relative))),('resolved_reference_relative_bias',resolved),('resolved_reference_absolute_relative_bias',list(map(abs,resolved)))]:compare_dist(claim[k],dist(v))
    for key,value in [('reference_precision_unresolved',sum(not v['reference_resolved'] for v in calculated)),('negative_reference_groups',sum(v['negative_reference'] for v in calculated)),('zero_reference_groups',sum(v['zero_reference'] for v in calculated)),('over_2_paired_SE',sum(v['over_2SE'] for v in calculated))]:assert claim[key]==value
    mc.append({k:v for k,v in claim.items() if k!='cases'})
pop=csvread(E/'corrected_validation/selection_population_audit.csv');selected=[r for r in pop if int(r['selected_case'])>0];coverage=existing['selection']
assert len(pop)==coverage['nonzero_population'] and len(selected)==coverage['selected_pairs']
for feature in ['zenith_deg','slant_m','cell_z_m','gradient']:
    for label,rr in [('population',pop),('selected',selected)]:compare_dist(coverage['coverage'][feature][label],dist(abs(float(r[feature])) if feature=='gradient' else float(r[feature]) for r in rr))
strata=lambda rr:{tuple(r[k] for k in ['zenith_quartile','slant_quartile','height_quartile','sensitivity_quartile']) for r in rr}
assert len(strata(pop))==coverage['populated_joint_strata'] and len(strata(selected))==coverage['selected_joint_strata']
matrix=csvread(E/'final_field_mc_coverage_20261024/final_jacobian_sparse.csv');coverage_rows=csvread(E/'final_field_mc_coverage_20261024/final_field_coverage.csv');fit_ids={int(r['bin_idx']) for r in coverage_rows if r['fit_mask']=='true'}
for label,slots in [('all_covered',1003*147),('fitted',690*147)]:
    vals=[float(r['derivative']) for r in matrix if r['in_support']=='true' and (label=='all_covered' or int(r['bin_idx']) in fit_ids)]
    claim=existing['current_field_support_jacobian'][label]
    assert claim['slots']==slots and claim['stored_entries']==len(vals) and claim['nonzero']==sum(v!=0 for v in vals) and claim['negative']==sum(v<0 for v in vals) and claim['zero_including_unstored']==slots-sum(v!=0 for v in vals)
    compare_dist(claim['absolute_nonzero_distribution'],dist(abs(v) for v in vals if v!=0))
report=dict(passed=True,scope='Independent corrected v2 actual-stencil/source/provenance and raw arithmetic checks; deterministic finite-step/case-specific evidence, no global derivative certificate.',effective_cpu_affinity=sorted(os.sched_getaffinity(0)),address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],tail=tail,existing_selection=coverage,existing_deterministic_distributions=existing['deterministic_step_distributions'],mc_bias_distributions=mc,current_field_support_jacobian=existing['current_field_support_jacobian'],input_sha256=inputs,review_source_sha256=sha(Path(__file__)))
(OUT/'tail_existing_distributions_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Independent corrected tail/existing distributions review: PASS')
