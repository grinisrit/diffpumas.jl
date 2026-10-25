#!/usr/bin/env python3
"""Independent read-only arithmetic/identity review; no transport or MC draws."""
import csv,hashlib,json,math,statistics,struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];E=ROOT/'docs/revision_evidence';A=E/'focused_revision_20261025';OUT=Path(__file__).resolve().parent
inputs={};sourcecache={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):inputs[str(p.relative_to(ROOT))]=sha(p);return list(csv.DictReader(p.open()))
def readjson(p):inputs[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def same(a,b):return struct.pack('>d',float(a))==struct.pack('>d',float(b))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-13,abs_tol=1e-18)
identity={}
for run in ['finite_contrast_run','scaling_run']:
    end=readjson(A/run/'end_status.json');exe=readjson(A/run/'execution.json')
    assert end['execution_completed'] and end['source_unchanged'] and end['runtime_unchanged']
    for f,h in end['output_sha256'].items():assert sha(A/run/f)==h,(run,f)
    for f,h in exe['source_sha256'].items():
        p=Path(f) if Path(f).is_absolute() else ROOT/f
        if p not in sourcecache:sourcecache[p]=sha(p)
        assert sourcecache[p]==h,(run,f)
    identity[run]=dict(output_hashes=len(end['output_sha256']),source_hashes=len(exe['source_sha256']),passed=True)
rows=readcsv(A/'finite_contrast_run/results/finite_contrast_vs_local_ad.csv');assert len(rows)==168
old=readcsv(E/'corrected_validation/mc_jacobian.csv');new=readcsv(E/'surrogate_v2/production/heldout_derivatives.csv');low=readcsv(E/'lowfraction_round3_20261025/pilot/results/heldout_derivatives.csv')
expected={}
for name,rr in [('historical',old),('new',new),('lowfraction_control',[r for r in low if r['arm']=='control'])]:
    for r in rr:
        k=(name,int(r['bin']),int(r['cell']),float(r['fraction'] if name=='historical' else r['field']))
        value=(float(r['lo'] if name=='lowfraction_control' else r['fraction_lo']),float(r['hi'] if name=='lowfraction_control' else r['fraction_hi']))
        if k in expected:assert expected[k]==value
        expected[k]=value
actual={(r['population'],int(r['bin']),int(r['cell']),float(r['field'])):r for r in rows};assert set(actual)==set(expected) and len(actual)==168
adrows=readcsv(E/'surrogate_v2/forwarddiff_replay/ad_fd_replay.csv');lowad=readcsv(E/'lowfraction_round3_20261025/pilot/results/ad_fd_steps.csv')
stored={(r['population'],int(r['bin']),int(r['cell']),float(r['field'])):float(r['ad']) for r in adrows if float(r['step'])==1e-6}
stored.update({('lowfraction_control',int(r['bin']),int(r['cell']),float(r['field'])):float(r['ad']) for r in lowad if r['arm']=='control' and float(r['step'])==1e-6})
for key,r in actual.items():
    lo,hi=expected[key];assert same(lo,r['fraction_lo']) and same(hi,r['fraction_hi'])
    assert same(r['ad'],stored[key]) and same(r['stored_ad'],stored[key]) and r['stored_ad_bitwise_equal']=='true'
    sec=(float(r['flux_hi'])-float(r['flux_lo']))/(hi-lo);diff=sec-float(r['ad']);relative=abs(diff)/max(abs(sec),abs(float(r['ad'])),1e-30)
    for name,value in [('midpoint_secant',sec),('secant_minus_ad',diff),('absolute_difference',abs(diff)),('relative_difference',relative)]:assert close(r[name],value),(key,name)
claimed=readjson(A/'finite_contrast_summary.json');contrast={}
for name,n in [('historical',72),('new',48),('lowfraction_control',48)]:
    rr=[r for r in rows if r['population']==name];assert len(rr)==n
    rel=np.array([float(r['relative_difference']) for r in rr]);absolute=np.array([float(r['absolute_difference']) for r in rr]);signed=np.array([float(r['secant_minus_ad']) for r in rr])
    checked=dict(cases=n,relative_median=float(np.median(rel)),relative_p95=float(np.quantile(rel,.95)),relative_max=float(rel.max()),absolute_median=float(np.median(absolute)),absolute_p95=float(np.quantile(absolute,.95)),signed_median=float(np.median(signed)),over_1pct=int(sum(rel>.01)),over_5pct=int(sum(rel>.05)),all_stored_ad_bitwise_equal=True)
    for k,v in checked.items():assert close(claimed['groups'][name][k],v),(name,k)
    contrast[name]=checked
assert claimed['source_output_sha256']==sha(A/'finite_contrast_run/results/finite_contrast_vs_local_ad.csv')
scaling=readcsv(A/'scaling_run/results/scaling_repeats.csv');assert len(scaling)==6*7*3
claimed_scaling=readjson(A/'scaling_summary.json');groups={}
for r in scaling:groups.setdefault((int(r['cells']),int(r['rays']),r['stage']),[]).append(r)
assert len(groups)==42 and {(k[0],k[1]) for k in groups}=={(n,m) for n in [32,64,128] for m in [24,96]}
recomputed=[]
for (cells,rays,stage),rr in sorted(groups.items()):
    assert sorted(int(r['repeat']) for r in rr)==[1,2,3]
    assert all(float(r['seconds'])>0 and int(r['allocated_bytes'])>=0 for r in rr)
    assert len({(r['nnz'],r['intersections'],r['retained_jacobian_bytes'],r['dense_normal_bytes']) for r in rr})==1
    assert all(int(r['dense_normal_bytes'])==8*cells*cells+48 for r in rr)
    assert all(int(r['retained_jacobian_bytes'])==16*int(r['nnz'])+8*(cells+1)+160 for r in rr)
    checked=dict(cells=cells,rays=rays,stage=stage,intersections=int(rr[0]['intersections']),nnz=int(rr[0]['nnz']),median_seconds=statistics.median(float(r['seconds']) for r in rr),minimum_seconds=min(float(r['seconds']) for r in rr),maximum_seconds=max(float(r['seconds']) for r in rr),median_allocated_bytes=statistics.median(int(r['allocated_bytes']) for r in rr),process_peak_rss_bytes=max(int(r['process_peak_rss_bytes']) for r in rr),retained_jacobian_bytes=int(rr[0]['retained_jacobian_bytes']),dense_normal_bytes=int(rr[0]['dense_normal_bytes']))
    claim=next(r for r in claimed_scaling['rows'] if r['cells']==cells and r['rays']==rays and r['stage']==stage)
    for k,v in checked.items():assert v==claim[k] if isinstance(v,str) else close(v,claim[k]),(cells,rays,stage,k)
    recomputed.append(checked)
report=dict(passed=True,independence='Source inspection and separately implemented arithmetic/provenance checks; no producer helpers imported and no transport rerun.',run_identity=identity,contrast=contrast,scaling_groups_checked=len(recomputed),scaling_rows_checked=len(scaling),qualifications=['Finite deterministic secant windows may be asymmetric and are not local derivatives or stochastic-gradient error bounds.','Scaling kernels are serial and BLAS1 although three Julia threads were available; small corrected-disabled grids do not establish full corrected-pipeline capacity.','Allocation totals are cumulative allocations; process peak RSS is cumulative, not stage incremental memory.','One warmup per stage is the measured protocol, not a formal guarantee excluding all latent compilation.'],input_sha256=inputs,review_source_sha256=sha(Path(__file__)))
(OUT/'contrast_scaling_review.json').write_text(json.dumps(report,indent=2)+'\n');print('Independent contrast/scaling review: PASS')
