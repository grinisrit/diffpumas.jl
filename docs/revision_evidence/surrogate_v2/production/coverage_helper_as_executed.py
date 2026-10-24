#!/usr/bin/env python3
"""Post-run conditional diagnostics; no test-driven coefficient fitting."""
import csv,json,statistics as st,math,sys,hashlib,datetime,tomllib
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1]);end=json.loads((root/"end_status.json").read_text());assert end["execution_completed"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in end["output_sha256"].items():assert sha(root/name)==digest,name
artifact=tomllib.loads((root/"frozen_corrections.toml").read_text())["metadata"]
result={"scope":"descriptive conditional diagnostics; jackknife SE not calibrated confidence interval; nonsmooth absolute bias can yield degenerate gain SE when bias sign is uncertain; flags are not significance or acceptance tests; no coefficient uncertainty or multiplicity adjustment", "training_q_min":artifact["training_q_min"],"training_q_max":artifact["training_q_max"]}
for stem,truth in [("heldout_flux","mc"),("heldout_derivatives","mc_secant")]:
 groups=defaultdict(list)
 for r in csv.DictReader((root/(stem+".csv")).open()):
  r={k:float(v) for k,v in r.items()};groups[(r["bin"],r["field"])].append(r)
 rows=[]
 for key,reps in groups.items():
  n=len(reps);b=[r["before"]-r[truth] for r in reps];a=[r["after"]-r[truth] for r in reps];mc=st.mean(r[truth] for r in reps)
  gain=abs(st.mean(b))-abs(st.mean(a))
  jk=[abs(st.mean(b[:i]+b[i+1:]))-abs(st.mean(a[:i]+a[i+1:])) for i in range(n)]
  jkmean=st.mean(jk);jkse=math.sqrt((n-1)/n*sum((v-jkmean)**2 for v in jk))
  row=dict(bin=key[0],field=key[1],zenith=reps[0]["zenith"],mc_mean=mc,before_bias=st.mean(b),after_bias=st.mean(a),before_relative_bias=st.mean(b)/max(abs(mc),1e-30),after_relative_bias=st.mean(a)/max(abs(mc),1e-30),absolute_bias_gain=gain,absolute_bias_gain_jackknife_se=jkse,jackknife_negative_gain_flag=gain < -2*jkse)
  for label,values in [("before",b),("after",a)]:
   se=st.stdev(values)/math.sqrt(n);row[label+"_se"]=se;row[label+"_over_2se"]=abs(st.mean(values))>2*se
  if "column_water_fraction" in reps[0]:
   q=reps[0]["column_water_fraction"];row.update(q=q,q_below_training=q<artifact["training_q_min"]-1e-12,q_above_training=q>artifact["training_q_max"]+1e-12,q_zero=q==0)
  if "ad" in reps[0]:
   row["ad_fd_relative_error"]=max(abs(r["ad"]-r["fd"])/max(abs(r["ad"]),abs(r["fd"]),1e-30) for r in reps);row["one_sided"]=reps[0]["fraction_lo"]==0
  rows.append(row)
 def stats(subset):
  d=dict(comparisons=len(subset),before_median_abs_relative_bias=st.median(abs(r["before_relative_bias"]) for r in subset),after_median_abs_relative_bias=st.median(abs(r["after_relative_bias"]) for r in subset),before_max_abs_relative_bias=max(abs(r["before_relative_bias"]) for r in subset),after_max_abs_relative_bias=max(abs(r["after_relative_bias"]) for r in subset),before_count_over_2se=sum(r["before_over_2se"] for r in subset),after_count_over_2se=sum(r["after_over_2se"] for r in subset),jackknife_negative_gain_flags=sum(r["jackknife_negative_gain_flag"] for r in subset),new_discrepancies_over_2se=sum((not r["before_over_2se"]) and r["after_over_2se"] for r in subset))
  if "q" in subset[0]:d.update(q_min=min(r["q"] for r in subset),q_max=max(r["q"] for r in subset),q_below_training=sum(r["q_below_training"] for r in subset),q_zero=sum(r["q_zero"] for r in subset))
  if "ad_fd_relative_error" in subset[0]:d.update(max_ad_fd_relative_error=max(r["ad_fd_relative_error"] for r in subset),one_sided_comparisons=sum(r["one_sided"] for r in subset))
  return d
 result[stem]=dict(all=stats(rows),by_field={str(int(k)):stats([r for r in rows if r["field"]==k]) for k in sorted(set(r["field"] for r in rows))},by_zenith={str(int(k)):stats([r for r in rows if r["zenith"]==k]) for k in sorted(set(r["zenith"] for r in rows))},rows=rows)
(root/"coverage_assessment.json").write_text(json.dumps(result,indent=2)+"\n")
provenance=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),end_status_sha256=sha(root/"end_status.json"),raw_sha256={n:sha(root/n) for n in ["heldout_flux.csv","heldout_derivatives.csv","frozen_corrections.toml"]},helper_sha256=sha(Path(__file__)),assessment_sha256=sha(root/"coverage_assessment.json"))
(root/"coverage_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
print(json.dumps({k:{a:b for a,b in v.items() if a!="rows"} if isinstance(v,dict) else v for k,v in result.items()},indent=2))
