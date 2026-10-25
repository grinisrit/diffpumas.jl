#!/usr/bin/env python3
import csv,json,math,tomllib
from pathlib import Path
root=Path(__file__).resolve().parents[3];base=root/'docs/revision_evidence/material_main_recovery_20261024/completed/main/data';ev=root/'docs/revision_evidence/convergence_round3_20261025'
preds=list(csv.DictReader((base/'lvd_fit_predictions.csv').open()));obs=list(csv.DictReader((root/'examples/data/lvd_conf/lvd_single_muon_flux_2d.csv').open()));trace=tomllib.loads((ev/'trace375_run/results/trace.toml').read_text());directions=list(csv.DictReader((ev/'diagnostic_run/results/directional_checks.csv').open()))
score={};total=0
for p in preds:
 b=int(p['bin_idx']);z=float(obs[b-1]['zenith_center_deg']);W=float(p['fit_mask'])/(1+(.15*max(1/math.cos(math.radians(z))-1,0))**2);pred=float(p['matched_csda_flux']);t=math.log(pred)+math.log(float(p['normalization']))-math.log(float(p['measured_flux']));s=W*t/math.sqrt(1+(t/.3)**2);score[b]=s/pred;total+=s
rows=[]
for h in [1e-6,1e-7,1e-8]:
 jump=0;parts=[]
 for b in sorted({r['bin_idx'] for r in trace['reports']}):
  rr={r['delta']:r for r in trace['reports'] if r['bin_idx']==b};df=(rr[h]['flux_change']+rr[-h]['flux_change'])/h;term=score[b]*df;jump+=term;parts.append((abs(term),b,term))
 plus=next(r for r in directions if r['direction']=='coordinate_375_1' and float(r['step'])==h);minus=next(r for r in directions if r['direction']=='coordinate_375_-1' and float(r['step'])==h);full=float(plus['slope'])+float(minus['slope']);dominant=max(parts)
 rows.append(dict(step=h,full_objective_one_sided_slope_jump=full,all_trace_ray_first_order_jump=jump,dominant_bin=dominant[1],dominant_ray_jump=dominant[2],dominant_fraction_of_full=dominant[2]/full))
print(json.dumps(dict(profile_score_sum=total,checks=rows,scope='Finite-step envelope first-order decomposition. Residual difference includes curvature/subtraction; observed transition is not a limiting nonsmooth-stationarity proof.'),indent=2))
