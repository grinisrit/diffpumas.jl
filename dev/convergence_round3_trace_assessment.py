#!/usr/bin/env python3
"""Finite-scale envelope-score attribution; not a derivative-limit proof."""
import csv,hashlib,json,math,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'docs/revision_evidence/convergence_round3_20261025'
trace=r/'trace375_run/results/trace.toml';checks=r/'diagnostic_run/results/directional_checks.csv'
preds=ROOT/'docs/revision_evidence/material_main_recovery_20261024/completed/main/data/lvd_fit_predictions.csv'
obs=ROOT/'examples/data/lvd_conf/lvd_single_muon_flux_2d.csv'
reports=tomllib.load(trace.open('rb'))['reports'];rows=list(csv.DictReader(checks.open()))
p=next(x for x in csv.DictReader(preds.open()) if x['bin_idx']=='831')
b=next(x for x in csv.DictReader(obs.open()) if x['bin_index']=='831')
flux=float(p['matched_csda_flux']);scale=float(p['normalization']);measured=float(p['measured_flux'])
t=math.log(scale*flux/measured);theta=float(b['zenith_center_deg']);c=.3
W=float(p['fit_mask'])/(1+(.15*max(1/math.cos(math.radians(theta))-1,0))**2)
score=W*t/math.sqrt(1+(t/c)**2)/flux
result=[]
for h in (1e-6,1e-7,1e-8):
    left=next(x for x in reports if x['bin_idx']==831 and x['delta']==-h)
    right=next(x for x in reports if x['bin_idx']==831 and x['delta']==h)
    fluxleft=left['flux_change']/(-h);fluxright=right['flux_change']/h
    a=next(x for x in rows if x['direction']=='coordinate_375_-1' and float(x['step'])==h)
    b=next(x for x in rows if x['direction']=='coordinate_375_1' and float(x['step'])==h)
    observed=float(a['slope'])+float(b['slope']);attributed=score*(fluxright-fluxleft)
    result.append(dict(step=h,left_flux_secant=fluxleft,right_flux_secant=fluxright,
        observed_objective_secant_jump=observed,score_times_flux_secant_jump=attributed,
        attributed_fraction=attributed/observed,remainder=observed-attributed))
harmonic=r/'polish_v2_run/results/harmonic_reduction.toml'
air=tomllib.load(harmonic.open('rb'))
material=air['air_material_index']
crossing=next(x for x in reports if x['bin_idx']==831 and x['delta']==1e-8)['first_30_changed_events'][0]
tables=len(crossing['base_intervals'])//3
actual_air=dict(material_index=material,energy_knots_118_to_120=air['air_table_energy_knots_118_to_120'],
    base_intervals_dedx0_predictor_next=[crossing['base_intervals'][material-1+tables*k] for k in range(3)],
    trial_intervals_dedx0_predictor_next=[crossing['trial_intervals'][material-1+tables*k] for k in range(3)],
    used_by='propagate_csda_air_segment: property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy/energy_predict/energy_next)')
bindings={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [trace,checks,preds,obs,harmonic,ROOT/"src/Tomography.jl",ROOT/"src/Physics.jl",Path(__file__)]}
summary=dict(bin_idx=831,water_cell=375,robust_log_scale=c,dem_slope=.15,W=W,t=t,profile_scale=scale,
    envelope_score_dF_dp=score,comparisons=result,actual_air_table_crossing=actual_air,input_sha256=bindings,
    algebra='Fdata=sum W*c²*(sqrt(1+(t/c)²)-1), t=log(p)+a-log(obs). Profile score sum W*t/sqrt(1+(t/c)²)=0, so da/dw cancels in dF/dw; each flux coefficient is W*t/(p*sqrt(1+(t/c)²)).',
    caveat='Finite h secant attribution, not derivative-limit or causal proof. Smooth prior and other rays contribute finite-step curvature; interval tracing records predictor/next-energy table crossings, with unchanged adaptive step counts/limiters at h=1e-8.')
(r/'trace_mechanism_assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
