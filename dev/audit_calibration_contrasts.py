#!/usr/bin/env python3
"""Compare every original training contrast with eight repeated MC calibrations.
No observation, coefficient, or training ray is removed or retuned.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-training',type=Path,required=True)
    p.add_argument('--ensemble',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('fresh output required')
    args.output.mkdir(parents=True)
    original=read(args.original_training)
    plan=tomllib.loads((args.ensemble/'preregistered_plan.toml').read_text())
    coefficients=read(args.ensemble/'coefficients.csv')
    assert len(coefficients)==8
    beta=np.array([[float(r['linear']),float(r['quadratic'])] for r in coefficients])
    oldbeta=np.array(plan['coefficient_origin'])
    records=[];sources=[Path(__file__),args.original_training,args.ensemble/'preregistered_plan.toml',args.ensemble/'coefficients.csv']
    for ray in range(1,13):
        old=sorted([r for r in original if int(r['bin'])==ray],key=lambda r:int(r['field']))
        assert len(old)==2
        oldcontrast=np.log(np.array([float(r['mc_wet']) for r in old])/float(old[0]['mc_dry']))
        repeated=[]
        for rep in range(1,9):
            path=args.ensemble/f'replicates/rep_{rep}_ray_{ray}.csv';sources.append(path)
            sample=sorted(read(path),key=lambda r:float(r['water_fraction']))
            assert [float(r['water_fraction']) for r in sample]==[0.,.2,.6]
            repeated.append(np.log(np.array([float(r['mc_flux']) for r in sample[1:]])/float(sample[0]['mc_flux'])))
        repeated=np.array(repeated)
        for k,q in enumerate((.2,.6)):
            mean=float(repeated[:,k].mean());sd=float(repeated[:,k].std(ddof=1));difference=float(oldcontrast[k]-mean)
            records.append(dict(ray=ray,zenith_deg=float(old[k]['zenith']),azimuth_deg=float(old[k]['azimuth']),training_fraction=q,original_log_mc_wet_dry_ratio=float(oldcontrast[k]),ensemble_mean_log_ratio=mean,ensemble_sample_sd_log_ratio=sd,original_minus_ensemble_mean=difference,descriptive_difference_over_sample_sd=difference/sd,ensemble_min_log_ratio=float(repeated[:,k].min()),ensemble_max_log_ratio=float(repeated[:,k].max())))
    # In this original uniform-field geometry q equals .2/.6 up to FP rounding.
    # Check that this analytical ridge influence recovers the coefficient shift.
    A=np.tile([[.2,.04],[.6,.36]],(12,1));H=np.linalg.solve(A.T@A+1e-6*np.eye(2),A.T)
    delta=np.array([r['original_minus_ensemble_mean'] for r in records])
    reconstructed=H@delta;actual=oldbeta-beta.mean(axis=0)
    assert np.allclose(reconstructed,actual,rtol=1e-12,atol=1e-12)
    contributions=[]
    for ray in range(1,13):
        contribution=H[:,2*(ray-1):2*ray]@delta[2*(ray-1):2*ray]
        contributions.append(dict(ray=ray,linear_coefficient_shift_contribution=float(contribution[0]),quadratic_coefficient_shift_contribution=float(contribution[1])))
    for name,rows in [('contrast_comparison.csv',records),('coefficient_shift_contributions.csv',contributions)]:
        with (args.output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(scope='Descriptive fixed-design contrast comparison and linear ridge influence only. Eight sample dispersions are not calibrated standard errors of truth, z-tests or confidence bounds. Every original and new training row is retained; no retuning.',original_coefficients=oldbeta.tolist(),ensemble_mean=beta.mean(axis=0).tolist(),ensemble_min=beta.min(axis=0).tolist(),ensemble_max=beta.max(axis=0).tolist(),original_outside_eight_replicate_ranges=((oldbeta<beta.min(axis=0))|(oldbeta>beta.max(axis=0))).tolist(),actual_coefficient_shift=actual.tolist(),reconstructed_coefficient_shift=reconstructed.tolist(),maximum_shift_reconstruction_error=float(np.max(abs(actual-reconstructed))),design='Original twelve-ray uniform .2/.6 geometry; q=.2/.6 up to floating rounding. H=(A.T A+1e-6 I)^-1 A.T. CSDA contrast cancels in old-minus-repeated target differences.',input_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (args.output/'contrast_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
