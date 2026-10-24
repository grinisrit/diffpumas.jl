#!/usr/bin/env python3
"""Independent arithmetic summaries of eight empirical calibration replicates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(minimum=float(x.min()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('fresh output required')
    coeff=list(csv.DictReader((args.input/'coefficients.csv').open()))
    rows=list(csv.DictReader((args.input/'fixed_field_ensemble.csv').open()))
    assert [int(x['replicate']) for x in coeff]==list(range(1,9))
    beta=np.array([[float(x['linear']),float(x['quadratic'])] for x in coeff])
    grouped={r:[x for x in rows if int(x['replicate'])==r] for r in range(1,9)}
    ids=[int(x['bin_idx']) for x in grouped[1]]
    assert len(ids)==len(set(ids))==1003
    assert all([int(x['bin_idx']) for x in grouped[r]]==ids for r in range(1,9))
    fit=np.array([x['fit_mask']=='true' for x in grouped[1]]);assert fit.sum()==690
    q=np.array([float(x['column_water_fraction']) for x in grouped[1]])
    coverage={int(x['bin_idx']):x for x in csv.DictReader(args.coverage.open())}
    assert set(ids)==set(coverage)
    result=dict(scope='Empirical sample dispersion from eight finite-training-MC material-coefficient fits with fixed dry calibration and fixed recovered field. No structural model discrepancy, inverse uncertainty, posterior or calibrated interval.',replicates=8,coefficient_mean=beta.mean(axis=0).tolist(),coefficient_sample_std=beta.std(axis=0,ddof=1).tolist(),coefficient_sample_covariance=np.cov(beta,rowvar=False).tolist(),ray_groups={})
    julia=tomllib.loads((args.input/'ensemble_summary.toml').read_text())
    assert np.allclose(result['coefficient_sample_covariance'],julia['coefficient_covariance_rows'],rtol=1e-12,atol=1e-15)
    for name,mask in [('covered',np.ones(1003,dtype=bool)),('fitted',fit),('dry_columns',q==0),('positive_columns',q>0)]:
        info={'rays':int(mask.sum()),'dry_columns':int(((q==0)&mask).sum())}
        for key,basekey in [('final_flux','final_flux'),('support_abs_jacobian','final_support_abs_jacobian'),('support_abs_logflux_sensitivity','final_support_abs_logflux_sensitivity')]:
            values=np.array([[float(x[key]) for x in grouped[r]] for r in range(1,9)])
            baseline=np.array([float(coverage[i][basekey]) for i in ids])
            positive=mask&(baseline>0)
            relative_sd=values.std(axis=0,ddof=1)[positive]/baseline[positive]
            mean_shift=values.mean(axis=0)[positive]/baseline[positive]-1
            info[key]={'positive_baseline_rays':int(positive.sum()),'zero_baseline_rays':int((mask&(baseline==0)).sum()),'pointwise_relative_sample_sd':stats(relative_sd) if len(relative_sd) else {},'pointwise_ensemble_mean_relative_shift_from_original':stats(mean_shift) if len(mean_shift) else {}}
        result['ray_groups'][name]=info
    sources=[Path(__file__),args.input/'coefficients.csv',args.input/'fixed_field_ensemble.csv',args.input/'ensemble_summary.toml',args.coverage]
    result['numpy_version']=np.__version__
    result['input_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
