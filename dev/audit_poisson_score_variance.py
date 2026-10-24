#!/usr/bin/env python3
"""Audit the alternative/null variance ratio of the fixed first-order score."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--templates',type=Path,required=True)
    p.add_argument('--shares',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('fresh output required')
    with args.templates.open() as stream:templates=list(csv.DictReader(stream))
    with args.shares.open() as stream:shares=list(csv.DictReader(stream))
    assert len(templates)==len(shares)==2700
    for rows in (templates,shares):assert np.array_equal([float(x['bin']) for x in rows],np.arange(1,2701))
    f=np.array([float(x['dry_flux']) for x in templates]);s=np.array([float(x['linear_signal']) for x in templates])
    assert np.all(np.isfinite(f)) and np.all(f>0) and np.all(np.isfinite(s)) and np.all(f+s>=0)
    z=s/f;records=[]
    for profile in ('legacy_equal_bin','uniform_directional_area','cosine','azimuth_modulated'):
        a=np.array([float(x[profile]) for x in shares]);assert np.all(a>0) and np.isclose(a.sum(),1,rtol=0,atol=1e-14)
        null=float(np.sum(z*z*a*f));alternative=float(np.sum(z*z*a*(f+s)))
        records.append(dict(profile=profile,null_score_variance_per_E=null,first_order_alt_variance_per_E=alternative,alt_to_null_sd_ratio=float(np.sqrt(alternative/null)),expected_signal_to_null_count_ratio=float(np.sum(a*(f+s))/np.sum(a*f))))
    files=[Path(__file__),args.templates,args.shares]
    result=dict(scope='Frozen first-order template variance audit only; no new draws or power interpolation. Gaussian null-standardized planning omits changed alternative variances. Power shortfall cannot be attributed solely to Poisson skewness.',definition='sqrt(sum((s/f)^2*p*(f+s))/sum((s/f)^2*p*f))',rows=records,input_sha256={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in files})
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
