#!/usr/bin/env python3
"""Durably verify completed main recovery and record a derived assessment."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'diffmuontomoLVD/revision'))
from stage_result_parsers import completed_stage

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    folder=a.main.resolve();output=a.output.resolve()
    if output.exists():p.error('fresh assessment required')
    helpers=[Path(__file__).resolve(),ROOT/'diffmuontomoLVD/revision/stage_result_parsers.py',ROOT/'diffmuontomoLVD/revision/build_surrogate_stability_report.py']
    hashes={str(x):sha(x) for x in helpers}
    while not (folder/'end_status.json').exists():time.sleep(30)
    result=dict(schema_version=1,scope='Derived completed-main recovery audit; no refit or new simulation',review_required=True,revision_complete=False,main=str(folder),helper_sha256=hashes)
    try:
        if any(sha(x)!=digest for x,digest in hashes.items()):raise ValueError('assessment helper changed while waiting')
        parsed=completed_stage(folder,'main')
        if not parsed.get('recovery'):raise ValueError('main is not an audited completed-fit recovery')
        with (folder/'data/lvd_stable_curvature.csv').open() as stream:curvature=list(csv.DictReader(stream))
        if len(curvature)!=600 or [int(r['cell_idx']) for r in curvature]!=list(range(1,601)):raise ValueError('curvature grid mismatch')
        if any(not math.isfinite(float(r['residual_gauss_newton_curvature_diagonal'])) or float(r['residual_gauss_newton_curvature_diagonal'])<0 for r in curvature):raise ValueError('invalid curvature diagnostic')
        with (folder/'data/lvd_mc_coverage.csv').open() as stream:coverage=list(csv.DictReader(stream))
        selected=[r for r in coverage if r['selected_mc']=='true']
        with (folder/'data/lvd_mc_common_subset.csv').open() as stream:closure=list(csv.DictReader(stream))
        if {int(r['bin_idx']) for r in closure}!={int(r['bin_idx']) for r in selected} or len(closure)!=len(selected):raise ValueError('actual MC closure directions differ from selection')
        if len(closure)!=630 or sum(float(r['zenith_deg'])<60 for r in closure)!=550:raise ValueError('unexpected closure angular counts')
        record=json.loads((folder/'recovery_provenance.json').read_text());command=record['command']
        if command[command.index('--seed')+1]!='42' or command[command.index('--papermatch-mc-samples')+1]!='8192':raise ValueError('unexpected MC seed/count')
        result.update(passed=True,main_end_sha256=sha(folder/'end_status.json'),recovery=parsed['recovery'],
            measured=parsed['measured'],curvature_rows=600,curvature_scope='Local residual Gauss-Newton diagonal, not posterior uncertainty',
            actual_mc_bins=630,actual_mc_bins_below60=550,actual_mc_bins_atleast60=80,mc_histories_per_bin=8192,
            energy_seed=819,transport_seed_formula='900042 + covered-position k (Julia 1-based)',
            unchanged_tail_sha256=record['unchanged_tail_sha256'],permanent_original_scope_fix_pending=True)
    except Exception as exc:result.update(passed=False,error=str(exc))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
