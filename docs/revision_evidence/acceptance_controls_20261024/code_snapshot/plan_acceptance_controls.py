#!/usr/bin/env python3
"""Prepare auditable preflight/full configurations; never launch a job."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection-execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.detection_execution.read_text())
    command = record['command']
    i = next(i for i, token in enumerate(command) if token.endswith('/examples/lvd_tomography.jl'))
    base_args = command[i + 1:]
    if '--detection-benchmark' not in base_args or '--correction-artifact' not in base_args:
        raise ValueError('Require audited material-corrected headline detection command')
    if args.output.exists():
        raise ValueError('Refuse existing plan output')
    args.output.mkdir(parents=True)
    headline_report = args.detection_execution.parent / 'data/lvd_detection_benchmark.txt'
    text = headline_report.read_text()
    psi = float(re.search(r'Ψ matched filter.*?τ₉₅\s*=\s*([\deE.+-]+)', text).group(1))
    roi = float(re.search(r'Φ unweighted plume-ROI sum\s+τ₉₅\s*=\s*([\deE.+-]+)', text).group(1))
    roi_bins = int(re.search(r'ROI:\s*(\d+)\s*/\s*2700', text).group(1))
    shared = dict(base_args=base_args, seed=20261024, benchmark_draws=32,
                  gaussian_null=3000, gaussian_signal=600, poisson_pilot=True,
                  poisson_calibration_null=3000, poisson_test_null=3000,
                  poisson_signal=600, pilot_exposure_factors=[.5, 1., 2.],
                  detection_execution=str(args.detection_execution.resolve()),
                  detection_execution_sha256=hashlib.sha256(args.detection_execution.read_bytes()).hexdigest(),
                  headline_report=str(headline_report.resolve()),
                  headline_report_sha256=hashlib.sha256(headline_report.read_bytes()).hexdigest(),
                  headline_matched_tau=psi, headline_roi_tau=roi, headline_roi_bins=roi_bins)
    for name, preflight in (('load_smoke', True), ('preflight', True), ('full', False)):
        config = dict(shared, preflight_only=preflight,
                      load_only=name == 'load_smoke',
                      output=str((args.output / name / 'results').resolve()))
        def toml(value):
            return json.dumps(value, ensure_ascii=True)
        (args.output / (name + '.toml')).write_text(
            '\n'.join(f'{key} = {toml(value)}' for key, value in config.items()) + '\n')
    (args.output / 'REVIEW_REQUIRED.md').write_text(
        'Prepared only. Run preflight under a recorded bounded launcher, review '
        'legacy parity and timing, then independently approve the full job. '
        'Use disjoint source-frozen execution scopes; do not overwrite outputs.\n')


if __name__ == '__main__':
    main()
