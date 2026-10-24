#!/usr/bin/env python3
"""Generate fresh finite-Poisson counts or exact binomial MC intervals.

No detector model is inferred here. Rates and seeds are explicit inputs from
the independently recorded acceptance-control experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    draw = sub.add_parser('draw')
    draw.add_argument('--rates', type=Path, required=True)
    draw.add_argument('--exposure', type=float, required=True)
    draw.add_argument('--draws', type=int, required=True)
    draw.add_argument('--seed', type=int, required=True)
    draw.add_argument('--output', type=Path, required=True)
    interval = sub.add_parser('intervals')
    interval.add_argument('--input', type=Path, required=True)
    interval.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'draw':
        rates = np.loadtxt(args.rates, ndmin=1)
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError('Invalid Poisson rate')
        if args.exposure <= 0 or args.draws <= 0 or args.output.exists():
            raise ValueError('Require positive exposure/draw count and new output')
        # Draw-major contiguous storage equals Julia columns of (bins, draws).
        values = np.random.default_rng(args.seed).poisson(
            args.exposure * rates, size=(args.draws, len(rates)))
        values.astype('<i8', copy=False).tofile(args.output)
        args.output.with_suffix('.json').write_text(json.dumps(dict(
            numpy=np.__version__, scipy=scipy.__version__, seed=args.seed,
            draws=args.draws, bins=len(rates), exposure=args.exposure,
            generator='numpy.default_rng PCG64; Poisson variates, no Gaussian substitution',
            format='little-endian signed int64, draw-major'), indent=2) + '\n')
    else:
        rows = list(csv.DictReader(args.input.open()))
        for row in rows:
            for prefix in ('null', 'signal'):
                result = binomtest(int(row[prefix + '_exceedances']),
                                  int(row[prefix + '_draws']))
                ci = result.proportion_ci(confidence_level=.95, method='exact')
                row[prefix + '_rate'] = result.statistic
                row[prefix + '_ci95_lower'] = ci.low
                row[prefix + '_ci95_upper'] = ci.high
        if not rows:
            raise ValueError('No pilot results')
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
