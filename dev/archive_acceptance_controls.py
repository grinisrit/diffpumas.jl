#!/usr/bin/env python3
"""Archive compact audited acceptance summaries; preserve large raw counts in place."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    end = json.loads((root / 'end_status.json').read_text())
    if not (end.get('execution_completed') and end.get('source_unchanged') and end.get('runtime_unchanged') and end.get('returncode') == 0):
        raise ValueError('Unclean experiment: ' + str(root))
    for name, expected in end['output_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Changed experiment output: ' + name)
    return end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstorder', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True, help='Extension queue root, containing job/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh compact archive')
    first = args.firstorder.resolve()
    extension = (args.extension / 'job').resolve()
    ends = dict(firstorder=audit(first), extension=audit(extension))
    old = pd.read_csv(first / 'results/poisson_pilot_intervals.csv')
    paired = pd.read_csv(extension / 'results/paired_pilot_intervals.csv')
    if len(old) != 36 or len(paired) != 144:
        raise ValueError('Incomplete detector/profile/hypothesis/exposure grid')
    if paired.duplicated(['profile', 'method', 'hypothesis', 'exposure_factor']).any():
        raise ValueError('Duplicate paired result')
    keys = ['profile', 'method', 'exposure_factor']
    merged = old.merge(paired[(paired.hypothesis == 'first_order') & (paired.exposure_factor <= 2)], on=keys, suffixes=('_old', '_paired'), validate='one_to_one')
    fields = ['total_E', 'expected_null_counts', 'expected_signal_counts', 'threshold', 'null_exceedances', 'null_draws', 'signal_exceedances', 'signal_draws']
    for field in fields:
        if not (merged[field + '_old'].to_numpy() == merged[field + '_paired'].to_numpy()).all():
            raise ValueError('Original first-order results changed: ' + field)
    for _, group in paired.groupby(keys):
        if len(group) != 2 or group.threshold.nunique() != 1 or group.null_exceedances.nunique() != 1:
            raise ValueError('Signal hypotheses do not share identical nulls/threshold')
    for _, row in paired.iterrows():
        for prefix in ('null', 'signal'):
            result = binomtest(int(row[prefix + '_exceedances']), int(row[prefix + '_draws']))
            ci = result.proportion_ci(.95, method='exact')
            if abs(ci.low - row[prefix + '_ci95_lower']) > 1e-12 or abs(ci.high - row[prefix + '_ci95_upper']) > 1e-12:
                raise ValueError('Exact binomial interval mismatch')
    replay = pd.read_csv(extension / 'results/reused_statistic_replay.csv')
    if len(replay) != 108 or (replay.max_abs_difference > 1e-10).any():
        raise ValueError('Reused-source statistic replay is incomplete or changed')
    args.output.mkdir(parents=True)
    selected = {
        'firstorder': ['execution.json', 'end_status.json', 'results/config.toml', 'results/preflight.toml', 'results/experiment_scope.toml', 'results/planning.csv', 'results/poisson_pilot_raw.csv', 'results/poisson_pilot_intervals.csv', 'results/directional_templates.csv', 'results/exposure_shares.csv'],
        'extension': ['execution.json', 'end_status.json', 'results/config.toml', 'results/scope.toml', 'results/paired_pilot_raw.csv', 'results/paired_pilot_intervals.csv', 'results/reused_statistic_replay.csv']}
    for label, source in [('firstorder', first), ('extension', extension)]:
        out = args.output / label
        out.mkdir()
        for name in selected[label]:
            shutil.copy2(source / name, out / Path(name).name)
        for name in ends[label]['output_sha256']:
            relative = Path(name)
            if relative.suffix == '.json' and '_poisson_' in relative.parent.name:
                metadata = args.output / 'rng_metadata' / label / relative.parent.name / relative.name
                metadata.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, metadata)
    shutil.copy2(args.extension / 'reuse_certificate.json', args.output / 'reuse_certificate.json')
    root = Path(__file__).resolve().parents[1]
    code = args.output / 'code_snapshot'; code.mkdir()
    names = ['detection_acceptance_controls.jl', 'acceptance_poisson_draws.py', 'plan_acceptance_controls.py', 'detection_acceptance_finite_extension.jl', 'run_acceptance_finite_extension.py', 'archive_acceptance_controls.py']
    for name in names:
        shutil.copy2(root / 'dev' / name, code / name)
    summary = dict(
        scope='Four assumed acceptance profiles at equal total integrated exposure; same fixed first-order statistics, separate first-order and finite-CSDA signal hypotheses. Not measured LVD acceptance, stochastic-transport truth or certified physical power.',
        raw_roots=dict(firstorder=str(first), extension=str(extension)),
        raw_outputs_sha256={label: value['output_sha256'] for label, value in ends.items()},
        all_raw_output_hashes_verified=True, original_first36_results_exact=True,
        shared_null_thresholds=True, exact_binomial_interval_values_recomputed=True,
        reused_statistic_replay_count=len(replay), replay_max_abs=float(replay.max_abs_difference.max()),
        raw_count_policy='Large .bin count populations and per-draw statistics remain at raw_roots, with hashes preserved here; they are intentionally not copied into Git.',
        compact_archive_sha256={str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output / 'archive_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'README.md').write_text('''# Acceptance and finite-count detection controls

The first-order experiment and paired finite-CSDA extension completed with unchanged source/runtime audits. Compact configuration, exposure/power tables, exact binomial Monte Carlo intervals, directional templates, exposure shares and code snapshots are retained here. Large count binaries and per-draw statistics remain under the raw roots in archive_audit.json; their full output hashes were verified and preserved.

All four profiles use the common exposure definition E=sum(tau_m), with tau_m=E*p_m and sum(p_m)=1. At a given absolute E their exposure budgets match. The actual pilot grid is E=Epsi_profile times 0.5/1/2/4/8/16, where Epsi_profile is each profile's Gaussian matched-score estimate; equal factors are not equal absolute E across profiles. Profiles are equal-per-bin legacy, uniform directional area, cosine and azimuth-modulated cosine. The legacy conversion is E=2700*tau_old. Expected counts differ by profile and are reported explicitly; these assumed profiles are not calibrated LVD acceptance.

Gaussian planning uses the dry-Jacobian first-order w=0.9 four-cell lens. Fresh Poisson simulations evaluate the same fixed first-order matched score, intrinsic-template ROI count and 200-step SART on two signal hypotheses: first-order flux and finite nonlinear CSDA flux. Neither is stochastic transport truth. Null calibration/test populations are shared between alternatives, while their signal counts use independent streams. The extension reuses the original first three exposure factors exactly and adds factors 4/8/16, with 108 source-statistic replays before new simulation.

Intervals quantify binomial Monte Carlo uncertainty conditional on each realized calibration threshold; threshold uncertainty and structural model uncertainty are not propagated. Three thousand null draws give only about four expected events at the nominal three-sigma tail. No exact three-sigma false-positive rate, physical finite-anomaly power, or universal matched-filter superiority is certified.
''')

    summary['compact_archive_sha256'] = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*') if p.is_file() and p.name != 'archive_audit.json'}
    (args.output / 'archive_audit.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
