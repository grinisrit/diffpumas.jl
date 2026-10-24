#!/usr/bin/env python3
"""Export audited stage evidence and reviewed candidate figures; never edit paper prose."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {
    'topo': ['fig_topography_3d'],
    'main': ['fig_sensitivity_volume_3d', 'fig_detection_volume_3d', 'fig_detector_2d_measured', 'fig_validation_zscores', 'fig_inverse_metrics', 'fig_inverse_field_3d',
             'fig_resolution_map', 'fig_water_increase_3d', 'fig_detector_water_field_3d',
             'fig_detector_2d_match', 'fig_mc_validation', 'fig_mc_flux_band'],
    'detection': [],  # The reviewed two-panel figure also requires contrast.
    'contrast': ['fig_detection_tau95'],
    'sweep': ['fig_reconstruction_sweep'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    end = json.loads((folder/'end_status.json').read_text())
    if not (end['execution_completed'] and end['source_unchanged'] and end['runtime_unchanged']):
        raise ValueError(f'unclean stage: {folder}')
    for name, expected in end['output_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError(f'stage output changed: {folder/name}')
    return end


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--stage', choices=BUILDERS, required=True)
    ap.add_argument('--output', type=Path, help='fresh isolated retry export destination')
    ap.add_argument('--figure-helper', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    reconcile = ROOT/'dev/reconcile_manuscript_outputs.py'
    stage_parser = ROOT/'diffmuontomoLVD/revision/stage_result_parsers.py'
    helper_hashes = {str(p):digest(p) for p in (Path(__file__).resolve(), reconcile, stage_parser, stage_parser.with_name('build_surrogate_stability_report.py'), args.figure_helper.resolve())}
    completed = json.loads((root/'status.json').read_text())['completed_stages']
    if args.stage not in completed:
        raise ValueError('requested stage not recorded complete')
    target = args.output.resolve() if args.output else root/'candidate_exports'/args.stage
    target.mkdir(parents=True, exist_ok=False)
    bundle = target/'data'
    bundle.mkdir()
    sources = {}
    ends = {}
    for stage in completed:
        folder = root/'stages'/stage
        audit(folder)
        ends[stage] = digest(folder/'end_status.json')
        for file in sorted((folder/'data').rglob('lvd_*')):
            if not file.is_file() or file.suffix not in ('.csv', '.txt', '.html', '.toml'):
                continue
            destination = bundle/file.name
            if destination.exists() and digest(destination) != digest(file):
                raise ValueError(f'ambiguous stage artifact basename: {file.name}')
            shutil.copy2(file, destination)
            sources.setdefault(file.name, []).append(dict(path=str(file), sha256=digest(file)))
    if 'main' in completed:
        log = root/'stages/main/run.log'
        shutil.copy2(log, bundle/'log_main.txt')
        sources['log_main.txt'] = [dict(path=str(log), sha256=digest(log))]
    auxiliary_controls = {}
    qfolder = root/'controls/column_domain'
    if (qfolder/'end_status.json').is_file():
        audit(qfolder)
        auxiliary_controls['column_domain'] = dict(end_status_sha256=digest(qfolder/'end_status.json'),
            outputs_sha256={name:digest(qfolder/name) for name in ('column_domain.csv','column_domain_summary.toml')})
        (target/'column_domain').mkdir()
        for name in ('column_domain.csv','column_domain_summary.toml','end_status.json','execution.json'):
            shutil.copy2(qfolder/name,target/'column_domain'/name)
    commands = []
    command = [sys.executable, str(stage_parser), '--stage', args.stage,
               '--directory', str(root/'stages'/args.stage), '--output', str(target/'stage_results.json')]
    subprocess.run(command, cwd=ROOT, check=True)
    commands.append(command)
    # Reconciliation preserves exact printed lines/precision; it invents no claims.
    command = [sys.executable, str(reconcile)]
    for stage in completed:
        command += ['--run', f'{stage}={root/"stages"/stage/"data"}']
    command += ['--output', str(target/'reconciliation')]
    subprocess.run(command, cwd=ROOT, check=True)
    commands.append(command)
    figures = target/'figures'
    selected = BUILDERS[args.stage]
    figure_status = 'not_requested_until_complementary_stage' if not selected else 'pending'
    if selected:
        command = [sys.executable, str(args.figure_helper.resolve()), '--data-dir', str(bundle),
                   '--output-dir', str(figures), '--only', ','.join(selected)]
        with (target/'figure_export.log').open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        commands.append(command)
        expected_images = [name+'.png' for name in selected]
        if 'fig_detector_water_field_3d' in selected:
            expected_images += ['fig_detector_water_depth_profile.png']
        missing = [name for name in expected_images if not (figures/name).is_file() or not (figures/name).stat().st_size]
        if result.returncode or missing:
            figure_status = dict(returncode=result.returncode, missing=missing)
        else:
            figure_status = 'candidate_figures_exported'
    changed_helpers = [name for name, value in helper_hashes.items() if digest(Path(name)) != value]
    if changed_helpers:
        raise ValueError('export helper changed during rendering: '+', '.join(changed_helpers))
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Audited generated candidates only; execution is separate from scientific acceptance; no paper prose changed',
        stage=args.stage, completed_stages=completed, stage_end_sha256=ends,
        artifact_sources=sources, auxiliary_controls=auxiliary_controls, commands=commands, figure_status=figure_status,
        helper_sha256=helper_hashes, helpers_unchanged=True,
        outputs_sha256={str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()})
    (target/'export_provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(stage=args.stage, candidate_exports=str(target), figure_status=figure_status)))
    return 0 if isinstance(figure_status, str) else 1


if __name__ == '__main__':
    raise SystemExit(main())
