#!/usr/bin/env python3
"""Read production evidence; never edit manuscript claims or infer completion.

Usage: python3 dev/reconcile_manuscript_outputs.py --run archived=PATH \
    --run corrected=PATH --output REPORT_DIRECTORY
No current-source defaults are substituted for absent historical metadata.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re

STAGES = {
    'topo': ['lvd_topography_grid.csv'],
    'main': ['lvd_tomography_inverse.txt', 'lvd_single_muon_angular_distribution.txt',
             'lvd_mc_common_subset_summary.csv'],
    'sweep': ['lvd_reconstruction_sweep.txt'],
    'detection': ['lvd_detection_benchmark.txt'],
    'contrast': ['lvd_detection_contrast_sweep.txt'],
}
TABLES = ['lvd_recursion_history.csv', 'lvd_mc_coverage_summary.csv',
          'lvd_mc_common_subset_summary.csv']
REPORTS = ['lvd_tomography_inverse.txt', 'lvd_reconstruction_sweep.txt',
           'lvd_detection_benchmark.txt', 'lvd_detection_contrast_sweep.txt',
           'lvd_single_muon_angular_distribution.txt', 'lvd_tomography_validation.txt']


def read_csv(file):
    with file.open(newline='') as stream:
        return list(csv.DictReader(stream))


def identify(file, latest_manifest_time=None):
    stat = file.stat()
    return dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                bytes=stat.st_size, modified_utc=datetime.datetime.fromtimestamp(
                    stat.st_mtime, datetime.timezone.utc).isoformat(),
                predates_latest_manifest=(stat.st_mtime < latest_manifest_time)
                    if latest_manifest_time is not None else None)


def summarize(label, directory):
    directory = directory.resolve()
    manifests = sorted(directory.glob('run_manifest_*.json'))
    latest_time = manifests[-1].stat().st_mtime if manifests else None
    report = dict(label=label, directory=str(directory), exists=directory.is_dir(),
                  scope='Recorded evidence only. Artifacts do not establish successful completion, '
                        'scientific validity, convergence, or calibrated uncertainty coverage.',
                  run_manifests=[], stages={}, metrics={}, artifacts={})
    def artifact(file):
        report['artifacts'][file.name] = identify(file, latest_time)
    for file in manifests:
        report['run_manifests'].append(dict(identity=identify(file), recorded=json.loads(file.read_text())))
    report['metadata_status'] = 'recorded_manifests_available' if manifests else 'unknown_no_run_manifest'
    # Launcher OK/FAILED records are explicit observations, never inferred from CSVs.
    logs = sorted(set(directory.glob('launcher*.log')) | set(directory.glob('*launcher*.txt')))
    events = {stage: [] for stage in STAGES}
    for file in logs:
        for number, line in enumerate(file.read_text(errors='replace').splitlines(), 1):
            match = re.search(r'\[(topo|main|sweep|detection|contrast)\] (OK|FAILED)\b', line)
            if match:
                events[match[1]].append(dict(status=match[2], file=str(file), line=number, text=line))
    for stage, expected in STAGES.items():
        present = [name for name in expected if (directory/name).is_file()]
        report['stages'][stage] = dict(
            artifact_status='missing_pending' if not present else
                ('partial_outputs' if len(present)<len(expected) else 'outputs_available_completion_unverified'),
            expected=expected, available=present, missing=[name for name in expected if name not in present],
            explicit_launcher_events=events[stage])
    for name in ('STOP_REASON.txt', 'attempt_status.json', 'failed.json', 'completed.json'):
        file=directory/name
        if file.is_file():
            artifact(file)
            report.setdefault('administrative_records', {})[name]=file.read_text()
    status=directory/'lvd_recursion_status.txt'
    if status.is_file():
        artifact(status)
        report['metrics']['recursion_configuration_and_status']={
            key.strip():value.strip() for line in status.read_text().splitlines()
            if '=' in line for key,value in [line.split('=',1)]}
    for name in TABLES:
        file=directory/name
        if file.is_file():
            artifact(file);report['metrics'][name]=read_csv(file)
    mask=directory/'lvd_fit_mask.csv'
    if mask.is_file():
        artifact(mask);rows=read_csv(mask)
        report['metrics']['exact_fit_mask_counts']=dict(
            all_bins=len(rows), covered_bins=sum(r.get('covered','').lower()=='true' for r in rows),
            ray_valid_bins=sum(r.get('ray_valid','').lower()=='true' for r in rows),
            fit_bins=sum(float(r['fit_mask'])>0 for r in rows))
    common=directory/'lvd_mc_common_subset.csv'
    if common.is_file():
        artifact(common);rows=read_csv(common)
        report['metrics']['common_subset_identity']=dict(bins=len(rows),
            unique_bins=len({r['bin_idx'] for r in rows}),
            common_scales=sorted({r['common_scale'] for r in rows}),
            scope='Dry CSDA uses initial calibration; wet CSDA final calibration. '
                  'Wet MC uses prescribed rays; inspect run provenance for transport scope. '
                  'This measured-data residual check is not independent held-out closure.')
    # Preserve source line numbers and printed precision. No regex-derived or
    # rounded numerical claim is silently promoted to an exact computed value.
    for name in REPORTS:
        file=directory/name
        if file.is_file():
            artifact(file)
            report['metrics'][name]=[dict(line=i,text=line) for i,line in
                enumerate(file.read_text().splitlines(),1) if line.strip()]
    # Keep the large topography table out of JSON while identifying it exactly.
    topo=directory/'lvd_topography_grid.csv'
    if topo.is_file(): artifact(topo)
    report['missing_metadata_policy']='Unknown sample counts/configuration remain unknown. '
    report['missing_metadata_policy']+='Recorded manifests/recursion status and report lines are reproduced; '
    report['missing_metadata_policy']+='no sample-count defaults are inferred from current source.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True,metavar='LABEL=PATH')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs=[]
    for value in args.run:
        label,separator,path=value.partition('=')
        if not separator or not label or not path: parser.error('--run requires LABEL=PATH')
        runs.append(summarize(label,Path(path)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'reconciliation.json').write_text(json.dumps(runs,indent=2)+'\n')
    lines=['# Manuscript numerical reconciliation','',
           'Read-only evidence index. Missing stages remain pending; source data are never replaced by current defaults.', '']
    for run in runs:
        lines += [f"## {run['label']}",'',f"Directory: `{run['directory']}`",'',
                  f"Metadata: {run['metadata_status']}",'',
                  '| Stage | Artifact status | Missing outputs |','|---|---|---|']
        for name,stage in run['stages'].items():
            lines.append(f"| {name} | {stage['artifact_status']} | {', '.join(stage['missing']) or 'none'} |")
        lines += ['', 'Exact recorded configurations, source/input hashes, CSV metrics and numbered report lines are in `reconciliation.json`.', '']
        stale=[name for name,record in run['artifacts'].items() if record['predates_latest_manifest']]
        if stale: lines += ['Artifacts predating the latest manifest (possible mixed provenance): '+', '.join(stale),'']
        if 'administrative_records' in run: lines += ['Administrative stop/completion records are present; inspect JSON before treating this run as final.', '']
    (args.output/'reconciliation.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
