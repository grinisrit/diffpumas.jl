#!/usr/bin/env python3
"""Plan or execute sequential frozen-artifact, full-grid reconstruction controls.

Run --execute inside tmux only after the coordinator freezes/tests source.
Every fit uses run_revision_job.py in a fresh directory. Scientific failures
(nonstationarity, unequal masks, field dependence) remain visible in summary.json.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib

from record_revision_run import record_sources

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def status(path):
    return dict(line.rstrip().split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def norm(xs):
    return math.sqrt(sum(x*x for x in xs))


def hash_values(xs):
    return hashlib.sha256(json.dumps(xs, separators=(',', ':')).encode()).hexdigest()


def result(folder):
    data = folder / 'results'
    pred = {int(r['bin_idx']): r for r in rows(data / 'lvd_fit_predictions.csv')}
    field = rows(data / 'lvd_fit_field.csv')
    solver = status(data / 'lvd_solver_status.txt')
    execution = json.loads((folder / 'end_status.json').read_text())
    return dict(pred=pred, field=field, solver=solver, execution=execution)


def compare(reference, candidate):
    rp, cp = reference['pred'], candidate['pred']
    covered = sorted(set(rp) & set(cp))
    common_fit = [k for k in covered if float(rp[k]['fit_mask']) > 0 and float(cp[k]['fit_mask']) > 0]
    metrics = {}
    for label, keys in [('common_covered', covered), ('common_fit', common_fit)]:
        def residuals(pred):
            obs = [float(pred[k]['measured_flux']) for k in keys]
            fitted = [float(pred[k]['matched_csda_flux'])*float(pred[k]['normalization']) for k in keys]
            return dict(relative_l2=norm([x-y for x,y in zip(fitted,obs)])/max(norm(obs),1e-300),
                        log_rmse=math.sqrt(sum(math.log(x/y)**2 for x,y in zip(fitted,obs))/len(keys)))
        metrics[label] = dict(n_bins=len(keys), baseline=residuals(rp), candidate=residuals(cp),
            prediction_relative_l2=norm([float(cp[k]['matched_csda_flux'])*float(cp[k]['normalization'])-
                float(rp[k]['matched_csda_flux'])*float(rp[k]['normalization']) for k in keys]) /
                max(norm([float(rp[k]['matched_csda_flux'])*float(rp[k]['normalization']) for k in keys]),1e-300))
    rf, cf = reference['field'], candidate['field']
    identity = lambda fs: [(r['cell_idx'],r['x_m'],r['y_m'],r['z_m'],r['aquifer_mask']) for r in fs]
    same_support = identity(rf) == identity(cf)
    metrics['same_support'] = same_support
    metrics['same_screen'] = [(k,rp[k]['fit_mask']) for k in sorted(rp)] == [(k,cp[k]['fit_mask']) for k in sorted(cp)]
    if same_support:
        active = [i for i,r in enumerate(rf) if r['aquifer_mask'].lower() == 'true']
        difference = [float(cf[i]['w'])-float(rf[i]['w']) for i in active]
        metrics.update(field_relative_l2=norm(difference)/max(norm([float(rf[i]['w']) for i in active]),1e-300),
                       field_rmse=norm(difference)/math.sqrt(len(active)), field_max_abs=max(map(abs,difference)))
    return metrics


def summarize(output, names):
    parsed = {}
    report = {'runs': {}, 'comparisons': {}}
    for name in names:
        folder = output / name
        if not folder.exists():
            continue
        try:
            r = result(folder)
            parsed[name] = r
            solver = r['solver']
            report['runs'][name] = dict(execution_completed=r['execution']['execution_completed'],
                kkt_converged=solver.get('kkt_converged') == 'true',
                failed_stationarity=solver.get('kkt_converged') != 'true', solver=solver,
                support_sha256=hash_values([(x['cell_idx'],x['aquifer_mask']) for x in r['field']]),
                screen_sha256=hash_values([(k,r['pred'][k]['fit_mask']) for k in sorted(r['pred'])]),
                cells=len(r['field']), covered_bins=len(r['pred']),
                support_cells=sum(x['aquifer_mask'].lower()=='true' for x in r['field']),
                fit_bins=sum(float(x['fit_mask'])>0 for x in r['pred'].values()),
                expected_full_grid=len(r['field'])==600 and len(r['pred'])==1003)
        except (OSError, KeyError, ValueError) as error:
            report['runs'][name] = dict(error=str(error), execution_completed=False, failed_stationarity=True)
    for name, value in parsed.items():
        baseline = name.split('/')[0] + '/start0'
        if baseline in parsed:
            report['comparisons'][name] = compare(parsed[baseline], value)
    if 'dry/start0' in parsed and 'corrected/start0' in parsed:
        report['comparisons']['corrected_vs_dry'] = compare(parsed['dry/start0'], parsed['corrected/start0'])
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--julia', default='julia')
    parser.add_argument('--calibration', type=Path, default=ROOT/'docs/revision_evidence/corrected_validation/calibration.toml')
    parser.add_argument('--corrected-artifact', type=Path)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--cpu-start', type=int, default=8)
    parser.add_argument('--memory-gib', type=float, default=16)
    parser.add_argument('--pilot', action='store_true', help='Only dry-artifact start0 for cost/KKT pilot')
    parser.add_argument('--execute', action='store_true', help='Actually run; default prints plan only')
    args = parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a fresh directory')
    calibration=args.calibration.resolve(strict=True)
    dump=args.dump.resolve(strict=True)
    base=tomllib.loads(calibration.read_text())
    required=('enabled','kappa_strag','kappa_hard','resid_a','resid_b','resid_c','resid_d')
    if set(required)-set(base):
        parser.error('baseline calibration is missing fields')
    if args.corrected_artifact:
        args.corrected_artifact=args.corrected_artifact.resolve(strict=True)
        corrected=tomllib.loads(args.corrected_artifact.read_text())
        if corrected.get('schema_version')!=1 or corrected.get('csda')!={k:base[k] for k in required}:
            parser.error('corrected artifact must share exactly the baseline dry coefficients')
    groups=['dry'] + (['corrected'] if args.corrected_artifact and not args.pilot else [])
    scenarios=['start0'] if args.pilot else ['start0','start015','start03','scale_minus5pct','scale_plus5pct','maskInf']
    names=[f'{group}/{scenario}' for group in groups for scenario in scenarios]
    print(json.dumps(dict(output=str(output),fits=names,cpus=args.cpus,cpu_start=args.cpu_start,
                          memory_gib=args.memory_gib,execute=args.execute),indent=2),flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True)
    dry_artifact=output/'dry_correction.toml'
    text='schema_version = 1\n\n[csda]\n'
    text+='\n'.join(f'{key} = {str(base[key]).lower() if isinstance(base[key],bool) else repr(base[key])}' for key in required)
    text+='\n\n[material]\nenabled = false\nlinear = 0.0\nquadratic = 0.0\n\n[metadata]\n'
    text+=f'baseline_calibration_sha256 = "{sha(calibration)}"\n'
    dry_artifact.write_text(text)
    (output/'plan.json').write_text(json.dumps(dict(fits=names,calibration_sha256=sha(calibration),
        dry_artifact_sha256=sha(dry_artifact),corrected_artifact_sha256=sha(args.corrected_artifact) if args.corrected_artifact else None,
        physics_sha256=sha(dump)),indent=2)+'\n')
    frozen_sources=record_sources(ROOT,dump)
    frozen_sources[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    (output/'frozen_sources.json').write_text(json.dumps(frozen_sources,indent=2)+'\n')
    failed=False
    for group in groups:
        artifact=dry_artifact if group=='dry' else args.corrected_artifact
        for scenario in scenarios:
            name=f'{group}/{scenario}'
            changed=[key for key,value in frozen_sources.items()
                if sha(Path(key) if Path(key).is_absolute() else ROOT/key)!=value]
            if changed:
                raise RuntimeError(f'Source changed between suite fits: {changed}')
            folder=output/name
            command=[args.julia,'--startup-file=no','--project=.',str(ROOT/'examples/lvd_tomography.jl'),
                '--dump',str(dump),'--output-dir',str(folder/'results'),'--papermatch-fit-only',
                '--geometry','grid','--mesh-half-km','5','--surface-step-km','1','--grid-nz','6',
                '--zenith-step','10','--azimuth-step','15','--samples','32','--csda-quadrature','midpoint',
                '--rock-density','2710','--seed','20261022','--correction-artifact',str(artifact),
                '--papermatch-fit-mode','stable','--papermatch-gn-iters','100',
                '--papermatch-gradient-tol','1e-6','--papermatch-log-scale','0.3']
            if scenario.startswith('start'):
                command+=['--papermatch-initial-water',{'start0':'0','start015':'0.15','start03':'0.3'}[scenario]]
            elif scenario=='maskInf':
                command+=['--papermatch-mask-ratio','Inf']
            else:
                baseline=result(output/group/'start0')
                scale=float(next(iter(baseline['pred'].values()))['normalization'])
                command+=['--papermatch-abs-scale',repr(scale*(0.95 if 'minus' in scenario else 1.05))]
            wrapped=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(folder),
                '--dump',str(dump),'--julia',args.julia,'--cpus',str(args.cpus),'--cpu-start',str(args.cpu_start),
                '--memory-gib',str(args.memory_gib),'--input',str(artifact),'--input',str(calibration),
                '--input',str(Path(__file__).resolve()),'--',*command]
            completed=subprocess.run(wrapped,cwd=ROOT).returncode==0
            summarize(output,names)
            if not completed:
                failed=True
                print(f'Execution/provenance failure: {name}; stopping further fits',file=sys.stderr)
                break
        if failed:
            break
    report=summarize(output,names)
    # Failed stationarity remains a scientific result rather than an execution success claim.
    print(json.dumps({'summary':str(output/'summary.json'),'failed_stationarity':
        [name for name,r in report['runs'].items() if r.get('failed_stationarity')]}),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
