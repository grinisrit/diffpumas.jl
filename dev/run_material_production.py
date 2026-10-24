#!/usr/bin/env python3
"""Durable five-stage production with immutable per-stage audits and automatic exports.

Default invocation only prints the exact plan. Use --execute inside tmux after
the coordinating agent authorizes the tested source freeze. Never resumes or
overwrites an existing output. Numerical execution and stationarity are separate.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib

from record_revision_run import record_sources
from robustness_run_identity import julia_runtime

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('topo', 'main', 'detection', 'contrast', 'sweep')
REQUIRED = {
    'topo': ('lvd_topography_grid.csv',),
    'main': ('lvd_tomography_inverse.txt', 'lvd_tomography_validation.txt',
             'lvd_single_muon_angular_distribution.txt', 'lvd_solver_status.txt',
             'lvd_fit_field.csv', 'lvd_fit_predictions.csv', 'lvd_mc_common_subset_summary.csv'),
    'detection': ('lvd_detection_benchmark.txt',),
    'contrast': ('lvd_detection_contrast_sweep.txt',),
    'sweep': ('lvd_reconstruction_sweep.txt',),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def commands(args):
    common = [args.julia, '--startup-file=no', '--project=.', str(ROOT/'examples/lvd_tomography.jl'),
        '--dump', str(args.dump), '--rock-density', '2710', '--geometry', 'grid',
        '--mesh-half-km', '5', '--surface-step-km', '1', '--grid-nz', '6',
        '--zenith-max', '60', '--zenith-step', '1', '--azimuth-step', '1',
        '--primary-altitude-km', '30', '--seed', '42', '--samples', '32',
        '--csda-quadrature', 'midpoint', '--energy-min', '0.001', '--energy-max', '1e9',
        '--threshold', '100', '--calibration-mc-samples', '8192', '--inverse-mc-samples', '1024',
        '--mc-samples', '1024', '--papermatch-mc-samples', '8192',
        '--correction-artifact', str(args.artifact), '--solver-mode', 'stable',
        '--papermatch-fit-mode', 'stable', '--papermatch-initial-water', str(args.initial_water),
        '--papermatch-gn-iters', str(args.gn_iters), '--papermatch-gradient-tol', str(args.gradient_tol),
        '--papermatch-log-scale', '0.3', '--papermatch-absolute-prior-weight', str(args.absolute_prior_weight)]
    result = {}
    for stage in ORDER:
        data = args.output/'stages'/stage/'data'
        if stage == 'topo':
            command = [args.julia, '--startup-file=no', '--project=.',
                       str(ROOT/'examples/export_lvd_topography.jl'), '--output-dir', str(data)]
        else:
            command = common + ['--output-dir', str(data)]
            if stage == 'main':
                command += ['--validation-cases', '12', '--papermatch', '--papermatch-recursions', '6', '--papermatch-recur-tol', '0.02']
            else:
                command += ['--zenith-step', '2', '--azimuth-step', '4']
                command += dict(detection=['--detection-benchmark'], contrast=['--detection-contrast-sweep'],
                                sweep=['--reconstruction-sweep', '--sweep-realizations', '40', '--gd-iters', '30'])[stage]
        result[stage] = command
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--julia', required=True)
    ap.add_argument('--absolute-prior-weight', type=float, required=True)
    ap.add_argument('--initial-water', type=float, default=0.)
    ap.add_argument('--gn-iters', type=int, default=100)
    ap.add_argument('--gradient-tol', type=float, default=1e-6)
    ap.add_argument('--figure-helper', type=Path, default=ROOT/'diffmuontomoLVD/make_figures.py')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    a.output = a.output.resolve()
    a.dump = a.dump.resolve(strict=True)
    a.artifact = a.artifact.resolve(strict=True)
    a.figure_helper = a.figure_helper.resolve(strict=True)
    if not all(map(math.isfinite,(a.absolute_prior_weight,a.initial_water,a.gradient_tol))) or a.absolute_prior_weight < 0 or not 0 <= a.initial_water <= .9 or a.gn_iters <= 0 or a.gradient_tol <= 0:
        ap.error('invalid fixed fit configuration')
    if a.output.exists():
        ap.error('fresh output required; existing evidence is never overwritten')
    artifact_data = tomllib.loads(a.artifact.read_text())
    metadata = artifact_data.get('metadata', {})
    if artifact_data.get('schema_version') != 1 or artifact_data.get('material', {}).get('enabled') is not True:
        ap.error('final production requires an enabled version-1 material correction artifact')
    expected_metadata = dict(physics_dump_sha256=digest(a.dump), primary_altitude_m=30000.,
                             rock_density=2710., csda_nodes=32, energy_min=1e-3, energy_max=1e9)
    if any(metadata.get(k) != v for k,v in expected_metadata.items()):
        ap.error('correction artifact is incompatible with frozen production physics/grid energy settings')
    plan = commands(a)
    if not a.execute:
        print(json.dumps(dict(order=ORDER, commands=plan, main_cpus=list(range(16)), memory_gib=36,
            postprocessing_cpus=[20], postprocessing_memory_gib=2,
            sweep_requested_realizations=40, sweep_actual_realizations=4, cache_reuse=False), indent=2))
        return 0
    a.output.mkdir(parents=True)
    helpers = [Path(__file__).resolve(), ROOT/'dev/run_revision_job.py']
    expected = record_sources(ROOT, a.dump)
    expected.update({str(p):digest(p) for p in helpers+[a.artifact]})
    runtime = julia_runtime(a.julia)
    identity = dict(started_utc=utc(), orchestrator_pid=os.getpid(), physics_dump=str(a.dump),
        correction_artifact=str(a.artifact), correction_metadata=metadata, commands=plan, source_sha256=expected, julia_runtime=runtime,
        configuration=dict(absolute_prior_weight=a.absolute_prior_weight, initial_water=a.initial_water,
            gn_iters=a.gn_iters, gradient_tol=a.gradient_tol, sweep_actual_realizations=4,
            cache_reuse=False), scope='Source/runtime/output execution audits do not establish stationarity or calibrated inference')
    write(a.output/'execution.json', identity)
    state = dict(started_utc=utc(), current_stage=None, completed_stages=[], postprocessing={}, execution_completed=False)
    write(a.output/'status.json', state)
    failure = None
    try:
        for stage in ORDER:
            changes = [n for n,d in expected.items() if not (Path(n) if Path(n).is_absolute() else ROOT/n).is_file() or digest(Path(n) if Path(n).is_absolute() else ROOT/n) != d]
            if changes:
                raise RuntimeError('frozen input changed: '+', '.join(changes))
            state.update(current_stage=stage, stage_started_utc=utc())
            write(a.output/'status.json', state)
            folder = a.output/'stages'/stage
            wrapper = [sys.executable, str(ROOT/'dev/run_revision_job.py'), '--output', str(folder),
                '--dump', str(a.dump), '--julia', a.julia, '--cpus', '16', '--cpu-start', '0', '--memory-gib', '36',
                '--input', str(a.artifact), '--input', str(Path(__file__).resolve()), '--', *plan[stage]]
            with (a.output/(stage+'_launcher.log')).open('wb') as stream:
                returncode = subprocess.run(wrapper, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            end = json.loads((folder/'end_status.json').read_text())
            if returncode or not end['execution_completed']:
                raise RuntimeError('stage execution audit failed: '+stage)
            for name, expected_digest in end['output_sha256'].items():
                if digest(folder/name) != expected_digest:
                    raise RuntimeError('stage raw output hash changed: '+str(folder/name))
            missing = [n for n in REQUIRED[stage] if not (folder/'data'/n).is_file() or not (folder/'data'/n).stat().st_size]
            if missing:
                raise RuntimeError(f'{stage} missing required outputs: {missing}')
            if stage == 'sweep' and '4 realizations/point;' not in (folder/'data/lvd_reconstruction_sweep.txt').read_text():
                raise RuntimeError('sweep actual realization count differs from frozen plan')
            state['completed_stages'].append(stage)
            if stage == 'main':
                solver = dict(line.split('=',1) for line in (folder/'data/lvd_solver_status.txt').read_text().splitlines() if '=' in line)
                state['main_solver_status'] = solver
                state['main_kkt_converged'] = solver.get('kkt_converged') == 'true'
            write(a.output/'status.json', state)
            post = [sys.executable, str(ROOT/'dev/run_bounded.py'), '--cpus', '1', '--memory-gib', '2',
                '--', 'taskset', '-c', '20', sys.executable, str(ROOT/'dev/postprocess_material_production.py'),
                '--root', str(a.output), '--stage', stage, '--figure-helper', str(a.figure_helper)]
            with (a.output/(stage+'_postprocess.log')).open('wb') as stream:
                code = subprocess.run(post, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            state['postprocessing'][stage] = dict(returncode=code, completed_utc=utc())
            write(a.output/'status.json', state)
        state['execution_completed'] = True
    except Exception as error:
        failure = str(error)
    finally:
        changes = [n for n,d in expected.items() if not (Path(n) if Path(n).is_absolute() else ROOT/n).is_file() or digest(Path(n) if Path(n).is_absolute() else ROOT/n) != d]
        runtime_same = julia_runtime(a.julia) == runtime
        state.update(ended_utc=utc(), failure=failure, source_unchanged=not changes,
            source_changes=changes, runtime_unchanged=runtime_same,
            execution_completed=state['execution_completed'] and not changes and runtime_same,
            candidate_exports_complete=len(state['postprocessing'])==5 and all(v['returncode']==0 for v in state['postprocessing'].values()))
        write(a.output/'status.json', state)
        write(a.output/'end_status.json', state)
    print(json.dumps(state), flush=True)
    return 0 if state['execution_completed'] and state['candidate_exports_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
