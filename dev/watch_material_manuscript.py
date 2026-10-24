#!/usr/bin/env python3
"""Refresh isolated manuscript candidates as immutable audited evidence arrives.

No scientific computation or authoritative paper overwrite. One attempt per
distinct input/helper signature; rejected signatures wait for a meaningful
change. CPU20 Python work shares the stage-export lock and a 2-GiB limit.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCIENCE=('main','detection','contrast','sweep')
PYTHON_LOCK=Path('/mnt/big_data/agent-builds/surrogate-stability-20261024/cpu20-python-postprocess.lock')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,data):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_hashes(builder,dependencies=(),compiler=None):
    paper=builder.parent.parent
    paths=[builder,builder.with_name('stage_result_parsers.py'),builder.with_name('build_surrogate_stability_report.py'),
        paper/'main.tex',paper/'references.bib',paper/'elsarticle.cls',paper/'elsarticle-num.bst']
    paths += [paper/'revision'/name for name in ('surrogate_stability_v2_methods.tex','surrogate_v2_results.tex',
                                              'validation_results.tex','ml_surrogate_comparison.tex',
                                              'stability_v2_progress.tex','stability_v2_progress.json')]
    paths += [Path(p) for p in dependencies]
    if compiler:paths.append(compiler)
    return {str(p.resolve()):sha(p) for p in sorted(set(paths)) if p.is_file()}


def evidence_snapshot(run,controls):
    """Inspect only small completed-record manifests while waiting, not large raw files."""
    state=json.loads((run/'status.json').read_text())
    indexpath=run/'candidate_exports/index.json'
    index=json.loads(indexpath.read_text()) if indexpath.exists() else {}
    entries=index.get('stages',index) if isinstance(index,dict) else {}
    stages={}
    for stage in SCIENCE:
        if stage not in state.get('completed_stages',[]):continue
        entry=entries.get(stage,{})
        export=run/'candidate_exports'/entry.get('path',stage)
        export=export.resolve()
        if not export.is_relative_to((run/'candidate_exports').resolve()):
            raise ValueError('export pointer escapes owned candidate directory')
        end=run/'stages'/stage/'end_status.json';provenance=export/'export_provenance.json'
        if end.is_file() and provenance.is_file():
            stages[stage]=dict(end_status_sha256=sha(end),export=str(export),export_provenance_sha256=sha(provenance))
    aux={}
    for name in ('column_domain','final_field_quadrature'):
        folder=run/'controls'/name;end=folder/'end_status.json';derived=folder/'derived_provenance.json'
        if end.is_file():
            data=json.loads(end.read_text())
            clean=all(data.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')) and data.get('returncode')==0
            aux[name]=dict(path=str(folder),end_status_sha256=sha(end),execution_clean=clean)
            if derived.is_file():aux[name]['derived_provenance_sha256']=sha(derived)
        else:aux[name]=dict(path=str(folder),execution_clean=False,status='pending')
    collector=None
    if controls and controls.is_dir():
        # A writer's incomplete final JSON is ignored until a later poll; this
        # never creates an invented or partially parsed control result.
        for file in sorted(controls.glob('collector_*.json'),reverse=True):
            try:data=json.loads(file.read_text())
            except (OSError,json.JSONDecodeError):continue
            if isinstance(data.get('runs'),dict):
                collector=dict(path=str(file.resolve()),sha256=sha(file),reported_cases=len(data['runs']),
                    reported_clean_executions=sum(all(row.get(k) is True for k in ('execution_completed','source_unchanged','runtime_unchanged')) for row in data['runs'].values()))
                break
    return dict(stages=stages,auxiliary=aux,collector=collector,
        root_execution_sha256=sha(run/'execution.json'),
        export_index_sha256=sha(indexpath) if indexpath.exists() else None)


def assembly_ready(snapshot):
    aux=snapshot['auxiliary']
    return (set(snapshot['stages'])==set(SCIENCE) and aux['column_domain']['execution_clean']
        and aux['final_field_quadrature']['execution_clean']
        and 'derived_provenance_sha256' in aux['final_field_quadrature'])


def verify_candidate(folder):
    manifest=json.loads((folder/'integration_manifest.json').read_text())
    for name,expected in manifest['outputs_sha256'].items():
        path=folder/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or sha(path)!=expected:
            raise ValueError('candidate output changed or unsafe: '+name)
    return manifest


def compile_candidate(folder, compiler, locks):
    command=[str(compiler),'-X','compile','candidate_main.tex','--keep-logs']
    before=sha(folder/'integration_manifest.json');binary=sha(compiler)
    with (folder/'tectonic_stdout.log').open('wb') as stream:
        child=subprocess.Popen(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,pass_fds=locks)
        code=child.wait()
    status=dict(command=command,cwd=str(folder),returncode=code,compiler_sha256=binary,
        builder_manifest_sha256=before,review_required=True)
    try:
        verify_candidate(folder)
        if sha(compiler)!=binary or sha(folder/'integration_manifest.json')!=before:
            raise ValueError('compiler or builder manifest changed during compilation')
        pdf=folder/'candidate_main.pdf'
        success=code==0 and pdf.is_file() and pdf.stat().st_size>0
        log=folder/'candidate_main.log'
        text=log.read_text(errors='replace') if log.exists() else ''
        warnings={name:text.count(name) for name in ('Float too large','Overfull \\hbox','undefined references','undefined citations')}
        status.update(status=('compiled_with_warnings' if any(warnings.values()) else 'compiled') if success else 'failed',
            diagnostic_counts=warnings,pdf_sha256=sha(pdf) if pdf.is_file() else None,
            log_sha256=sha(log) if log.is_file() else None,stdout_sha256=sha(folder/'tectonic_stdout.log'))
    except Exception as error:status.update(status='failed',error=str(error))
    write(folder/'compile_provenance.json',status)
    status['provenance_sha256']=sha(folder/'compile_provenance.json')
    return status


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--controls-root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--builder',type=Path,default=ROOT/'diffmuontomoLVD/revision/build_stage_manuscript.py')
    ap.add_argument('--tectonic',type=Path,help='optional compiler for assembled candidates only')
    ap.add_argument('--postprocess-lock',type=Path,default=PYTHON_LOCK,help='shared resource lock; override only for isolated fixtures')
    ap.add_argument('--poll-seconds',type=float,default=30.)
    ap.add_argument('--expected-stability-cases',type=int,default=12)
    ap.add_argument('--once',action='store_true',help='one observation/attempt for bounded fixture checks')
    args=ap.parse_args()
    if args.poll_seconds<1:ap.error('poll interval must be at least one second')
    run=args.run_root.resolve(strict=True);builder=args.builder.resolve(strict=True);out=args.output.resolve()
    controls=args.controls_root.resolve() if args.controls_root else None
    compiler=args.tectonic.resolve(strict=True) if args.tectonic else None
    post_lock_path=args.postprocess_lock.resolve()
    out.mkdir(parents=True,exist_ok=True);(out/'revisions').mkdir(exist_ok=True)
    available_affinity=set(os.sched_getaffinity(0))
    if 20 not in available_affinity:ap.error('CPU20 unavailable')
    os.sched_setaffinity(0,{20});resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    own_hash=sha(Path(__file__).resolve())
    with (out/'watcher.lock').open('a+') as ownership:
        try:fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:ap.error('another watcher or its rendering child owns this output')
        statefile=out/'watcher_state.json'
        state=json.loads(statefile.read_text()) if statefile.exists() else dict(attempted_signatures=[],retained_dependencies=[])
        if state.get('run_root',str(run))!=str(run):ap.error('watcher root identity differs')
        state.update(run_root=str(run),controls_root=str(controls) if controls else None,
            started_or_resumed_utc=utc(),watcher_sha256=own_hash,review_required=True,revision_complete=False,
            cpu_pool=[20],address_space_bytes=2*1024**3,python_postprocess_lock=str(post_lock_path))
        def save(**changes):state.update(updated_utc=utc(),**changes);write(statefile,state)
        def event(**data):
            with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),**data),allow_nan=False)+'\n')
        if state.get('phase') in ('rendering_candidate','compiling_candidate'):
            # Ownership is inherited by live children. Reaching this point
            # proves any orphaned builder/compiler ended; its unverified files
            # remain preserved and the same inputs may get one fresh attempt.
            interrupted=dict(signature=state.get('signature'),output=state.get('output'),
                command=state.get('command'),observed_utc=utc(),status='interrupted_attempt')
            state.setdefault('interrupted_attempts',[]).append(interrupted)
            state['attempted_signatures']=[s for s in state['attempted_signatures'] if s!=state.get('signature')]
            event(kind='interrupted_attempt',**interrupted)
        save(phase='waiting_evidence')
        while True:
            if sha(Path(__file__).resolve())!=own_hash:
                save(phase='watcher_source_changed',error='restart required to load changed watcher code');return 1
            try:
                snapshot=evidence_snapshot(run,controls)
                helpers=source_hashes(builder,state.get('retained_dependencies',[]),compiler)
                signature=hashlib.sha256(json.dumps(dict(evidence=snapshot,helpers=helpers,watcher_sha256=own_hash),sort_keys=True).encode()).hexdigest()
                if not snapshot['stages']:
                    save(phase='waiting_scientific_exports',evidence=snapshot)
                elif signature not in state['attempted_signatures']:
                    save(phase='waiting_python_postprocess_lock',evidence=snapshot)
                    post_lock_path.parent.mkdir(parents=True,exist_ok=True)
                    with post_lock_path.open('a+') as postlock:
                        fcntl.flock(postlock,fcntl.LOCK_EX)
                        # Evidence/helpers may change while another export owns
                        # the lock. Re-observe before committing an attempt.
                        snapshot=evidence_snapshot(run,controls);helpers=source_hashes(builder,state.get('retained_dependencies',[]),compiler)
                        signature=hashlib.sha256(json.dumps(dict(evidence=snapshot,helpers=helpers,watcher_sha256=own_hash),sort_keys=True).encode()).hexdigest()
                        if signature in state['attempted_signatures']:
                            if args.once:return 0
                            continue
                        previous=[int(p.name.split('_',1)[0]) for p in (out/'revisions').iterdir() if p.is_dir() and p.name.split('_',1)[0].isdigit()]
                        previous += [int(p.stem.split('_')[1]) for p in out.glob('attempt_*.log') if p.stem.split('_')[1].isdigit()]
                        number=1+max(previous+[state.get('attempt_counter',0)])
                        state['attempt_counter']=number
                        destination=out/'revisions'/f'{number:04d}_{signature[:12]}'
                        assemble=assembly_ready(snapshot)
                        # Both builder and optional compiler inherit this
                        # watcher's CPU20 affinity and fixed 2-GiB limit.
                        command=[sys.executable,str(builder),'--root',str(run),'--output',str(destination)]
                        if assemble:command+=['--assemble']
                        if snapshot['collector']:command+=['--stability-report',snapshot['collector']['path'],'--expected-stability-cases',str(args.expected_stability_cases)]
                        state['attempted_signatures'].append(signature)
                        save(phase='rendering_candidate',signature=signature,command=command,output=str(destination))
                        log=out/f'attempt_{number:04d}.log'
                        with log.open('wb') as stream:
                            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                pass_fds=(ownership.fileno(),postlock.fileno()))
                            save(rendering_pid=child.pid);code=child.wait()
                        changed=[p for p,d in helpers.items() if not Path(p).is_file() or sha(Path(p))!=d]
                        attempt=dict(signature=signature,command=command,returncode=code,helpers_sha256=helpers,
                                     helper_changes=changed,evidence=snapshot,output=str(destination),cpu_pool=[20],address_space_bytes=2*1024**3)
                        if code or changed:
                            attempt['status']='rejected_candidate'
                            save(phase='rejected_candidate',last_attempt=attempt)
                        else:
                            try:manifest=verify_candidate(destination)
                            except Exception as error:
                                attempt.update(status='rejected_candidate',error=str(error))
                                save(phase='rejected_candidate',last_attempt=attempt)
                            else:
                                attempt['status']='assembled_candidate' if assemble else 'incomplete_candidate'
                                attempt['integration_manifest_sha256']=sha(destination/'integration_manifest.json')
                                if assemble and compiler:
                                    save(phase='compiling_candidate',signature=signature,output=str(destination),command=command)
                                    attempt['compile']=compile_candidate(destination,compiler,(ownership.fileno(),postlock.fileno()))
                                state['retained_dependencies']=list(manifest.get('stage_bindings',{}).get('retained_sources_sha256',{}))
                                write(out/'latest_candidate.json',dict(**attempt,review_required=True,revision_complete=False))
                                save(phase=attempt['status'],last_attempt=attempt)
                        event(kind='candidate_attempt',**attempt)
                        if destination.exists():write(destination/'watcher_attempt.json',attempt)
                        controls_done=(snapshot['collector'] is not None and
                            snapshot['collector']['reported_cases']==args.expected_stability_cases and
                            snapshot['collector']['reported_clean_executions']==args.expected_stability_cases)
                        compile_ok=not compiler or attempt.get('compile',{}).get('status') in ('compiled','compiled_with_warnings')
                        if attempt['status']=='assembled_candidate' and compiler and not compile_ok:
                            save(phase='assembled_candidate_compile_failed',last_attempt=attempt)
                        if attempt['status']=='assembled_candidate' and controls_done and compile_ok:
                            save(phase='assembled_candidate_awaiting_review',ended_utc=utc(),watcher_finished=True,
                                 stop_reason='All scheduled stage/auxiliary/control evidence arrived; final scientific review remains required')
                            return 0
                else:
                    save(evidence=snapshot)
            except Exception as error:
                # Invalid or partially written evidence is never accepted. A
                # poll error updates one status, not a new candidate directory.
                save(phase='evidence_unavailable',error=str(error))
            if args.once:return 0
            time.sleep(args.poll_seconds)


if __name__=='__main__':raise SystemExit(main())
