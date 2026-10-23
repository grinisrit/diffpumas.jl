#!/usr/bin/env python3
"""One controlled no-air convention fit; no changes to the active pipelines."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/home/roland/devspace/diffpumas.jl')
sys.path.insert(0,str(ROOT/'dev'))
import robustness_run_identity as ident
import record_revision_run as recorder
from continue_revision_robustness import observe_child
parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
baseline_dir=ROOT/'examples/data/revision_robustness_corrected_20261023/baseline'
output=ROOT/'examples/data/revision_boundary_fit_20261023/no_air'
record=ident.read_record(baseline_dir);base=record['identity'];binary=base['runtime']['binary']
command=list(base['command']);command[0]=binary
assert command.count('--output-dir')==1 and '--primary-altitude-km' not in command
command[command.index('--output-dir')+1]=str(output)
command+=['--primary-altitude-km','0']
# Removing only output and the explicit altitude control must restore every
# baseline numerical argument, including the baseline seed and MC budgets.
assert command[1:-2]==[str(output) if x==str(baseline_dir) else x for x in base['command'][1:]]
bounded=[sys.executable,str(ROOT/'dev/run_bounded.py'),'--cpus','1','--memory-gib','2','--','taskset','-c','20']+command
plan=dict(command=command,bounded_command=bounded,output=str(output),cpus=1,address_space_gib=2,
          requested_affinity=[20],baseline_identity_sha256=record['sha256'],baseline_identity=base,
          allowed_changes=['primary_altitude_km:30->0','output_directory','command[0]','cpus:4->1','address_space_gib:12->2'],
          scope='No-air convention control, not an altitude-corrected physical truth. Existing valid upward rays have positive exit height; max(0, primary_altitude-surface_exit_z) is zero at primary_altitude=0.')
if args.dry_run:
    print(json.dumps(plan,indent=2));raise SystemExit(0)
if output.exists():raise SystemExit('Existing output refused; preserve any failed attempt')
original_affinity=os.sched_getaffinity(0)
os.sched_setaffinity(0,{20}) # Runtime-identification work uses the spare slot too.
runtime=ident.julia_runtime(binary)
assert runtime==base['runtime'],'Actual Julia runtime changed'
assert ident.build_identity(ROOT,Path(base['dump']),base['command'],base['cpus'],base['address_space_gib'],runtime=runtime)==base,'Baseline source/dump/runtime/environment changed'
original_execution=json.loads((baseline_dir.parent/'execution.json').read_text())
for name,sha in original_execution['expected_source_sha256'].items():
    p=Path(name) if Path(name).is_absolute() else ROOT/name
    assert ident.digest_file(p)==sha,'Original pipeline source/wrapper changed: '+name
new_identity=ident.build_identity(ROOT,Path(base['dump']),command,1,2,runtime=runtime)
ident.prepare(output,new_identity)
manifest=recorder.build_record(ROOT,Path(base['dump']),bounded,binary)
manifest_path=output/'run_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
expected=dict(manifest['sha256'])
for p in [Path(__file__).resolve(),ROOT/'dev/continue_revision_robustness.py']:
    expected[str(p)]=ident.digest_file(p)
plan.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime=runtime,
            expected_source_sha256=expected,source_manifest_sha256=ident.digest_file(manifest_path))
(output/'execution.json').write_text(json.dumps(plan,indent=2)+'\n')
status=dict(returncode=None,execution_completed=False)
try:
    os.sched_setaffinity(0,original_affinity) # Bounded helper must see the host quota before inner taskset20.
    with (output/'run.log').open('x') as stream:
        child=subprocess.Popen(bounded,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        observation=observe_child(child,binary,[20])
        (output/'runtime_observation.json').write_text(json.dumps(observation,indent=2)+'\n')
        status['returncode']=child.wait()
    if status['returncode']==0 and observation.get('affinity_matches') is True:
        os.sched_setaffinity(0,{20})
        ident.complete(output)
except Exception as error:
    status['exception']=repr(error)
finally:
    os.sched_setaffinity(0,{20})
    changes=[]
    for name,sha in expected.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if not p.is_file() or ident.digest_file(p)!=sha:changes.append(name)
    runtime_unchanged=ident.digest_file(Path(binary))==runtime['sha256']
    status.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_unchanged=not changes,
                  source_changes=changes,runtime_unchanged=runtime_unchanged,
                  actual_affinity_record=json.loads((output/'runtime_observation.json').read_text()) if (output/'runtime_observation.json').exists() else None,
                  fit_status=(output/'lvd_recursion_status.txt').read_text() if (output/'lvd_recursion_status.txt').exists() else None,
                  output_sha256={p.name:ident.digest_file(p) for p in output.iterdir() if p.is_file()},
                  failed_attempt_preserved=True)
    status['execution_completed']=status['returncode']==0 and not changes and runtime_unchanged and (output/'completed.json').exists()
    (output/'end_status.json').write_text(json.dumps(status,indent=2)+'\n')
    os.sched_setaffinity(0,original_affinity)
raise SystemExit(0 if status['execution_completed'] else 1)
