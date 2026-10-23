#!/usr/bin/env python3
"""Bounded deterministic final-field probe; leaves active runs unchanged."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
REPO=Path('/home/roland/devspace/diffpumas.jl');sys.path.insert(0,str(REPO/'dev'))
import record_revision_run as recorder
import robustness_run_identity as identity
from continue_revision_robustness import observe_child
p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args()
script=Path(__file__).with_name('run.jl').resolve()
output=REPO/'examples/data/revision_quadrature_final_field_20261023'
main=REPO/'examples/data/lvd_revision_corrected_20261023'
execution=json.loads((main/'execution.json').read_text())
source_manifest=json.loads((main/execution['source_manifest']).read_text())
binary=source_manifest['julia_runtime']['binary'];dump=Path(source_manifest['physics_dump']['path'])
command=[binary,'--startup-file=no','--project=.',str(script),str(output/'data')]
bounded=[sys.executable,str(REPO/'dev/run_bounded.py'),'--cpus','1','--memory-gib','2','--','taskset','-c','20']+command
plan=dict(command=command,bounded_command=bounded,output=str(output),cpus=1,address_space_gib=2,requested_affinity=[20],
          scope='Deterministic convergence on production geometry and final field;512nodes comparator not truth; noMC uncertainty; no-free-parameter rays have flux only.')
if args.dry_run:print(json.dumps(plan,indent=2));raise SystemExit(0)
if output.exists():raise SystemExit('Refuse existing output; preserve every attempt')
for name,digest in execution['expected_source_sha256'].items():
 f=Path(name) if Path(name).is_absolute() else REPO/name
 assert f.is_file() and identity.digest_file(f)==digest,'Main frozen source changed: '+name
original=os.sched_getaffinity(0);os.sched_setaffinity(0,{20})
runtime=identity.julia_runtime(binary);assert runtime==source_manifest['julia_runtime'],'Main actualruntime changed'
manifest=recorder.build_record(REPO,dump,bounded,binary)
extras=[script,Path(__file__).resolve(),REPO/'dev/energy_quadrature_probe.jl',REPO/'examples/revision_suite.jl',
        REPO/'dev/continue_revision_robustness.py',main/'lvd_fit_field.csv',main/'lvd_recursion_status.txt',
        main/'lvd_recursion_history.csv',main/'lvd_mc_calibration_de8834f1c7261ba2.toml',main/'execution.json']
expected=dict(manifest['sha256']);expected.update({str(f):identity.digest_file(f) for f in extras})
output.mkdir(parents=True)
(output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime=runtime,expected_source_sha256=expected,
            run_manifest_sha256=identity.digest_file(output/'run_manifest.json'))
(output/'execution.json').write_text(json.dumps(plan,indent=2)+'\n')
status=dict(returncode=None,execution_completed=False)
try:
 os.sched_setaffinity(0,original)
 with (output/'run.log').open('x') as stream:
  child=subprocess.Popen(bounded,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
  observation=observe_child(child,binary,[20]);(output/'runtime_observation.json').write_text(json.dumps(observation,indent=2)+'\n')
  status['returncode']=child.wait()
except Exception as error:status['exception']=repr(error)
finally:
 os.sched_setaffinity(0,{20})
 changed=[]
 for name,digest in expected.items():
  f=Path(name) if Path(name).is_absolute() else REPO/name
  if not f.is_file() or identity.digest_file(f)!=digest:changed.append(name)
 observed=json.loads((output/'runtime_observation.json').read_text()) if (output/'runtime_observation.json').exists() else {}
 status.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_unchanged=not changed,source_changes=changed,
   runtime_unchanged=identity.digest_file(Path(binary))==runtime['sha256'],actual_affinity_record=observed,
   outputs_sha256={str(f.relative_to(output)):identity.digest_file(f) for f in output.rglob('*') if f.is_file()})
 status['execution_completed']=(status['returncode']==0 and not changed and status['runtime_unchanged'] and
    observed.get('affinity_matches') is True and (output/'data/status.toml').is_file())
 (output/'end_status.json').write_text(json.dumps(status,indent=2)+'\n');os.sched_setaffinity(0,original)
raise SystemExit(0 if status['execution_completed'] else 1)
