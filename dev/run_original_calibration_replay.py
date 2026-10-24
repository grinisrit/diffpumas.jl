#!/usr/bin/env python3
"""Wait for eight-replicate completion, verify unchanged MC, replay one old ray."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from robustness_run_identity import julia_runtime

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('fresh output root required')
    out.mkdir(parents=True);job=args.ensemble_job.resolve()
    while not (job/'end_status.json').exists():time.sleep(30)
    end=json.loads((job/'end_status.json').read_text())
    if end.get('returncode')!=0 or not all(end.get(k)is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):raise SystemExit('ensemble did not end cleanly')
    original=ROOT/'docs/revision_evidence/surrogate_v2/production/execution.json'
    original_end=original.with_name('end_status.json');reference_end=json.loads(original_end.read_text())
    if reference_end.get('returncode')!=0 or not all(reference_end.get(k)is True for k in ('execution_completed','source_unchanged','runtime_unchanged')):raise SystemExit('original calibration did not end cleanly')
    for filename in ('training.csv','frozen_corrections.toml'):
        if sha(original.with_name(filename))!=reference_end['output_sha256'][filename]:raise SystemExit('original saved calibration output changed: '+filename)
    reference=json.loads(original.read_text());commit=reference['git_commit'];bindings={}
    for name in ('MixedTransport','Materials','Context','GaisserFlux','Transport','Constants','DEDXLoader','AtomicShellData','Coulomb'):
        relative='src/'+name+'.jl';path=ROOT/relative;expected=reference['source_sha256'][relative]
        if sha(path)!=expected:raise SystemExit('original MC dependency changed: '+relative)
        bindings[str(path)]=expected
    def unchanged_region(relative,anchor,stop=None):
        old=subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT).decode();new=(ROOT/relative).read_text()
        def region(s):
            start=s.index(anchor);finish=s.find(stop,start+len(anchor)) if stop else len(s)
            return s[start:finish if finish>=0 else len(s)]
        a,b=region(old),region(new)
        if a!=b:raise SystemExit('MC/sampling source region changed: '+relative)
        return hashlib.sha256(a.encode()).hexdigest()
    region_hashes={'Tomography_MC_tail':unchanged_region('src/Tomography.jl','# Monte Carlo cross-check'),'energy_sampler':unchanged_region('src/Tomography.jl','function sample_energy_set(', '\nfunction ')}
    proof=dict(original_execution_sha256=sha(original),original_end_sha256=sha(original_end),original_commit=commit,unchanged_dependencies=bindings,unchanged_regions_sha256=region_hashes,scope='Physics.jl dual overloads and CSDA derivative methods changed since original; MC tail, energy sampling and transport dependencies verified unchanged. Numeric old-ray replay separately gates exact MC primal parity.')
    proofpath=out/'mc_source_gate.json';proofpath.write_text(json.dumps(proof,indent=2)+'\n')
    julia=reference['julia_runtime']['binary'];helper=ROOT/'dev/replay_original_calibration_ray.jl';dump=ROOT/'examples/data/revision_physics_v2_final_20261023/materials.pumas';artifact=ROOT/'docs/revision_evidence/surrogate_v2/production/frozen_corrections.toml';training=ROOT/'docs/revision_evidence/surrogate_v2/production/training.csv'
    if julia_runtime(julia)!=reference['julia_runtime']:raise SystemExit('original Julia runtime identity changed')
    command=[sys.executable,str(ROOT/'dev/run_revision_job.py'),'--output',str(out/'job'),'--dump',str(dump),'--julia',julia,'--cpus','4','--cpu-start','16','--memory-gib','8']
    for p in [Path(__file__).resolve(),helper,proofpath,original,original_end,artifact,training,ROOT/'dev/validate_surrogate_v2.jl',ROOT/'examples/revision_suite.jl',job/'end_status.json']:command+=['--input',str(p)]
    command+=['--',julia,'--startup-file=no','--project=.',str(helper),'--dump',str(dump),'--artifact',str(artifact),'--training',str(training),'--output',str(out/'job/data')]
    (out/'plan.json').write_text(json.dumps({'command':command,'scope':'Original-ray verification only, not a ninth ensemble replicate'},indent=2)+'\n')
    if not args.prepare_only:raise SystemExit(subprocess.call(command,cwd=ROOT))
if __name__=='__main__':main()
