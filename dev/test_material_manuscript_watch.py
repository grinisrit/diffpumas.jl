"""Explicit unit fixtures only; no production evidence or numerical results are fabricated."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_watch',HERE/'watch_material_manuscript.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
spec2=importlib.util.spec_from_file_location('stage_post',HERE/'postprocess_material_production.py')
stage_post=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(stage_post)


class CandidateWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='candidate-watcher-unit-',dir='/mnt/big_data/agent-builds')
        self.base=Path(self.tmp.name);self.run=self.base/'unit_run'
        (self.run/'stages/main').mkdir(parents=True)
        (self.run/'candidate_exports/main_retry').mkdir(parents=True)
        (self.run/'status.json').write_text(json.dumps({'completed_stages':['main'],'unit_fixture':True}))
        (self.run/'execution.json').write_text('{"unit_fixture":true}')
        (self.run/'stages/main/end_status.json').write_text('{"unit_fixture":true}')
        (self.run/'candidate_exports/main_retry/export_provenance.json').write_text('{"unit_fixture":true}')
        (self.run/'candidate_exports/index.json').write_text(json.dumps({'schema_version':1,'stages':{'main':{'path':'main_retry'}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_retry_pointer_and_auxiliary_gate(self):
        snapshot=w.evidence_snapshot(self.run,None)
        self.assertEqual(snapshot['stages']['main']['export'],str((self.run/'candidate_exports/main_retry').resolve()))
        self.assertFalse(w.assembly_ready(snapshot))
        snapshot['stages']={stage:{} for stage in w.SCIENCE}
        snapshot['auxiliary']={'column_domain':{'execution_clean':True},'final_field_quadrature':{'execution_clean':True}}
        self.assertFalse(w.assembly_ready(snapshot))
        snapshot['auxiliary']['final_field_quadrature']['derived_provenance_sha256']='UNIT'
        self.assertTrue(w.assembly_ready(snapshot))

    def test_export_index_writer_migrates_flat_schema(self):
        index=self.run/'candidate_exports/index.json'
        index.write_text(json.dumps({'topo':{'path':'topo','unit_fixture':True}}))
        target=self.run/'candidate_exports/main_retry'
        stage_post.publish_index(self.run,target,'main','UNIT_END')
        data=json.loads(index.read_text())
        self.assertEqual(data['schema_version'],1)
        self.assertEqual(data['stages']['main']['path'],'main_retry')
        self.assertEqual(data['stages']['main']['stage_end_sha256'],'UNIT_END')
        self.assertEqual(data['stages']['topo']['path'],'topo')

    def test_candidate_tamper_rejected(self):
        folder=self.base/'candidate';folder.mkdir();file=folder/'unit.tex';file.write_text('UNIT FIXTURE')
        (folder/'integration_manifest.json').write_text(json.dumps({'outputs_sha256':{'unit.tex':w.sha(file)}}))
        w.verify_candidate(folder);file.write_text('CHANGED UNIT FIXTURE')
        with self.assertRaises(ValueError):w.verify_candidate(folder)

    def test_one_attempt_per_signature_and_helper_fix(self):
        builder=self.base/'unit_paper/revision/unit_builder.py';builder.parent.mkdir(parents=True)
        source='''import argparse,json,hashlib,os,resource
from pathlib import Path
REJECT=True
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--root');a,_=p.parse_known_args()
if REJECT:print("UNIT REJECTION BEFORE OUTPUT");raise SystemExit(2)
out=Path(a.output);out.mkdir(parents=True)
file=out/'unit_fragment.tex';file.write_text('UNIT FIXTURE: no numerical paper results')
proof=out/'resource.json';proof.write_text(json.dumps(dict(cpus=sorted(os.sched_getaffinity(0)),address_space=resource.getrlimit(resource.RLIMIT_AS)[0])))
manifest=dict(unit_fixture=True,stage_bindings={},outputs_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [file,proof]})
(out/'integration_manifest.json').write_text(json.dumps(manifest))
'''
        builder.write_text(source)
        output=self.base/'watcher'
        command=[sys.executable,str(HERE/'watch_material_manuscript.py'),'--run-root',str(self.run),
                 '--output',str(output),'--builder',str(builder),'--postprocess-lock',str(self.base/'unit_postprocess.lock'),'--once']
        subprocess.run(command,check=True,capture_output=True)
        state=json.loads((output/'watcher_state.json').read_text())
        self.assertEqual(state['phase'],'rejected_candidate');self.assertEqual(len(state['attempted_signatures']),1)
        subprocess.run(command,check=True,capture_output=True)
        self.assertEqual(len((output/'events.jsonl').read_text().splitlines()),1)
        builder.write_text(source.replace('REJECT=True','REJECT=False'))
        subprocess.run(command,check=True,capture_output=True)
        state=json.loads((output/'watcher_state.json').read_text())
        self.assertEqual(state['phase'],'incomplete_candidate');self.assertEqual(len(state['attempted_signatures']),2)
        self.assertEqual(state['attempt_counter'],2)
        self.assertIn('UNIT REJECTION',(output/'attempt_0001.log').read_text())
        self.assertTrue((output/'attempt_0002.log').exists())
        latest=json.loads((output/'latest_candidate.json').read_text());self.assertFalse(latest['revision_complete'])
        self.assertTrue(latest['review_required'])
        proof=json.loads((Path(latest['output'])/'resource.json').read_text())
        self.assertEqual(proof,dict(cpus=[20],address_space=2*1024**3))

    def test_interrupted_resume_is_explicit_and_retries_fresh(self):
        builder=self.base/'unit_paper/revision/reject_builder.py';builder.parent.mkdir(parents=True)
        builder.write_text('print("UNIT INTERRUPTED ATTEMPT FIXTURE")\nraise SystemExit(2)\n')
        output=self.base/'interrupted_watcher'
        command=[sys.executable,str(HERE/'watch_material_manuscript.py'),'--run-root',str(self.run),
            '--output',str(output),'--builder',str(builder),'--postprocess-lock',str(self.base/'unit_lock'),'--once']
        subprocess.run(command,check=True,capture_output=True)
        state=json.loads((output/'watcher_state.json').read_text())
        state['phase']='rendering_candidate'  # Simulated prior controller interruption, not scientific evidence.
        (output/'watcher_state.json').write_text(json.dumps(state))
        subprocess.run(command,check=True,capture_output=True)
        state=json.loads((output/'watcher_state.json').read_text())
        self.assertEqual(state['phase'],'rejected_candidate')
        self.assertEqual(len(state['interrupted_attempts']),1)
        self.assertEqual(state['attempt_counter'],2)
        self.assertTrue((output/'attempt_0001.log').exists())
        self.assertTrue((output/'attempt_0002.log').exists())
        self.assertIn('interrupted_attempt',(output/'events.jsonl').read_text())

    def test_real_minimal_compile_provenance(self):
        compiler=Path('/mnt/big_data/agent-builds/diffpumas-latex/tectonic')
        if not compiler.is_file():self.skipTest('configured Tectonic unavailable')
        folder=self.base/'unit_compile';folder.mkdir()
        tex=folder/'candidate_main.tex'
        tex.write_text('\\documentclass{article}\n\\begin{document}Unit fixture only.\\end{document}\n')
        (folder/'integration_manifest.json').write_text(json.dumps({'unit_fixture':True,'outputs_sha256':{tex.name:w.sha(tex)}}))
        result=w.compile_candidate(folder,compiler,())
        self.assertIn(result['status'],('compiled','compiled_with_warnings'))
        self.assertTrue((folder/'candidate_main.pdf').read_bytes().startswith(b'%PDF-'))
        self.assertTrue(result['pdf_sha256']);self.assertTrue(result['provenance_sha256'])


if __name__=='__main__':unittest.main()
