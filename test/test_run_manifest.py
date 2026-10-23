"""Run-manifest source coverage and selected-runtime CLI smoke."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dev'))
spec=importlib.util.spec_from_file_location('record_run',ROOT/'dev/record_revision_run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ManifestTest(unittest.TestCase):
    def test_transitive_sources_and_missing_dump(self):
        with tempfile.TemporaryDirectory() as name:
            dump=Path(name)/'missing.pumas'
            hashes=m.record_sources(ROOT,dump)
            for source in ('examples/lvd_muography.jl','examples/export_lvd_topography.jl',
                           'dev/record_revision_run.py','dev/robustness_run_identity.py'):
                self.assertEqual(hashes[source],hashlib.sha256((ROOT/source).read_bytes()).hexdigest())
            dump.write_bytes(b'physics fixture')
            self.assertEqual(m.record_sources(ROOT,dump)[str(dump)],hashlib.sha256(dump.read_bytes()).hexdigest())

    def test_selected_runtime_and_nonoverwriting_manifests(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            # Executable query fixture speaks the documented two-line Julia
            # identity protocol, so this test requires no Julia installation.
            executable=directory/'selected-julia'
            executable.write_text('#!/bin/sh\nprintf "fixture-1.10\\n%s\\n" "$0"\n')
            executable.chmod(0o755)
            output=directory/'out';dump=directory/'missing.pumas'
            command=[sys.executable,str(ROOT/'dev/record_revision_run.py'),'--julia',str(executable),
                     str(output),str(dump),'bash','dev/run_lvd_paper.sh','--stages','topo']
            subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
            first=next(output.glob('run_manifest_*.json'))
            before=first.read_bytes()
            record=json.loads(before)
            self.assertEqual(record['julia_runtime']['version'],'fixture-1.10')
            self.assertEqual(record['julia_runtime']['binary'],str(executable))
            self.assertEqual(record['julia_runtime']['sha256'],hashlib.sha256(executable.read_bytes()).hexdigest())
            self.assertEqual(record['physics_dump'],dict(path=str(dump),exists=False,sha256=None))
            self.assertIn('examples/lvd_muography.jl',record['sha256'])
            subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
            self.assertEqual(len(list(output.glob('run_manifest_*.json'))),2)
            self.assertEqual(first.read_bytes(),before)


if __name__=='__main__':unittest.main()
