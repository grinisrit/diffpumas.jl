import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
SPEC=importlib.util.spec_from_file_location('reserved_resume',Path(__file__).with_name('reserved_terrain_resume.py'))
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

class ReservedTerrainTests(unittest.TestCase):
    def temporary(self):return tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025')
    def test_only_resource_and_output_path_changes(self):
        command=['python3','audit.py','--output','old','--cpu-start','0','--cpus','8','--memory-gib','16',
                 '--','julia','model.jl','--output-dir','old/data','--seed','42']
        changed=MODULE.relocate_command(command,Path('/fresh/alternate'))
        self.assertEqual([i for i,(a,b) in enumerate(zip(command,changed)) if a!=b],[3,5,14])
        self.assertEqual(command[-2:],changed[-2:]);self.assertEqual(command[7:10],changed[7:10])
    def test_atomic_reservation_never_overwrites_science(self):
        with self.temporary() as tmp:
            folder=Path(tmp)/'case';folder.mkdir();sentinel=folder/'scientific.csv';sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):MODULE.reserve(folder,{'owner':'fixture'})
            self.assertEqual(sentinel.read_text(),'preserve')
    def test_guard_rejects_unknown_files_or_changed_marker(self):
        with self.temporary() as tmp:
            folder=Path(tmp)/'case';identity=MODULE.reserve(folder,{'owner':'fixture'})
            MODULE.verify_guard(identity);(folder/'unknown').write_text('preserve')
            with self.assertRaises(ValueError):MODULE.verify_guard(identity)
            (folder/'unknown').unlink();(folder/'orchestration_guard.json').write_text('changed')
            with self.assertRaises(ValueError):MODULE.verify_guard(identity)
    def test_exact_guard_release_and_relative_alias(self):
        with self.temporary() as tmp:
            root=Path(tmp);folder=root/'original';identity=MODULE.reserve(folder,{'owner':'fixture'})
            alternate=root/'alternate';alternate.mkdir();(alternate/'end_status.json').write_text('{}')
            result=MODULE.release_and_alias(identity,alternate)
            self.assertTrue(folder.is_symlink());self.assertEqual(folder.resolve(),alternate.resolve())
            self.assertFalse(os.path.isabs(result['relative_target']))
    def test_original_launcher_refuses_guard_before_any_child(self):
        with self.temporary() as tmp:
            root=Path(tmp);guard=root/'guard';identity=MODULE.reserve(guard,{'owner':'fixture'});sentinel=root/'launched'
            wrapper=MODULE.ROOT/'dev/run_revision_job.py'
            command=[sys.executable,str(wrapper),'--output',str(guard),'--dump',str(wrapper),'--julia','/nonexistent',
                '--cpus','8','--cpu-start','0','--memory-gib','16','--',sys.executable,'-c',f'open({str(sentinel)!r},"w").write("bad")']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('output exists',result.stderr)
            self.assertFalse(sentinel.exists());MODULE.verify_guard(identity)
            source=wrapper.read_text()
            self.assertLess(source.index('if output.exists():'),source.index('output.mkdir(parents=True)'))
            self.assertLess(source.index('output.mkdir(parents=True)'),source.index('child = subprocess.Popen('))
    def test_parent_exit_identity_and_timeout(self):
        identity=dict(pid=123,start_ticks=10)
        with patch.object(MODULE.common,'process_identity',return_value=dict(pid=123,start_ticks=11,state='R')):
            MODULE.wait_parent_exit(identity,timeout=0.)
        with patch.object(MODULE.common,'process_identity',return_value=dict(pid=123,start_ticks=10,state='R')):
            with self.assertRaises(ValueError):MODULE.wait_parent_exit(identity,timeout=0.)
if __name__=='__main__':unittest.main()
