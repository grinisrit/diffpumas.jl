import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('terrain_resume',Path(__file__).with_name('resume_terrain_stress.py'))
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

class TerrainResumeTests(unittest.TestCase):
    def test_resource_diff_leaves_numerical_argv_exact(self):
        command=['python3','audit.py','--cpu-start','0','--cpus','8','--memory-gib','16','--',
                 'julia','model.jl','--output-dir','fresh','--seed','42']
        changed=MODULE.resource_command(command,8)
        self.assertEqual(changed[changed.index('--'):],command[command.index('--'):])
        self.assertEqual([i for i,(a,b) in enumerate(zip(command,changed)) if a!=b],[3])
        self.assertEqual(command[3],'0')
        with self.assertRaises(ValueError):MODULE.resource_command(command,16)
    def test_pid_reuse_and_wrong_target_rejected(self):
        queue=Path('/tmp/fixture-only-no-files')
        expected=dict(pid=123,start_ticks=10,command=['python3',str(MODULE.ROOT/'dev/run_terrain_stress.py'),
            '--execute','--output',str(queue)])
        with patch.object(MODULE,'process_identity',return_value=dict(expected,start_ticks=11)):
            with self.assertRaises(ValueError):MODULE.verify_parent(expected,queue)
        with patch.object(MODULE,'process_identity',return_value=expected):
            self.assertEqual(MODULE.verify_parent(expected,queue),expected)
            with self.assertRaises(ValueError):MODULE.verify_parent(expected,Path('/tmp/another-fixture'))
    def test_completed_hashes_and_no_pending_overwrite(self):
        with tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025') as tmp:
            root=Path(tmp);folder=root/'done';folder.mkdir();item=folder/'value.csv';item.write_text('frozen\n')
            end=dict(returncode=0,execution_completed=True,source_unchanged=True,runtime_unchanged=True,
                output_sha256={'value.csv':hashlib.sha256(item.read_bytes()).hexdigest()})
            (folder/'end_status.json').write_text(json.dumps(end));(root/'inflight').mkdir()
            plan=dict(cases=[dict(name='done'),dict(name='pending'),dict(name='inflight')])
            completed,pending,active=MODULE.pending_cases(root,plan)
            self.assertEqual(list(completed),['done']);self.assertEqual([c['name'] for c in pending],['pending'])
            self.assertEqual(active,['inflight'])
            item.write_text('modified\n')
            with self.assertRaises(ValueError):MODULE.pending_cases(root,plan)
    def test_schedule_rejects_numeric_drift_and_duplicate_names(self):
        command=['python3','audit.py','--cpu-start','0','--cpus','8','--memory-gib','16','--','julia','model.jl','--seed','42']
        plan=dict(cases=[dict(name='one',command=command),dict(name='two',command=command)])
        cases=[dict(name=name,cpu_start=start,original_command=list(command),resource_only_command=MODULE.resource_command(command,start))
            for name,start in [('one',0),('two',8)]]
        import copy
        schedule=dict(completed_before_resume={},cases=cases)
        self.assertEqual(len(MODULE.validate_schedule(schedule,plan,Path('/nonexistent-fixture'))),2)
        bad=copy.deepcopy(schedule);bad['cases'][1]['name']='one'
        with self.assertRaises(ValueError):MODULE.validate_schedule(bad,plan,Path('/nonexistent-fixture'))
        bad=copy.deepcopy(schedule);bad['cases'][1]['resource_only_command'][-1]='43'
        with self.assertRaises(ValueError):MODULE.validate_schedule(bad,plan,Path('/nonexistent-fixture'))
    def test_missing_manifest_timeout_and_dead_wrapper(self):
        with tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025') as tmp:
            folder=Path(tmp)
            with self.assertRaises(ValueError):MODULE.wait_current_case(folder,dict(pid=1),folder,manifest_timeout=0.)
            (folder/'pid.json').write_text(json.dumps(dict(launcher_pid=2,pid=3)))
            wrapper=dict(pid=2,ppid=1,start_ticks=10,state='Z')
            with patch.object(MODULE,'process_identity',return_value=wrapper):
                with self.assertRaises(ValueError):MODULE.wait_current_case(folder,dict(pid=1),folder)
    def test_no_actual_signals_in_prepare_or_resource_helpers(self):
        import os
        identity=MODULE.process_identity(os.getpid())
        self.assertEqual(identity['pid'],os.getpid());self.assertGreater(identity['start_ticks'],0)
        self.assertIn(identity['state'],('R','S'))
if __name__=='__main__':unittest.main()
