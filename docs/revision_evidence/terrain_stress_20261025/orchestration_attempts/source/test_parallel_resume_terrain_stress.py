import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
SPEC=importlib.util.spec_from_file_location('parallel_resume',Path(__file__).with_name('parallel_resume_terrain_stress.py'))
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

class ParallelTerrainTests(unittest.TestCase):
    def fixture(self):
        command=['python3','audit.py','--cpu-start','0','--cpus','8','--memory-gib','16','--','julia','model.jl','--seed','42']
        sources=[dict(name='fixed_plus1',screen='fixed',command=command),dict(name='all_zero',screen='all',command=command),dict(name='all_minus5',screen='all',command=command)]
        cases=[]
        for source in sources:
            starts=[0,8] if source['screen']=='all' else [0]
            cases.append(dict(name=source['name'],original=source,allowed_cpu_starts=starts,
                resource_only_commands={str(start):MODULE.common.resource_command(command,start) for start in starts}))
        return dict(cases=sources),dict(completed_before_resume={},foreign_case=None,cases=cases)
    def test_unique_claims_and_secondary_all_mask_only(self):
        original,_=self.fixture();ownership={case['name']:dict(phase='pending') for case in original['cases']}
        self.assertEqual(MODULE.claim_case(original['cases'],ownership,8)['name'],'all_zero')
        self.assertEqual(MODULE.claim_case(original['cases'],ownership,0)['name'],'fixed_plus1')
        self.assertEqual(MODULE.claim_case(original['cases'],ownership,0)['name'],'all_minus5')
        self.assertIsNone(MODULE.claim_case(original['cases'],ownership,8))
    def test_rollback_forbidden_after_external_commitment(self):
        self.assertEqual(MODULE.rollback_action(False),'resume_original_parent')
        self.assertEqual(MODULE.rollback_action(True),'preserve_original_parent_paused')
    def test_schedule_exact_numerical_commands_and_unique_cases(self):
        original,schedule=self.fixture();self.assertEqual(len(MODULE.validate_schedule(schedule,original)),3)
        for mutate in ('seed','duplicate','pool'):
            bad=copy.deepcopy(schedule)
            if mutate=='seed':bad['cases'][1]['resource_only_commands']['8'][-1]='43'
            elif mutate=='duplicate':bad['cases'][1]['name']='fixed_plus1'
            else:bad['cases'][0]['allowed_cpu_starts']=[0,8]
            with self.assertRaises(ValueError):MODULE.validate_schedule(bad,original)
    def test_foreign_wrapper_liveness_and_manifest_timeout(self):
        with tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025') as tmp:
            folder=Path(tmp)
            with self.assertRaises(ValueError):MODULE.foreign_snapshot(folder,dict(pid=1),timeout=0.)
            snapshot=dict(already_completed=False,wrapper=dict(pid=2,start_ticks=10),child=dict(pid=3,already_exited_before_snapshot=True))
            with patch.object(MODULE.common,'process_identity',return_value=dict(pid=2,start_ticks=10,state='Z')):
                with self.assertRaises(ValueError):MODULE.wait_foreign(folder,snapshot,poll=0.)
    def test_foreign_julia_must_have_exited_after_clean_end(self):
        with tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025') as tmp:
            folder=Path(tmp);(folder/'end_status.json').write_text('{}')
            snapshot=dict(already_completed=False,wrapper=dict(pid=2,start_ticks=10),child=dict(pid=3,start_ticks=20))
            with patch.object(MODULE.common,'audit_case',return_value={'proof':True}),patch.object(MODULE.common,'process_identity',return_value=dict(pid=3,start_ticks=20,state='R')):
                with self.assertRaises(ValueError):MODULE.wait_foreign(folder,snapshot,poll=0.)
            with patch.object(MODULE.common,'audit_case',return_value={'proof':True}),patch.object(MODULE.common,'process_identity',side_effect=FileNotFoundError):
                self.assertEqual(MODULE.wait_foreign(folder,snapshot,poll=0.),{'proof':True})
if __name__=='__main__':unittest.main()
