import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('terrain_summary',Path(__file__).with_name('summarize_terrain_stress.py'))
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

class TerrainSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025')
        self.root=Path(self.temp.name)
        self.cases=[dict(name='fixed_zero',screen='fixed',amplitude=0.,forward_only=False),
                    dict(name='fixed_plus5',screen='fixed',amplitude=.05,forward_only=False)]
        (self.root/'preregistered_plan.json').write_text(json.dumps(dict(scope='synthetic fixture only',cases=self.cases)))
    def tearDown(self):self.temp.cleanup()
    def case(self,name):
        folder=self.root/name;data=folder/'data';data.mkdir(parents=True)
        (data/'report.toml').write_text('''objective=1.0
projected_gradient_inf=0.01
stationary_at_requested_tolerance=false
max_abs_slant_change_m=0.0
max_abs_map_height_change_m=0.0
off_sector_changed_paths=0
changed_supported_cell_intersections=0
''')
        (data/'predictions.csv').write_text('bin_idx,sector,fit_mask,fixed_field_flux,fixed_log_residual,refit_log_residual\n1,true,1,2.0,0.1,0.1\n2,false,1,1.0,0.2,0.2\n')
        (data/'fit_field.csv').write_text('cell_idx,support,water\n1,true,0.2\n2,false,0.0\n')
        outputs={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
        (folder/'end_status.json').write_text(json.dumps(dict(execution_completed=True,source_unchanged=True,runtime_unchanged=True,output_sha256=outputs)))
    def test_partial_skips_unfinished(self):
        self.case('fixed_zero');value=MODULE.summarize(self.root,True)
        self.assertEqual(len(value['cases']),1);self.assertFalse(value['complete'])
        with self.assertRaises(ValueError):MODULE.summarize(self.root)
    def test_matching_zero_must_be_complete(self):
        self.case('fixed_plus5');value=MODULE.summarize(self.root,True)
        self.assertEqual(value['cases'],[])
        self.assertTrue(any('matching zero' in item for item in value['pending']))
    def test_tampered_output_never_adopted(self):
        self.case('fixed_zero');self.case('fixed_plus5')
        (self.root/'fixed_plus5/data/fit_field.csv').write_text('tampered')
        with self.assertRaises(ValueError):MODULE.summarize(self.root)
        value=MODULE.summarize(self.root,True)
        self.assertEqual([r['case'] for r in value['cases']],['fixed_zero'])
        self.assertEqual(len(value['failed']),1)
    def test_pair_and_provenance(self):
        self.case('fixed_zero');self.case('fixed_plus5');value=MODULE.summarize(self.root)
        self.assertTrue(value['complete']);self.assertEqual(value['cases'][1]['field_relative_to_matching_zero'],0.)
        self.assertIn('data/report.toml',value['cases'][1]['case_provenance']['adopted_output_sha256'])
if __name__=='__main__':unittest.main()
