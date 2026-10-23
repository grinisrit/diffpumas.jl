"""Read-only fixture checks for missing stages and mixed-provenance reporting."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('reconcile',Path(__file__).resolve().parents[1]/'dev/reconcile_manuscript_outputs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReconciliationTest(unittest.TestCase):
    def test_missing_and_stale_are_not_success(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            mask=root/'lvd_fit_mask.csv'
            mask.write_text('bin_idx,covered,ray_valid,fit_mask\n1,true,true,1.0\n2,true,true,0.0\n3,false,false,0.0\n')
            os.utime(mask,(100,100))
            status=root/'lvd_recursion_status.txt'
            status.write_text('converged=false\npasses=3\ncalibration_mc_samples=128\n')
            manifest=root/'run_manifest_20261023.json'
            manifest.write_text('{"command":["--samples","64"],"sha256":{"src/Physics.jl":"fixture-hash"}}')
            (root/'STOP_REASON.txt').write_text('administratively stopped before fit completion')
            report=module.summarize('fixture',root)
            self.assertEqual(report['metrics']['exact_fit_mask_counts'],dict(all_bins=3,covered_bins=2,ray_valid_bins=2,fit_bins=1))
            self.assertEqual(report['stages']['main']['artifact_status'],'missing_pending')
            self.assertEqual(report['metrics']['recursion_configuration_and_status']['converged'],'false')
            self.assertTrue(report['artifacts'][mask.name]['predates_latest_manifest'])
            self.assertEqual(report['run_manifests'][0]['recorded']['sha256']['src/Physics.jl'],'fixture-hash')
            self.assertIn('STOP_REASON.txt',report['administrative_records'])
            self.assertFalse((root/'reconciliation.json').exists())
            absent=module.summarize('pending',root/'missing')
            self.assertFalse(absent['exists'])
            self.assertEqual(absent['metadata_status'],'unknown_no_run_manifest')
            self.assertEqual(absent['metrics'],{})

if __name__=='__main__': unittest.main()
