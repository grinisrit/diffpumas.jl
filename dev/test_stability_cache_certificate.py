#!/usr/bin/env python3
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import stability_cache_certificate as producer
import run_cached_stability_fit as runner

class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cache-certificate-test-')
        self.factory = Path(self.temp.name)/'factory.jl'
        self.factory.write_text('# exact closure factory fixture\n')
        source, _, _ = runner.transformed_source()
        self.proof = dict(schema_version=1,gate_kind='dry_serial',passed=True,
            scope='fixture parity only', sample_scope='deterministic CSDA nodes; no MC',
            sample_count=32,case_count=126,ray_count=1003,threads=1,cpu_affinity=[20],
            closure_factory_path=str(self.factory),closure_factory_sha256=runner.sha(self.factory),
            transformed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            checks={key:True for key in producer.CHECKS})
    def tearDown(self): self.temp.cleanup()
    def test_serial_and_threaded(self):
        producer.validate_proof(self.proof)
        p=copy.deepcopy(self.proof);p.update(gate_kind='corrected_threaded',threads=4)
        p['checks']['threaded_parity']=True
        producer.validate_proof(p)
        p['threads']=1
        with self.assertRaises(ValueError):producer.validate_proof(p)
    def test_false_missing_and_unexpected_counts(self):
        for key in producer.CHECKS:
            p=copy.deepcopy(self.proof);p['checks'][key]=False
            with self.assertRaises(ValueError):producer.validate_proof(p)
        for key,value in [('sample_count',1024),('ray_count',1002),('case_count',0),('threads',False)]:
            p=copy.deepcopy(self.proof);p[key]=value
            with self.assertRaises(ValueError):producer.validate_proof(p)
    def test_changed_factory_and_transformation(self):
        p=copy.deepcopy(self.proof);p['transformed_source_sha256']='0'*64
        with self.assertRaises(ValueError):producer.validate_proof(p)
        self.factory.write_text('changed')
        with self.assertRaises(ValueError):producer.validate_proof(self.proof)
    def test_gn_complete_stream(self):
        p=copy.deepcopy(self.proof);p.update(gate_kind='actual_gn_trial',expected_trial_count=124,observed_trial_count=124)
        p['checks']['trial_stream_complete']=True
        for prefix in ('recorded_stream','reference_end'):
            p[prefix+'_path']=str(self.factory);p[prefix+'_sha256']=runner.sha(self.factory)
        producer.validate_proof(p)
        p['observed_trial_count']=123
        with self.assertRaises(ValueError):producer.validate_proof(p)
    def test_external_recorded_input_and_fresh_output(self):
        self.assertEqual(runner.recorded_digest({str(self.factory):'x'},self.factory),'x')
        output=Path(self.temp.name)/'proof.json';producer.write_fresh(output,self.proof)
        with self.assertRaises(FileExistsError):producer.write_fresh(output,self.proof)
    def test_exact_replacement_preserves_other_source(self):
        source,old,new=runner.transformed_source()
        original=runner.EXAMPLE.read_text()
        self.assertEqual(source,original.replace(old,new,1))
        self.assertIn('physics_identity=objectid(physics)',new)
        self.assertEqual(source.count('flux_only_at = (ρ, wv) -> begin'),1)

if __name__=='__main__': unittest.main()
