import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import poisson

SPEC=importlib.util.spec_from_file_location('poisson_reference',Path(__file__).with_name('poisson_likelihood_reference.py'))
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

class NoAudit:
    def verify(self,path):return path

class LikelihoodReferenceTests(unittest.TestCase):
    def test_numeric_csv_bin_identifiers(self):
        self.assertEqual(MODULE.ordered_bins([{'bin':'1.0'},{'bin':'2'}]),[1,2])
        with self.assertRaises(ValueError):MODULE.ordered_bins([{'bin':'1.5'}])
    def test_scipy_logpmf_oracle(self):
        counts=np.array([[0,0,0],[1,2,3],[10,0,1]],dtype=np.int64)
        null=np.array([.01,2.,.8]);alternative=np.array([.02,1.,3.])
        for exposure in (.1,4.):
            actual=MODULE.likelihood_ratio_scores(counts,null,alternative,exposure)
            oracle=np.sum(poisson.logpmf(counts,exposure*alternative)-poisson.logpmf(counts,exposure*null),axis=1)
            np.testing.assert_allclose(actual,oracle,rtol=2e-14,atol=2e-14)
    def test_chunk_boundary_and_identical_hypotheses(self):
        counts=np.arange(387,dtype=np.int64).reshape(129,3)%5
        null=np.array([.2,.3,.4]);alternative=np.array([.3,.4,.5])
        actual=MODULE.likelihood_ratio_scores(counts,null,alternative,2.)
        oracle=np.sum(poisson.logpmf(counts,2*alternative)-poisson.logpmf(counts,2*null),axis=1)
        np.testing.assert_allclose(actual,oracle,rtol=2e-14,atol=2e-14)
        same=MODULE.likelihood_ratio_scores(counts,null,null,2.)
        self.assertTrue(np.array_equal(same,np.zeros(129)))
        self.assertEqual(np.count_nonzero(same>np.quantile(same,1-MODULE.ALPHA)),0)
    def test_invalid_counts_and_rates(self):
        for counts,null,alternative in [(np.array([[-1,1]],dtype=np.int64),[1,1],[2,2]),
                (np.array([[1.,1.]]),[1,1],[2,2]),
                (np.array([[1,1]],dtype=np.int64),[0,1],[2,2])]:
            with self.assertRaises(ValueError):MODULE.likelihood_ratio_scores(counts,null,alternative,1.)
    def test_saved_metadata_byte_shape(self):
        with tempfile.TemporaryDirectory(dir='/mnt/big_data/agent-builds/terrain-stress-20261025') as tmp:
            path=Path(tmp)/'counts.bin';np.array([[0,1],[2,3]],dtype='<i8').tofile(path)
            metadata=dict(format=MODULE.FORMAT,bins=2,draws=2,exposure=3.,seed=1,
                generator='numpy.default_rng PCG64; Poisson variates, no Gaussian substitution')
            path.with_suffix('.json').write_text(json.dumps(metadata))
            counts,_=MODULE.open_counts(path,3.,2,2,NoAudit())
            np.testing.assert_array_equal(counts,[[0,1],[2,3]]);del counts
            with self.assertRaises(ValueError):MODULE.open_counts(path,4.,2,2,NoAudit())
            with path.open('ab') as stream:stream.write(b'x')
            with self.assertRaises(ValueError):MODULE.open_counts(path,3.,2,2,NoAudit())
if __name__=='__main__':unittest.main()
