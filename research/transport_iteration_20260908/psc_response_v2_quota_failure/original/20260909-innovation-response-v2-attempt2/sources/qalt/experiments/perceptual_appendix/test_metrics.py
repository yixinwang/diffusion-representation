"""Fabricated-array tests only; never access CIFAR or preserved banks."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('appendix_metrics',Path(__file__).with_name('metrics.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MetricsTest(unittest.TestCase):
    def test_kid_unequal_sizes_direct_pair_sum(self):
        rng=np.random.default_rng(821)
        a,b=rng.normal(size=(7,3)),rng.normal(size=(9,3))
        kernel=lambda x,y:(np.dot(x,y)/3+1)**3
        direct=(sum(kernel(x,y) for i,x in enumerate(a) for j,y in enumerate(a) if i!=j)/42
                +sum(kernel(x,y) for i,x in enumerate(b) for j,y in enumerate(b) if i!=j)/72
                -2*sum(kernel(x,y) for x in a for y in b)/63)
        self.assertAlmostEqual(metrics.polynomial_kid(a,b),direct,places=12)
        self.assertAlmostEqual(metrics.polynomial_kid(a,b),metrics.polynomial_kid(b,a),places=12)

    def test_kid_negative_allowed_and_constant_zero(self):
        a=np.array([[-1.,0.],[1.,0.]])
        self.assertLess(metrics.polynomial_kid(a,a),0)
        self.assertAlmostEqual(metrics.polynomial_kid(np.ones((6,3)),np.ones((8,3))),0)

    def test_prdc_direct_euclidean(self):
        rng=np.random.default_rng(44)
        a,b=rng.normal(size=(8,3)),rng.normal(size=(10,3))
        ar=np.array([sorted(np.linalg.norm(x-y) for j,y in enumerate(a) if j!=i)[4] for i,x in enumerate(a)])
        br=np.array([sorted(np.linalg.norm(x-y) for j,y in enumerate(b) if j!=i)[4] for i,x in enumerate(b)])
        cross=np.array([[np.linalg.norm(x-y) for y in b] for x in a])
        expect={'precision':(cross<ar[:,None]).any(0).mean(),
                'recall':(cross<br[None,:]).any(1).mean(),
                'density':(cross<ar[:,None]).sum(0).mean()/5,
                'coverage':(cross.min(1)<ar).mean()}
        out=metrics.prdc(a,b)
        for key,value in expect.items():self.assertAlmostEqual(out[key],value)

    def test_separated_and_identical_clouds(self):
        a=np.random.default_rng(9).normal(size=(12,3))
        same=metrics.prdc(a,a)
        self.assertEqual(same['coverage'],1)
        self.assertEqual(same['precision'],1)
        apart=metrics.prdc(a,a+100)
        for k in ('coverage','precision','recall','density'):self.assertEqual(apart[k],0)

    def test_duplicate_ties_and_input_rejection(self):
        a=np.zeros((6,3)); out=metrics.prdc(a,a)
        self.assertEqual(out['real_duplicate_rows'],5)
        self.assertEqual(out['cross_radius_ties_real'],36)
        self.assertEqual(out['coverage'],0) # strict neighborhood excludes zero-radius ties
        nonzero=np.tile(np.random.default_rng(77).normal(size=2048),(6,1))
        self.assertTrue(np.array_equal(metrics.squared_distances(nonzero,nonzero),np.zeros((6,6))))
        self.assertEqual(metrics.prdc(nonzero,nonzero)['coverage'],0)
        with self.assertRaises(ValueError):metrics.prdc(np.zeros((5,3)),a)
        with self.assertRaises(ValueError):metrics.polynomial_kid(np.full((6,3),np.nan),a)


if __name__=='__main__':unittest.main()
