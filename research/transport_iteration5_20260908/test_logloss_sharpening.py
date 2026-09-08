"""Deterministic tests for the supplemental log-loss calculation."""
import math
import unittest
from logloss_sharpening import logloss_bernstein, sharper_bound

class SharpeningTests(unittest.TestCase):
    def test_scalar_integral_bound_and_continuous_limit(self):
        self.assertEqual(logloss_bernstein(0),2)
        self.assertAlmostEqual(logloss_bernstein(1e-8),2,places=7)
        for B in (.01,.2,math.log(12),10):
            V=logloss_bernstein(B)
            for frac in (-1.,-.75,-.1,0.,.1,.75,1.):
                r=B*frac
                self.assertLessEqual(r*r,V*(math.expm1(-r)+r)+1e-11)

    def test_misspecified_population_convex_optimum(self):
        # Bernoulli target .8; model probabilities constrained to [.4,.6].
        # Population optimum .6 is misspecified; compare feasible model .4.
        weights=(.8,.2); ratios=(.4/.6,.6/.4)
        losses=tuple(-math.log(v) for v in ratios)
        self.assertLessEqual(sum(w*v for w,v in zip(weights,ratios)),1)
        risk=sum(w*r for w,r in zip(weights,losses))
        second=sum(w*r*r for w,r in zip(weights,losses))
        self.assertLessEqual(second,logloss_bernstein(math.log(1.5))*risk)

    def test_finite_bound_still_exceeds_pair_gap(self):
        rows=[sharper_bound(40000,b,1,.5,1.5,10.5) for b in range(2,100)]
        best=min(rows,key=lambda r:r['total'])
        self.assertEqual(best['b'],13)
        self.assertAlmostEqual(best['excess_coefficient'],11.187974367351192)
        self.assertAlmostEqual(best['total'],1.3118432844697976)
        self.assertGreater(best['total'],.01965986394557823)

if __name__ == '__main__': unittest.main()
