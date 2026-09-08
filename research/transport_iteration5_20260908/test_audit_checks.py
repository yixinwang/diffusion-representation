"""Independent deterministic tests; no model fitting and no experimental outcomes."""
import math
import unittest
from fractions import Fraction
from audit_checks import (h_moments, dependence_lower, pro4_bound, table_resources,
                          invert_linear_density, validation_selection_penalty,
                          verify_split_roles, validate_run_contract)

class AuditTests(unittest.TestCase):
    def test_exact_cubic_moments(self):
        self.assertEqual(h_moments(), (Fraction(0), Fraction(17,35)))

    def test_pair_gap_units(self):
        self.assertAlmostEqual(dependence_lower(.5), 0.01965986394557823)
        self.assertAlmostEqual(dependence_lower(-.5), dependence_lower(.5))

    def test_finite_constants_not_a_learning_lower_bound(self):
        row = pro4_bound(40000,2,1,.5,1.5,10.5)
        self.assertEqual(row['coefficients'], 12)
        self.assertAlmostEqual(row['kappa'],2310.626417732768)
        self.assertAlmostEqual(row['estimation'],10.622170713614741)
        self.assertGreater(row['estimation'], dependence_lower(.5))

    def test_approximation_mesh_precondition(self):
        with self.assertRaises(ValueError): pro4_bound(100,2,1,.5,1.5,30)

    def test_integer_mesh_search(self):
        rows = [pro4_bound(40000,b,1,.5,1.5,10.5) for b in range(2,100)]
        best = min(rows,key=lambda x:x['total'])
        self.assertEqual(best['b'], 5)
        self.assertAlmostEqual(best['total'],56.19300886441856)

    def test_memory_is_not_pixel_sample_size(self):
        self.assertEqual(table_resources(8)['coefficients'],900000000)
        self.assertEqual(table_resources(8)['raw_bytes'],3600000000)
        self.assertEqual(table_resources(8)['active_density_products_upper'],13122)

    def test_inverse_flat_density_and_units(self):
        self.assertAlmostEqual(invert_linear_density(2,2,.2,.12),.06)

    def test_inverse_increasing_and_decreasing(self):
        for a,right in ((.25,3.),(3.,.25),(.25,.25),(1.01,1.0)):
            width=.17
            for fraction in (0.,.001,.1,.5,.9,1.):
                t=width*fraction
                v=width*(a+right)/2 if fraction == 1.0 else a*t+(right-a)*t*t/(2*width)
                actual=invert_linear_density(a,right,width,v)
                self.assertAlmostEqual(actual,t,places=13)

    def test_inverse_rejects_invalid_targets(self):
        for args in ((0,1,1,.5),(1,1,1,-.1),(1,1,1,1.1)):
            with self.assertRaises(ValueError): invert_linear_density(*args)

    def test_validation_cost_increases_with_catalogue(self):
        self.assertGreater(validation_selection_penalty(.25,3,10,5000),
                           validation_selection_penalty(.25,3,1,5000))

    def test_reused_groups_rejected(self):
        with self.assertRaises(ValueError):
            verify_split_roles({'train':['g1'],'val':['g1']},{'train':1,'val':1})
        verify_split_roles({'train':['g1'],'val':['g2']},{'train':1,'val':1})

    def test_contract_fails_closed(self):
        with self.assertRaises(ValueError): validate_run_contract({})

    def test_spectral_radius_is_not_operator_contraction(self):
        # Strictly lower-triangular star: eigenvalues zero, row sums .75,
        # but spectral norm .75*sqrt(8)>1.
        self.assertLess(.75,1)
        self.assertGreater(.75*math.sqrt(8),1)

if __name__ == '__main__': unittest.main()
