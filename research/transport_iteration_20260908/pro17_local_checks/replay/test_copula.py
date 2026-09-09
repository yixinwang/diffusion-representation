import math
import unittest
import numpy as np
from scipy.special import ndtr, ndtri
from copula import (Model, RHO, psi, inverse_conditional_cdf, conditional_cdf,
                    fit, compile_equal_information_copy, generate_observed,
                    copula_kl, expected_kl, oracle_cell_means, read_observations)

class CopulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        z, w = np.polynomial.legendre.leggauss(16)
        cls.v = (z+1)/2
        cls.w = w/2
        cls.x = np.einsum('i,j,k->ijk', psi(cls.v), psi(cls.v), psi(cls.v))
        cls.w3 = np.einsum('i,j,k->ijk', cls.w, cls.w, cls.w)

    def test_01_normalization(self):
        for t in (-.9, -.4, 0, .4, .9):
            self.assertAlmostEqual(float(np.sum(self.w3*(1+t*self.x))), 1., places=13)

    def test_02_positivity(self):
        x = np.linspace(-1,1,101)
        self.assertGreaterEqual(float(np.min(1+.9*x)), .1-1e-15)

    def test_03_inverse_roundtrip(self):
        e = np.linspace(0,1,129)[None, :]
        a = np.linspace(-.9,.9,43)[:, None]
        v = inverse_conditional_cdf(e, a)
        self.assertLess(float(np.max(np.abs(conditional_cdf(v,a)-e))), 8e-15)

    def test_04_inverse_no_repair_invalid(self):
        with self.assertRaises(ValueError):
            inverse_conditional_cdf(np.array([.5]), np.array([1.]))

    def test_05_jacobian(self):
        v = np.linspace(.01,.99,73)
        a = .83
        eps = 1e-6
        numerical = (conditional_cdf(v+eps,a)-conditional_cdf(v-eps,a))/(2*eps)
        self.assertLess(float(np.max(np.abs(numerical-(1+a*psi(v))))), 1e-8)

    def test_06_full_dimensional_gaussian_roundtrip(self):
        rng = np.random.default_rng(1701706)
        model = Model(np.array([[.8],[-.4]]), 16, 4)
        z = rng.normal(size=(100,29))
        decoded = model.decode(z)
        self.assertLess(float(np.max(np.abs(model.encode(decoded)-z))), 5e-11)
        np.testing.assert_array_equal(decoded[:,:17], z[:,:17])

    def test_07_all_pair_marginals_independent(self):
        density = 1+.87*self.x
        for axis in range(3):
            marginal = np.tensordot(density, self.w, axes=(axis,0))
            np.testing.assert_allclose(marginal, np.ones_like(marginal), atol=1e-14)

    def test_08_moment_identity(self):
        for t in (-.9, 0., .9):
            moment = np.sum(self.w3*self.x*(1+t*self.x))
            self.assertAlmostEqual(float(moment), t/27., places=14)

    def test_09_second_moment(self):
        for t in (-.9, 0., .9):
            self.assertAlmostEqual(float(np.sum(self.w3*self.x**2*(1+t*self.x))), 1/27, places=14)

    def test_10_kl_series_quadrature(self):
        z,w = np.polynomial.legendre.leggauss(32)
        x = np.einsum('i,j,k->ijk',z,z,z)
        w3 = np.einsum('i,j,k->ijk',w/2,w/2,w/2)
        for t,e in ((.85,-.8),(-.6,.7),(.8,.8),(.0,.5)):
            q = np.sum(w3*(1+t*x)*(np.log1p(t*x)-np.log1p(e*x)))
            self.assertAlmostEqual(float(copula_kl(t,e)), float(q), places=11)

    def test_11_kl_bounds(self):
        grid = np.linspace(-.9,.9,13)
        t,e = np.meshgrid(grid,grid)
        kl = copula_kl(t,e)
        lo = (t-e)**2/(54*(1+.9))
        hi = (t-e)**2/(54*(1-.9))
        self.assertTrue(np.all(kl >= lo-1e-13))
        self.assertTrue(np.all(kl <= hi+1e-13))

    def test_12_copy_is_same_map(self):
        rng = np.random.default_rng(1701712)
        model = Model(np.array([[.5],[-.7]]), 16, 4)
        copy = compile_equal_information_copy(model)
        z = rng.normal(size=(50,29))
        np.testing.assert_array_equal(model.decode(z),copy.decode(z))
        np.testing.assert_array_equal(model.log_prob(z),copy.log_prob(z))

    def test_13_shared_failure_population_projection(self):
        product = Model(np.zeros((8,1)),16,4)
        shared = expected_kl(product,.85,np.ones(4))
        alternating = expected_kl(product,.85,np.array([1,-1,1,-1]))
        self.assertAlmostEqual(shared, alternating, places=13)
        self.assertGreater(shared,0)

    def test_14_lipschitz_bin_approximation(self):
        B=8; a=.85
        roots, weights = np.polynomial.legendre.leggauss(64)
        means = oracle_cell_means(B,a)
        mse=0.
        for b in range(B):
            c=(b+(roots+1)/2)/B
            mse += np.dot(weights,(a*np.sin(2*np.pi*c)-means[b])**2)/(2*B)
        self.assertLessEqual(mse,(2*np.pi*a)**2/(12*B**2)+1e-13)

    def test_15_rank16_does_not_span17_responses(self):
        U=np.eye(17)[:,:16]
        residual=np.eye(17)-U@U.T
        self.assertAlmostEqual(float(np.trace(residual)),1.)
        self.assertAlmostEqual(float(np.trace(residual))/8.,.125)

    def test_16_fit_only_observations(self):
        rng=np.random.default_rng(1701716)
        observed=generate_observed(rng,128,4,16,4,.85,np.ones(4))
        model=fit(observed,4,16,4,True)
        self.assertEqual(model.theta.shape,(4,1))
        self.assertTrue(np.all(np.abs(model.theta)<=RHO))

    def test_17_full_log_density_formula(self):
        rng=np.random.default_rng(1701717)
        model=Model(np.array([[.5],[-.7]]),16,4)
        z=rng.normal(size=(31,29))
        c,v=read_observations(z,16,4)
        base=-.5*np.sum(z*z+math.log(2*math.pi),axis=1)
        ratio=np.sum(np.log1p(model.response(c)*np.prod(psi(v),axis=-1)),axis=1)
        np.testing.assert_allclose(model.log_prob(z)-base,ratio,atol=1e-13)

    def test_18_non_gaussian_joint_witness(self):
        self.assertGreater(float(copula_kl(.8,0.)),0.)
        self.assertAlmostEqual(float(np.sum(self.w3*self.x*(1+.8*self.x))),.8/27,places=14)

    def test_19_empty_cell_fails_without_adaptation(self):
        z=np.zeros((10,29))
        with self.assertRaises(ValueError):
            fit(z,8,16,4,True)

    def test_20_oracle_bin_projection_improves_product(self):
        signs=np.ones(4)
        q0=Model(np.zeros((8,1)),16,4)
        qb=Model(oracle_cell_means(8,.85)[:,None],16,4)
        self.assertLess(expected_kl(qb,.85,signs),expected_kl(q0,.85,signs))

if __name__ == '__main__':
    unittest.main(verbosity=2)
