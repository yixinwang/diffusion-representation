import numpy as np
import pytest
from scipy.integrate import quad
from reference import A, KNOTS, HEIGHTS, psi, integrated_psi, encode, decode


def test_feature_moments_and_reflection():
    for power, expected in ((1, 0), (2, 1), (3, 0)):
        value = sum(quad(lambda u: float(psi(u)) ** power, a, b)[0]
                    for a, b in zip(KNOTS[:-1], KNOTS[1:]))
        assert abs(value - expected) < 2e-14
    u = np.linspace(0, 1, 1025)
    np.testing.assert_allclose(psi(u), psi(1-u), atol=2e-15, rtol=0)
    np.testing.assert_allclose(integrated_psi([0, 1]), 0, atol=0)


def test_conditional_density_normalization():
    for u in (0., .2, .5, .8, 1.):
        for theta in (-.45, 0., .45):
            integral = sum(quad(lambda v: np.exp(encode(u, v, theta)[1]), a, b)[0]
                           for a, b in zip(KNOTS[:-1], KNOTS[1:]))
            assert abs(integral - 1) < 2e-15


def test_all_knots_neighbors_and_extreme_parameters():
    v = np.unique(np.concatenate([KNOTS, np.nextafter(KNOTS[1:], 0),
                                  np.nextafter(KNOTS[:-1], 1)]))
    for u in np.r_[KNOTS, .21, .77]:
        for theta in (-.45, -.001, 0., .001, .45):
            p, ld = encode(u, v, theta)
            recovered, invld = decode(u, p, theta)
            np.testing.assert_allclose(recovered, v, atol=5e-16, rtol=0)
            np.testing.assert_allclose(ld+invld, 0, atol=3e-15, rtol=0)


def test_fabricated_broadcast_and_nonidentity_jacobian():
    rng = np.random.default_rng(1309101)
    u, p = rng.random((2, 32, 1440))
    theta = rng.uniform(-.45, .45, (32, 1))
    v, ld = decode(u, p, theta)
    recovered, invld = encode(u, v, theta)
    np.testing.assert_allclose(recovered, p, atol=5e-16, rtol=0)
    np.testing.assert_allclose(ld+invld, 0, atol=4e-15, rtol=0)
    x = np.array([.19, .63]);theta=.41;h=1e-6
    f = lambda q: np.array([q[0], decode(q[0], q[1], theta)[0]])
    jac = np.column_stack([(f(x+h*e)-f(x-h*e))/(2*h) for e in np.eye(2)])
    assert abs(jac[1, 0]) > .01
    assert abs(np.log(np.linalg.det(jac))-decode(*x, theta)[1]) < 2e-9


def test_discovered_endpoint_cancellation():
    # Independent broad sweep found these failures in the original formula.
    for u, theta in ((.873046875, -.45), (.373046875, .45)):
        p = np.array([0., np.nextafter(0., 1.), np.nextafter(1., 0.), 1.])
        v, _ = decode(u, p, theta)
        assert v[0] == 0 and v[-1] == 1
        np.testing.assert_allclose(encode(u, v, theta)[0], p, atol=3e-16, rtol=0)


@pytest.mark.parametrize('args', [(np.nan,.5,0),(.5,-.1,0),(.5,1.1,0),(.5,.5,.451),(.5,.5,np.inf)])
def test_invalid_inputs_rejected(args):
    for fn in (encode, decode):
        with pytest.raises(ValueError):fn(*args)
