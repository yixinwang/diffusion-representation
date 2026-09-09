import numpy as np
import pytest
from scipy.special import gammainc, gammaincc
from qalt.rgb_block import FixedShapeGSM
from qalt.radial_transport import decode_radial, encode_radial, _chi_logs


def model():
    raw = np.array([[1., .4, -.1], [.4, 1.3, .2], [-.1, .2, .8]])
    return FixedShapeGSM(np.array([.2, .3, .5]), np.array([.08, .5, 1.8]), raw / np.linalg.det(raw)**(1/3))


def test_radial_cdf_formula_and_tails():
    r = np.array([1e-100, 1e-8, .1, .999, 1., 1.001, 2., 8., 35., 50.])
    lc, ls = _chi_logs(r)
    np.testing.assert_allclose(lc[:-1], np.log(gammainc(1.5, r[:-1]**2/2)), atol=2e-13)
    np.testing.assert_allclose(ls[2:-1], np.log(gammaincc(1.5, r[2:-1]**2/2)), atol=2e-13)
    assert np.isfinite(ls[-1]) and ls[-1] < -1200


def test_roundtrip_origins_small_radii_extreme_tails_and_shape():
    rng = np.random.default_rng(307)
    u = np.concatenate([rng.normal(size=(100, 3)), np.array([[0,0,0], [1e-100,0,0], [50,0,0], [-30,20,10]])])
    decoded = decode_radial(model(), u)
    encoded = encode_radial(model(), decoded.values)
    np.testing.assert_allclose(encoded.values, u, rtol=3e-12, atol=3e-12)
    assert abs(encoded.values[-3, 0] / 1e-100 - 1) < 3e-12
    np.testing.assert_allclose(encoded.log_abs_det, -decoded.log_abs_det, rtol=1e-10, atol=1e-8)
    assert decoded.max_relative_bracket <= 1e-12
    assert decoded.iterations <= 128
    assert decode_radial(model(), u.reshape(2,52,3)).values.shape == (2,52,3)


def test_single_gaussian_reduces_to_linear_map():
    m = FixedShapeGSM(np.array([1.]), np.array([.7]), model().shape)
    u = np.array([[.4, -.2, 1.], [0., 0., 0.]])
    result = decode_radial(m, u)
    np.testing.assert_allclose(result.values, .7 * u @ m.cholesky.T, atol=1e-15)
    np.testing.assert_allclose(result.log_abs_det, 3*np.log(.7), atol=2e-15)
    assert result.iterations == 0


def test_logdet_matches_independent_finite_difference_jacobian():
    m = model()
    for u in [np.array([.3, -.8, .5]), np.zeros(3)]:
        h = 1e-5
        jac = np.column_stack([(decode_radial(m, u+h*e, rtol=1e-14).values - decode_radial(m, u-h*e, rtol=1e-14).values)/(2*h) for e in np.eye(3)])
        sign, logdet = np.linalg.slogdet(jac)
        assert sign > 0
        np.testing.assert_allclose(logdet, decode_radial(m, u, rtol=1e-14).log_abs_det, atol=2e-8)


def test_nonconvergence_and_invalid_input_fail_explicitly():
    with pytest.raises(FloatingPointError, match="did not converge"):
        decode_radial(model(), np.ones((2,3)), max_iterations=1)
    with pytest.raises(ValueError):
        decode_radial(model(), np.array([np.nan,0,0]))
    with pytest.raises(ValueError):
        decode_radial(model(), np.ones(3), rtol=0)
