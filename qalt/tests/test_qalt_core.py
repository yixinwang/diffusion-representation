import numpy as np

from qalt.core import (
    decoder,
    euler_scale,
    fiber_kl,
    fiber_w2_squared,
    pooled_variance,
    inverse_decoder,
    sample_active,
    separate_variances,
)


def test_exact_solver_ties_and_euler_is_strict() -> None:
    scale = np.full(12, 2.0)
    assert fiber_kl(scale, scale) == 0.0
    assert fiber_w2_squared(scale, scale) == 0.0
    assert fiber_kl(scale, euler_scale(scale, 10)) > 0.0
    assert fiber_w2_squared(scale, euler_scale(scale, 10)) > 0.0


def test_pooled_and_separate_estimators_share_information() -> None:
    rng = np.random.default_rng(7)
    samples = 2.0 * rng.normal(size=(30, 12))
    assert pooled_variance(samples).shape == (12,)
    assert separate_variances(samples).shape == (12,)
    assert np.allclose(pooled_variance(samples), np.mean(samples**2))


def test_nonlinear_non_gaussian_observation_shape() -> None:
    rng = np.random.default_rng(8)
    z = sample_active(100, 4, rng)
    r = rng.normal(size=(100, 12))
    x = decoder(z, r)
    recovered_z, recovered_r = inverse_decoder(x, 4)
    assert x.shape == (100, 16)
    assert np.max(np.abs(recovered_z - z)) < 1e-12
    assert np.max(np.abs(recovered_r - r)) == 0.0
