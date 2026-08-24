import numpy as np

from qalt.multiscale import (
    MixtureFiber,
    band_slices,
    fit_mixture_fiber,
    haar_forward,
    haar_inverse,
    token_benchmark,
)


def test_image_and_video_haar_are_exact_and_norm_preserving() -> None:
    rng = np.random.default_rng(31)
    for shape, axes in (((3, 32, 32), (1, 2)), ((2, 16, 32, 32), (1, 2, 3))):
        values = rng.normal(size=shape)
        coefficients = haar_forward(values, axes)
        assert np.max(np.abs(haar_inverse(coefficients, axes) - values)) < 1e-12
        assert abs(np.sum(coefficients**2) - np.sum(values**2)) < 1e-9
        assert sum(np.prod(coefficients[index].shape) for index in band_slices(shape, axes)) == values.size


def test_train_fitted_mixture_recovers_parent_gating() -> None:
    rng = np.random.default_rng(37)
    parent = rng.normal(size=20_000)
    truth = MixtureFiber(np.array([-0.3, 1.1, -0.5]), np.array([0.45, 1.7]))
    detail = truth.sample(parent, rng)
    fitted = fit_mixture_fiber(detail, parent)
    grid = np.linspace(-2.0, 2.0, 101)
    assert np.corrcoef(truth.probabilities(grid), fitted.probabilities(grid))[0, 1] > 0.98
    assert np.max(np.abs(truth.scales - fitted.scales)) < 0.1


def test_token_accounting_and_memory_are_strict() -> None:
    result = token_benchmark(shape=(32, 32), steps=20, repeats=3)
    assert result["qalt_token_updates"] < result["full_token_updates"]
    assert result["memory_ratio"] < 1.0
