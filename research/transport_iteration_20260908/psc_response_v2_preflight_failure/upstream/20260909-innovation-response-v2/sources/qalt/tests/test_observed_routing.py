import numpy as np

from qalt.observed_routing import (
    empirical_bernstein_upper,
    fit_conditional_model,
    image_haar,
    mixture_log_bounds,
    paired_dequantize,
    per_image_channel_log_prob,
)


def test_dequantization_is_record_keyed_and_haar_has_expected_shapes() -> None:
    images = np.arange(4 * 3 * 32 * 32, dtype=np.uint32).reshape(4, 3, 32, 32).astype(np.uint8)
    records = np.array([4, 1, 9, 2])
    first = paired_dequantize(images, records, 1100)
    second = paired_dequantize(images[[2, 0]], records[[2, 0]], 1100)
    assert np.array_equal(first[[2, 0]], second)
    coarse, detail = image_haar(first)
    assert coarse.shape == (4, 3, 16, 16)
    assert detail.shape == (4, 9, 16, 16)
    assert np.isclose(np.sum(first**2), np.sum(coarse**2) + np.sum(detail**2), rtol=1e-6)


def test_small_conditional_models_are_finite_and_copy_ties() -> None:
    rng = np.random.default_rng(71)
    images = rng.integers(0, 256, size=(32, 3, 32, 32), dtype=np.uint8)
    values = paired_dequantize(images, np.arange(32), 1100)
    coarse, detail = image_haar(values)
    model, sample_hash = fit_conditional_model(coarse, detail, 2, False, True, rng, maximum_residuals=2_000)
    scores = per_image_channel_log_prob(model, coarse, detail)
    assert len(sample_hash) == 64
    assert scores.shape == (32, 9)
    assert np.all(np.isfinite(scores))
    assert np.array_equal(scores, per_image_channel_log_prob(model, coarse, detail))


def test_empirical_bernstein_bound_respects_declared_range() -> None:
    lower_log, upper_log = mixture_log_bounds()
    width = 2.0 * (upper_log - lower_log) / 9.0
    values = np.linspace(-0.01, 0.01, 100)
    upper = empirical_bernstein_upper(values, -width, width, family_size=512, alpha=0.05)
    assert upper > np.mean(values)
