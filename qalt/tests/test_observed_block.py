from __future__ import annotations

import json
import numpy as np
import pytest

from qalt.observed_block import (
    ARM_NAMES,
    BAND_COUNT,
    COLOR_COUNT,
    DETAIL_DIMENSION,
    RidgeLocation,
    blocks_to_detail,
    canonical_site_sample,
    declared_parent_masks,
    detail_to_blocks,
    diagnostic_sufficient_statistics,
    fit_observed_block_models,
    fit_scalar_gsm,
    image_haar_inverse,
    per_image_band_log_scores,
    site_band_log_scores,
)
from qalt.rgb_block import FixedShapeGSM
from qalt.observed_routing import image_haar


@pytest.fixture(scope="module")
def synthetic_fit():
    rng = np.random.default_rng(401)
    images, rows, columns = 48, 4, 4
    coarse = rng.normal(0.0, 0.35, size=(images, COLOR_COUNT, rows, columns))
    coarse_sites = np.moveaxis(coarse, 1, -1).reshape(-1, COLOR_COUNT)
    blocks = np.empty((images * rows * columns, BAND_COUNT, COLOR_COUNT))
    raw_shape = np.array([[1.0, 0.45, -0.2], [0.45, 1.3, 0.25], [-0.2, 0.25, 0.9]])
    shape = raw_shape / np.linalg.det(raw_shape) ** (1.0 / 3.0)
    truth = FixedShapeGSM(np.array([0.7, 0.3]), np.array([0.18, 0.65]), shape)
    for band in range(BAND_COUNT):
        mean = np.column_stack(
            (
                0.12 * coarse_sites[:, 0] + 0.02 * band,
                -0.09 * coarse_sites[:, 1] + 0.03 * band,
                0.07 * coarse_sites[:, 2] - 0.01 * band,
            )
        )
        blocks[:, band] = mean + truth.sample(rng.normal(size=(len(blocks), COLOR_COUNT)))
    blocks = blocks.reshape(images, rows, columns, BAND_COUNT, COLOR_COUNT)
    sample = np.arange(images * rows * columns)
    progress = []
    models = fit_observed_block_models(
        coarse,
        blocks,
        sample=sample,
        max_iterations=80,
        tolerance=1e-7,
        progress=progress.append,
    )
    assert progress[0] == f"sample:sites={len(sample)}:done"
    assert "b4:band=2:stratum=3:done" in progress
    assert "i8:coordinate=8:stratum=3:done" in progress
    assert progress[-1] == "all_models:done"
    return coarse, blocks, models


def test_explicit_block_layout_is_lossless_and_rejects_implicit_channels() -> None:
    detail = np.arange(2 * DETAIL_DIMENSION * 5 * 4, dtype=float).reshape(2, DETAIL_DIMENSION, 5, 4)
    blocks = detail_to_blocks(detail)
    assert blocks.shape == (2, 5, 4, BAND_COUNT, COLOR_COUNT)
    assert np.array_equal(blocks_to_detail(blocks), detail)
    with pytest.raises(ValueError, match="explicit"):
        blocks_to_detail(detail)

    rng = np.random.default_rng(402)
    images = rng.random((3, COLOR_COUNT, 32, 32), dtype=np.float32)
    coarse, haar_detail = image_haar(images)
    reconstructed = image_haar_inverse(coarse, detail_to_blocks(haar_detail))
    np.testing.assert_allclose(reconstructed, images, atol=2e-7, rtol=0.0)


def test_parent_masks_are_declared_acyclic_and_no_future_values_enter_predictions() -> None:
    assert declared_parent_masks("coarse") == ((),) * DETAIL_DIMENSION
    assert declared_parent_masks("within_band") == (
        (),
        (0,),
        (0, 1),
        (),
        (3,),
        (3, 4),
        (),
        (6,),
        (6, 7),
    )
    assert declared_parent_masks("full")[-1] == tuple(range(8))

    rng = np.random.default_rng(403)
    coarse = rng.normal(size=(3, COLOR_COUNT, 2, 2))
    blocks = rng.normal(size=(3, 2, 2, BAND_COUNT, COLOR_COUNT))
    for mode in ("coarse", "within_band", "full"):
        masks = declared_parent_masks(mode)
        coefficients = tuple(rng.normal(size=13 + len(mask)) for mask in masks)
        location = RidgeLocation(coefficients, masks, 1e-3)
        reference = location.predict_flat(coarse, blocks, clip=False).reshape(-1, DETAIL_DIMENSION)
        flat = blocks.reshape(-1, DETAIL_DIMENSION)
        for target, parents in enumerate(masks):
            perturbed = flat.copy()
            forbidden = [index for index in range(DETAIL_DIMENSION) if index not in parents]
            perturbed[:, forbidden] += rng.normal(10.0, 1.0, size=(len(flat), len(forbidden)))
            candidate = location.predict_flat(
                coarse,
                perturbed.reshape(blocks.shape),
                clip=False,
            ).reshape(-1, DETAIL_DIMENSION)
            np.testing.assert_array_equal(candidate[:, target], reference[:, target])

    invalid = list(declared_parent_masks("coarse"))
    invalid[0] = (0,)
    coefficients = tuple(np.zeros(13 + len(mask)) for mask in invalid)
    with pytest.raises(ValueError, match="strictly earlier"):
        RidgeLocation(coefficients, tuple(invalid), 1e-3)


def test_site_sample_is_canonical_and_scalar_em_is_exactly_monotone() -> None:
    shuffled = np.array([9, 1, 5, 3])
    assert np.array_equal(canonical_site_sample(12, shuffled), [1, 3, 5, 9])
    rng = np.random.default_rng(409)
    residual = np.concatenate((rng.normal(0.0, 0.2, 3_000), rng.normal(0.0, 0.8, 1_000)))
    fitted, diagnostics = fit_scalar_gsm(residual, 2)
    assert diagnostics.converged
    assert np.all(np.diff(diagnostics.log_likelihood) >= -1e-8)
    assert np.all(fitted.weights >= 1e-4)
    assert np.all((fitted.scales >= 0.05) & (fitted.scales <= 2.0))


def test_all_registered_scores_have_image_band_shape_and_exact_ties(synthetic_fit) -> None:
    coarse, blocks, models = synthetic_fit
    scores = per_image_band_log_scores(models, coarse, blocks)
    assert tuple(scores) == ARM_NAMES
    assert all(score.shape == (len(blocks), BAND_COUNT) for score in scores.values())
    assert all(np.all(np.isfinite(score)) for score in scores.values())
    np.testing.assert_allclose(scores["b4"], scores["e4"], atol=2e-10, rtol=2e-12)
    assert models.p4.mixtures is models.b4.mixtures
    assert models.e4.mixtures is models.b4.mixtures
    assert models.p4.fit_diagnostics is models.b4.fit_diagnostics
    assert models.z4.fit_diagnostics is models.b4.fit_diagnostics
    assert models.e4.fit_diagnostics is models.b4.fit_diagnostics
    assert models.b1.location is models.b4.location is models.b8.location
    for band in range(BAND_COUNT):
        for stratum in range(4):
            np.testing.assert_array_equal(
                models.b1.mixtures[band][stratum].shape,
                models.b4.mixtures[band][stratum].shape,
            )
            np.testing.assert_array_equal(
                models.b4.mixtures[band][stratum].shape,
                models.b8.mixtures[band][stratum].shape,
            )
            diagonal_shape = models.d4.mixtures[band][stratum].shape
            assert np.allclose(diagonal_shape, np.diag(np.diag(diagonal_shape)))

    counts = models.parameter_counts()
    assert counts["p4"]["independently_fitted"] == 0
    assert counts["z4"]["independently_fitted"] == 0
    assert counts["e4"]["independently_fitted"] == 0
    assert counts["b8"]["represented"] > counts["b4"]["represented"] > counts["b1"]["represented"]
    traces = models.fit_trace_export()
    json.dumps(traces)
    assert len(traces["b4"]["cells"]) == BAND_COUNT
    assert len(traces["b4"]["cells"][0]) == 4
    assert len(traces["i8"]["cells"]) == DETAIL_DIMENSION
    assert all(len(cell["log_likelihood"]) >= 2 for band in traces["b4"]["cells"] for cell in band)

    rng = np.random.default_rng(419)
    coarse_16 = rng.normal(size=(2, COLOR_COUNT, 16, 16))
    blocks_16 = rng.normal(size=(2, 16, 16, BAND_COUNT, COLOR_COUNT))
    site_scores = site_band_log_scores(models, coarse_16, blocks_16)
    image_scores = per_image_band_log_scores(models, coarse_16, blocks_16)
    assert site_scores["b4"].shape[1] * site_scores["b4"].shape[2] == 256
    for name in ARM_NAMES:
        np.testing.assert_array_equal(image_scores[name], np.sum(site_scores[name], axis=(1, 2)))


def test_scoring_is_bitwise_invariant_to_image_chunking(synthetic_fit) -> None:
    coarse, blocks, models = synthetic_fit
    whole = per_image_band_log_scores(models, coarse, blocks)
    one = per_image_band_log_scores(models, coarse, blocks, chunk_images=1)
    seven = per_image_band_log_scores(models, coarse, blocks, chunk_images=7)
    for name in ARM_NAMES:
        np.testing.assert_array_equal(one[name], whole[name])
        np.testing.assert_array_equal(seven[name], whole[name])


def test_synthetic_diagnostics_have_complete_sufficient_statistics(synthetic_fit) -> None:
    coarse, blocks, models = synthetic_fit
    diagnostics = diagnostic_sufficient_statistics(models, coarse, blocks)
    site_count = int(np.prod(blocks.shape[:3]))
    assert diagnostics.counts.shape == (BAND_COUNT, 4)
    assert diagnostics.per_image_counts.shape == (len(blocks), BAND_COUNT, 4)
    np.testing.assert_array_equal(diagnostics.counts, np.sum(diagnostics.per_image_counts, axis=0))
    assert np.all(np.sum(diagnostics.counts, axis=1) == site_count)
    assert diagnostics.pit_leq_counts.shape == (BAND_COUNT, 4, 19)
    np.testing.assert_array_equal(
        diagnostics.pit_leq_counts,
        np.sum(diagnostics.per_image_pit_leq_counts, axis=0),
    )
    assert np.all(np.diff(diagnostics.pit_leq_counts, axis=-1) >= 0)
    assert diagnostics.angular_second_sums.shape == (BAND_COUNT, 4, COLOR_COUNT, COLOR_COUNT)
    assert diagnostics.angular_fourth_sums.shape == diagnostics.angular_second_sums.shape
    np.testing.assert_allclose(
        diagnostics.angular_second_sums,
        np.sum(diagnostics.per_image_angular_second_sums, axis=0),
    )
    np.testing.assert_allclose(
        diagnostics.angular_fourth_sums,
        np.sum(diagnostics.per_image_angular_fourth_sums, axis=0),
    )
    assert diagnostics.responsibility_sums.shape == (BAND_COUNT, 4, 4)
    np.testing.assert_allclose(
        diagnostics.responsibility_sums,
        np.sum(diagnostics.per_image_responsibility_sums, axis=0),
    )
    np.testing.assert_allclose(np.sum(diagnostics.responsibility_sums, axis=-1), diagnostics.counts)
    assert diagnostics.heatmap_b4_minus_i8_sum.shape == (BAND_COUNT, blocks.shape[1], blocks.shape[2])
    assert diagnostics.normalized_energy_count == site_count
    assert np.sum(diagnostics.per_image_normalized_energy_count) == site_count
    np.testing.assert_allclose(
        diagnostics.normalized_energy_sum,
        np.sum(diagnostics.per_image_normalized_energy_sum, axis=0),
    )
    np.testing.assert_allclose(
        diagnostics.normalized_energy_outer_sum,
        np.sum(diagnostics.per_image_normalized_energy_outer_sum, axis=0),
    )
    np.testing.assert_allclose(
        diagnostics.heatmap_b4_minus_i8_sum,
        np.sum(diagnostics.per_image_heatmap_b4_minus_i8, axis=0),
    )
    assert diagnostics.location_prediction_count == site_count * DETAIL_DIMENSION
    assert diagnostics.score_count_per_arm == site_count * BAND_COUNT
    assert diagnostics.finite_score_counts == {name: site_count * BAND_COUNT for name in ARM_NAMES}
    assert diagnostics.b4_e4_max_abs < 2e-12
    assert diagnostics.p4_product_max_abs == 0.0
    assert diagnostics.z4_component_marginal_max_abs < 1e-14
    assert np.all(np.isfinite(diagnostics.cross_band_energy_correlation()))
    second_deviation, fourth_deviation = diagnostics.angular_max_deviations()
    assert second_deviation < 0.12
    assert fourth_deviation < 0.12
