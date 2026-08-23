import numpy as np

from fiqfm.rotation import (
    block_subspace_metrics,
    cell_covariance_contrasts,
    commutant_gram,
    cross_fitted_covariance_contrasts,
    haar_orthogonal,
    fit_feature_map,
    fit_balanced_cell_partition,
    learn_commutant_blocks,
    learn_pair_partition,
    population_contrasts,
    predicted_covariance_contrasts,
    rotate_residual,
    sample_rotation_unit,
    signed_permutation,
    true_covariances,
)


def test_rotation_target_is_positive_and_marginally_tied() -> None:
    unit = sample_rotation_unit(80_000, 3)
    minimum = np.linalg.eigvalsh(unit.covariance).min()
    assert minimum > 0.35
    assert np.max(np.abs(unit.covariance.mean(axis=0) - np.eye(4))) < 0.01
    standardized = (unit.active[:, 0] - unit.active[:, 0].mean()) / unit.active[:, 0].std()
    assert abs(np.mean(standardized**4) - 3.0) > 0.02


def test_commutant_spectral_jbd_recovers_haar_blocks() -> None:
    rotation = haar_orthogonal(4, 11)
    estimate = learn_commutant_blocks(population_contrasts(rotation))
    metrics = block_subspace_metrics(estimate.axes, rotation)
    assert metrics["projector_error"] < 1e-10
    assert metrics["maximum_principal_sine"] < 1e-7
    assert estimate.commutant_eigenvalues[1] > 0.1


def test_permutation_search_recovers_permuted_blocks() -> None:
    rotation = signed_permutation(4, 17)
    axes = learn_pair_partition(population_contrasts(rotation))
    metrics = block_subspace_metrics(axes, rotation)
    assert metrics["projector_error"] < 1e-12


def test_one_contrast_is_not_block_identifying() -> None:
    contrast = np.diag([1.0, 2.0, 3.0, 4.0])[None]
    gram, _ = commutant_gram(contrast)
    eigenvalues = np.linalg.eigvalsh(gram)
    assert np.sum(eigenvalues < 1e-10) >= 3


def test_commuting_contrasts_are_not_block_identifying() -> None:
    contrasts = np.stack(
        [np.diag([1.0, 2.0, 3.0, 4.0]), np.diag([4.0, 1.0, 2.0, 3.0])]
    )
    gram, _ = commutant_gram(contrasts)
    assert np.sum(np.linalg.eigvalsh(gram) < 1e-10) >= 3


def test_equivalent_noncommuting_blocks_are_not_identifying() -> None:
    diagonal = np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]])
    contrasts = np.stack(
        [
            np.block([[diagonal, np.zeros((2, 2))], [np.zeros((2, 2)), diagonal]]),
            np.block([[exchange, np.zeros((2, 2))], [np.zeros((2, 2)), exchange]]),
        ]
    )
    gram, _ = commutant_gram(contrasts)
    assert np.sum(np.linalg.eigvalsh(gram) < 1e-10) >= 2


def test_covariance_formula_matches_sampler_dimension() -> None:
    unit = sample_rotation_unit(12, 5)
    expected = true_covariances(unit.active)
    assert unit.residual.shape == (12, 4)
    assert np.array_equal(unit.covariance, expected)


def test_predicted_contrasts_ignore_feature_reparameterization() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(200, 7))
    coefficients = rng.normal(size=(7, 10))
    change = rng.normal(size=(7, 7)) + 4.0 * np.eye(7)
    original, original_values = predicted_covariance_contrasts(features, coefficients)
    transformed, transformed_values = predicted_covariance_contrasts(
        features @ change, np.linalg.solve(change, coefficients)
    )
    assert np.allclose(original_values, transformed_values)
    assert np.allclose(commutant_gram(original)[0], commutant_gram(transformed)[0])


def test_finite_predicted_contrasts_recover_haar_blocks() -> None:
    train = sample_rotation_unit(12_000, 1)
    rotation = haar_orthogonal(4, 6)
    feature_map = fit_feature_map(train.active, n_random=2, seed=4)
    contrasts, eigenvalues = cross_fitted_covariance_contrasts(
        feature_map.transform(train.active),
        rotate_residual(train.residual, rotation),
        ridge=1.0,
    )
    estimate = learn_commutant_blocks(contrasts)
    metrics = block_subspace_metrics(estimate.axes, rotation)
    assert metrics["projector_error"] < 0.1
    assert (eigenvalues[3] - eigenvalues[4]) / eigenvalues[3] > 0.1


def test_balanced_cells_use_train_only_thresholds() -> None:
    train = sample_rotation_unit(12_000, 43)
    validation = sample_rotation_unit(4_000, 47)
    partition = fit_balanced_cell_partition(train.active, depth=4)
    train_cells = partition.transform(train.active)
    assert np.ptp(np.bincount(train_cells)) <= 1
    contrasts, counts = cell_covariance_contrasts(
        validation.active, validation.residual, partition
    )
    assert contrasts.shape == (16, 4, 4)
    assert counts.min() > 50
