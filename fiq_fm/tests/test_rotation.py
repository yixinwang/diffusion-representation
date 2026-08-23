import numpy as np

from fiqfm.rotation import (
    block_subspace_metrics,
    commutant_gram,
    haar_orthogonal,
    learn_commutant_blocks,
    learn_pair_partition,
    population_contrasts,
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
