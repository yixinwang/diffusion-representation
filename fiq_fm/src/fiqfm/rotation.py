from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


@dataclass(frozen=True)
class RotationUnit:
    source_active: np.ndarray
    source_residual: np.ndarray
    active: np.ndarray
    residual: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class FeatureMap:
    location: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    feature_location: np.ndarray
    feature_scale: np.ndarray

    def transform(self, active: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(active) - self.location) / self.scale
        raw = _raw_features(standardized, self.weights)
        normalized = (raw - self.feature_location) / self.feature_scale
        return np.concatenate([np.ones((len(active), 1)), normalized], axis=1)


@dataclass(frozen=True)
class RotationEstimate:
    axes: np.ndarray
    separator: np.ndarray
    commutant_eigenvalues: np.ndarray
    separator_eigenvalues: np.ndarray


def active_map(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source must have shape [n, 2]")
    first = 1.3 * source[:, 0] + 0.35 * np.tanh(source[:, 0])
    second = 1.1 * source[:, 1] + 0.40 * np.tanh(source[:, 0])
    return np.stack([first, second], axis=1)


def coefficient_functions(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=float)

    def bounded_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = np.tanh(first)
        right = np.tanh(second)
        denominator = np.sqrt(1.0 + left**2 + right**2)
        return left / denominator, right / denominator

    a1, b1 = bounded_pair(active[:, 0], active[:, 1])
    a2, b2 = bounded_pair(
        0.8 * active[:, 0] + 0.6 * active[:, 1],
        -0.6 * active[:, 0] + 0.8 * active[:, 1],
    )
    return np.stack([a1, b1, a2, b2], axis=1)


def true_covariances(active: np.ndarray) -> np.ndarray:
    coefficients = coefficient_functions(active)
    diagonal = np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]])
    covariance = np.zeros((len(active), 4, 4), dtype=float)
    covariance[:, :2, :2] = (
        np.eye(2)
        + 0.65 * coefficients[:, 0, None, None] * diagonal
        + 0.65 * coefficients[:, 1, None, None] * exchange
    )
    covariance[:, 2:, 2:] = (
        np.eye(2)
        + 0.65 * coefficients[:, 2, None, None] * diagonal
        + 0.65 * coefficients[:, 3, None, None] * exchange
    )
    return covariance


def sample_rotation_unit(n: int, seed: int) -> RotationUnit:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    source_active = rng.normal(size=(n, 2))
    source_residual = rng.normal(size=(n, 4))
    active = active_map(source_active)
    covariance = true_covariances(active)
    factors = np.linalg.cholesky(covariance)
    residual = np.einsum("nij,nj->ni", factors, source_residual)
    return RotationUnit(source_active, source_residual, active, residual, covariance)


def signed_permutation(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dimension)
    signs = rng.choice([-1.0, 1.0], size=dimension)
    return np.eye(dimension)[:, permutation] * signs[None, :]


def haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    axes, triangular = np.linalg.qr(matrix)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    return axes * signs[None, :]


def rotate_residual(residual: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.asarray(residual) @ np.asarray(rotation).T


def rotate_covariance(covariance: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk,lk->nil", rotation, covariance, rotation)


def _raw_features(standardized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    projection = standardized @ weights
    return np.concatenate(
        [
            standardized,
            standardized**2,
            (standardized[:, :1] * standardized[:, 1:2]),
            np.tanh(standardized),
            np.sin(standardized),
            np.tanh(projection),
            np.sin(projection),
        ],
        axis=1,
    )


def fit_feature_map(active: np.ndarray, n_random: int = 16, seed: int = 0) -> FeatureMap:
    active = np.asarray(active, dtype=float)
    location = active.mean(axis=0, keepdims=True)
    scale = np.maximum(active.std(axis=0, ddof=1, keepdims=True), 1e-8)
    standardized = (active - location) / scale
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(active.shape[1], n_random))
    weights /= np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)
    raw = _raw_features(standardized, weights)
    feature_location = raw.mean(axis=0, keepdims=True)
    feature_scale = np.maximum(raw.std(axis=0, ddof=1, keepdims=True), 1e-8)
    return FeatureMap(location, scale, weights, feature_location, feature_scale)


def symmetric_to_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    dimension = matrix.shape[-1]
    values = [matrix[..., index, index] for index in range(dimension)]
    values.extend(
        np.sqrt(2.0) * matrix[..., first, second]
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )
    return np.stack(values, axis=-1)


def vector_to_symmetric(vector: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    output = np.zeros((*vector.shape[:-1], dimension, dimension), dtype=float)
    cursor = 0
    for index in range(dimension):
        output[..., index, index] = vector[..., cursor]
        cursor += 1
    for first in range(dimension):
        for second in range(first + 1, dimension):
            value = vector[..., cursor] / np.sqrt(2.0)
            output[..., first, second] = value
            output[..., second, first] = value
            cursor += 1
    if cursor != vector.shape[-1]:
        raise ValueError("vector has the wrong symmetric dimension")
    return output


def fit_covariance_regression(
    features: np.ndarray,
    residual: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    outer = np.einsum("ni,nj->nij", residual, residual)
    targets = symmetric_to_vector(outer)
    penalty = np.eye(features.shape[1])
    penalty[0, 0] = 0.0
    gram = features.T @ features + ridge * penalty
    return np.linalg.solve(gram, features.T @ targets)


def predicted_covariance_contrasts(
    features: np.ndarray,
    coefficients: np.ndarray,
    dimension: int = 4,
    rank: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(features) @ np.asarray(coefficients)
    predicted -= predicted.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(predicted, full_matrices=False)
    retained = min(rank, len(singular_values))
    scale = np.sqrt(max(len(predicted) - 1, 1))
    vectors = singular_values[:retained, None] * right[:retained] / scale
    contrasts = vector_to_symmetric(vectors, dimension)
    identity = np.eye(dimension)
    contrasts -= np.trace(contrasts, axis1=1, axis2=2)[:, None, None] * identity / dimension
    return contrasts, singular_values


def _traceless_symmetric_basis(dimension: int) -> np.ndarray:
    basis: list[np.ndarray] = []
    for index in range(dimension - 1):
        diagonal = np.zeros(dimension)
        diagonal[: index + 1] = 1.0
        diagonal[index + 1] = -(index + 1)
        diagonal /= np.linalg.norm(diagonal)
        basis.append(np.diag(diagonal))
    for first in range(dimension):
        for second in range(first + 1, dimension):
            matrix = np.zeros((dimension, dimension))
            matrix[first, second] = matrix[second, first] = 1.0 / np.sqrt(2.0)
            basis.append(matrix)
    return np.stack(basis)


def commutant_gram(contrasts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimension = contrasts.shape[-1]
    basis = _traceless_symmetric_basis(dimension)
    commutators = np.stack(
        [
            np.concatenate([(contrast @ element - element @ contrast).ravel() for contrast in contrasts])
            for element in basis
        ]
    )
    return commutators @ commutators.T, basis


def learn_commutant_blocks(contrasts: np.ndarray) -> RotationEstimate:
    gram, basis = commutant_gram(np.asarray(contrasts, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    separator = np.einsum("a,aij->ij", eigenvectors[:, 0], basis)
    separator = 0.5 * (separator + separator.T)
    separator_values, separator_axes = np.linalg.eigh(separator)
    order = np.argsort(separator_values)
    axes = separator_axes[:, order]
    return RotationEstimate(axes, separator, eigenvalues, separator_values[order])


def offblock_energy(contrasts: np.ndarray, axes: np.ndarray, block_size: int = 2) -> float:
    aligned = np.einsum("ij,njk,kl->nil", axes.T, contrasts, axes)
    cross = aligned[:, :block_size, block_size:]
    numerator = 2.0 * float(np.sum(cross**2))
    trace = np.trace(contrasts, axis1=1, axis2=2)
    centered = contrasts - trace[:, None, None] * np.eye(contrasts.shape[-1]) / contrasts.shape[-1]
    denominator = float(np.sum(centered**2))
    return numerator / max(denominator, 1e-15)


def learn_pair_partition(contrasts: np.ndarray) -> np.ndarray:
    candidates = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]
    scored = []
    for blocks in candidates:
        ordering = [*blocks[0], *blocks[1]]
        axes = np.eye(4)[:, ordering]
        scored.append((offblock_energy(contrasts, axes), ordering))
    _, ordering = min(scored, key=lambda item: (item[0], item[1]))
    return np.eye(4)[:, ordering]


def predict_covariance(features: np.ndarray, coefficients: np.ndarray, dimension: int = 4) -> np.ndarray:
    return vector_to_symmetric(features @ coefficients, dimension)


def _clip_spd(matrix: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    values, axes = np.linalg.eigh(0.5 * (matrix + np.swapaxes(matrix, -1, -2)))
    values = np.clip(values, floor, ceiling)
    return np.einsum("...ij,...j,...kj->...ik", axes, values, axes)


def impose_covariance_family(
    covariance: np.ndarray,
    family: str,
    block_size: int = 2,
    floor: float = 0.05,
    ceiling: float = 20.0,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    dimension = covariance.shape[-1]
    if family == "full":
        return _clip_spd(covariance, floor, ceiling)
    output = np.zeros_like(covariance)
    if family == "diagonal":
        diagonal = np.clip(np.diagonal(covariance, axis1=-2, axis2=-1), floor, ceiling)
        indices = np.arange(dimension)
        output[..., indices, indices] = diagonal
        return output
    if family != "block":
        raise ValueError(f"unknown covariance family: {family}")
    for start in range(0, dimension, block_size):
        stop = min(start + block_size, dimension)
        output[..., start:stop, start:stop] = _clip_spd(
            covariance[..., start:stop, start:stop], floor, ceiling
        )
    return output


def rotate_batch_covariance(covariance: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk,kl->nil", axes.T, covariance, axes)


def conditional_gaussian_nll(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    if not np.all(sign > 0):
        raise ValueError("covariance must be positive definite")
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    quadratic = np.sum(residual * solved, axis=1)
    dimension = residual.shape[1]
    return 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic) / dimension


def _matrix_square_root(covariance: np.ndarray) -> np.ndarray:
    values, axes = np.linalg.eigh(covariance)
    values = np.sqrt(np.maximum(values, 0.0))
    return np.einsum("...ij,...j,...kj->...ik", axes, values, axes)


def gaussian_w2_squared(target: np.ndarray, model: np.ndarray) -> np.ndarray:
    target_root = _matrix_square_root(target)
    middle = np.einsum("nij,njk,nkl->nil", target_root, model, target_root)
    middle_root = _matrix_square_root(middle)
    distance = (
        np.trace(target, axis1=1, axis2=2)
        + np.trace(model, axis1=1, axis2=2)
        - 2.0 * np.trace(middle_root, axis1=1, axis2=2)
    )
    return np.maximum(distance, 0.0) / target.shape[-1]


def block_subspace_metrics(axes: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    truth_first = truth[:, :2]
    truth_second = truth[:, 2:]
    estimate = axes[:, :2]

    def metrics(reference: np.ndarray) -> tuple[float, float]:
        projector_error = np.linalg.norm(estimate @ estimate.T - reference @ reference.T, "fro") / np.sqrt(2.0)
        singular_values = np.linalg.svd(reference.T @ estimate, compute_uv=False)
        max_sine = np.sqrt(max(0.0, 1.0 - float(np.min(singular_values)) ** 2))
        return float(projector_error), float(max_sine)

    direct = metrics(truth_first)
    swapped = metrics(truth_second)
    best = min((direct, swapped), key=lambda value: (value[0], value[1]))
    return {"projector_error": best[0], "maximum_principal_sine": best[1]}


def population_contrasts(rotation: np.ndarray | None = None) -> np.ndarray:
    diagonal = np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = np.array([[0.0, 1.0], [1.0, 0.0]])
    contrasts = []
    for block, matrix in itertools.product(range(2), (diagonal, exchange)):
        contrast = np.zeros((4, 4))
        start = 2 * block
        contrast[start : start + 2, start : start + 2] = 0.65 * matrix
        contrasts.append(contrast)
    output = np.stack(contrasts)
    return output if rotation is None else rotate_covariance(output, rotation)
