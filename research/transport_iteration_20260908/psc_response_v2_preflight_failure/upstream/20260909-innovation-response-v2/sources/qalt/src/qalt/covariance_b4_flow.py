"""Global covariance-only flows for nine-dimensional B4 coordinates.

The fitted map is the unique symmetric, positive-definite whitening map. It is
zero mean by construction: fitting estimates a second moment, not a centered
covariance, so this layer cannot absorb or conceal a location correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DIMENSION = 9
BLOCK_SIZE = 3
MIN_EIGENVALUE = 1.0e-4
MAX_CONDITION_NUMBER = 1.0e6
LOG_2PI = float(np.log(2.0 * np.pi))

_BLOCK3_MASK = (
    np.arange(DIMENSION)[:, None] // BLOCK_SIZE
    == np.arange(DIMENSION)[None, :] // BLOCK_SIZE
)


def _as_vectors(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != DIMENSION:
        raise ValueError(f"values must have final dimension {DIMENSION}")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    leading_shape = array.shape[:-1]
    return array.reshape(-1, DIMENSION), leading_shape


def standard_normal_log_prob(values: np.ndarray) -> np.ndarray:
    """Return the standard-normal log density along the final dimension."""

    rows, leading_shape = _as_vectors(values)
    result = -0.5 * (DIMENSION * LOG_2PI + np.einsum("ni,ni->n", rows, rows))
    return result.reshape(leading_shape)


@dataclass(frozen=True)
class CovarianceB4Flow:
    """A global symmetric covariance whitening flow with immutable parameters."""

    structure: str
    covariance: np.ndarray
    whitening: np.ndarray
    coloring: np.ndarray
    eigenvalues: np.ndarray
    log_det_whitening: float

    def __post_init__(self) -> None:
        if self.structure not in {"diagonal", "block3", "full"}:
            raise ValueError("structure must be 'diagonal', 'block3', or 'full'")

        covariance = np.asarray(self.covariance, dtype=np.float64)
        whitening = np.asarray(self.whitening, dtype=np.float64)
        coloring = np.asarray(self.coloring, dtype=np.float64)
        eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        log_det = float(self.log_det_whitening)
        if covariance.shape != (DIMENSION, DIMENSION):
            raise ValueError("covariance must have shape (9, 9)")
        if whitening.shape != covariance.shape or coloring.shape != covariance.shape:
            raise ValueError("whitening and coloring must match covariance shape")
        if eigenvalues.shape != (DIMENSION,):
            raise ValueError("eigenvalues must have shape (9,)")
        arrays = (covariance, whitening, coloring, eigenvalues)
        if any(not np.all(np.isfinite(array)) for array in arrays) or not np.isfinite(
            log_det
        ):
            raise ValueError("flow parameters must be finite")
        if np.any(eigenvalues < MIN_EIGENVALUE):
            raise ValueError(f"minimum eigenvalue must be at least {MIN_EIGENVALUE:g}")
        condition = float(eigenvalues[-1] / eigenvalues[0])
        if condition > MAX_CONDITION_NUMBER:
            raise ValueError(
                f"covariance condition number exceeds {MAX_CONDITION_NUMBER:g}"
            )

        for name, matrix in (
            ("covariance", covariance),
            ("whitening", whitening),
            ("coloring", coloring),
        ):
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
                raise ValueError(f"{name} must be symmetric")
        identity = np.eye(DIMENSION, dtype=np.float64)
        if not np.allclose(whitening @ coloring, identity, rtol=1e-10, atol=1e-11):
            raise ValueError("whitening and coloring must be mutual inverses")
        if not np.allclose(
            whitening @ covariance @ whitening, identity, rtol=1e-10, atol=1e-11
        ):
            raise ValueError("whitening must map covariance to identity")
        expected_log_det = -0.5 * float(np.sum(np.log(eigenvalues)))
        if not np.isclose(log_det, expected_log_det, rtol=1e-12, atol=1e-12):
            raise ValueError("log determinant is inconsistent with eigenvalues")

        if self.structure == "diagonal":
            mask = np.eye(DIMENSION, dtype=bool)
        elif self.structure == "block3":
            mask = _BLOCK3_MASK
        else:
            mask = np.ones((DIMENSION, DIMENSION), dtype=bool)
        for name, matrix in (
            ("covariance", covariance),
            ("whitening", whitening),
            ("coloring", coloring),
        ):
            if not np.allclose(matrix[~mask], 0.0, rtol=0.0, atol=1e-13):
                raise ValueError(f"{name} violates the {self.structure} structure")

        for field_name, array in (
            ("covariance", covariance),
            ("whitening", whitening),
            ("coloring", coloring),
            ("eigenvalues", eigenvalues),
        ):
            frozen = np.array(array, dtype=np.float64, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "log_det_whitening", log_det)

    @property
    def parameter_count(self) -> int:
        """Number of independently fitted second-moment entries."""

        return {"diagonal": 9, "block3": 18, "full": 45}[self.structure]

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whiten values and return the exact forward log determinant."""

        rows, leading_shape = _as_vectors(values)
        transformed = rows @ self.whitening
        log_det = np.full(leading_shape, self.log_det_whitening, dtype=np.float64)
        return transformed.reshape((*leading_shape, DIMENSION)), log_det

    def inverse(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Color base coordinates and return the exact inverse log determinant."""

        rows, leading_shape = _as_vectors(base)
        transformed = rows @ self.coloring
        log_det = np.full(leading_shape, -self.log_det_whitening, dtype=np.float64)
        return transformed.reshape((*leading_shape, DIMENSION)), log_det

    def log_ratio(self, values: np.ndarray) -> np.ndarray:
        """Return log p_flow(values) minus log standard_normal(values)."""

        base, log_det = self.forward(values)
        return standard_normal_log_prob(base) + log_det - standard_normal_log_prob(
            values
        )


def fit_covariance_b4_flow(
    values: np.ndarray, *, structure: str = "full"
) -> CovarianceB4Flow:
    """Fit a global zero-mean second moment and its symmetric whitening map.

    No shrinkage, jitter, centering, or learned location is applied. Matrices
    failing the frozen eigenvalue or condition-number gates are rejected.
    """

    if structure not in {"diagonal", "block3", "full"}:
        raise ValueError("structure must be 'diagonal', 'block3', or 'full'")
    rows, _ = _as_vectors(values)
    if rows.shape[0] < 1:
        raise ValueError("at least one vector is required")
    second_moment = rows.T @ rows / rows.shape[0]
    second_moment = 0.5 * (second_moment + second_moment.T)
    if structure == "diagonal":
        covariance = np.diag(np.diag(second_moment))
    elif structure == "block3":
        covariance = np.where(_BLOCK3_MASK, second_moment, 0.0)
    else:
        covariance = second_moment

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("covariance eigenvalues must be finite")
    if np.any(eigenvalues < MIN_EIGENVALUE):
        raise ValueError(f"minimum eigenvalue must be at least {MIN_EIGENVALUE:g}")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if condition > MAX_CONDITION_NUMBER:
        raise ValueError(
            f"covariance condition number {condition:g} exceeds "
            f"{MAX_CONDITION_NUMBER:g}"
        )

    whitening = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    coloring = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    whitening = 0.5 * (whitening + whitening.T)
    coloring = 0.5 * (coloring + coloring.T)
    log_det = -0.5 * float(np.sum(np.log(eigenvalues)))
    return CovarianceB4Flow(
        structure=structure,
        covariance=covariance,
        whitening=whitening,
        coloring=coloring,
        eigenvalues=eigenvalues,
        log_det_whitening=log_det,
    )


__all__ = [
    "BLOCK_SIZE",
    "CovarianceB4Flow",
    "DIMENSION",
    "MAX_CONDITION_NUMBER",
    "MIN_EIGENVALUE",
    "fit_covariance_b4_flow",
    "standard_normal_log_prob",
]
