"""Observation-fitted conditional CDF flow with one observed parent per node.

Each conditional is an add-one response histogram. Linear interpolation between
context-cell centers makes the triangular transport continuous, with an exact
Jacobian almost everywhere. No smoothness or empirical-quality claim is made
for a fitted data set. Each row supplied to ``fit`` is one independent cluster
when applying the companion finite-sample learning theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri


_LOG_2PI = float(np.log(2.0 * np.pi))


def _validated_bins(bins: int) -> int:
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)):
        raise ValueError("bins must be an integer at least two")
    if bins < 2:
        raise ValueError("bins must be an integer at least two")
    return int(bins)


def _validated_parents(parents: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(parents)
    if result.shape != (dimension,) or result.dtype.kind not in "iu":
        raise ValueError("parents must be an integer vector with one entry per coordinate")
    if np.any(result < -1) or np.any(result >= np.arange(dimension)):
        raise ValueError("each parent must be -1 or an earlier coordinate index")
    return np.array(result, dtype=np.int64, copy=True)


@dataclass(frozen=True)
class ConditionalCDFFlow:
    """A full-dimensional standard-Gaussian-to-unit-cube transport.

    ``encode`` and ``decode`` return ``(values, log_abs_det)``; the determinant
    is for the returned transformation. Arbitrary leading batch dimensions are
    preserved. ``log_prob`` returns the joint density's logarithm on the open
    unit cube and minus infinity outside it. Nonfinite inputs are rejected.

    Floating-point CDF saturation is rejected explicitly rather than clipped.
    Consequently, extremely large finite Gaussian inputs may not be encodable
    as finite-precision interior unit-cube values.
    """

    parents: np.ndarray
    probabilities: tuple[np.ndarray, ...]
    bins: int = 8
    _cdfs: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bins = _validated_bins(self.bins)
        dimension = len(self.probabilities)
        if dimension < 1:
            raise ValueError("at least one coordinate is required")
        parents = _validated_parents(self.parents, dimension)
        tables, cdfs = [], []
        for parent, raw in zip(parents, self.probabilities):
            table = np.asarray(raw, dtype=np.float64)
            expected_shape = (1 if parent == -1 else bins, bins)
            if table.shape != expected_shape:
                raise ValueError(f"conditional table must have shape {expected_shape}")
            if not np.all(np.isfinite(table)) or np.any(table <= 0.0):
                raise ValueError("conditional probabilities must be finite and positive")
            if not np.allclose(table.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError("conditional probability rows must sum to one")
            cdf = np.concatenate(
                (np.zeros((table.shape[0], 1)), np.cumsum(table, axis=1)), axis=1
            )
            cdf[:, -1] = 1.0
            # Store differences of the canonical CDF so inverses and densities
            # use the same masses, including final-bin rounding.
            table = np.diff(cdf, axis=1)
            if np.any(table <= 0.0):
                raise ValueError("conditional probabilities are below CDF precision")
            table.setflags(write=False)
            cdf.setflags(write=False)
            tables.append(table)
            cdfs.append(cdf)
        parents.setflags(write=False)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "probabilities", tuple(tables))
        object.__setattr__(self, "_cdfs", tuple(cdfs))

    @property
    def dimension(self) -> int:
        return len(self.parents)

    @classmethod
    def fit(
        cls, data: np.ndarray, parents: np.ndarray, bins: int = 8
    ) -> "ConditionalCDFFlow":
        """Fit add-one counts from observed rows strictly inside (0, 1)^D.

        Parents and bins must be chosen without inspecting evaluation data.
        This method does not select a graph, learn an observation chart, or
        treat multiple correlated sites as independent training clusters.
        """
        bins = _validated_bins(bins)
        rows = np.asarray(data, dtype=np.float64)
        if rows.ndim != 2 or min(rows.shape) < 1:
            raise ValueError("data must be a nonempty [n, D] array")
        if not np.all(np.isfinite(rows)) or np.any((rows <= 0.0) | (rows >= 1.0)):
            raise ValueError("data must be finite and strictly inside the unit cube")
        parents = _validated_parents(parents, rows.shape[1])
        indices = np.minimum((rows * bins).astype(np.int64), bins - 1)
        tables = []
        for coordinate, parent in enumerate(parents):
            if parent == -1:
                counts = np.bincount(indices[:, coordinate], minlength=bins)[None, :]
            else:
                flat_indices = indices[:, parent] * bins + indices[:, coordinate]
                counts = np.bincount(flat_indices, minlength=bins * bins).reshape(
                    bins, bins
                )
            table = counts.astype(np.float64) + 1.0
            table /= table.sum(axis=1, keepdims=True)
            tables.append(table)
        return cls(parents=parents, probabilities=tuple(tables), bins=bins)

    def _as_rows(self, values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != self.dimension:
            raise ValueError(f"values must have final dimension {self.dimension}")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must be finite")
        return array.reshape(-1, self.dimension), array.shape[:-1]

    def _context(
        self, rows: np.ndarray, coordinate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parent = self.parents[coordinate]
        if parent == -1:
            index = np.zeros(rows.shape[0], dtype=np.int64)
            return index, index, np.zeros(rows.shape[0])
        position = np.clip(rows[:, parent] * self.bins - 0.5, 0.0, self.bins - 1.0)
        left = np.floor(position).astype(np.int64)
        right = np.minimum(left + 1, self.bins - 1)
        return left, right, position - left

    @staticmethod
    def _interpolate(
        table: np.ndarray,
        index: np.ndarray,
        context: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        left, right, weight = context
        return (1.0 - weight) * table[left, index] + weight * table[right, index]

    def _response(
        self, rows: np.ndarray, coordinate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        position = rows[:, coordinate] * self.bins
        index = np.minimum(position.astype(np.int64), self.bins - 1)
        context = self._context(rows, coordinate)
        mass = self._interpolate(self.probabilities[coordinate], index, context)
        start = self._interpolate(self._cdfs[coordinate], index, context)
        cdf = start + (position - index) * mass
        return cdf, self.bins * mass

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        rows, leading_shape = self._as_rows(values)
        inside = np.all((rows > 0.0) & (rows < 1.0), axis=1)
        safe_rows = np.where(inside[:, None], rows, 0.5)
        result = np.zeros(rows.shape[0])
        for coordinate in range(self.dimension):
            _, density = self._response(safe_rows, coordinate)
            result += np.log(density)
        result[~inside] = -np.inf
        return result.reshape(leading_shape)

    def encode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map unit-cube observations to Gaussian inputs, with forward log determinant."""
        rows, leading_shape = self._as_rows(values)
        if np.any((rows <= 0.0) | (rows >= 1.0)):
            raise ValueError("encode values must be strictly inside the unit cube")
        uniforms = np.empty_like(rows)
        log_density = np.zeros(rows.shape[0])
        for coordinate in range(self.dimension):
            uniforms[:, coordinate], density = self._response(rows, coordinate)
            log_density += np.log(density)
        if np.any((uniforms <= 0.0) | (uniforms >= 1.0)):
            raise ValueError("conditional CDF reached a floating-point unit-cube boundary")
        gaussian = ndtri(uniforms)
        log_base = -0.5 * (self.dimension * _LOG_2PI + np.sum(gaussian**2, axis=1))
        return (
            gaussian.reshape((*leading_shape, self.dimension)),
            (log_density - log_base).reshape(leading_shape),
        )

    def decode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map all Gaussian coordinates to observations, with inverse log determinant."""
        gaussian, leading_shape = self._as_rows(values)
        uniforms = ndtr(gaussian)
        if np.any((uniforms <= 0.0) | (uniforms >= 1.0)):
            raise ValueError("Gaussian CDF reached a floating-point unit-cube boundary")
        rows = np.empty_like(gaussian)
        log_density = np.zeros(rows.shape[0])
        for coordinate in range(self.dimension):
            context = self._context(rows, coordinate)
            lower = np.zeros(rows.shape[0], dtype=np.int64)
            upper = np.full(rows.shape[0], self.bins, dtype=np.int64)
            target = uniforms[:, coordinate]
            while np.any(upper - lower > 1):
                midpoint = (lower + upper) // 2
                threshold = self._interpolate(self._cdfs[coordinate], midpoint, context)
                choose_lower = target >= threshold
                lower = np.where(choose_lower, midpoint, lower)
                upper = np.where(choose_lower, upper, midpoint)
            mass = self._interpolate(self.probabilities[coordinate], lower, context)
            start = self._interpolate(self._cdfs[coordinate], lower, context)
            rows[:, coordinate] = (lower + (target - start) / mass) / self.bins
            log_density += np.log(self.bins * mass)
        if np.any((rows <= 0.0) | (rows >= 1.0)):
            raise ValueError("decoded values reached a floating-point unit-cube boundary")
        log_base = -0.5 * (self.dimension * _LOG_2PI + np.sum(gaussian**2, axis=1))
        return (
            rows.reshape((*leading_shape, self.dimension)),
            (log_base - log_density).reshape(leading_shape),
        )


__all__ = ["ConditionalCDFFlow"]
