"""Positive conditional density splines fitted by convex Frank-Wolfe steps.

The context basis is the open-uniform quadratic B-spline tensor product;
response densities interpolate positive coefficients with linear hat bases.
These are established spline constructions, not a novelty claim. Basis
implementation follows SciPy's official BSpline documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

All computations are float64 NumPy/SciPy. There is no data loader, hidden
teacher input, complete image model, or optimization-success claim by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq
from scipy.sparse import csr_matrix


CONSTRAINT_TOLERANCE = 2e-12


def _configuration(bins, context_dimension, lower, upper):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer at least two")
    if isinstance(context_dimension, bool) or context_dimension not in (0, 1, 2):
        raise ValueError("context dimension must be zero, one, or two")
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < 1 < upper):
        raise ValueError("coefficient bounds must satisfy 0 < lower < 1 < upper")


def response_integral_weights(bins: int) -> np.ndarray:
    """Integrals of the b+1 equally spaced response hats on [0,1]."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer at least two")
    weights = np.full(bins + 1, 1.0 / bins)
    weights[[0, -1]] *= 0.5
    return weights


def _contexts(contexts, shape, dimension):
    if contexts is None:
        if dimension != 0:
            raise ValueError("nonzero context dimension requires context values")
        return np.empty((int(np.prod(shape, dtype=int)), 0))
    values = np.asarray(contexts, dtype=np.float64)
    if values.shape != shape + (dimension,):
        raise ValueError("context shape must equal response shape plus context axis")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("contexts must be finite and lie in [0,1]")
    return values.reshape(-1, dimension) if dimension else np.empty((int(np.prod(shape, dtype=int)), 0))


def _unit_values(values):
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError("response or probability values must be finite and lie in [0,1]")
    return array


def _context_basis(contexts: np.ndarray, bins: int) -> csr_matrix:
    """At most 3**s entries per row; no dense context-grid design matrix."""
    count, dimension = contexts.shape
    rows = np.arange(count)
    if count == 0:
        return csr_matrix((0, (bins + 2) ** dimension))
    if dimension == 0:
        return csr_matrix((np.ones(count), (rows, np.zeros(count, dtype=int))), shape=(count, 1))
    knots = np.concatenate((np.zeros(3), np.arange(1, bins) / bins, np.ones(3)))
    indices = np.zeros((count, 1), dtype=int)
    entries = np.ones((count, 1))
    for coordinate in range(dimension):
        basis = BSpline.design_matrix(contexts[:, coordinate], knots, 2, extrapolate=False)
        # Open-uniform quadratic design_matrix explicitly stores three local
        # entries per row, including boundary zeros.
        if np.any(np.diff(basis.indptr) != 3):
            raise RuntimeError("unexpected quadratic B-spline sparse structure")
        local_indices, local_entries = basis.indices.reshape(count, 3), basis.data.reshape(count, 3)
        indices = (indices[..., None] * (bins + 2) + local_indices[:, None, :]).reshape(count, -1)
        entries = (entries[..., None] * local_entries[:, None, :]).reshape(count, -1)
    result = csr_matrix((entries.ravel(), (np.repeat(rows, entries.shape[1]), indices.ravel())),
                        shape=(count, (bins + 2) ** dimension))
    result.eliminate_zeros()
    return result


def _joint_features(contexts, response, bins):
    basis = _context_basis(contexts, bins).tocoo()
    index = np.minimum((response * bins).astype(int), bins - 1)
    fraction = response * bins - index
    left_column = basis.col * (bins + 1) + index[basis.row]
    row = np.concatenate((basis.row, basis.row))
    column = np.concatenate((left_column, left_column + 1))
    data = np.concatenate((basis.data * (1 - fraction[basis.row]), basis.data * fraction[basis.row]))
    result = csr_matrix((data, (row, column)), shape=(len(response), basis.shape[1] * (bins + 1)))
    result.eliminate_zeros()
    return result


@dataclass(frozen=True)
class PositiveDensitySpline:
    """Normalized scalar conditional density with zero to two context inputs.

    coefficients has shape ((bins+2)**context_dimension, bins+1). Each row
    lies in [lower,upper] and integrates to one under the response hats. Query
    contexts have shape response.shape+(context_dimension,); None is allowed
    only for zero context. Inputs and uniforms may include 0 and 1. No
    extrapolation or probability clipping is performed. cdf is C1 in response
    and context in the unit-cube interior; density is C1 only in context.
    """
    coefficients: np.ndarray
    bins: int
    context_dimension: int
    lower: float = 0.25
    upper: float = 4.0

    def __post_init__(self):
        _configuration(self.bins, self.context_dimension, self.lower, self.upper)
        coefficients = np.array(self.coefficients, dtype=np.float64, copy=True)
        expected = ((self.bins + 2) ** self.context_dimension, self.bins + 1)
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"coefficients must be finite with shape {expected}")
        if np.any(coefficients < self.lower - CONSTRAINT_TOLERANCE) or np.any(coefficients > self.upper + CONSTRAINT_TOLERANCE):
            raise ValueError("coefficients violate the positive box constraints")
        integral = coefficients @ response_integral_weights(self.bins)
        if np.any(np.abs(integral - 1) > CONSTRAINT_TOLERANCE):
            raise ValueError("every response coefficient row must integrate to one")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)

    def _query(self, contexts, values):
        array = _unit_values(values)
        context = _contexts(contexts, array.shape, self.context_dimension)
        coefficient = _context_basis(context, self.bins) @ self.coefficients
        return array, coefficient

    def density(self, contexts, response):
        array, coefficient = self._query(contexts, response)
        flat = array.ravel()
        index = np.minimum((flat * self.bins).astype(int), self.bins - 1)
        fraction = flat * self.bins - index
        rows = np.arange(flat.size)
        density = coefficient[rows, index] * (1 - fraction) + coefficient[rows, index + 1] * fraction
        return density.reshape(array.shape)

    def log_prob(self, contexts, response):
        return np.log(self.density(contexts, response))

    def _cumulative(self, coefficient):
        mass = (coefficient[:, :-1] + coefficient[:, 1:]) / (2 * self.bins)
        return np.concatenate((np.zeros((len(coefficient), 1)), np.cumsum(mass, axis=1)), axis=1)

    def cdf(self, contexts, response):
        array, coefficient = self._query(contexts, response)
        flat = array.ravel()
        index = np.minimum((flat * self.bins).astype(int), self.bins - 1)
        distance = flat - index / self.bins
        rows = np.arange(flat.size)
        start = coefficient[rows, index]
        slope = self.bins * (coefficient[rows, index + 1] - start)
        cumulative = self._cumulative(coefficient)
        result = cumulative[rows, index] + start * distance + 0.5 * slope * distance**2
        result[flat == 0] = 0
        result[flat == 1] = 1
        return result.reshape(array.shape)

    def icdf(self, contexts, uniform):
        array, coefficient = self._query(contexts, uniform)
        flat = array.ravel()
        cumulative = self._cumulative(coefficient)
        index = np.sum(flat[:, None] >= cumulative[:, 1:-1], axis=1)
        rows = np.arange(flat.size)
        increment = flat - cumulative[rows, index]
        start = coefficient[rows, index]
        slope = self.bins * (coefficient[rows, index + 1] - start)
        discriminant = start**2 + 2 * slope * increment
        if np.any(discriminant <= 0) or not np.all(np.isfinite(discriminant)):
            raise FloatingPointError("positive density inverse lost its quadratic discriminant")
        distance = 2 * increment / (start + np.sqrt(discriminant))
        result = index / self.bins + distance
        result[flat == 0] = 0
        result[flat == 1] = 1
        if np.any((result < 0) | (result > 1)) or not np.all(np.isfinite(result)):
            raise FloatingPointError("conditional inverse is outside numerical range")
        return result.reshape(array.shape)


def bounded_row_linear_oracle(gradient, weights, lower=0.25, upper=4.0):
    """Minimize each row's linear objective under box and weighted-sum limits.

    A bounded fractional-knapsack solution allocates mass in increasing order
    of gradient/weight, with deterministic stable sorting of equal costs.
    """
    gradient, weights = np.asarray(gradient, dtype=float), np.asarray(weights, dtype=float)
    if gradient.ndim != 2 or weights.shape != (gradient.shape[1],) or not np.all(weights > 0):
        raise ValueError("invalid gradient/weight shapes")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(weights)):
        raise ValueError("oracle inputs must be finite")
    if not (0 < lower < 1 < upper) or abs(np.sum(weights) - 1) > CONSTRAINT_TOLERANCE:
        raise ValueError("oracle requires unit-sum positive weights and feasible bounds")
    order = np.argsort(gradient / weights[None], axis=1, kind="stable")
    ordered_weights = np.broadcast_to(weights, gradient.shape)[np.arange(len(gradient))[:, None], order]
    capacity = (upper - lower) * ordered_weights
    before = np.cumsum(capacity, axis=1) - capacity
    allocation = np.minimum(capacity, np.maximum(0, 1 - lower - before))
    ordered_values = lower + allocation / ordered_weights
    result = np.empty_like(gradient)
    np.put_along_axis(result, order, ordered_values, axis=1)
    return result


@dataclass(frozen=True)
class PositiveSplineFitDiagnostics:
    iterations: int
    objective: float
    frank_wolfe_gap: float
    requested_gap: float
    converged: bool
    objective_trace: tuple[float, ...]
    gap_trace: tuple[float, ...]
    step_sizes: tuple[float, ...]
    sample_count: int
    feature_nonzeros: int


def fit_positive_density_spline(contexts, response, *, bins=8, max_iterations=500,
                                gap_tolerance=1e-6, lower=0.25, upper=4.0):
    """Fit from observed [N,s] contexts and [N] responses by convex likelihood.

    Returns (model, diagnostics). No validation/test input or parameter tuning
    occurs here. Each iteration uses sparse local features, the exact bounded
    row linear oracle, and a scalar convex line search. The Frank-Wolfe gap
    bounds the empirical objective suboptimality in exact arithmetic; recorded
    floating-point gaps have ordinary numerical tolerance, not interval-
    arithmetic certification. A capped run is not converged unless its final
    gap is at or below the requested tolerance. It gives no population bound.
    """
    response = _unit_values(response)
    if response.ndim != 1 or response.size < 1:
        raise ValueError("fitting requires a nonempty response vector")
    if contexts is not None and np.asarray(contexts).ndim != 2:
        raise ValueError("fitting contexts must have shape [N,s]")
    dimension = 0 if contexts is None else np.asarray(contexts).shape[-1]
    _configuration(bins, dimension, lower, upper)
    context = _contexts(contexts, response.shape, dimension)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not np.isfinite(gap_tolerance) or gap_tolerance < 1e-12:
        raise ValueError("gap_tolerance must be finite and at least 1e-12")
    features = _joint_features(context, response, bins)
    coefficients = np.ones(((bins + 2) ** dimension, bins + 1))
    weights = response_integral_weights(bins)
    objectives, gaps, steps = [], [], []
    for iteration in range(max_iterations + 1):
        density = features @ coefficients.ravel()
        objective = -float(np.mean(np.log(density)))
        gradient = np.asarray(features.T @ (-1 / (len(response) * density))).reshape(coefficients.shape)
        vertex = bounded_row_linear_oracle(gradient, weights, lower, upper)
        direction = vertex - coefficients
        gap = -float(np.sum(gradient * direction))
        if gap < -1e-11 or not np.isfinite(gap):
            raise FloatingPointError("invalid Frank-Wolfe gap")
        gap = max(0.0, gap)
        objectives.append(objective)
        gaps.append(gap)
        if gap <= gap_tolerance or iteration == max_iterations:
            break
        change = features @ direction.ravel()
        def derivative(step):
            return -float(np.mean(change / (density + step * change)))
        if derivative(1.0) <= 0:
            step = 1.0
        else:
            step = float(brentq(derivative, 0.0, 1.0, xtol=1e-14, rtol=1e-14))
        updated = coefficients + step * direction
        new_objective = -float(np.mean(np.log(features @ updated.ravel())))
        if new_objective > objective + 1e-12:
            raise FloatingPointError("convex line search increased negative log likelihood")
        coefficients = updated
        steps.append(step)
    model = PositiveDensitySpline(coefficients, bins, dimension, lower, upper)
    diagnostics = PositiveSplineFitDiagnostics(iteration, objectives[-1], gaps[-1], gap_tolerance,
        gaps[-1] <= gap_tolerance, tuple(objectives), tuple(gaps), tuple(steps), len(response), features.nnz)
    return model, diagnostics


__all__ = ["PositiveDensitySpline", "PositiveSplineFitDiagnostics", "response_integral_weights",
           "bounded_row_linear_oracle", "fit_positive_density_spline"]
