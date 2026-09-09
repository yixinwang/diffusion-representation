"""Feasible accelerated optimization of the unchanged positive-spline family.

This module leaves the original Frank-Wolfe fitter untouched. It uses the
same features, coefficient polytope, likelihood, and final Frank-Wolfe gap.
The feasible similar-triangles construction is established accelerated
optimization (Gasnikov and Nesterov, https://arxiv.org/abs/1604.05275).
Its objective rate does not imply an equally accelerated gap rate or an
empirical runtime advantage. No data loading or population claim occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .positive_density_spline import (
    CONSTRAINT_TOLERANCE, PositiveDensitySpline, _configuration, _contexts,
    _unit_values, _joint_features, response_integral_weights,
    bounded_row_linear_oracle,
)


@dataclass(frozen=True)
class RowProjectionDiagnostics:
    multipliers: tuple[float, ...]
    row_integral_max_error: float
    box_max_violation: float
    kkt_max_residual: float
    row_count: int


def weighted_row_projection(values, weights, lower=0.25, upper=4.0):
    """Return (Euclidean projected rows, diagnostics) using sorted breakpoints.

    Each row satisfies lower<=a<=upper and weights@a=1. Its unique projection
    is clip(v-lambda*weights, lower, upper). The scalar equation is piecewise
    affine, solved by sweeping its sorted entry/exit breakpoints. This is an
    exact real-arithmetic projection, implemented in float64 with explicit
    feasibility/KKT residuals. It is not a clip-and-renormalize procedure.
    """
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or weights.shape != (values.shape[1],):
        raise ValueError("projection expects a row matrix and one weight per column")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("projection values must be finite and weights strictly positive")
    if abs(weights.sum() - 1) > CONSTRAINT_TOLERANCE or not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < 1 < upper):
        raise ValueError("unit-sum weights and feasible finite coefficient bounds required")
    projected = np.empty_like(values)
    multipliers = []
    columns = len(weights)
    for row_index, row in enumerate(values):
        entry = (row - upper) / weights
        leave = (row - lower) / weights
        points = np.concatenate((entry, leave))
        if not np.all(np.isfinite(points)):
            raise FloatingPointError("projection breakpoints exceed floating-point range")
        order = np.argsort(points, kind="stable")
        active = np.zeros(columns, dtype=bool)
        active_count = 0
        at_upper = np.ones(columns, dtype=bool)
        intercept = float(upper * weights.sum())
        slope = 0.0
        position = 0
        multiplier = None
        while position < len(order):
            left = points[order[position]]
            stop = position
            while stop < len(order) and points[order[stop]] == left:
                event = int(order[stop])
                j = event % columns
                if event < columns:
                    active[j] = True
                    active_count += 1
                    at_upper[j] = False
                    intercept += weights[j] * (row[j] - upper)
                    slope += weights[j] ** 2
                else:
                    active[j] = False
                    active_count -= 1
                    intercept += weights[j] * (lower - row[j])
                    slope -= weights[j] ** 2
                stop += 1
            right = points[order[stop]] if stop < len(order) else left
            if active_count == 0:
                if abs(intercept - 1) <= CONSTRAINT_TOLERANCE:
                    exact_constant = upper * weights[at_upper].sum() + lower * weights[~at_upper].sum()
                    if abs(exact_constant - 1) <= CONSTRAINT_TOLERANCE:
                        multiplier = float(left + 0.5 * (right - left))
                        break
                slope = 0.0
            else:
                candidate = (intercept - 1) / slope
                tolerance = 64 * np.finfo(float).eps * max(1.0, abs(left), abs(right))
                if left - tolerance <= candidate <= right + tolerance:
                    # Recompute at the selected interval to limit accumulation
                    # error from the preceding breakpoint updates.
                    at_lower = ~active & ~at_upper
                    exact_intercept = float(weights[active] @ row[active]
                        + upper * weights[at_upper].sum() + lower * weights[at_lower].sum())
                    candidate = (exact_intercept - 1) / float(weights[active] @ weights[active])
                    if not left - tolerance <= candidate <= right + tolerance:
                        raise FloatingPointError("projection active interval lost numerical consistency")
                    multiplier = float(min(right, max(left, candidate)))
                    break
            position = stop
        if multiplier is None:
            raise FloatingPointError("no feasible projection breakpoint interval found")
        projected[row_index] = np.clip(row - multiplier * weights, lower, upper)
        multipliers.append(multiplier)
    multiplier_array = np.asarray(multipliers)
    residual = projected - values + multiplier_array[:, None] * weights
    at_lower = projected <= lower
    at_upper = projected >= upper
    stationarity = np.where(at_lower, np.maximum(-residual, 0),
                            np.where(at_upper, np.maximum(residual, 0), np.abs(residual)))
    integral_error = float(np.max(np.abs(projected @ weights - 1), initial=0))
    box_violation = float(max(np.max(lower - projected, initial=0), np.max(projected - upper, initial=0)))
    kkt_error = float(np.max(stationarity, initial=0))
    if integral_error > CONSTRAINT_TOLERANCE or box_violation > CONSTRAINT_TOLERANCE:
        raise FloatingPointError("row projection failed its feasibility tolerance")
    return projected, RowProjectionDiagnostics(tuple(multipliers), integral_error, box_violation, kkt_error, len(values))


def _similar_triangles_point(x, z, accumulated_weight, lipschitz):
    if not (math.isfinite(lipschitz) and lipschitz > 0 and math.isfinite(accumulated_weight) and accumulated_weight >= 0):
        raise ValueError("positive finite L and nonnegative accumulated weight required")
    alpha = (1 + math.sqrt(1 + 4 * lipschitz * accumulated_weight)) / (2 * lipschitz)
    next_weight = accumulated_weight + alpha
    fraction = alpha / next_weight
    y = (1 - fraction) * x + fraction * z
    return y, alpha, next_weight


def _similar_triangles_update(x, z, gradient_at_y, alpha, next_weight, weights, lower, upper):
    next_z, projection = weighted_row_projection(z - alpha * gradient_at_y, weights, lower, upper)
    fraction = alpha / next_weight
    next_x = (1 - fraction) * x + fraction * next_z
    return next_x, next_z, projection


@dataclass(frozen=True)
class AcceleratedSplineFitDiagnostics:
    iterations: int
    objective: float
    frank_wolfe_gap: float
    requested_gap: float
    converged: bool
    termination: str
    objective_trace: tuple[float, ...]
    gradient_objective_trace: tuple[float, ...]
    incumbent_objective_trace: tuple[float, ...]
    gap_trace: tuple[float, ...]
    final_gap_recomputed: bool
    gradient_step_weights: tuple[float, ...]
    accumulated_weights: tuple[float, ...]
    lipschitz_constant: float
    gradient_evaluations: int
    objective_evaluations: int
    forward_feature_products: int
    transpose_feature_products: int
    projection_calls: int
    projected_row_count: int
    linear_minimizer_calls: int
    feature_construction_count: int
    curvature_column_sum_passes: int
    maximum_evaluation_feasibility_error: float
    maximum_projection_integral_error: float
    maximum_projection_kkt_residual: float
    sample_count: int
    feature_nonzeros: int
    solver: str = "feasible_similar_triangles_with_fresh_frank_wolfe_gap"


def fit_positive_density_spline_accelerated(contexts, response, *, bins=8,
        max_iterations=500, gap_tolerance=1e-6, lower=0.25, upper=4.0):
    """Fit the existing density family; return a model and charged diagnostics.

    Input contracts match the original fitter. max_iterations counts projected
    updates, not evaluations. Every gradient point is a feasible convex
    combination. A training-objective incumbent is retained because the raw
    accelerated objective need not decrease monotonically. Success requires
    the ORIGINAL Frank-Wolfe gap at the returned point <= gap_tolerance; all
    returns recompute that gap. No step-size or optimizer-status surrogate is
    accepted. Counts include initial/monitoring/final evaluations and the
    final independent linear minimization. Pooled response count is not an
    assertion of independent statistical samples.
    """
    response = _unit_values(response)
    if response.ndim != 1 or response.size < 1:
        raise ValueError("fitting requires a nonempty response vector")
    if contexts is not None and np.asarray(contexts).ndim != 2:
        raise ValueError("fitting contexts require shape [N,s]")
    dimension = 0 if contexts is None else np.asarray(contexts).shape[-1]
    _configuration(bins, dimension, lower, upper)
    context = _contexts(contexts, response.shape, dimension)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not np.isfinite(gap_tolerance) or gap_tolerance < 1e-12:
        raise ValueError("gap tolerance must be finite and at least 1e-12")
    features = _joint_features(context, response, bins)
    weights = response_integral_weights(bins)
    shape = ((bins + 2) ** dimension, bins + 1)
    x = np.ones(shape)
    z = x.copy()
    accumulated = 0.0
    column_sums = np.asarray(features.sum(axis=0)).ravel()
    # A small upward rounding factor is numerical headroom, not an interval
    # proof. The underlying bound is ||F||_2^2 <= ||F||_1 ||F||_infinity.
    lipschitz = float(np.max(column_sums) / (len(response) * lower**2)) * (1 + 64 * np.finfo(float).eps)
    counts = {"gradient": 0, "objective": 0, "forward": 0, "transpose": 0,
              "projection": 0, "rows": 0, "linear": 0}
    feasibility_max = 0.0
    projection_integral_max = projection_kkt_max = 0.0

    def evaluate(point, with_gradient=False):
        nonlocal feasibility_max
        if not np.all(np.isfinite(point)):
            raise FloatingPointError("nonfinite accelerated evaluation point")
        violation = float(max(np.max(lower - point), np.max(point - upper),
                              np.max(np.abs(point @ weights - 1)), 0))
        feasibility_max = max(feasibility_max, violation)
        if violation > CONSTRAINT_TOLERANCE:
            raise FloatingPointError("accelerated likelihood evaluated outside the feasible polytope")
        density = features @ point.ravel()
        counts["forward"] += 1
        counts["objective"] += 1
        if np.any(density <= 0) or not np.all(np.isfinite(density)):
            raise FloatingPointError("invalid accelerated density evaluation")
        objective = -float(np.mean(np.log(density)))
        if not with_gradient:
            return objective
        gradient = np.asarray(features.T @ (-1 / (len(response) * density))).reshape(shape)
        counts["gradient"] += 1
        counts["transpose"] += 1
        return objective, gradient

    def gap_at(point, gradient):
        vertex = bounded_row_linear_oracle(gradient, weights, lower, upper)
        counts["linear"] += 1
        gap = float(np.sum(gradient * (point - vertex)))
        if not np.isfinite(gap) or gap < -1e-11:
            raise FloatingPointError("invalid fresh Frank-Wolfe gap")
        return max(0.0, gap)

    initial_objective = evaluate(x)
    best, best_objective = x.copy(), initial_objective
    objective_trace, gradient_objectives = [initial_objective], []
    incumbents, gaps, alphas, accumulated_values = [initial_objective], [], [], []
    returned = None
    updates = 0
    for _ in range(max_iterations):
        y, alpha, next_weight = _similar_triangles_point(x, z, accumulated, lipschitz)
        objective_y, gradient = evaluate(y, with_gradient=True)
        gap_y = gap_at(y, gradient)
        gradient_objectives.append(objective_y)
        gaps.append(gap_y)
        if objective_y < best_objective:
            best, best_objective = y.copy(), objective_y
        if gap_y <= gap_tolerance:
            returned = y.copy()
            incumbents.append(best_objective)
            break
        x, z, projection = _similar_triangles_update(x, z, gradient, alpha, next_weight, weights, lower, upper)
        counts["projection"] += 1
        counts["rows"] += shape[0]
        projection_integral_max = max(projection_integral_max, projection.row_integral_max_error)
        projection_kkt_max = max(projection_kkt_max, projection.kkt_max_residual)
        accumulated = next_weight
        alphas.append(alpha)
        accumulated_values.append(accumulated)
        objective_x = evaluate(x)
        objective_trace.append(objective_x)
        if objective_x < best_objective:
            best, best_objective = x.copy(), objective_x
        incumbents.append(best_objective)
        updates += 1
    if returned is None:
        returned = best
    # Always evaluate the actual returned coefficients again. A gap at the
    # latest y or x cannot certify a different incumbent.
    final_objective, final_gradient = evaluate(returned, with_gradient=True)
    final_gap = gap_at(returned, final_gradient)
    converged = final_gap <= gap_tolerance
    model = PositiveDensitySpline(returned, bins, dimension, lower, upper)
    diagnostics = AcceleratedSplineFitDiagnostics(
        iterations=updates, objective=final_objective, frank_wolfe_gap=final_gap,
        requested_gap=gap_tolerance, converged=converged,
        termination="gap_satisfied" if converged else "projected_update_cap",
        objective_trace=tuple(objective_trace), gradient_objective_trace=tuple(gradient_objectives),
        incumbent_objective_trace=tuple(incumbents), gap_trace=tuple(gaps), final_gap_recomputed=True,
        gradient_step_weights=tuple(alphas), accumulated_weights=tuple(accumulated_values),
        lipschitz_constant=lipschitz, gradient_evaluations=counts["gradient"],
        objective_evaluations=counts["objective"], forward_feature_products=counts["forward"],
        transpose_feature_products=counts["transpose"], projection_calls=counts["projection"],
        projected_row_count=counts["rows"], linear_minimizer_calls=counts["linear"],
        feature_construction_count=1, curvature_column_sum_passes=1,
        maximum_evaluation_feasibility_error=feasibility_max,
        maximum_projection_integral_error=projection_integral_max,
        maximum_projection_kkt_residual=projection_kkt_max,
        sample_count=len(response), feature_nonzeros=features.nnz,
    )
    return model, diagnostics


__all__ = ["weighted_row_projection", "RowProjectionDiagnostics", "AcceleratedSplineFitDiagnostics",
           "fit_positive_density_spline_accelerated"]
