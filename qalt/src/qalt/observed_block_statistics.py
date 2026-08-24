"""Data-agnostic statistics for the frozen observed RGB-block study.

This module intentionally performs no file or dataset I/O.  It consumes
already-frozen per-seed, per-image, per-band conditional log scores.  The
repair holdout is adaptive development data, so reported intervals, tests,
and empirical-Bernstein values are diagnostic and carry no coverage claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy import stats


SEED_COUNT = 5
BAND_COUNT = 3
SITES_PER_BAND = 256
COORDINATES_PER_BAND = 3
CLASS_COUNT = 10
DETAIL_COUNT = BAND_COUNT * SITES_PER_BAND * COORDINATES_PER_BAND
ROUTE_COUNT = 1 << BAND_COUNT
BOOTSTRAP_DRAWS = 9_999
BOOTSTRAP_SEED = 20_260_824
ALPHA = 0.05
EQUIVALENCE_MARGIN = 0.01


@dataclass(frozen=True)
class GateSpec:
    name: str
    left: str
    right: str
    kind: str
    margin: float


REGISTERED_GATES = (
    GateSpec("o4_minus_b4", "o4", "b4", "superiority", 0.01),
    GateSpec("p4_minus_b4", "p4", "b4", "superiority", 0.01),
    GateSpec("b1_minus_b4", "b1", "b4", "superiority", 0.01),
    GateSpec("z4_minus_b4", "z4", "b4", "superiority", 0.01),
    GateSpec("d4_minus_b4", "d4", "b4", "superiority", 0.01),
    GateSpec("b4_unconditional_minus_b4", "b4_unconditional", "b4", "superiority", 0.0),
    GateSpec("b4_minus_b8", "b4", "b8", "equivalence", EQUIVALENCE_MARGIN),
    GateSpec("b4_minus_a8", "b4", "a8", "equivalence", EQUIVALENCE_MARGIN),
    GateSpec("b4_minus_i8", "b4", "i8", "equivalence", EQUIVALENCE_MARGIN),
    GateSpec("i4_minus_i8", "i4", "i8", "equivalence", EQUIVALENCE_MARGIN),
)


def _class_indices(labels: np.ndarray, expected_per_class: int | None = None) -> list[np.ndarray]:
    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if not np.array_equal(np.unique(values), np.arange(CLASS_COUNT)):
        raise ValueError("the frozen study requires classes 0,...,9")
    groups = [np.flatnonzero(values == class_id) for class_id in range(CLASS_COUNT)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("each class needs at least two source images")
    if expected_per_class is not None and any(len(group) != expected_per_class for group in groups):
        raise ValueError(f"each class must contain exactly {expected_per_class} images")
    return groups


def _validated_score_arrays(scores: Mapping[str, np.ndarray], required: set[str]) -> dict[str, np.ndarray]:
    missing = required - set(scores)
    if missing:
        raise ValueError(f"missing registered score arms: {sorted(missing)}")
    output: dict[str, np.ndarray] = {}
    image_count: int | None = None
    for name in required:
        values = np.asarray(scores[name], dtype=float)
        if values.ndim != 3 or values.shape[0] != SEED_COUNT or values.shape[2] != BAND_COUNT:
            raise ValueError(f"{name} scores must have shape (5, images, 3)")
        if image_count is None:
            image_count = values.shape[1]
        if values.shape[1] != image_count:
            raise ValueError("all score arms must use the same ordered images")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} scores contain nonfinite values")
        output[name] = values
    return output


def seed_averaged_nll(scores: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize band-total log scores after averaging fixed seeds inside image.

    Each input entry is the sum over all 256 sites and three RGB coordinates
    in that band, not a per-site or per-coordinate mean.
    """
    arrays = _validated_score_arrays(scores, set(scores))
    return {
        name: -np.mean(values.sum(axis=2), axis=0) / DETAIL_COUNT
        for name, values in arrays.items()
    }


def balanced_class_mean(values: np.ndarray, labels: np.ndarray) -> float:
    """Estimate the frozen uniform-over-classes target."""
    sample = np.asarray(values, dtype=float)
    groups = _class_indices(labels)
    if sample.shape != (len(labels),) or not np.all(np.isfinite(sample)):
        raise ValueError("values must contain one finite number per image")
    return float(np.mean([np.mean(sample[group]) for group in groups]))


def stratified_welch_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the frozen balanced mean and Welch--Satterthwaite uncertainty."""
    sample = np.asarray(values, dtype=float)
    groups = _class_indices(labels)
    if sample.shape != (len(labels),) or not np.all(np.isfinite(sample)):
        raise ValueError("values must contain one finite number per image")
    means = np.array([np.mean(sample[group]) for group in groups])
    components = np.array(
        [(1.0 / CLASS_COUNT) ** 2 * np.var(sample[group], ddof=1) / len(group) for group in groups]
    )
    variance = float(np.sum(components))
    if variance == 0.0:
        degrees = math.inf
    else:
        denominator = sum(component**2 / (len(group) - 1) for component, group in zip(components, groups))
        degrees = math.inf if denominator == 0.0 else variance**2 / denominator
    return {
        "mean": float(np.mean(means)),
        "variance": variance,
        "standard_error": math.sqrt(variance),
        "degrees_of_freedom": float(degrees),
    }


def superiority_pvalue(values: np.ndarray, labels: np.ndarray, margin: float) -> tuple[float, dict[str, float]]:
    summary = stratified_welch_summary(values, labels)
    if summary["standard_error"] == 0.0:
        pvalue = 0.0 if summary["mean"] > margin else 1.0
    else:
        statistic = (summary["mean"] - margin) / summary["standard_error"]
        pvalue = float(stats.t.sf(statistic, summary["degrees_of_freedom"]))
    return pvalue, summary


def equivalence_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    half_width: float = EQUIVALENCE_MARGIN,
) -> tuple[float, dict[str, float]]:
    if half_width <= 0.0:
        raise ValueError("equivalence half-width must be positive")
    summary = stratified_welch_summary(values, labels)
    if summary["standard_error"] == 0.0:
        p_lower = 0.0 if summary["mean"] > -half_width else 1.0
        p_upper = 0.0 if summary["mean"] < half_width else 1.0
    else:
        lower_statistic = (summary["mean"] + half_width) / summary["standard_error"]
        upper_statistic = (summary["mean"] - half_width) / summary["standard_error"]
        p_lower = float(stats.t.sf(lower_statistic, summary["degrees_of_freedom"]))
        p_upper = float(stats.t.cdf(upper_statistic, summary["degrees_of_freedom"]))
    summary = {**summary, "p_lower": p_lower, "p_upper": p_upper}
    return max(p_lower, p_upper), summary


def holm_adjust(pvalues: Mapping[str, float], expected_count: int = len(REGISTERED_GATES)) -> dict[str, float]:
    """Holm-adjust a frozen claim family under arbitrary dependence."""
    if len(pvalues) != expected_count:
        raise ValueError(f"expected exactly {expected_count} p-values")
    names = list(pvalues)
    raw = np.array([pvalues[name] for name in names], dtype=float)
    if np.any(~np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("p-values must be finite and lie in [0,1]")
    order = sorted(range(len(names)), key=lambda index: (raw[index], index))
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(names) - rank) * raw[index])
        adjusted_sorted.append(min(1.0, running))
    adjusted = np.empty(len(names))
    for index, value in zip(order, adjusted_sorted):
        adjusted[index] = value
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def stratified_percentile_intervals(
    values: np.ndarray,
    labels: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 64,
) -> np.ndarray:
    """Paired percentile intervals for one or more image-level contrasts."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim == 1:
        sample = sample[:, None]
    if sample.ndim != 2 or sample.shape[0] != len(labels) or not np.all(np.isfinite(sample)):
        raise ValueError("values must have shape (images, contrasts) and be finite")
    if draws <= 0 or batch_size <= 0:
        raise ValueError("draws and batch size must be positive")
    groups = _class_indices(labels)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, sample.shape[1]))
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        batch = np.zeros((stop - start, sample.shape[1]))
        for group in groups:
            indices = rng.integers(0, len(group), size=(stop - start, len(group)))
            batch += np.mean(sample[group][indices], axis=1) / CLASS_COUNT
        boot[start:stop] = batch
    return np.quantile(boot, (0.025, 0.975), axis=0).T


def evaluate_registered_gates(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Evaluate the ten frozen adaptive-development gates."""
    required = {spec.left for spec in REGISTERED_GATES} | {spec.right for spec in REGISTERED_GATES}
    arrays = _validated_score_arrays(scores, required)
    if next(iter(arrays.values())).shape[1] != len(labels):
        raise ValueError("labels do not match score images")
    _class_indices(labels)
    nll = seed_averaged_nll(arrays)
    contrast_values = np.column_stack([nll[spec.left] - nll[spec.right] for spec in REGISTERED_GATES])
    intervals = stratified_percentile_intervals(
        contrast_values,
        labels,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    raw_pvalues: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}
    for column, spec in enumerate(REGISTERED_GATES):
        if spec.kind == "superiority":
            pvalue, summary = superiority_pvalue(contrast_values[:, column], labels, spec.margin)
        else:
            pvalue, summary = equivalence_pvalue(contrast_values[:, column], labels, spec.margin)
        raw_pvalues[spec.name] = pvalue
        summaries[spec.name] = summary
    adjusted = holm_adjust(raw_pvalues)
    gates: dict[str, dict[str, object]] = {}
    for column, spec in enumerate(REGISTERED_GATES):
        estimate = summaries[spec.name]["mean"]
        effect_pass = estimate > spec.margin if spec.kind == "superiority" else abs(estimate) < spec.margin
        gates[spec.name] = {
            **summaries[spec.name],
            "kind": spec.kind,
            "margin": spec.margin,
            "bootstrap_95": [float(value) for value in intervals[column]],
            "pvalue": raw_pvalues[spec.name],
            "holm_adjusted_pvalue": adjusted[spec.name],
            "passes": bool(effect_pass and adjusted[spec.name] <= ALPHA),
        }
    return {
        "status": "adaptive_development_no_coverage",
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "gates": gates,
        "all_gates_pass": all(bool(gate["passes"]) for gate in gates.values()),
    }


def construct_route_regrets(scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return direct I8 regret for all eight B4/A8 band routes."""
    arrays = _validated_score_arrays(scores, {"b4", "a8", "i8"})
    b4, a8, i8 = arrays["b4"], arrays["a8"], arrays["i8"]
    bounds = registered_density_log_bounds()
    scalar_lower, scalar_upper = bounds["scalar"]
    block_lower, block_upper = bounds["block"]
    declared = {
        "b4": (SITES_PER_BAND * block_lower, SITES_PER_BAND * block_upper),
        "a8": (
            SITES_PER_BAND * COORDINATES_PER_BAND * scalar_lower,
            SITES_PER_BAND * COORDINATES_PER_BAND * scalar_upper,
        ),
        "i8": (
            SITES_PER_BAND * COORDINATES_PER_BAND * scalar_lower,
            SITES_PER_BAND * COORDINATES_PER_BAND * scalar_upper,
        ),
    }
    for name, values in arrays.items():
        lower, upper = declared[name]
        if np.any(values < lower - 1e-8) or np.any(values > upper + 1e-8):
            raise ValueError(f"{name} band score violates its registered density range")
    output = np.empty((ROUTE_COUNT, b4.shape[1]))
    for mask in range(ROUTE_COUNT):
        selected = np.array([bool(mask & (1 << band)) for band in range(BAND_COUNT)])
        route_score = np.where(selected[None, None, :], b4, a8).sum(axis=2)
        output[mask] = np.mean(i8.sum(axis=2) - route_score, axis=0) / DETAIL_COUNT
    return output


def registered_density_log_bounds(
    weight_floor: float = 1e-4,
    scale_floor: float = 0.05,
    scale_ceiling: float = 2.0,
    shape_eigenvalue_floor: float = 0.1,
) -> dict[str, tuple[float, float]]:
    """Return deterministic scalar and RGB-block log-density bounds."""
    if not (0.0 < weight_floor < 1.0 and 0.0 < scale_floor <= scale_ceiling):
        raise ValueError("invalid mixture bounds")
    if shape_eigenvalue_floor <= 0.0:
        raise ValueError("shape eigenvalue floor must be positive")
    scalar_lower = (
        math.log(weight_floor)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(scale_ceiling)
        - 4.0 / (2.0 * scale_floor**2)
    )
    scalar_upper = -0.5 * math.log(2.0 * math.pi) - math.log(scale_floor)
    block_lower = (
        math.log(weight_floor)
        - 1.5 * math.log(2.0 * math.pi)
        - 3.0 * math.log(scale_ceiling)
        - 12.0 / (2.0 * scale_floor**2 * shape_eigenvalue_floor)
    )
    block_upper = -1.5 * math.log(2.0 * math.pi) - 3.0 * math.log(scale_floor)
    return {"scalar": (scalar_lower, scalar_upper), "block": (block_lower, block_upper)}


def registered_route_regret_ranges() -> np.ndarray:
    """Return one deterministic normalized regret interval per route."""
    bounds = registered_density_log_bounds()
    scalar_lower, scalar_upper = bounds["scalar"]
    block_lower, block_upper = bounds["block"]
    b_lower = 3.0 * scalar_lower - block_upper
    b_upper = 3.0 * scalar_upper - block_lower
    a_lower = 3.0 * scalar_lower - 3.0 * scalar_upper
    a_upper = 3.0 * scalar_upper - 3.0 * scalar_lower
    output = np.empty((ROUTE_COUNT, 2))
    for mask in range(ROUTE_COUNT):
        joint = mask.bit_count()
        output[mask] = (
            (joint * b_lower + (BAND_COUNT - joint) * a_lower) / 9.0,
            (joint * b_upper + (BAND_COUNT - joint) * a_upper) / 9.0,
        )
    return output


def evaluate_registered_routes(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    expected_per_class: int = 500,
    alpha: float = ALPHA,
    epsilon: float = EQUIVALENCE_MARGIN,
) -> dict[str, object]:
    """Compute the frozen classwise diagnostic pseudo-EB route statistic."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    groups = _class_indices(labels, expected_per_class=expected_per_class)
    regrets = construct_route_regrets(scores)
    if regrets.shape[1] != len(labels):
        raise ValueError("labels do not match route score images")
    ranges = registered_route_regret_ranges()
    log_term = math.log(2.0 * ROUTE_COUNT * CLASS_COUNT / alpha)
    routes = []
    eligible = []
    for mask in range(ROUTE_COUNT):
        lower, upper = ranges[mask]
        class_uppers = []
        variance_radii = []
        range_radii = []
        for group in groups:
            values = regrets[mask, group]
            if np.any(values < lower - 1e-10) or np.any(values > upper + 1e-10):
                raise ValueError("route regret violates its deterministic range")
            variance = float(np.var(values, ddof=1))
            variance_radius = math.sqrt(2.0 * variance * log_term / len(group))
            range_radius = 7.0 * (upper - lower) * log_term / (3.0 * (len(group) - 1))
            variance_radii.append(variance_radius)
            range_radii.append(range_radius)
            class_uppers.append(float(np.mean(values) + variance_radius + range_radius))
        upper_bound = float(np.mean(class_uppers))
        joint_bands = mask.bit_count()
        metadata = {
            "mask": mask,
            "joint_bands": [band for band in range(BAND_COUNT) if mask & (1 << band)],
            "mean_regret": balanced_class_mean(regrets[mask], labels),
            "class_upper_bounds": class_uppers,
            "class_variance_radii": variance_radii,
            "class_range_radii": range_radii,
            "upper_pseudo_bound": upper_bound,
            "analytic_interval": [float(lower), float(upper)],
            "critical_depth": 1 if joint_bands == BAND_COUNT else 3,
            "unbatched_calls": 9 - 2 * joint_bands,
            "batched_head_invocations": 3 * int(joint_bands < BAND_COUNT) + int(joint_bands > 0),
            "eligible": bool(upper_bound <= epsilon),
        }
        routes.append(metadata)
        if metadata["eligible"]:
            eligible.append(metadata)
    selected = min(
        eligible,
        key=lambda route: (route["critical_depth"], route["unbatched_calls"], route["mask"]),
        default=None,
    )
    return {
        "status": "diagnostic_pseudo_certificate_no_coverage",
        "alpha": alpha,
        "epsilon": epsilon,
        "multiplicity_log_term": log_term,
        "routes": routes,
        "decision": "route" if selected is not None else "fallback_i8",
        "selected_mask": None if selected is None else selected["mask"],
        "selected_route": selected,
    }
