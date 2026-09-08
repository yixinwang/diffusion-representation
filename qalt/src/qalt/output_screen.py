"""Output-space rejection screen for a frozen, coupled generator pair.

This module never accepts reference/test observations. Its bound can reject
an energy-score improvement target; failure to reject supplies no evidence
of quality. The caller must justify the displacement bound and iid sampling.
"""
from __future__ import annotations

import math
import numpy as np


def displacement_screen(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    displacement_bound: float,
    margin: float,
    alpha: float = 0.05,
    family_size: int = 1,
    distance_scale: float = 1.0,
) -> dict:
    """Bound absolute population energy-score change by Hoeffding.

    Arrays have paired iid rows from frozen generators under a common source
    coupling. Distances are Euclidean divided by ``distance_scale``. The
    caller supplies an a priori almost-sure bound on those scaled distances,
    never an observed maximum. ``family_size`` is the number of prespecified
    comparisons. Adaptive generator selection requires independent new rows.
    """
    left, right = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if left.ndim < 2 or left.shape != right.shape or len(left) == 0 or left[0].size == 0:
        raise ValueError("nonempty paired arrays with identical shapes required")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("outputs must be finite")
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("distance_scale must be positive and finite")
    if not math.isfinite(displacement_bound) or displacement_bound < 0:
        raise ValueError("displacement_bound must be finite and nonnegative")
    if not math.isfinite(margin) or margin <= 0 or not 0 < alpha < 1:
        raise ValueError("positive finite margin and alpha in (0, 1) required")
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    displacement_bound, margin, alpha, distance_scale = map(float, (displacement_bound, margin, alpha, distance_scale))
    distances = np.linalg.norm((left - right).reshape(len(left), -1), axis=1) / distance_scale
    if not np.all(np.isfinite(distances)) or np.any(distances > displacement_bound):
        raise ValueError("output displacement violates the supplied bound")
    mean = float(np.mean(distances))
    radius = displacement_bound * math.sqrt(math.log(family_size / alpha) / (2 * len(left)))
    mean_upper = min(displacement_bound, mean + radius)
    score_upper = 2 * mean_upper
    return {
        "n_source_pairs": len(left), "mean_displacement": mean,
        "declared_displacement_bound": displacement_bound,
        "distance_scale": distance_scale, "alpha": alpha, "family_size": family_size,
        "mean_displacement_upper": mean_upper,
        "absolute_energy_score_change_upper": score_upper,
        "requested_margin": margin, "reject_margin": score_upper < margin,
        "quality_established": False,
        "assumptions": "finite first moments; frozen generators; iid paired sources; a priori almost-sure bound; prespecified family",
    }
