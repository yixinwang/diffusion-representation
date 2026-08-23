from __future__ import annotations

import numpy as np
from scipy import stats


def student_mean_interval(
    values: list[float] | np.ndarray, confidence: float = 0.95
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a nonempty vector")
    mean = float(array.mean())
    if len(array) == 1:
        return {"mean": mean, "low": mean, "high": mean}
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    radius = float(stats.t.ppf(0.5 + confidence / 2.0, len(array) - 1) * standard_error)
    return {"mean": mean, "low": mean - radius, "high": mean + radius}


def one_sided_mean_test(
    values: list[float] | np.ndarray, null: float, alternative: str
) -> dict[str, float | str]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("one-sided tests need at least two paired units")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    if standard_error <= np.finfo(float).eps:
        passes_direction = mean > null if alternative == "greater" else mean < null
        raw_p = 0.0 if passes_direction else 1.0
    else:
        statistic = (mean - null) / standard_error
        raw_p = float(
            stats.t.sf(statistic, len(array) - 1)
            if alternative == "greater"
            else stats.t.cdf(statistic, len(array) - 1)
        )
    return {
        "mean": mean,
        "null": float(null),
        "alternative": alternative,
        "standard_error": standard_error,
        "raw_p": raw_p,
    }


def holm_adjust(raw_p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, raw_p) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(raw_p)))
        adjusted[name] = running
    return adjusted
