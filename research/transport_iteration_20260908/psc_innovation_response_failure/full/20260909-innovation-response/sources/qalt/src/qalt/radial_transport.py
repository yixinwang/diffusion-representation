"""Invertible radial transport for the existing three-dimensional GSM density.

This is a numerical inverse-CDF transport, not a constant-cost sampler. Each
bisection iteration evaluates K radial component probabilities per vector.
No probabilities are clipped. A finite input outside numerical range, or an
unconverged root, raises explicitly instead of silently changing the law.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import erfcx, hyp1f1, logsumexp

from .rgb_block import FixedShapeGSM


@dataclass(frozen=True)
class RadialTransportResult:
    values: np.ndarray
    log_abs_det: np.ndarray
    iterations: int
    max_relative_bracket: float


def _rows(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("expected finite vectors with final dimension three")
    return array.reshape(-1, 3), array.shape[:-1]


def _chi_logs(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log CDF and survival of chi(3), retaining small and large tails."""
    r = np.asarray(radius, dtype=np.float64)
    if np.any(r < 0) or np.any(~np.isfinite(r)):
        raise FloatingPointError("radial probability argument outside numerical range")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_sf = -0.5 * r * r + np.log(erfcx(r / math.sqrt(2)) + math.sqrt(2 / math.pi) * r)
        log_cdf = np.empty_like(r)
        small = r <= 1.0
        # Integral of sqrt(2/pi) t^2 exp(-t^2/2), stable even as r -> 0.
        log_cdf[small] = (0.5 * math.log(2 / math.pi) - math.log(3)
                          + 3 * np.log(r[small])
                          + np.log(hyp1f1(1.5, 2.5, -0.5 * r[small] ** 2)))
        log_cdf[~small] = np.log(-np.expm1(log_sf[~small]))
    if np.any(np.isnan(log_sf)) or np.any(np.isnan(log_cdf)):
        raise FloatingPointError("radial probability evaluation failed")
    return log_cdf, log_sf


def _mixture_logs(model: FixedShapeGSM, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lc, ls = _chi_logs(r[:, None] / model.scales[None, :])
    lw = np.log(model.weights)[None, :]
    return logsumexp(lc + lw, axis=1), logsumexp(ls + lw, axis=1)


def _bisect(lower: np.ndarray, upper: np.ndarray, target: np.ndarray,
            use_cdf: np.ndarray, evaluate, rtol: float, max_iterations: int):
    if not (np.isfinite(rtol) and 4 * np.finfo(float).eps <= rtol < 1):
        raise ValueError("rtol must be at least four machine epsilons and below one")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if np.any(~np.isfinite(target)) or np.any(~np.isfinite(upper)) or np.any(lower <= 0):
        raise FloatingPointError("inverse-CDF bracket or target outside numerical range")
    iterations = 0
    for iterations in range(max_iterations + 1):
        width = (upper - lower) / (lower + 0.5 * (upper - lower))
        if np.all(width <= rtol):
            return lower + 0.5 * (upper - lower), iterations, float(np.max(width, initial=0))
        if iterations == max_iterations:
            break
        midpoint = lower + 0.5 * (upper - lower)
        lc, ls = evaluate(midpoint)
        # Both decisions mean the required root is above the midpoint.
        go_right = np.where(use_cdf, lc < target, ls > target)
        active = width > rtol
        lower = np.where(active & go_right, midpoint, lower)
        upper = np.where(active & ~go_right, midpoint, upper)
    raise FloatingPointError(f"radial inverse CDF did not converge in {max_iterations} iterations")


def _transform(model: FixedShapeGSM, values: np.ndarray, inverse: bool,
               rtol: float, max_iterations: int) -> RadialTransportResult:
    rows, leading = _rows(values)
    whitened = np.linalg.solve(model.cholesky, rows.T).T if inverse else rows.copy()
    radii = np.hypot(np.hypot(whitened[:, 0], whitened[:, 1]), whitened[:, 2])
    positive = radii > 0
    source_r = radii[positive]
    smin, smax = float(np.min(model.scales)), float(np.max(model.scales))
    if inverse:
        lc, ls = _mixture_logs(model, source_r)
        lo, hi = source_r / smax, source_r / smin
        evaluate = _chi_logs
    else:
        lc, ls = _chi_logs(source_r)
        lo, hi = source_r * smin, source_r * smax
        evaluate = lambda r: _mixture_logs(model, r)
    use_cdf = lc <= -math.log(2)
    target = np.where(use_cdf, lc, ls)
    solved, iterations, width = _bisect(lo, hi, target, use_cdf, evaluate, rtol, max_iterations)
    mapped = np.zeros_like(rows)
    mapped[positive] = (whitened[positive] / source_r[:, None]) * solved[:, None]
    if not inverse:
        mapped = mapped @ model.cholesky.T
    if np.any(~np.isfinite(mapped)):
        raise FloatingPointError("radial transport output outside numerical range")
    base = mapped if inverse else rows
    endpoint = rows if inverse else mapped
    normal_log_prob = -1.5 * math.log(2 * math.pi) - 0.5 * np.sum(base * base, axis=1)
    forward_log_det = normal_log_prob - model.log_prob(endpoint)
    log_det = -forward_log_det if inverse else forward_log_det
    if np.any(~np.isfinite(log_det)):
        raise FloatingPointError("radial transport log determinant outside numerical range")
    return RadialTransportResult(mapped.reshape(leading + (3,)), log_det.reshape(leading), iterations, width)


def decode_radial(model: FixedShapeGSM, base_normal: np.ndarray, *,
                  rtol: float = 1e-12, max_iterations: int = 128) -> RadialTransportResult:
    """Map standard-normal vectors to the unchanged GSM; return forward logdet.

    Positive-radius roots have certified relative bracket width <= rtol. The
    origin maps exactly to zero. Density-ratio logdet uses the solved endpoint,
    so its numerical accuracy also depends on the inverse-CDF root tolerance.
    """
    return _transform(model, base_normal, False, rtol, max_iterations)


def encode_radial(model: FixedShapeGSM, residual: np.ndarray, *,
                  rtol: float = 1e-12, max_iterations: int = 128) -> RadialTransportResult:
    """Invert decode_radial and return the inverse log determinant."""
    return _transform(model, residual, True, rtol, max_iterations)
