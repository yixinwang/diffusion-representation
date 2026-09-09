"""Scalar rational-quadratic splines with analytic inverse and linear tails.

Provenance: Durkan, Bekasov, Murray and Papamakarios, Neural Spline Flows,
NeurIPS 2019, section 3.1 and appendix A. This implements their established
spline equations and makes no novelty claim.
https://papers.nips.cc/paper_files/paper/2019/file/7ac71d433f282034e088473244df8c02-Paper.pdf

PyTorch is optional for the rest of qalt: only importing this module loads it.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return transformed values and elementwise log absolute derivatives.

    For input shape S, raw widths/heights have shape S+(K,) and raw internal
    derivatives S+(K-1,). All tensors share float32/float64 dtype and device.
    No parameter broadcasting is implicit. Width/height minimums are fractions
    of the whole interval [-tail_bound, tail_bound]. Internal derivatives are
    min_derivative + softplus(raw + softplus_inverse(1-min_derivative)); the
    two endpoint derivatives are one. All-zero raw parameters initialize the
    identity (up to floating-point rounding). The returned logdet changes sign
    for inverse=True. Sum it over event dimensions in a multivariate flow.

    Outside the open interval, including both endpoints, the map is exactly
    the identity and logdet is zero. Autograd supports inputs and parameters
    within bins; matching derivatives ensure C1 transitions at knots. Extreme
    finite parameters can still exceed floating-point resolution; invalid
    finite arithmetic raises instead of concealing a failed inverse.
    """
    params = (unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
    if not isinstance(inputs, torch.Tensor) or inputs.dtype not in (torch.float32, torch.float64):
        raise ValueError("inputs must be a float32 or float64 tensor")
    if any(not isinstance(p, torch.Tensor) for p in params):
        raise ValueError("parameters must be tensors")
    if any(p.dtype != inputs.dtype or p.device != inputs.device for p in params):
        raise ValueError("all tensors must share dtype and device")
    if unnormalized_widths.ndim != inputs.ndim + 1:
        raise ValueError("widths must append one bin axis to input shape")
    bins = unnormalized_widths.shape[-1]
    expected = inputs.shape + (bins,)
    if bins < 2 or unnormalized_widths.shape != expected or unnormalized_heights.shape != expected:
        raise ValueError("widths and heights require matching input shape and at least two bins")
    if unnormalized_derivatives.shape != inputs.shape + (bins - 1,):
        raise ValueError("internal derivatives require K-1 entries per input")
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be boolean")
    if not math.isfinite(tail_bound) or tail_bound <= 0:
        raise ValueError("tail_bound must be finite and positive")
    if not (0 < min_bin_width < 1 / bins and 0 < min_bin_height < 1 / bins):
        raise ValueError("minimum bin fractions must be positive and less than 1/K")
    if not 0 < min_derivative < 1:
        raise ValueError("min_derivative must be between zero and one")
    if not bool(torch.isfinite(inputs).all()) or any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError("inputs and parameters must be finite")

    flat = inputs.reshape(-1)
    inside = (flat > -tail_bound) & (flat < tail_bound)
    # Empty indexing keeps zero parameter gradients connected for all-tail
    # batches without evaluating a rational function on out-of-domain inputs.
    x = flat[inside]
    raw_w, raw_h, raw_d = [p.reshape(-1, p.shape[-1])[inside] for p in params]
    widths = min_bin_width + (1 - bins * min_bin_width) * torch.softmax(raw_w, dim=-1)
    heights = min_bin_height + (1 - bins * min_bin_height) * torch.softmax(raw_h, dim=-1)

    def knots(fractions):
        middle = -tail_bound + 2 * tail_bound * torch.cumsum(fractions, dim=-1)[..., :-1]
        left = torch.full_like(fractions[..., :1], -tail_bound)
        right = torch.full_like(fractions[..., :1], tail_bound)
        return torch.cat((left, middle, right), dim=-1)

    xknots, yknots = knots(widths), knots(heights)
    widths = xknots[..., 1:] - xknots[..., :-1]
    heights = yknots[..., 1:] - yknots[..., :-1]
    offset = math.log(math.expm1(1 - min_derivative))
    inner_derivatives = min_derivative + F.softplus(raw_d + offset)
    endpoint = torch.ones_like(widths[..., :1])
    derivatives = torch.cat((endpoint, inner_derivatives, endpoint), dim=-1)
    search_knots = yknots if inverse else xknots
    indices = torch.sum(x[:, None] >= search_knots[..., 1:-1], dim=-1, keepdim=True)

    def take(array):
        return array.gather(-1, indices).squeeze(-1)

    xleft, yleft = take(xknots), take(yknots)
    width, height = take(widths), take(heights)
    dleft, dright = take(derivatives), take(derivatives[..., 1:])
    delta = height / width
    curvature = dleft + dright - 2 * delta
    if inverse:
        eta = (x - yleft) / height
        a = delta - dleft + eta * curvature
        b = dleft - eta * curvature
        c = -delta * eta
        # Equivalent to b^2-4ac, expressed as a sum of nonnegative
        # terms to avoid catastrophic cancellation for steep bins.
        discriminant = (dleft * (1 - eta) - dright * eta).square() + 4 * delta.square() * eta * (1 - eta)
        if bool((discriminant < 0).any()) or not bool(torch.isfinite(discriminant).all()):
            raise FloatingPointError("spline inverse quadratic has invalid discriminant")
        root = torch.sqrt(discriminant)
        # Stable positive root: avoid cancellation for either sign of b.
        # When b<0, a=delta-b>0. Mask unused denominators to avoid NaN gradients.
        negative_b = b < 0
        safe_a = torch.where(negative_b, a, torch.ones_like(a))
        safe_denominator = torch.where(negative_b, torch.ones_like(b), b + root)
        theta = torch.where(negative_b, (-b + root) / (2 * safe_a), -2 * c / safe_denominator)
    else:
        theta = (x - xleft) / width
    complement = 1 - theta
    cross = theta * complement
    denominator = delta + curvature * cross
    derivative_numerator = dright * theta.square() + 2 * delta * cross + dleft * complement.square()
    forward_logdet = 2 * torch.log(delta) + torch.log(derivative_numerator) - 2 * torch.log(denominator)
    if inverse:
        mapped = xleft + theta * width
        element_logdet = -forward_logdet
    else:
        mapped = yleft + height * (delta * theta.square() + dleft * cross) / denominator
        element_logdet = forward_logdet
    if not bool(torch.isfinite(mapped).all()) or not bool(torch.isfinite(element_logdet).all()):
        raise FloatingPointError("spline evaluation exceeded numerical range")
    outputs = flat.clone()
    logabsdet = torch.zeros_like(flat)
    outputs[inside] = mapped
    logabsdet[inside] = element_logdet
    return outputs.reshape(inputs.shape), logabsdet.reshape(inputs.shape)


__all__ = ["rational_quadratic_spline"]
