"""Pro16 standalone candidate, NOT installed in qalt or native validated.

Same rational-quadratic real equations, raw-parameter transform and identity
 tails as the frozen implementation. Binary64 scalar work, reflected direct
 distances, positive odds coordinates and Bernstein-form logdet. Returned values
 and logdet retain the caller dtype. This is NOT a bitwise-compatible program.
 No interior clipping, discriminant clamping, detached gradients or fallback.
 A caller MUST reject aggregate validity=false before backward or using outputs.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def knots_and_derivatives(rw, rh, rd, bound=3., mw=1e-3, mh=1e-3, md=1e-3):
    """Use the supplied dtype. Raw-to-parameter operations match the reference."""
    k = rw.shape[-1]
    def knots(raw, minimum):
        f = minimum + (1-k*minimum)*torch.softmax(raw, dim=-1)
        return torch.cat((torch.full_like(f[..., :1], -bound),
                          -bound+2*bound*torch.cumsum(f, dim=-1)[..., :-1],
                          torch.full_like(f[..., :1], bound)), dim=-1)
    xk, yk = knots(rw, mw), knots(rh, mh)
    one = torch.ones_like(xk[..., :1])
    d = torch.cat((one, md+F.softplus(rd+math.log(math.expm1(1-md))), one), -1)
    return xk, yk, d


def stable_bin(z, xl, xr, yl, yr, dl, dr, *, inverse=False):
    """Binary64 bin primitive, including both endpoints; per-element validity.

    Inputs must have identical shape/device and dtype float64. Boundaries are
    actual knots: differences are formed here, not rounded in a lower dtype.
    Invalid arguments use containment placeholders AND remain invalid. These
    placeholders never authorize acceptance, and no successful value is clipped.
    details contains masks/intermediates for forensic tests, not model metadata.
    """
    args = (z, xl, xr, yl, yr, dl, dr)
    if any(a.dtype != torch.float64 or a.shape != z.shape or a.device != z.device for a in args):
        raise ValueError('stable_bin requires equal-shape float64 tensors')
    lo, hi = (yl, yr) if inverse else (xl, xr)
    admitted = torch.stack([torch.isfinite(a) for a in args]).all(0)
    admitted = admitted & (xr > xl) & (yr > yl) & (dl > 0) & (dr > 0) & (z >= lo) & (z <= hi)
    # Invalid-only containment. Keep admitted in the final conjunction.
    z = torch.where(admitted, z, torch.full_like(z, .5))
    xl = torch.where(admitted, xl, torch.zeros_like(xl))
    yl = torch.where(admitted, yl, torch.zeros_like(yl))
    xr = torch.where(admitted, xr, torch.ones_like(xr))
    yr = torch.where(admitted, yr, torch.ones_like(yr))
    dl = torch.where(admitted, dl, torch.ones_like(dl))
    dr = torch.where(admitted, dr, torch.ones_like(dr))
    w, h = xr-xl, yr-yl
    delta = h/w
    lo, hi = (yl, yr) if inverse else (xl, xr)
    left, right = z-lo, hi-z
    reflected = right < left
    near = torch.where(reflected, right, left)
    span = h if inverse else w
    e = near/span
    ec = 1-e
    d0, d1 = torch.where(reflected, dr, dl), torch.where(reflected, dl, dr)
    scale = torch.maximum(delta, torch.maximum(d0, d1))
    s, l, r = delta/scale, d0/scale, d1/scale
    ratios_ok = torch.isfinite(scale) & (scale > 0) & torch.isfinite(delta) & (delta > 0)
    ratios_ok = ratios_ok & torch.isfinite(e) & (e >= 0) & (e <= 1) & (s > 0) & (l > 0) & (r > 0)
    disc_ok = torch.ones_like(admitted)
    root_den_ok = torch.ones_like(admitted)
    if inverse:
        beta = l*ec-r*e
        disc = beta.square()+4*s.square()*e*ec
        # Strict positivity follows from positive real slopes; zero is not accepted.
        disc_ok = torch.isfinite(disc) & (disc > 0)
        root = torch.sqrt(torch.where(disc_ok, disc, torch.ones_like(disc)))
        negative = beta < 0
        n = torch.where(negative, root-beta, 2*s*e)
        d = torch.where(negative, 2*s*ec, root+beta)
        den = n+d
        root_den_ok = torch.isfinite(n) & torch.isfinite(d) & (n >= 0) & (d >= 0) & torch.isfinite(den) & (den > 0)
        safe_den = torch.where(root_den_ok, den, torch.ones_like(den))
        t, c = n/safe_den, d/safe_den
    else:
        t, c = e, ec
    coord_ok = torch.isfinite(t) & torch.isfinite(c) & (t >= 0) & (t <= 1) & (c >= 0) & (c <= 1)
    cross = t*c
    Q = s*(t.square()+c.square())+(l+r)*cross
    A = r*t.square()+2*s*cross+l*c.square()
    terms_ok = ratios_ok & torch.isfinite(Q) & (Q > 0) & torch.isfinite(A) & (A > 0)
    qs = torch.where(terms_ok, Q, torch.ones_like(Q))
    aa = torch.where(terms_ok, A, torch.ones_like(A))
    ds = torch.where(terms_ok, delta, torch.ones_like(delta))
    ss = torch.where(terms_ok, scale, torch.ones_like(scale))
    forward_ld = 2*torch.log(ds)-torch.log(ss)+torch.log(aa)-2*torch.log(qs)
    if inverse:
        value = torch.where(reflected, xr-w*t, xl+w*t)
        ld, out_lo, out_hi = -forward_ld, xl, xr
    else:
        fraction = (s*t.square()+l*cross)/qs
        value = torch.where(reflected, yr-h*fraction, yl+h*fraction)
        ld, out_lo, out_hi = forward_ld, yl, yr
    output_ok = torch.isfinite(value) & torch.isfinite(ld) & (value >= out_lo) & (value <= out_hi)
    valid = admitted & ratios_ok & disc_ok & root_den_ok & coord_ok & terms_ok & output_ok
    details = dict(admitted=admitted, ratios_ok=ratios_ok, disc_ok=disc_ok,
                   root_den_ok=root_den_ok, coord_ok=coord_ok, terms_ok=terms_ok,
                   output_ok=output_ok, reflected=reflected, near=e, t=t, c=c, Q=Q, A=A)
    return value, ld, valid, details


def stable_spline(inputs, raw_w, raw_h, raw_d, *, inverse=False,
                  tail_bound=3., min_bin_width=1e-3,
                  min_bin_height=1e-3, min_derivative=1e-3):
    """Dense scalar spline, binary64 work and caller-dtype outputs/status.

    All original shape/configuration/input finite requirements remain. Whole
    flow callers must ALSO keep raw affine, masked-coordinate, accumulated LD,
    loss, parameter and gradient finite checks; this primitive cannot replace them.
    No new parameters/state/priors. binary64 is unconditional, not failure-selected.
    """
    params = (raw_w, raw_h, raw_d)
    if not isinstance(inputs, torch.Tensor) or inputs.dtype not in (torch.float32, torch.float64):
        raise ValueError('input must be float32/float64 tensor')
    if any(not isinstance(p, torch.Tensor) or p.dtype != inputs.dtype or p.device != inputs.device for p in params):
        raise ValueError('model parameters must share tensor dtype/device')
    if raw_w.ndim != inputs.ndim+1:
        raise ValueError('width tensor needs one appended bin dimension')
    k = raw_w.shape[-1]
    if k < 2 or raw_w.shape != inputs.shape+(k,) or raw_h.shape != raw_w.shape or raw_d.shape != inputs.shape+(k-1,):
        raise ValueError('parameter shapes must exactly match input and bins')
    if not isinstance(inverse, bool) or not math.isfinite(tail_bound) or tail_bound <= 0:
        raise ValueError('invalid direction or tail bound')
    if not (0 < min_bin_width < 1/k and 0 < min_bin_height < 1/k and 0 < min_derivative < 1):
        raise ValueError('invalid minimum spline parameters')
    finite = torch.isfinite(inputs).all() & torch.stack([torch.isfinite(p).all() for p in params]).all()
    inside = (inputs > -tail_bound) & (inputs < tail_bound)
    # Tails are evaluated with an inactive identity bin. Nonfinite original raw
    # parameters, even on tails, still fail the finite conjunction above.
    rw, rh, rd = [torch.where(inside[..., None], p, torch.zeros_like(p)).double() for p in params]
    z = torch.where(inside, inputs, torch.zeros_like(inputs)).double()
    xk, yk, derivatives = knots_and_derivatives(rw, rh, rd, tail_bound, min_bin_width, min_bin_height, min_derivative)
    widths, heights = xk[..., 1:]-xk[..., :-1], yk[..., 1:]-yk[..., :-1]
    bins_ok = (torch.isfinite(widths) & (widths > 0) & torch.isfinite(heights) & (heights > 0)).all(-1)
    deriv_ok = (torch.isfinite(derivatives) & (derivatives > 0)).all(-1)
    search = yk if inverse else xk
    index = (z[..., None] >= search[..., 1:-1]).sum(-1, keepdim=True)
    def take(a):
        return a.gather(-1, index).squeeze(-1)
    value, ld, ok, _ = stable_bin(z, take(xk), take(xk[..., 1:]), take(yk), take(yk[..., 1:]),
                                 take(derivatives), take(derivatives[..., 1:]), inverse=inverse)
    value, ld = value.to(inputs.dtype), ld.to(inputs.dtype)
    cast_ok = torch.isfinite(value) & torch.isfinite(ld)
    valid = finite & ((~inside) | (bins_ok & deriv_ok & ok & cast_ok)).all()
    return torch.where(inside, value, inputs), torch.where(inside, ld, torch.zeros_like(inputs)), valid
