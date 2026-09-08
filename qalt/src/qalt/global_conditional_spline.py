"""Global conditional residual coupling flow on losslessly packed Haar details.

Established affine coupling and rational-quadratic spline transforms are composed
with global attention. This component makes no novelty or quality guarantee.
The coarse code is supplied by a separately charged generative model at sampling.
Its conditional likelihood is not the likelihood of that whole model.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .learned_latent_flow_matching import GlobalConditionalVelocity
from .spline import rational_quadratic_spline


class _GlobalSplineCoupling(nn.Module):
    def __init__(self, channels, context_channels, size, width, heads, bins, parity):
        super().__init__()
        self.channels, self.bins = channels, bins
        channel = torch.arange(channels)[:, None, None]
        row = torch.arange(size)[None, :, None]
        column = torch.arange(size)[None, None, :]
        self.register_buffer('mask', ((channel + row + column + parity) % 2 == 0)[None])
        self.conditioner = GlobalConditionalVelocity(channels, context_channels, size, width, heads)
        # Reuse its full-token attention features; time is fixed at zero.
        self.conditioner.output = nn.Conv2d(width, channels * (3 * bins + 1), 1)
        nn.init.zeros_(self.conditioner.output.weight)
        nn.init.zeros_(self.conditioner.output.bias)

    def forward(self, x, coarse, inverse=False):
        fixed = torch.where(self.mask, x, torch.zeros_like(x))
        raw = self.conditioner(fixed, x.new_zeros(x.shape[0]), coarse)
        raw = raw.reshape(x.shape[0], self.channels, 3*self.bins+1, *x.shape[2:]).permute(0, 1, 3, 4, 2)
        shift, log_scale = raw[..., 0], 2 * torch.tanh(raw[..., 1])
        widths = raw[..., 2:2+self.bins]
        heights = raw[..., 2+self.bins:2+2*self.bins]
        derivatives = raw[..., 2+2*self.bins:]
        if inverse:
            value, ld = rational_quadratic_spline(x, widths, heights, derivatives, inverse=True)
            value = (value-shift)*torch.exp(-log_scale)
            ld = ld-log_scale
        else:
            value = x*torch.exp(log_scale)+shift
            value, ld = rational_quadratic_spline(value, widths, heights, derivatives)
            ld = ld+log_scale
        value = torch.where(self.mask, x, value)
        ld = torch.where(self.mask, torch.zeros_like(ld), ld).flatten(1).sum(1)
        return value, ld


class GlobalConditionalSplineDecoder(nn.Module):
    """Exact conditional flow with all residual Gaussian coordinates retained.

    Inputs are packed NCHW residuals and a same-grid coarse tensor. ``decode``
    maps Gaussian residual noise to residual coefficients; ``encode`` inverts
    it. Both return values and per-sample log absolute Jacobian determinants.
    Coarse values are held fixed for these Jacobians. No observed residual or
    additional randomness is accepted during sampling. The caller must generate
    coarse values first and charge that model and the shared analysis separately.
    """
    def __init__(self, residual_channels: int, context_channels: int, size: int,
                 layers: int = 4, width: int = 32, bins: int = 8,
                 attention_heads: int = 4):
        super().__init__()
        values=(residual_channels, context_channels, size, layers, width, bins, attention_heads)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError('dimensions and layer counts must be positive integers')
        if layers < 2 or layers % 2 or bins < 2 or bins >= 1000:
            raise ValueError('use an even number of layers >=2 and 2<=bins<1000')
        if residual_channels*size*size < 2 or width % attention_heads:
            raise ValueError('need at least two residual coordinates and divisible attention width')
        self.residual_channels, self.context_channels, self.size = residual_channels, context_channels, size
        self.dimension = residual_channels*size*size
        self.layers = nn.ModuleList([
            _GlobalSplineCoupling(residual_channels, context_channels, size, width, attention_heads, bins, i%2)
            for i in range(layers)
        ])

    @property
    def parameter_counts(self):
        count = sum(p.numel() for p in self.parameters())
        return {'residual_decoder': count, 'total': count}

    def _validate(self, x, coarse):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] < 1 or x.shape[1:] != (self.residual_channels, self.size, self.size):
            raise ValueError('residual source must have the configured packed NCHW shape')
        if not isinstance(coarse, torch.Tensor) or coarse.shape != (x.shape[0], self.context_channels, self.size, self.size):
            raise ValueError('coarse context must have the configured NCHW shape')
        parameter = next(self.parameters())
        if x.dtype not in (torch.float32, torch.float64) or any(t.dtype != parameter.dtype or t.device != parameter.device for t in (x, coarse)):
            raise ValueError('residuals, coarse context and model must share float dtype and device')
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(coarse).all()):
            raise ValueError('residuals and coarse context must be finite')

    def _transform(self, x, coarse, inverse):
        self._validate(x, coarse)
        ld = x.new_zeros(x.shape[0])
        for layer in reversed(self.layers) if inverse else self.layers:
            x, increment = layer(x, coarse, inverse=inverse)
            ld = ld+increment
        return x, ld

    def decode(self, z, coarse):
        return self._transform(z, coarse, False)

    def encode(self, residual, coarse):
        return self._transform(residual, coarse, True)

    def sample_from_gaussian(self, z, coarse):
        return self.decode(z, coarse)[0]

    def log_prob(self, residual, coarse):
        z, ld = self.encode(residual, coarse)
        return -.5*(z.square()+math.log(2*math.pi)).flatten(1).sum(1)+ld


__all__ = ['GlobalConditionalSplineDecoder']
