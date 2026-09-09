"""Variance-normalized independent-source conditional flow matching.

For X_t=(1-t)Z+tR and s_t=sqrt((1-t)^2+t^2), integrate Y_t=X_t/s_t.
This deterministic path-coordinate change retains both endpoints and the entire
Gaussian source. Weighting its squared derivative regression by s_t^2 equals
original-velocity regression under v_theta(X,t)=a_t X+s_t w_theta(X/s_t,t),
a_t=(2t-1)/s_t^2. Raw regression loss includes irreducible conditional noise;
it is neither excess risk nor a generation-quality certificate.

When R and Z are independent standard Gaussian vectors conditional on context,
Y_t is standard Gaussian and E[dY_t/dt | Y_t,context]=0. A zero learned field
then integrates exactly at every Heun step count. This special reference property
is not a universal quality or efficiency guarantee for non-Gaussian targets.
Uses the conditional flow-matching regression construction of Lipman et al.,
Flow Matching for Generative Modeling (2022), https://arxiv.org/abs/2210.02747.
"""
from __future__ import annotations

import torch
from torch import nn
from .flow_matching import heun_integrate
from .learned_latent_flow_matching import GlobalConditionalVelocity


def gaussian_reference_path(source, target, time):
    """Return (Y_t, dY_t/dt, s_t^2); time shape N, states N x ... .

    Squared scale has singleton trailing dimensions and broadcasts over states.
    This pure function samples nothing and preserves autograd differentiation.
    """
    if source.shape != target.shape or source.ndim < 2 or len(source) == 0:
        raise ValueError('source and target need matching nonempty batch shapes')
    if source.dtype not in (torch.float32, torch.float64) or any(
        value.dtype != source.dtype or value.device != source.device
        for value in (target, time)):
        raise ValueError('states and times must share float32/64 dtype and device')
    if time.shape != (len(source),) or not bool(((time >= 0) & (time <= 1)).all()):
        raise ValueError('time must contain one value in [0,1] per example')
    if not all(bool(torch.isfinite(value).all()) for value in (source, target, time)):
        raise ValueError('path arguments must be finite')
    t = time.reshape(-1, *([1]*(source.ndim-1)))
    squared_scale = (1-t).square()+t.square()
    scale = squared_scale.sqrt()
    state = ((1-t)*source+t*target)/scale
    derivative = ((1-t)*target-t*source)/(squared_scale*scale)
    return state, derivative, squared_scale


def gaussian_reference_loss(velocity, target, context=None, generator=None,
                            *, source=None, time=None):
    """Mean scalar s_t^2-weighted regression; overrides support paired audits.

    Default independent standard Gaussian source and uniform time use only the
    supplied generator. Context is passed unchanged to the velocity network.
    """
    if source is None:
        source = torch.randn(target.shape, dtype=target.dtype, device=target.device,
                             generator=generator)
    if time is None:
        time = torch.rand(len(target), dtype=target.dtype, device=target.device,
                          generator=generator)
    state, derivative, weight = gaussian_reference_path(source, target, time)
    prediction = velocity(state, time, context)
    if prediction.shape != target.shape:
        raise ValueError('velocity must preserve state shape')
    return (weight*(prediction-derivative).square()).mean()


class GaussianReferenceResidualFlowMatching(nn.Module):
    """Residual-only comparator with the existing globally attending network.

    The caller supplies observed coarse codes for training and its own generated
    coarse codes for sampling. This module never fits or generates coarse codes.
    Source is packed NCHW, retains every residual coordinate, and is not drawn
    internally during sampling. ``steps`` means Heun steps, hence 2*steps NFEs.
    """
    def __init__(self, residual_channels=45, context_channels=3, size=8,
                 width=124, attention_heads=4):
        super().__init__()
        values = (residual_channels, context_channels, size, width, attention_heads)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError('dimensions must be positive integers')
        if width % attention_heads:
            raise ValueError('width must be divisible by attention heads')
        self.residual_channels, self.context_channels, self.size = residual_channels, context_channels, size
        self.dimension = residual_channels*size*size
        self.velocity = GlobalConditionalVelocity(residual_channels, context_channels,
                                                  size, width, attention_heads)

    def _validate(self, value, context):
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[1:] != (self.residual_channels, self.size, self.size):
            raise ValueError('expected configured packed residual NCHW array')
        if context.shape != (len(value), self.context_channels, self.size, self.size):
            raise ValueError('expected configured same-grid coarse context')
        parameter = next(self.parameters())
        if any(v.dtype != parameter.dtype or v.device != parameter.device or
               not bool(torch.isfinite(v).all()) for v in (value, context)):
            raise ValueError('finite model-matched residual and context required')

    def training_loss(self, residual, context, generator=None):
        self._validate(residual, context)
        return gaussian_reference_loss(self.velocity, residual, context, generator)

    @torch.no_grad()
    def sample_from_gaussian(self, source, context, steps=16):
        self._validate(source, context)
        result = heun_integrate(self.velocity, source, steps=steps, context=context)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('nonfinite residual integration')
        return result


__all__ = ['gaussian_reference_path', 'gaussian_reference_loss',
           'GaussianReferenceResidualFlowMatching']
