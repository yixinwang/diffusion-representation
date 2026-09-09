"""Small full-tensor and Haar-conditional flow-matching pilot baselines.

Independent Gaussian/data linear paths use x_t=(1-t)z+t*x and target x-z.
The conditional flow-matching objective follows Lipman et al., Flow Matching
for Generative Modeling, ICLR 2023: https://arxiv.org/abs/2210.02747 .
These are explicit finite-capacity, fixed-Heun baselines, with no VAE and no
claim to reproduce a trained state-of-the-art image or video system. They
share the exact fixed Haar primitives used by MultiscaleSplineFlow.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .multiscale_flow import haar_split, haar_merge


class ConvolutionalVelocity(nn.Module):
    """Small time-conditioned field; output starts at zero.

    Context is a fixed conditional input throughout each ODE integration.
    Batch statistics and training/evaluation-dependent layers are absent.
    """
    def __init__(self, channels: int, context_channels: int = 0, width: int = 32):
        super().__init__()
        if channels < 1 or context_channels < 0 or width < 1:
            raise ValueError("invalid velocity width or channel count")
        self.channels, self.context_channels = channels, context_channels
        self.net = nn.Sequential(
            nn.Conv2d(channels + context_channels + 1, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels or t.shape != (x.shape[0],):
            raise ValueError("velocity requires NCHW and one time per example")
        parts = [x, t.reshape(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])]
        if self.context_channels:
            if context is None or context.shape != (x.shape[0], self.context_channels, *x.shape[2:]):
                raise ValueError("conditional velocity received incompatible context")
            parts.append(context)
        elif context is not None:
            raise ValueError("unconditional velocity received context")
        return self.net(torch.cat(parts, dim=1))


def heun_integrate(velocity, initial: torch.Tensor, steps: int = 16,
                   context: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate dx/dt=v(x,t,context) from 0 to 1 with 2*steps field calls.

    No adaptive solver, endpoint clipping, or fresh randomness is introduced.
    This helper is differentiable; model sampling wrappers use no_grad.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if initial.ndim != 4 or not initial.is_floating_point():
        raise ValueError("Heun initial state must be floating NCHW")
    state = initial
    dt = 1.0 / steps
    for index in range(steps):
        t = initial.new_full((initial.shape[0],), index / steps)
        first = velocity(state, t, context)
        predictor = state + dt * first
        tnext = initial.new_full((initial.shape[0],), (index + 1) / steps)
        second = velocity(predictor, tnext, context)
        if first.shape != state.shape or second.shape != state.shape:
            raise ValueError("velocity output must preserve state shape")
        state = state + (0.5 * dt) * (first + second)
    return state


def _squared_error(velocity, target, source, generator=None, context=None):
    time = torch.rand((target.shape[0],), device=target.device, dtype=target.dtype,
                      generator=generator)
    interpolation = time.reshape(-1, 1, 1, 1)
    state = (1 - interpolation) * source + interpolation * target
    prediction = velocity(state, time, context)
    return (prediction - (target - source)).square().sum()


def _validate_configuration(channels, size, levels, width):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (channels, size, levels, width)):
        raise ValueError("channels, size, levels, and width must be positive integers")
    if size % (2 ** levels):
        raise ValueError("size must be divisible by 2**levels")


def _validate_images(x, channels, size):
    if x.ndim != 4 or x.shape[1:] != (channels, size, size) or x.shape[0] < 1:
        raise ValueError("expected a nonempty configured NCHW batch")
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError("images must be finite float32 or float64 values")


def _validate_source(z, dimension):
    if z.ndim != 2 or z.shape[1] != dimension or z.shape[0] < 1:
        raise ValueError("source must contain every Gaussian coordinate in a nonempty batch")
    if z.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(z).all()):
        raise ValueError("Gaussian source must be finite float32 or float64")


class FullTensorFlowMatching(nn.Module):
    """Unconditional ambient FM with the same supplied full Gaussian prior.

    `levels` is accepted as shared configuration metadata and imposes the same
    size divisibility constraint; the full model does not split its state.
    Loss is averaged over examples and all scalar coordinates.
    """
    def __init__(self, channels=3, size=32, levels=2, width=32):
        super().__init__()
        _validate_configuration(channels, size, levels, width)
        self.channels, self.size, self.levels = channels, size, levels
        self.dimension = channels * size * size
        self.velocity = ConvolutionalVelocity(channels, width=width)

    def training_loss(self, x, generator=None):
        _validate_images(x, self.channels, self.size)
        source = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return _squared_error(self.velocity, x, source, generator) / x.numel()

    @torch.no_grad()
    def sample_from_gaussian(self, z, steps=16):
        _validate_source(z, self.dimension)
        initial = z.reshape(z.shape[0], self.channels, self.size, self.size)
        return heun_integrate(self.velocity, initial, steps)


class HierarchicalFlowMatching(nn.Module):
    """Learned coarse FM plus stochastic conditional detail FMs.

    Every Gaussian source coordinate is retained. The source block order is
    identical to MultiscaleSplineFlow: deepest coarse, then details from
    deepest to finest. Training uses each observed parent coarse as a teacher
    context, which is valid conditional density training. Sampling supplies
    each detail field only its own already generated parent reconstruction.
    No observed context or cached training tensor can be passed to sampling.

    The aggregate loss is total squared velocity error / (batch*dimension),
    not an equally weighted sum of per-block means. Sampling costs
    2*steps*(levels+1) field calls across the whole hierarchy. This stochastic
    latent/coarse baseline uses no compression loss and no VAE.
    """
    def __init__(self, channels=3, size=32, levels=2, width=32):
        super().__init__()
        _validate_configuration(channels, size, levels, width)
        self.channels, self.size, self.levels = channels, size, levels
        self.dimension = channels * size * size
        self.coarse_size = size // (2 ** levels)
        self.block_shapes = [(channels, self.coarse_size, self.coarse_size)] + [
            (3 * channels, self.coarse_size * 2 ** i, self.coarse_size * 2 ** i)
            for i in range(levels)
        ]
        if sum(math.prod(s) for s in self.block_shapes) != self.dimension:
            raise AssertionError("Haar hierarchy did not preserve dimension")
        self.coarse_velocity = ConvolutionalVelocity(channels, width=width)
        self.detail_velocities = nn.ModuleList([
            ConvolutionalVelocity(3 * channels, context_channels=channels, width=width)
            for _ in range(levels)
        ])

    def _source_blocks(self, z):
        return [piece.reshape(z.shape[0], *shape)
                for piece, shape in zip(z.split([math.prod(s) for s in self.block_shapes], dim=1),
                                        self.block_shapes)]

    def training_loss(self, x, generator=None):
        _validate_images(x, self.channels, self.size)
        z = torch.randn((x.shape[0], self.dimension), device=x.device,
                        dtype=x.dtype, generator=generator)
        source = self._source_blocks(z)
        coarse = x
        levels = []
        for _ in range(self.levels):
            coarse, detail = haar_split(coarse)
            levels.append((coarse, detail))
        squared = _squared_error(self.coarse_velocity, coarse, source[0], generator)
        for velocity, noise, (parent, target) in zip(self.detail_velocities, source[1:], reversed(levels)):
            squared = squared + _squared_error(velocity, target, noise, generator, context=parent)
        return squared / (x.shape[0] * self.dimension)

    @torch.no_grad()
    def sample_from_gaussian(self, z, steps=16):
        _validate_source(z, self.dimension)
        source = self._source_blocks(z)
        coarse = heun_integrate(self.coarse_velocity, source[0], steps)
        for velocity, noise in zip(self.detail_velocities, source[1:]):
            detail = heun_integrate(velocity, noise, steps, context=coarse)
            coarse = haar_merge(coarse, detail)
        return coarse


__all__ = ["ConvolutionalVelocity", "heun_integrate", "FullTensorFlowMatching",
           "HierarchicalFlowMatching"]
