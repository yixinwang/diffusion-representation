"""Independently trained learned-analysis latent FM with a joint residual FM.

This is a strong finite-capacity comparator, not a semantic-code or quality
guarantee. The analysis is trained first and then frozen. Its entire fitting
and inversion cost belongs to this model. All D Gaussian source coordinates
are retained, including the stochastic residual decoder's noise.

Only the analysis has a tractable exact likelihood. Finite-step Heun samples
from the composed model must not be assigned that analysis likelihood.
"""

from __future__ import annotations

import math
from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F

from .flow_matching import (
    FullTensorFlowMatching,
    _squared_error,
    _validate_configuration,
    _validate_images,
    _validate_source,
    heun_integrate,
)
from .multiscale_flow import CouplingStack, MultiscaleSplineFlow


def pack_residuals(blocks: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Permute deepest-to-finest Haar-shaped detail blocks onto the coarse grid."""
    if not blocks or blocks[0].ndim != 4:
        raise ValueError("residual blocks must be a nonempty sequence of NCHW tensors")
    first = blocks[0]
    batch, channels, height, width = first.shape
    if channels < 3 or channels % 3 or min(batch, height, width) < 1:
        raise ValueError("invalid first residual block shape")
    packed = []
    for level, block in enumerate(blocks):
        factor = 2**level
        if block.shape != (batch, channels, height * factor, width * factor):
            raise ValueError("residual blocks must follow the declared dyadic pyramid")
        if block.dtype != first.dtype or block.device != first.device:
            raise ValueError("residual block dtype and device must agree")
        packed.append(F.pixel_unshuffle(block, factor))
    return torch.cat(packed, dim=1)


def unpack_residuals(
    packed: torch.Tensor, channels: int, levels: int
) -> tuple[torch.Tensor, ...]:
    """Invert ``pack_residuals`` without interpolation, averaging, or extra noise."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (channels, levels)):
        raise ValueError("channels and levels must be positive integers")
    if packed.ndim != 4 or min(packed.shape) < 1:
        raise ValueError("packed residuals must be nonempty NCHW")
    sizes = [3 * channels * 4**level for level in range(levels)]
    if packed.shape[1] != sum(sizes):
        raise ValueError("packed residual channel count loses or adds coordinates")
    return tuple(F.pixel_shuffle(block, 2**level)
                 for level, block in enumerate(packed.split(sizes, dim=1)))


class GlobalConditionalVelocity(nn.Module):
    """Nonlinear conditional field with all coarse-grid tokens in each evaluation.

    State, generated/observed coarse code, time, and fixed spatial coordinates
    enter the feature map. Attention has zero dropout and no batch statistics.
    Its final output starts at zero, matching the other FM implementations.
    """

    def __init__(self, channels: int, context_channels: int, size: int,
                 width: int = 32, heads: int = 4):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (channels, context_channels, size, width, heads)):
            raise ValueError("field dimensions and attention heads must be positive integers")
        if width % heads:
            raise ValueError("velocity width must be divisible by attention heads")
        self.channels, self.context_channels, self.size = channels, context_channels, size
        self.input = nn.Conv2d(channels + context_channels + 3, width, 3, padding=1)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        axis = torch.linspace(-1.0, 1.0, size)
        row, column = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("position", torch.stack((row, column))[None])

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        expected = (self.channels, self.size, self.size)
        if x.ndim != 4 or x.shape[1:] != expected or t.shape != (x.shape[0],):
            raise ValueError("global velocity requires configured NCHW and one time per sample")
        if context is None or context.shape != (
            x.shape[0], self.context_channels, self.size, self.size
        ):
            raise ValueError("global velocity requires its configured coarse context")
        clock = t[:, None, None, None].expand(-1, 1, self.size, self.size)
        position = self.position.expand(x.shape[0], -1, -1, -1)
        features = F.silu(self.input(torch.cat((x, context, clock, position), dim=1)))
        tokens = features.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        features = (tokens + attended).transpose(1, 2).reshape_as(features)
        return self.output(F.silu(self.local(features)))


class LearnedLatentFlowMatching(nn.Module):
    """Two-stage learned invertible analysis plus coarse and joint-residual FMs.

    ``train_analysis_loss`` fits full Gaussian likelihood through learned
    pre-Haar coupling and the multiscale analysis. ``freeze_analysis`` ends
    that stage. ``training_loss`` subsequently fits only the two FM fields.
    ``sample_from_gaussian`` accepts exactly D coordinates and no observed
    context. All source coordinates participate in generation.

    Optional unit-interval inputs use an exact outer logit, outside learned
    pre-Haar mixing. The fixed preprocessing must be shared by comparisons.
    """

    def __init__(self, channels: int = 3, size: int = 32, levels: int = 2,
                 pre_layers: int = 2, coarse_layers: int = 6,
                 detail_layers: int = 4, width: int = 32, bins: int = 8,
                 attention_heads: int = 4, unit_interval: bool = False):
        super().__init__()
        _validate_configuration(channels, size, levels, width)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2
               for v in (pre_layers, coarse_layers, detail_layers, bins)):
            raise ValueError("coupling layer counts and bins must be integers at least two")
        coarse_size = size // 2**levels
        if coarse_size < 2 or coarse_size % 2:
            raise ValueError("levels must leave an even coarse grid of side at least two")
        self.channels, self.size, self.levels = channels, size, levels
        self.coarse_size, self.unit_interval = coarse_size, bool(unit_interval)
        self.dimension = channels * size * size
        self.latent_dimension = channels * coarse_size * coarse_size
        self.residual_dimension = self.dimension - self.latent_dimension
        self.packed_residual_channels = channels * (4**levels - 1)
        self.pre_analysis = CouplingStack(channels, 0, pre_layers, width, bins)
        self.analysis = MultiscaleSplineFlow(
            channels=channels, size=size, levels=levels, coarse_layers=coarse_layers,
            detail_layers=detail_layers, width=width, bins=bins, unit_interval=False,
        )
        self.coarse_prior = FullTensorFlowMatching(
            channels=channels, size=coarse_size, levels=1, width=width,
        )
        self.residual_velocity = GlobalConditionalVelocity(
            self.packed_residual_channels, channels, coarse_size, width, attention_heads,
        )
        self.register_buffer("_analysis_frozen", torch.tensor(False))
        self.register_load_state_dict_post_hook(self._restore_stage_after_load)

    @property
    def analysis_frozen(self) -> bool:
        return bool(self._analysis_frozen.item())

    @property
    def parameter_counts(self) -> dict[str, int]:
        analysis = sum(p.numel() for p in self._analysis_parameters())
        coarse = sum(p.numel() for p in self.coarse_prior.parameters())
        residual = sum(p.numel() for p in self.residual_velocity.parameters())
        return {"analysis": analysis, "coarse_fm": coarse, "residual_fm": residual,
                "total": analysis + coarse + residual}

    def _analysis_parameters(self):
        return chain(self.pre_analysis.parameters(), self.analysis.parameters())

    def _restore_stage_after_load(self, module, incompatible_keys) -> None:
        frozen = self.analysis_frozen
        for parameter in self._analysis_parameters():
            parameter.requires_grad_(not frozen)
            if frozen:
                parameter.grad = None
        self.pre_analysis.train(self.training and not frozen)
        self.analysis.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.analysis_frozen:
            self.pre_analysis.eval()
            self.analysis.eval()
        return self

    def encode_analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the learned analysis only; its first block is the retained code."""
        _validate_images(x, self.channels, self.size)
        total = x.new_zeros(x.shape[0])
        if self.unit_interval:
            if not bool(torch.all((x > 0) & (x < 1))):
                raise ValueError("unit-interval observations must be strictly interior")
            total = (-torch.log(x) - torch.log1p(-x)).flatten(1).sum(1)
            x = torch.log(x) - torch.log1p(-x)
        mixed, pre_ld = self.pre_analysis(x)
        code, analysis_ld = self.analysis.encode(mixed)
        return code, total + pre_ld + analysis_ld

    def decode_analysis(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the learned analysis, without invoking either FM sampler."""
        _validate_source(code, self.dimension)
        mixed, total = self.analysis.decode(code)
        observation, pre_ld = self.pre_analysis(mixed, inverse=True)
        total = total + pre_ld
        if self.unit_interval:
            total = total + (-F.softplus(observation) - F.softplus(-observation)).flatten(1).sum(1)
            observation = torch.sigmoid(observation)
        return observation, total

    def train_analysis_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.analysis_frozen:
            raise RuntimeError("analysis is frozen; its training stage has ended")
        code, determinant = self.encode_analysis(x)
        log_base = -0.5 * (code.square() + math.log(2.0 * math.pi)).sum(1)
        return -(log_base + determinant).mean() / self.dimension

    def freeze_analysis(self) -> None:
        """Freeze and clear analysis gradients; subsequent fits train only the FMs."""
        self._analysis_frozen.fill_(True)
        self._restore_stage_after_load(self, None)

    def _require_frozen(self) -> None:
        if not self.analysis_frozen:
            raise RuntimeError("freeze_analysis must be called before FM fitting or sampling")

    def _split_code(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = [math.prod(shape) for shape in self.analysis.block_shapes]
        blocks = [piece.reshape(code.shape[0], *shape)
                  for piece, shape in zip(code.split(sizes, dim=1), self.analysis.block_shapes)]
        return blocks[0], pack_residuals(blocks[1:])

    def _join_code(self, coarse: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        blocks = unpack_residuals(residual, self.channels, self.levels)
        return torch.cat([coarse.flatten(1)] + [block.flatten(1) for block in blocks], dim=1)

    def training_loss(self, x: torch.Tensor,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        self._require_frozen()
        with torch.no_grad():
            code, _ = self.encode_analysis(x)
            coarse, residual = self._split_code(code)
        source = torch.randn((x.shape[0], self.dimension), device=x.device,
                             dtype=x.dtype, generator=generator)
        coarse_source = source[:, :self.latent_dimension].reshape_as(coarse)
        residual_source = source[:, self.latent_dimension:].reshape_as(residual)
        loss = _squared_error(self.coarse_prior.velocity, coarse, coarse_source, generator)
        loss = loss + _squared_error(self.residual_velocity, residual, residual_source,
                                    generator, context=coarse)
        return loss / (x.shape[0] * self.dimension)

    @torch.no_grad()
    def sample_from_gaussian(self, z: torch.Tensor, steps: int = 16) -> torch.Tensor:
        self._require_frozen()
        _validate_source(z, self.dimension)
        coarse = self.coarse_prior.sample_from_gaussian(z[:, :self.latent_dimension], steps)
        source = z[:, self.latent_dimension:].reshape(
            z.shape[0], self.packed_residual_channels, self.coarse_size, self.coarse_size,
        )
        residual = heun_integrate(self.residual_velocity, source, steps, context=coarse)
        observation, _ = self.decode_analysis(self._join_code(coarse, residual))
        return observation


__all__ = ["GlobalConditionalVelocity", "LearnedLatentFlowMatching",
           "pack_residuals", "unpack_residuals"]
