"""Pro9 read-only review reference: cached conditional global-innovation decoder.

This file neither loads native data nor submits jobs. It is a replacement *module*
for GlobalInnovationFlow.residual_decoder, not a trained model or a study runner.
The scalar transform is an integrated piecewise-linear derivative with identity
real-line tails. The conditional mixer is an orthogonal low-rank SPD update.
No CDF/quantile, rejection sampling, source clipping, or additional noise is used.

Public transforms return an aggregate on-device validity flag. Call require_valid
before accepting values or differentiating a loss. Frozen CUDA timings must charge
that check. Metadata checks are eager; tensor kernels contain no host .item calls.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Transform:
    value: Tensor
    logdet: Tensor
    valid: Tensor


def require_valid(result: Transform, label: str = "transform") -> Transform:
    """One host decision; no silently accepted clipped/fallback output."""
    if not bool(result.valid):
        raise FloatingPointError(f"{label}: nonfinite value or numerical invariant failed")
    return result


def derivative_heights(raw: Tensor, floor: float = 0.1) -> Tensor:
    """For K bins, raw has K-1 entries. Endpoint derivatives are exactly one.

    Interior heights sum to K-1 in exact arithmetic. Uniform logits initialize
    the identity map. For K=8 the derivative is between .1 and 6.4.
    """
    inner = floor + (1.0 - floor) * raw.shape[-1] * raw.softmax(dim=-1)
    one = torch.ones_like(inner[..., :1])
    return torch.cat((one, inner, one), dim=-1)


def integrated_linear_kernel(
    value: Tensor, raw: Tensor, *, inverse: bool = False,
    bound: float = 4.0, floor: float = 0.1,
) -> Transform:
    """Shape-static differentiable tensor kernel; value (...), raw (..., K-1).

    The inner scratch value is replaced by zero on the inactive tail branch,
    preventing overflow in unused quadratics. The actual tail is returned
    unchanged. Only bin indices are bounded, never source/output coordinates.
    A failed discriminant produces an explicitly invalid result, not a fallback.
    """
    heights = derivative_heights(raw, floor)
    k = raw.shape[-1] + 1
    dx = 2.0 * bound / k
    areas = 0.5 * dx * (heights[..., :-1] + heights[..., 1:])
    knots = torch.cat((torch.zeros_like(areas[..., :1]), areas.cumsum(-1)), -1) - bound
    inside = (value > -bound) & (value < bound)
    scratch = torch.where(inside, value, torch.zeros_like(value))
    if inverse:
        index = (scratch.unsqueeze(-1) >= knots[..., 1:-1]).sum(-1)
    else:
        index = torch.floor((scratch + bound) / dx).long().clamp(0, k - 1)
    gather = lambda t, j: t.gather(-1, j.unsqueeze(-1)).squeeze(-1)
    h0 = gather(heights, index)
    h1 = gather(heights, index + 1)
    y0 = gather(knots, index)
    x0 = -bound + dx * index.to(value.dtype)
    slope = (h1 - h0) / dx
    if inverse:
        distance = scratch - y0
        disc = h0.square() + 2.0 * slope * distance
        # Keep invalid arithmetic out of the backward graph, but mark invalid.
        root = torch.sqrt(torch.where(disc > 0, disc, torch.ones_like(disc)))
        offset = 2.0 * distance / (h0 + root)
        transformed = x0 + offset
        derivative = h0 + slope * offset
        branch_valid = (disc > 0) & (derivative > 0)
        logdet = -torch.log(torch.where(derivative > 0, derivative, torch.ones_like(derivative)))
    else:
        offset = scratch - x0
        transformed = y0 + offset * (h0 + 0.5 * slope * offset)
        derivative = h0 + slope * offset
        branch_valid = derivative > 0
        logdet = torch.log(torch.where(derivative > 0, derivative, torch.ones_like(derivative)))
    output = torch.where(inside, transformed, value)
    ld = torch.where(inside, logdet, torch.zeros_like(logdet))
    # Arithmetic tolerance, not a model tolerance or source clamp.
    eps = torch.finfo(value.dtype).eps
    knot_ok = ((knots[..., -1] - bound).abs() <= 128 * eps * bound).all()
    interval_ok = ((offset >= -128 * eps * bound) & (offset <= dx + 128 * eps * bound))
    valid = (torch.isfinite(value).all() & torch.isfinite(raw).all()
             & torch.isfinite(output).all() & torch.isfinite(ld).all()
             & torch.where(inside, branch_valid & interval_ok, True).all() & knot_ok)
    return Transform(output, ld, valid)


def integrated_linear(
    value: Tensor, raw: Tensor, *, inverse: bool = False,
    bound: float = 4.0, floor: float = 0.1,
) -> Transform:
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("Use float32 or float64 for all scalar flow arithmetic")
    if raw.shape[:-1] != value.shape or raw.shape[-1] < 2:
        raise ValueError("raw must have shape value.shape + (K-1,), K >= 3")
    if raw.device != value.device or raw.dtype != value.dtype:
        raise ValueError("value and raw must share device and dtype")
    if not (bound > 0 and 0 < floor < 1):
        raise ValueError("bound > 0 and 0 < floor < 1 required")
    return integrated_linear_kernel(value, raw, inverse=inverse, bound=bound, floor=floor)


class OrthonormalFrame(nn.Module):
    """Full-rank graph chart QR([I; W]) times a learned Cayley rotation.

    QR and the rank-sized Cayley solve occur once per block per training
    transform, NOT per sample. [I; W] has full column rank for every finite W.
    The rotation is needed: a subspace alone does not specify eigenvectors
    for unequal eigenvalues. The graph chart excludes singular leading minors;
    the Cayley chart excludes rotations with eigenvalue -1. Approximation error
    from these chart restrictions is not assumed away for native data.
    Inference caching is explicit, invalidated by train(True), and not persisted
    in checkpoints. The duplicate cache bytes must be included in memory costs.
    """
    def __init__(self, dimension: int, rank: int):
        super().__init__()
        if not 0 < rank < dimension:
            raise ValueError("0 < rank < block dimension required")
        self.dimension, self.rank = dimension, rank
        self.bottom = nn.Parameter(torch.randn(dimension - rank, rank) / math.sqrt(dimension - rank))
        self.register_buffer("identity", torch.eye(rank))
        self.register_buffer("upper", torch.triu_indices(rank, rank, offset=1))
        self.rotation = nn.Parameter(torch.zeros(rank * (rank - 1) // 2))
        self.register_buffer("cached", torch.zeros(dimension, rank), persistent=False)
        self.cache_ready = False
        self.register_load_state_dict_post_hook(self._invalidate_after_load)

    def _invalidate_after_load(self, module, incompatible_keys) -> None:
        self.cache_ready = False

    def train(self, mode: bool = True):
        if mode:
            self.cache_ready = False
        return super().train(mode)

    def matrix(self) -> Tensor:
        if self.cache_ready:
            return self.cached
        chart = torch.cat((self.identity, self.bottom), dim=0)
        q, r = torch.linalg.qr(chart, mode="reduced")
        signs = torch.where(r.diagonal() >= 0, 1.0, -1.0).detach()
        q = q * signs.unsqueeze(0)
        skew = torch.zeros_like(self.identity).index_put(
            (self.upper[0], self.upper[1]), self.rotation)
        skew = skew - skew.T
        rotation = torch.linalg.solve(self.identity - skew, self.identity + skew)
        return (q @ rotation).contiguous()

    @torch.no_grad()
    def prepare_inference(self) -> None:
        if self.training:
            raise RuntimeError("Call eval() before preparing frozen inference")
        self.cache_ready = False
        self.cached.copy_(self.matrix())
        self.cache_ready = True


def rank_mix(value: Tensor, frame: Tensor, alpha: Tensor, inverse: bool = False) -> Transform:
    """M=I+U diag(exp(alpha)-1) U.T; no dense n-by-n matrix is materialized."""
    signed = -alpha if inverse else alpha
    transformed = value + ((value @ frame) * torch.expm1(signed)) @ frame.T
    ld = signed.sum(-1)
    tolerance = 256 * torch.finfo(value.dtype).eps
    orth_error = (frame.T @ frame - torch.eye(frame.shape[1], dtype=frame.dtype, device=frame.device)).abs().amax()
    valid = (torch.isfinite(value).all() & torch.isfinite(frame).all()
             & torch.isfinite(alpha).all() & torch.isfinite(transformed).all()
             & torch.isfinite(ld).all() & (orth_error <= tolerance))
    return Transform(transformed, ld, valid)


@dataclass
class ContextCache:
    raw: Tensor              # (batch, active coordinates, 9)
    alpha: Tensor            # (batch, rank), bounded log-eigenvalues
    frame: Tensor            # (active coordinates, rank)


class PrefixConditioner(nn.Module):
    """Local features + four global queries -> 16 global scalars, once/block.

    All future/current residual inputs are masked *inside* this module. The
    local features and channel embeddings are not claimed to be 16-dimensional.
    Only active coordinates receive output heads, with no 45*64*9 dense head.
    """
    def __init__(self, channels: int, context_channels: int, size: int,
                 active: Tensor, observed: Tensor, width: int, rank: int,
                 channel_embedding: int = 12, bins: int = 8):
        super().__init__()
        self.channels, self.context_channels, self.size = channels, context_channels, size
        self.bins, self.rank = bins, rank
        self.register_buffer("active", active.long())
        self.register_buffer("spatial", active.long() % (size * size))
        self.register_buffer("channel", active.long() // (size * size))
        self.register_buffer("observed", observed.reshape(1, channels, size, size).bool())
        axis = torch.linspace(-1, 1, size)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("position", torch.stack((row, col))[None])
        input_channels = 2 * channels + context_channels + 2
        self.input = nn.Conv2d(input_channels, width, 3, padding=1)
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.keys = nn.Linear(width, 16)
        self.values = nn.Linear(width, 4)
        self.queries = nn.Parameter(torch.randn(4, 16) / 4.0)
        self.embedding = nn.Embedding(channels, channel_embedding)
        self.head = nn.Linear(width + channel_embedding + 16, bins + 1)
        self.eigen_head = nn.Linear(16, rank)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.eigen_head.weight)
        nn.init.zeros_(self.eigen_head.bias)

    def forward(self, residual: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor]:
        mask = self.observed.expand(residual.shape[0], -1, -1, -1)
        visible = torch.where(mask, residual, torch.zeros_like(residual))
        inp = torch.cat((visible, mask.to(residual.dtype), coarse,
                         self.position.expand(residual.shape[0], -1, -1, -1)), dim=1)
        features = F.silu(self.input(inp))
        features = features + F.silu(self.local(features))
        tokens = features.flatten(2).transpose(1, 2)
        keys, values = self.keys(tokens), self.values(tokens)
        weights = torch.softmax(torch.einsum("qd,bsd->bqs", self.queries, keys) / 4.0, dim=-1)
        summary = (weights @ values).flatten(1)  # (batch, 4*4), no extra noise
        local = tokens[:, self.spatial]
        embedding = self.embedding(self.channel)[None].expand(residual.shape[0], -1, -1)
        context = torch.cat((local, embedding, summary[:, None].expand(-1, len(self.active), -1)), -1)
        return self.head(context), math.log(2.0) * torch.tanh(self.eigen_head(summary))


class CachedGlobalInnovationDecoder(nn.Module):
    """Four observed-prefix blocks, scalar transform then conditional global mix.

    decode: e=g(z;H); r=mu(H)+exp(ell(H))*M(H)e.
    encode: e=M(H)^-1[(r-mu(H))*exp(-ell(H))]; z=g^-1(e;H).
    H=(coarse, previous residual blocks). Current-block values never enter H.
    The 45x8x8 default splits exactly into four interleaved 720-coordinate blocks.
    """
    def __init__(self, residual_channels: int = 45, context_channels: int = 3,
                 size: int = 8, blocks: int = 4, width: int = 32,
                 rank: int = 16, bins: int = 8, channel_embedding: int = 12,
                 use_mixer: bool = True):
        super().__init__()
        vals = (residual_channels, context_channels, size, blocks, width, rank, bins, channel_embedding)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in vals):
            raise ValueError("configuration fields must be positive integers")
        if bins < 3:
            raise ValueError("bins must be at least three")
        if size % blocks:
            raise ValueError("size must be divisible by blocks for equal interleaved blocks")
        self.residual_channels, self.context_channels, self.size = residual_channels, context_channels, size
        self.dimension, self.blocks, self.bins = residual_channels * size * size, blocks, bins
        self.use_mixer = use_mixer
        c = torch.arange(residual_channels)[:, None, None]
        row = torch.arange(size)[None, :, None]
        col = torch.arange(size)[None, None, :]
        assignment = ((c + row + col) % blocks).flatten()
        self.conditioners = nn.ModuleList()
        self.frames = nn.ModuleList()
        for b in range(blocks):
            active = torch.nonzero(assignment == b).flatten()
            self.conditioners.append(PrefixConditioner(residual_channels, context_channels,
                size, active, assignment < b, width, rank, channel_embedding, bins))
            # Mixer-free ablation has no unused frame/eigen-head parameters.
            if use_mixer:
                self.frames.append(OrthonormalFrame(len(active), rank))
            else:
                self.conditioners[-1].eigen_head = None
        if not use_mixer:
            # Avoid calling the regular forward's eigen_head in the ablation.
            for conditioner in self.conditioners:
                conditioner.eigen_head = _ZeroEigen(rank)

    @property
    def parameter_counts(self) -> dict[str, int]:
        return {"residual_decoder": sum(p.numel() for p in self.parameters()),
                "total": sum(p.numel() for p in self.parameters())}

    def _validate(self, residual: Tensor, coarse: Tensor) -> None:
        p = next(self.parameters())
        if residual.ndim != 4 or residual.shape[1:] != (self.residual_channels, self.size, self.size):
            raise ValueError("wrong residual NCHW shape")
        if coarse.shape != (residual.shape[0], self.context_channels, self.size, self.size):
            raise ValueError("wrong coarse NCHW shape")
        if residual.shape[0] < 1 or residual.dtype not in (torch.float32, torch.float64):
            raise ValueError("nonempty float32/float64 batch required")
        if any(t.dtype != p.dtype or t.device != p.device for t in (residual, coarse)):
            raise ValueError("tensors and model must share dtype/device")

    def cache_context(self, residual: Tensor, coarse: Tensor, block: int) -> ContextCache:
        raw, alpha = self.conditioners[block](residual, coarse)
        frame = self.frames[block].matrix() if self.use_mixer else raw.new_empty(raw.shape[1], 0)
        return ContextCache(raw, alpha, frame)

    def apply_cached(self, value: Tensor, cache: ContextCache, inverse: bool = False) -> Transform:
        raw = cache.raw
        mean = 4.0 * torch.tanh(raw[..., 0])
        ell = math.log(2.0) * torch.tanh(raw[..., 1])
        if inverse:
            centered = (value - mean) * torch.exp(-ell)
            mixed = rank_mix(centered, cache.frame, cache.alpha, True) if self.use_mixer else Transform(centered, value.new_zeros(value.shape[0]), torch.isfinite(centered).all())
            scalar = integrated_linear_kernel(mixed.value, raw[..., 2:], inverse=True)
            return Transform(scalar.value, scalar.logdet.sum(-1) + mixed.logdet - ell.sum(-1),
                             scalar.valid & mixed.valid & torch.isfinite(raw).all())
        scalar = integrated_linear_kernel(value, raw[..., 2:])
        mixed = rank_mix(scalar.value, cache.frame, cache.alpha) if self.use_mixer else Transform(scalar.value, value.new_zeros(value.shape[0]), scalar.valid)
        output = mean + torch.exp(ell) * mixed.value
        ld = scalar.logdet.sum(-1) + mixed.logdet + ell.sum(-1)
        return Transform(output, ld, scalar.valid & mixed.valid & torch.isfinite(output).all()
                         & torch.isfinite(ld).all() & torch.isfinite(raw).all())

    def transform(self, residual: Tensor, coarse: Tensor, inverse: bool = False) -> Transform:
        self._validate(residual, coarse)
        original = residual.flatten(1)
        # This is an output container, not a learned posterior draw.
        result = torch.zeros_like(original)
        prefix = torch.zeros_like(original)
        ld = residual.new_zeros(residual.shape[0])
        valid = torch.isfinite(residual).all() & torch.isfinite(coarse).all()
        for b, conditioner in enumerate(self.conditioners):
            cache = self.cache_context(prefix.reshape_as(residual), coarse, b)
            block = self.apply_cached(original[:, conditioner.active], cache, inverse)
            result = result.index_copy(1, conditioner.active, block.value)
            # Encode conditions on actual preceding residuals, NOT whitened codes.
            observed = original[:, conditioner.active] if inverse else block.value
            prefix = prefix.index_copy(1, conditioner.active, observed)
            ld = ld + block.logdet
            valid = valid & block.valid
        return Transform(result.reshape_as(residual), ld, valid)

    def decode(self, source: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor]:
        result = require_valid(self.transform(source, coarse), "decode")
        return result.value, result.logdet

    def encode(self, residual: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor]:
        result = require_valid(self.transform(residual, coarse, True), "encode")
        return result.value, result.logdet

    def sample_from_gaussian(self, source: Tensor, coarse: Tensor) -> Tensor:
        return self.decode(source, coarse)[0]

    def log_prob(self, residual: Tensor, coarse: Tensor) -> Tensor:
        result = require_valid(self.transform(residual, coarse, True), "log_prob")
        answer = -0.5 * (result.value.square() + math.log(2.0 * math.pi)).flatten(1).sum(1) + result.logdet
        if not bool(torch.isfinite(answer).all()):
            raise FloatingPointError("nonfinite conditional log density")
        return answer

    @torch.no_grad()
    def prepare_inference(self) -> None:
        if self.training:
            raise RuntimeError("Call eval() first; modifying weights invalidates all caches")
        for frame in self.frames:
            frame.prepare_inference()

    def cache_bytes(self) -> int:
        return sum(f.cached.numel() * f.cached.element_size() for f in self.frames)


class _ZeroEigen(nn.Module):
    """No learned parameters; used only by the fully retrained scalar ablation."""
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def forward(self, summary: Tensor) -> Tensor:
        return summary.new_zeros(summary.shape[0], self.rank)


def replace_in_memory(global_innovation_flow: Any, *, width: int = 32,
                      rank: int = 16, use_mixer: bool = True):
    """Deep copy; replace residual module only. Does not write any repo/file.

    Compatible with the interfaces read at bb024437... . Integration with the
    live repo has not been executed by this standalone review package.
    """
    result = copy.deepcopy(global_innovation_flow)
    parameter = next(result.parameters())
    result.residual_decoder = CachedGlobalInnovationDecoder(
        result.packed_residual_channels, result.channels, result.coarse_size,
        width=width, rank=rank, use_mixer=use_mixer).to(device=parameter.device, dtype=parameter.dtype)
    return result
