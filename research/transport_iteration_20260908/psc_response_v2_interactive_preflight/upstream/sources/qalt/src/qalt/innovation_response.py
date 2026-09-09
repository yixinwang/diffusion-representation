"""Opt-in triangular Gaussian-innovation response before the cached block.

This is precomposition B_H o F_H, distinct from a projected-rank transport.
No fitting, new random draws, quality guarantee or changed default decoder.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cached_global_innovation import (
    CachedGlobalInnovationDecoder, ContextCache, Transform,
)
from .global_innovation_flow import GlobalInnovationFlow


@dataclass
class ResponseContextCache(ContextCache):
    """Ephemeral cache; same prefix/weight/dtype lifetime as ContextCache."""
    summary: Tensor
    block: int


class InnovationResponse(nn.Module):
    """Keep rank anchors; shift/scale every follower using existing frame rows.

    Default 720 coordinates/rank16: 16 anchors,704 followers,2624 parameters.
    Both modes execute the same two linear layers and feature operations.
    Tiny ranks use the first rank summary entries for the prefix-only control.
    """
    def __init__(self, dimension: int, rank: int = 16, mode: str = "innovation"):
        super().__init__()
        if (not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 16
                or not isinstance(dimension, int) or isinstance(dimension, bool)
                or dimension <= rank):
            raise ValueError("integer dimension > rank and 1 <= rank <= 16 required")
        if mode not in ("innovation", "prefix"):
            raise ValueError("mode must be innovation or prefix")
        self.dimension, self.rank, self.mode = dimension, rank, mode
        # Endpoint-inclusive nearest integer, with exact half ties rounded up.
        # Integer arithmetic makes this independent of the default float dtype.
        anchors = (torch.zeros(1, dtype=torch.long) if rank == 1 else
                   (2 * torch.arange(rank) * (dimension - 1) + rank - 1)
                   // (2 * (rank - 1)))
        mask = torch.ones(dimension, dtype=torch.bool)
        mask[anchors] = False
        self.register_buffer("anchors", anchors)
        self.register_buffer("followers", torch.arange(dimension)[mask])
        self.input = nn.Linear(16 + 2 * rank, 32)
        self.output = nn.Linear(32, 2 * rank)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, summary: Tensor, frame: Tensor,
                inverse: bool = False) -> Transform:
        if (value.ndim != 2 or value.shape[1] != self.dimension
                or summary.shape != (len(value), 16)
                or frame.shape != (self.dimension, self.rank)):
            raise ValueError("response value/summary/frame shape mismatch")
        parameter = self.input.weight
        if any(t.dtype != parameter.dtype or t.device != parameter.device
               for t in (value, summary, frame)):
            raise ValueError("response tensors and parameters must share dtype/device")
        anchors = value[:, self.anchors]
        selected = anchors if self.mode == "innovation" else summary[:, :self.rank]
        bounded = torch.tanh(selected)
        features = torch.cat((summary, bounded, bounded.square()), dim=1)
        hidden = F.silu(self.input(features))
        coefficients = self.output(hidden)
        a, b = coefficients.split(self.rank, dim=1)
        rows = frame[self.followers]
        mean = a @ rows.T
        raw_scale = b @ rows.T
        scale = math.log(2.0) * torch.tanh(raw_scale)
        follower = value[:, self.followers]
        transformed = ((follower - mean) * torch.exp(-scale) if inverse
                       else mean + torch.exp(scale) * follower)
        output = value.index_copy(1, self.followers, transformed)
        logdet = scale.sum(1) * (-1 if inverse else 1)
        valid = torch.isfinite(value).all() & torch.isfinite(summary).all()
        # Check unbounded intermediates before tanh could conceal overflow.
        for tensor in (frame, features, hidden, coefficients, mean, raw_scale,
                       scale, transformed, output, logdet):
            valid = valid & torch.isfinite(tensor).all()
        return Transform(output, logdet, valid)


class InnovationResponseDecoder(CachedGlobalInnovationDecoder):
    """Same cached scalar/mixer B_H, precomposed with response F_H.

    Encode first applies B_H^-1, recovering the unchanged Gaussian anchors,
    then applies F_H^-1. Conditioner and frame are each evaluated once/block.
    """
    def __init__(self, *args, response_mode="innovation", **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_mixer:
            raise ValueError("response requires the existing shared mixer frame")
        self.responses = nn.ModuleList([
            InnovationResponse(len(c.active), c.rank, response_mode)
            for c in self.conditioners
        ])

    def cache_context(self, residual: Tensor, coarse: Tensor, block: int):
        raw, alpha, summary = self.conditioners[block](residual, coarse, return_summary=True)
        return ResponseContextCache(raw, alpha, self.frames[block].matrix(), summary, block)

    def apply_cached(self, value: Tensor, cache: ContextCache,
                     inverse: bool = False) -> Transform:
        if not isinstance(cache, ResponseContextCache):
            raise ValueError("response requires its ephemeral summary context cache")
        response = self.responses[cache.block]
        if inverse:
            old = super().apply_cached(value, cache, inverse=True)
            new = response(old.value, cache.summary, cache.frame, inverse=True)
            output = new.value
        else:
            new = response(value, cache.summary, cache.frame)
            old = super().apply_cached(new.value, cache)
            output = old.value
        logdet = old.logdet + new.logdet
        return Transform(output, logdet, old.valid & new.valid
                         & torch.isfinite(output).all() & torch.isfinite(logdet).all())


class InnovationResponseFlow(GlobalInnovationFlow):
    """Opt-in full Gaussian-source composition, with exact existing root.

    The superclass's temporary residual construction is setup work to charge;
    only the replacement remains registered. No unused FM/response parameters.
    """
    def __init__(self, *args, innovation_blocks=4, innovation_width=None,
                 innovation_rank=16, innovation_bins=8,
                 innovation_channel_embedding=12, response_mode="innovation", **kwargs):
        super().__init__(*args, **kwargs)
        parameter = next(self.parameters())
        width = (innovation_width if innovation_width is not None
                 else self.coarse_decoder.layers[0].conditioner.input.out_channels)
        self.residual_decoder = InnovationResponseDecoder(
            self.packed_residual_channels, self.channels, self.coarse_size,
            blocks=innovation_blocks, width=width, rank=innovation_rank,
            bins=innovation_bins, channel_embedding=innovation_channel_embedding,
            response_mode=response_mode).to(device=parameter.device, dtype=parameter.dtype)
