"""Opt-in sinh-arcsinh shape response; no change to the location-scale default.

Scalar shape arithmetic is promoted to float64, including for float32 models;
casts and scalar work must be charged in any cost comparison. Identity algebra
preserves ordinary zero-shape values without branching away shape gradients.
It is not universally stable: extreme opposing terms can cancel, overflow, or
round the log1p argument to -1. Such intermediates invalidate the transform;
no clipping, endpoint repair, or universal floating-point guarantee is used.
"""
import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cached_global_innovation import Transform
from .innovation_response import InnovationResponse, InnovationResponseFlow


def sinh_arcsinh_shape(value: Tensor, tau: Tensor, kappa: Tensor,
                      inverse: bool = False) -> Transform:
    """Elementwise SAS value and log derivative, with aggregate validity.

    Forward q=sinh((asinh(z)+tau)*exp(-kappa)); inverse reverses it.
    All inputs must have the same shape/device/dtype (float32 or float64).
    Invalid intermediates remain visible in outputs and valid=False; callers
    must respect validity, as the inherited full decoder does.
    """
    if (value.dtype not in (torch.float32, torch.float64)
            or any(t.shape != value.shape or t.dtype != value.dtype
                   or t.device != value.device for t in (tau, kappa))):
        raise ValueError('shape inputs require identical shape/device and float32/64 dtype')
    z, t, k = value.double(), tau.double(), kappa.double()
    v = torch.asinh(z)
    signed_k = k if inverse else -k
    exp_k = torch.exp(signed_k)
    expm1_k = torch.expm1(signed_k)
    delta = v * expm1_k - t if inverse else v * expm1_k + t * exp_k
    radius = torch.hypot(torch.ones_like(z), z)
    half_sinh = torch.sinh(delta / 2)
    sinh_delta = torch.sinh(delta)
    twice_square = 2 * half_sinh.square()
    term1, term2 = z * twice_square, radius * sinh_delta
    result64 = z + term1 + term2
    log_argument = twice_square + (z / radius) * sinh_delta
    log64 = signed_k + torch.log1p(log_argument)
    result, logdet = result64.to(value.dtype), log64.to(value.dtype)
    valid = torch.ones((), dtype=torch.bool, device=value.device)
    for tensor in (z, t, k, v, exp_k, expm1_k, delta, radius, half_sinh,
                   sinh_delta, twice_square, term1, term2, result64,
                   log_argument, log64, result, logdet):
        valid = valid & torch.isfinite(tensor).all()
    valid = valid & (log_argument > -1).all()
    return Transform(result, logdet, valid)


class SinhArcsinhResponse(InnovationResponse):
    """Existing anchors/frame/SiLU head, extended with two rank shape outputs.

    tau=tanh(raw_tau), kappa=log(2)*tanh(raw_kappa); m and s retain
    exactly the original dtype and operation order of the location-scale map.
    Prefix and innovation controls differ only in their prescribed features.
    """
    def __init__(self, dimension: int, rank: int = 16, mode: str = 'innovation'):
        super().__init__(dimension, rank, mode)
        self.output = nn.Linear(32, 4 * rank)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, summary: Tensor, frame: Tensor,
                inverse: bool = False) -> Transform:
        if (value.ndim != 2 or value.shape[1] != self.dimension
                or summary.shape != (len(value), 16)
                or frame.shape != (self.dimension, self.rank)):
            raise ValueError('response value/summary/frame shape mismatch')
        parameter = self.input.weight
        if (parameter.dtype not in (torch.float32, torch.float64)
                or any(t.dtype != parameter.dtype or t.device != parameter.device
                       for t in (value, summary, frame))):
            raise ValueError('response tensors and parameters require shared float32/64 dtype/device')
        valid = (torch.isfinite(value).all() & torch.isfinite(summary).all()
                 & torch.isfinite(frame).all())
        anchors = value[:, self.anchors]
        selected = anchors if self.mode == 'innovation' else summary[:, :self.rank]
        bounded = torch.tanh(selected)
        features = torch.cat((summary, bounded, bounded.square()), dim=1)
        hidden_raw = self.input(features)
        hidden = F.silu(hidden_raw)
        # Keep the LS projection's original GEMM shape for exact promotion.
        # Two 2*rank projections (same 4*rank parameters), charged as such.
        split = 2 * self.rank
        coefficients = torch.cat((
            F.linear(hidden, self.output.weight[:split], self.output.bias[:split]),
            F.linear(hidden, self.output.weight[split:], self.output.bias[split:])), dim=1)
        a, b, c, d = coefficients.split(self.rank, dim=1)
        rows = frame[self.followers]
        mean, raw_scale = a @ rows.T, b @ rows.T
        raw_tau, raw_kappa = c @ rows.T, d @ rows.T
        for unbounded in (features, hidden_raw, hidden, coefficients, mean,
                          raw_scale, raw_tau, raw_kappa):
            valid = valid & torch.isfinite(unbounded).all()
        scale = math.log(2.0) * torch.tanh(raw_scale)
        tau = torch.tanh(raw_tau)
        kappa = math.log(2.0) * torch.tanh(raw_kappa)
        follower = value[:, self.followers]
        if inverse:
            affine = (follower - mean) * torch.exp(-scale)
            shape = sinh_arcsinh_shape(affine, tau, kappa, inverse=True)
            transformed = shape.value
        else:
            shape = sinh_arcsinh_shape(follower, tau, kappa)
            affine = mean + torch.exp(scale) * shape.value
            transformed = affine
        output = value.index_copy(1, self.followers, transformed)
        logdet = scale.sum(1) * (-1 if inverse else 1) + shape.logdet.sum(1)
        valid = valid & shape.valid
        for tensor in (value, summary, frame, selected, features, hidden_raw,
                       hidden, coefficients, mean, raw_scale, raw_tau, raw_kappa,
                       scale, tau, kappa, affine, transformed, output, logdet):
            valid = valid & torch.isfinite(tensor).all()
        return Transform(output, logdet, valid)


def promote_location_scale_flow(flow: InnovationResponseFlow) -> InnovationResponseFlow:
    """Return an independent SAS copy, retaining B, frames, modes and LS rows.

    No old module is modified. New shape rows are zero; old head/input values
    and parameter requires_grad flags are copied. This allocates/copies state
    and consumes constructor RNG, so promotion/setup cost must be accounted.
    The returned flow remains compatible with InnovationResponseFlow methods;
    its response state layout is intentionally larger than the old checkpoint.
    """
    if not isinstance(flow, InnovationResponseFlow):
        raise ValueError('an InnovationResponseFlow is required')
    if any(type(r) is not InnovationResponse for r in flow.residual_decoder.responses):
        raise ValueError('promotion requires original location-scale response heads')
    result = copy.deepcopy(flow)
    for i, old in enumerate(result.residual_decoder.responses):
        new = SinhArcsinhResponse(old.dimension, old.rank, old.mode).to(
            device=old.input.weight.device, dtype=old.input.weight.dtype)
        new.input.load_state_dict(old.input.state_dict())
        with torch.no_grad():
            new.output.weight[:2*old.rank].copy_(old.output.weight)
            new.output.bias[:2*old.rank].copy_(old.output.bias)
            new.anchors.copy_(old.anchors); new.followers.copy_(old.followers)
        for name, parameter in new.named_parameters():
            parameter.requires_grad_(dict(old.named_parameters())[name].requires_grad)
        new.train(old.training)
        new.input.train(old.input.training); new.output.train(old.output.training)
        result.residual_decoder.responses[i] = new
    return result
