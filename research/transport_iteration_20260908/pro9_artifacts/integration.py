"""Prospective in-memory integration helpers; no loader, training loop, or jobs.

These helpers are not a substitute for the repository's canonical data ledger,
source guard, independent evaluator or whole-cost accounting. The complete
repository integration has not been executed in this local CPU package.
"""
from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any
import torch
from torch import nn


def whole_parameter_count(model: nn.Module) -> int:
    """Frozen learned weights are not free; count every registered parameter."""
    return sum(p.numel() for p in model.parameters())


def conservative_parameter_match(
    factory: Callable[[int], nn.Module], widths: Iterable[int],
    target: int, maximum_excess: float = .05,
) -> tuple[nn.Module, dict[str, Any]]:
    """Shape-only matching of WHOLE models; no data, padding or weak control.

    The caller supplies a complete model factory, including its analysis/root.
    Controls have at least target parameters and no more than 5% extra. Fails
    closed if widths cannot achieve this; never silently match only a decoder.
    Constructors may initialize weights: call under a separate construction RNG
    scope, then reset the final training initialization seed consistently.
    """
    if target <= 0 or not 0 <= maximum_excess <= .05:
        raise ValueError('invalid target or excessive matching tolerance')
    records = []
    selected = None
    for width in sorted(set(widths)):
        if not isinstance(width, int) or width <= 0:
            raise ValueError('widths must be positive integers')
        model = factory(width)
        count = whole_parameter_count(model)
        records.append({'width': width, 'whole_parameters': count})
        if target <= count <= target * (1 + maximum_excess):
            if selected is None or count < selected[0]:
                selected = (count, width, model)
    if selected is None:
        raise ValueError(f'No conservative whole-model match; shape census={records}')
    count, width, model = selected
    return model, {'target': target, 'selected_width': width,
                   'whole_parameters': count, 'excess_fraction': count / target - 1,
                   'shape_census': records}


def set_endpoint_phase(model: nn.Module, phase: str) -> list[nn.Parameter]:
    """For the repo GlobalInnovationFlow interface at bb024437.

    Caller must discard stale fit-code caches and reconstruct the optimizer at
    phase transitions. Optimizer/setup/cache costs consume the phase budget.
    Joint phase adapts A and the decoder, holding learned root weights fixed.
    Do not use no_grad around root evaluation: gradients through root inputs
    must still reach A. This isolates analysis adaptation from root refitting.
    """
    if phase not in {'analysis', 'root', 'decoder', 'joint'}:
        raise ValueError('unknown phase')
    required = ('_analysis_frozen', '_analysis_parameters', 'coarse_decoder', 'residual_decoder')
    if not all(hasattr(model, field) for field in required):
        raise TypeError('Expected GlobalInnovationFlow, not a finite-Heun wrapper')
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    with torch.no_grad():
        model._analysis_frozen.fill_(phase not in {'analysis', 'joint'})
    model.train(True)  # invalidates residual frame caches
    if phase in {'analysis', 'joint'}:
        for p in model._analysis_parameters():
            p.requires_grad_(True)
    if phase == 'root':
        for p in model.coarse_decoder.parameters():
            p.requires_grad_(True)
    if phase in {'decoder', 'joint'}:
        for p in model.residual_decoder.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def joint_endpoint_loss(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """Complete logit-space normalized NLL per coordinate; not Gaussian A NLL.

    Add the fixed outer logit Jacobian for image-space reporting. It is constant
    with respect to model parameters. Do not cache A(logits) in the joint phase.
    """
    if bool(model._analysis_frozen):
        raise RuntimeError('Joint phase requires the analysis freeze flag cleared')
    return -model.log_prob(logits).mean() / model.dimension
