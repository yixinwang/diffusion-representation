from __future__ import annotations

import torch

from diqfm.rotated import RotatedShearChart


def test_rotated_chart_cycle() -> None:
    model = RotatedShearChart(hidden=16)
    x = torch.randn(64, 8)
    reconstructed = model.decode(model.encode(x))
    assert torch.allclose(reconstructed, x, atol=2e-5, rtol=2e-5)
