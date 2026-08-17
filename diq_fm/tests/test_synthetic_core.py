from __future__ import annotations

import numpy as np
import torch

from diqfm.synthetic import ShearNet, h_true_np, make_params, sample


def test_pair_identity_matches_conditional_variance() -> None:
    params = make_params(seed=7)
    paired = sample(4000, params, seed=9, paired=True)
    za = paired["xa"][:, :2] - h_true_np(paired["xa"][:, 2:])
    zb = paired["xb"][:, :2] - h_true_np(paired["xb"][:, 2:])
    pair = 0.5 * np.mean(np.sum((za - zb) ** 2, axis=1))
    assert abs(pair - 2 * 0.035**2) < 3e-4


def test_shear_anchor() -> None:
    network = ShearNet(hidden=16)
    u = torch.randn(32, 6)
    assert torch.allclose(network(torch.zeros_like(u)), torch.zeros(32, 2), atol=1e-7)
