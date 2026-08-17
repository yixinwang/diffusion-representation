from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .synthetic import ShearNet, set_seed


torch.set_num_threads(4)


def make_orthogonal(seed: int, dimension: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return matrix


def rotate_data(data: dict[str, np.ndarray], matrix: np.ndarray, paired: bool) -> dict[str, np.ndarray]:
    output = dict(data)
    if paired:
        output["xa"] = data["xa"] @ matrix.T
        output["xb"] = data["xb"] @ matrix.T
    else:
        output["x"] = data["x"] @ matrix.T
    return output


class RotatedShearChart(nn.Module):
    """Orthogonal Cayley layer followed by an exactly invertible nonlinear shear."""

    def __init__(self, dimension: int = 8, state_dim: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.dimension = dimension
        self.state_dim = state_dim
        self.skew_parameter = nn.Parameter(torch.zeros(dimension, dimension))
        self.shear = ShearNet(hidden)

    def orthogonal(self) -> torch.Tensor:
        skew = self.skew_parameter - self.skew_parameter.T
        identity = torch.eye(self.dimension, device=skew.device, dtype=skew.dtype)
        return torch.linalg.solve(identity - skew, identity + skew)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.orthogonal()
        residual = base[:, self.state_dim :]
        state = base[:, : self.state_dim] - self.shear(residual)
        return torch.cat([state, residual], dim=1)

    def decode(self, coordinates: torch.Tensor) -> torch.Tensor:
        state = coordinates[:, : self.state_dim]
        residual = coordinates[:, self.state_dim :]
        base = torch.cat([state + self.shear(residual), residual], dim=1)
        return base @ self.orthogonal().T


def train_rotated_chart(
    train_pairs: dict[str, np.ndarray],
    validation_pairs: dict[str, np.ndarray],
    *,
    seed: int = 0,
    steps: int = 1500,
    batch_size: int = 512,
    spread_weight: float = 0.003,
) -> RotatedShearChart:
    set_seed(seed)
    model = RotatedShearChart()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    xa = torch.tensor(train_pairs["xa"], dtype=torch.float32)
    xb = torch.tensor(train_pairs["xb"], dtype=torch.float32)
    va = torch.tensor(validation_pairs["xa"], dtype=torch.float32)
    vb = torch.tensor(validation_pairs["xb"], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_state = None
    best_validation = float("inf")
    for step in range(steps):
        index = rng.integers(0, len(xa), size=min(batch_size, len(xa)))
        encoded_a = model.encode(xa[index])
        encoded_b = model.encode(xb[index])
        state_a = encoded_a[:, :2]
        state_b = encoded_b[:, :2]
        pair_loss = 0.5 * ((state_a - state_b) ** 2).sum(1).mean()
        state = torch.cat([state_a, state_b], dim=0)
        centered = state - state.mean(0)
        covariance = centered.T @ centered / len(centered)
        spread = -torch.logdet(covariance + 1e-3 * torch.eye(2))
        regularizer = 1e-5 * (
            model.skew_parameter.square().mean()
            + sum(parameter.square().mean() for parameter in model.shear.parameters())
        )
        loss = pair_loss + spread_weight * spread + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                state_va = model.encode(va)[:, :2]
                state_vb = model.encode(vb)[:, :2]
                validation = float((0.5 * ((state_va - state_vb) ** 2).sum(1).mean()).item())
            if validation < best_validation:
                best_validation = validation
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
