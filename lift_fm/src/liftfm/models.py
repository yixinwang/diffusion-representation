from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class TokenVelocity(nn.Module):
    def __init__(self, tokens: int, width: int = 48, depth: int = 2, heads: int = 4, ff_mult: int = 2, classes: int = 10):
        super().__init__()
        self.tokens = tokens
        self.width = width
        self.input = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.label = nn.Embedding(classes, width)
        self.time = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff_mult * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.output = nn.Linear(width, 1)
        self.depth = depth
        self.ff_mult = ff_mult

    @staticmethod
    def time_features(t: torch.Tensor) -> torch.Tensor:
        frequencies = torch.tensor([1.0, 2.0, 4.0], device=t.device, dtype=t.dtype)
        phase = 2.0 * math.pi * t[:, None] * frequencies[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = self.input(x[..., None])
        context = self.time(self.time_features(t)) + self.label(labels)
        h = h + self.position[None, :, :] + context[:, None, :]
        return self.output(self.blocks(h)).squeeze(-1)

    def flop_proxy(self, batch: int = 1) -> int:
        n, h, f = self.tokens, self.width, self.ff_mult * self.width
        # qkv+projection, attention products, two FF matrices; multiply-add counted as 2 FLOPs.
        per_layer = 2 * (4 * n * h * h + 2 * n * n * h + 2 * n * h * f)
        io = 2 * n * h * 2
        time_label = 2 * (6 * h + h * h)
        return int(batch * (self.depth * per_layer + io + time_label))

    def activation_proxy_bytes(self, batch: int = 1) -> int:
        n, h = self.tokens, self.width
        return int(4 * batch * self.depth * (6 * n * h + self.tokens * self.tokens))


@dataclass
class FlowTrainingResult:
    model: TokenVelocity
    losses: list[float]
    seconds: float


def train_rectified_flow(
    data: np.ndarray,
    labels: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    depth: int,
    heads: int,
    learning_rate: float,
) -> FlowTrainingResult:
    import time
    seed_everything(seed)
    x = torch.as_tensor(np.asarray(data, dtype=np.float32))
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    model = TokenVelocity(x.shape[1], width=width, depth=depth, heads=heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 91)
    losses: list[float] = []
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = torch.randint(0, len(x), (batch_size,), generator=generator)
        target = x[indices]
        label = y[indices]
        source = torch.randn(target.shape, generator=generator)
        t = torch.rand(batch_size, generator=generator)
        state = (1.0 - t[:, None]) * source + t[:, None] * target
        velocity = target - source
        prediction = model(state, t, label)
        loss = F.mse_loss(prediction, velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            losses.append(float(loss.detach()))
    return FlowTrainingResult(model, losses, time.perf_counter() - start)


@torch.no_grad()
def sample_rectified_flow(
    model: TokenVelocity,
    labels: np.ndarray,
    nfe: int,
    seed: int,
    source: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    label = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    generator = torch.Generator().manual_seed(seed)
    if source is None:
        state = torch.randn((len(label), model.tokens), generator=generator)
    else:
        state = torch.as_tensor(np.asarray(source, dtype=np.float32)).clone()
    step = 1.0 / nfe
    for index in range(nfe):
        t = torch.full((len(label),), index / nfe, dtype=state.dtype)
        state = state + step * model(state, t, label)
    return state.cpu().numpy().astype(np.float64)


class ConditionalVAE(nn.Module):
    def __init__(self, dimension: int = 64, hidden: int = 128, latent: int = 64, classes: int = 10):
        super().__init__()
        self.dimension = dimension
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(dimension + classes, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent + classes, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dimension), nn.Sigmoid())
        self.classes = classes

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(labels, self.classes).float()
        h = self.encoder(torch.cat((x, one_hot), dim=1))
        return self.mu(h), self.logvar(h).clamp(-12.0, 8.0)

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, self.classes).float()
        return self.decoder(torch.cat((z, one_hot), dim=1))

    def forward(self, x: torch.Tensor, labels: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decode(z, labels), mu, logvar

    def decoder_flop_proxy(self, batch: int = 1) -> int:
        layers = [(self.latent + self.classes, 128), (128, 128), (128, self.dimension)]
        return int(batch * 2 * sum(left * right for left, right in layers))


@dataclass
class VAETrainingResult:
    model: ConditionalVAE
    losses: list[float]
    seconds: float


def train_vae(
    data: np.ndarray,
    labels: np.ndarray,
    seed: int,
    beta: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> VAETrainingResult:
    import time
    seed_everything(seed)
    x = torch.as_tensor(np.asarray(data, dtype=np.float32).reshape(len(data), -1))
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    model = ConditionalVAE(dimension=x.shape[1], latent=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 127)
    losses = []
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = torch.randint(0, len(x), (batch_size,), generator=generator)
        batch = x[indices]
        label = y[indices]
        reconstruction, mu, logvar = model(batch, label, generator)
        distortion = F.mse_loss(reconstruction, batch, reduction="none").sum(dim=1).mean()
        rate = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1.0 - logvar, dim=1).mean()
        loss = distortion + beta * rate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            losses.append(float(loss.detach()))
    return VAETrainingResult(model, losses, time.perf_counter() - start)


@torch.no_grad()
def sample_vae(model: ConditionalVAE, labels: np.ndarray, seed: int) -> np.ndarray:
    model.eval()
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn((len(y), model.latent), generator=generator)
    return model.decode(z, y).cpu().numpy().reshape(len(y), 8, 8).astype(np.float64)


@torch.no_grad()
def reconstruct_vae(model: ConditionalVAE, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(np.asarray(data, dtype=np.float32).reshape(len(data), -1))
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
    mu, _ = model.encode(x, y)
    return model.decode(mu, y).cpu().numpy().reshape(len(data), 8, 8).astype(np.float64)
