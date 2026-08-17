from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn

from . import synthetic as synthetic


torch.set_num_threads(4)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 4, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_variance = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 8),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mean(hidden), torch.clamp(self.log_variance(hidden), -8, 5)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return self.decode(z), mean, log_variance


def train_one(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    beta: float,
    seed: int,
    steps: int = 1000,
) -> tuple[VAE, float]:
    synthetic.set_seed(seed)
    model = VAE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    train = torch.tensor(train_x, dtype=torch.float32)
    validation = torch.tensor(validation_x, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_state = None
    best_validation = float("inf")
    stale = 0
    for step in range(steps):
        index = rng.integers(len(train), size=min(256, len(train)))
        x = train[index]
        reconstruction, mean, log_variance = model(x)
        reconstruction_loss = ((reconstruction - x) ** 2).mean()
        kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).mean()
        loss = reconstruction_loss + beta * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 40 == 0:
            with torch.no_grad():
                mean, log_variance = model.encode(validation)
                reconstruction = model.decode(mean)
                validation_loss = float(
                    ((reconstruction - validation) ** 2).mean().item()
                    + beta * 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).mean().item()
                )
            if validation_loss < best_validation - 1e-6:
                best_validation = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 10 and step > 600:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_validation


def fit_context_gmms(z: np.ndarray, context: np.ndarray, n_context: int, seed: int, components: int = 3) -> list[GaussianMixture]:
    return [
        GaussianMixture(
            components,
            covariance_type="full",
            reg_covar=1e-4,
            n_init=1,
            max_iter=120,
            random_state=seed + c,
        ).fit(z[context == c])
        for c in range(n_context)
    ]


def sample_gmms(models: list[GaussianMixture], context: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((len(context), models[0].means_.shape[1]))
    for c, model in enumerate(models):
        index = np.flatnonzero(context == c)
        n = len(index)
        component = rng.choice(model.n_components, n, p=model.weights_)
        values = np.empty((n, output.shape[1]))
        for k in range(model.n_components):
            selected = np.flatnonzero(component == k)
            if len(selected):
                values[selected] = rng.multivariate_normal(model.means_[k], model.covariances_[k], size=len(selected))
        output[index] = values
    return output


def run(seed: int = 0, n_train: int = 40, n_validation: int = 80, n_test: int = 800) -> dict[str, float]:
    params = synthetic.make_params(seed=31415)
    train = synthetic.sample(n_train, params, seed + 10, paired=True)
    validation = synthetic.sample(n_validation, params, seed + 20, paired=True)
    test = synthetic.sample(n_test, params, seed + 30, paired=False)
    train_x = np.concatenate([train["xa"], train["xb"]])
    train_context = np.concatenate([train["c"], train["c"]])
    train_state = np.concatenate([train["za"], train["zb"]])
    validation_x = np.concatenate([validation["xa"], validation["xb"]])
    candidates = []
    for i, beta in enumerate([0.0, 1e-4, 1e-3, 1e-2]):
        model, objective = train_one(train_x, validation_x, beta, seed + 100 * i)
        candidates.append((objective, beta, model))
    validation_objective, beta, model = min(candidates, key=lambda value: value[0])
    with torch.no_grad():
        z_train = model.encode(torch.tensor(train_x, dtype=torch.float32))[0].numpy()
        z_test = model.encode(torch.tensor(test["x"], dtype=torch.float32))[0].numpy()
        reconstruction = model.decode(torch.tensor(z_train, dtype=torch.float32)).numpy()
    gmms = fit_context_gmms(z_train, train_context, 8, seed)
    generated_z = sample_gmms(gmms, test["c"], seed + 999)
    with torch.no_grad():
        generated_mean = model.decode(torch.tensor(generated_z, dtype=torch.float32)).numpy()
    decoder_variance = max(1e-5, float(np.mean((train_x - reconstruction) ** 2)))
    rng = np.random.default_rng(seed + 777)
    generated_stochastic = generated_mean + rng.normal(scale=np.sqrt(decoder_variance), size=generated_mean.shape)
    coefficient = np.linalg.lstsq(np.c_[np.ones(len(z_train)), z_train], train_state, rcond=None)[0]
    predicted_state = np.c_[np.ones(len(z_test)), z_test] @ coefficient
    state_r2 = 1 - np.sum((predicted_state - test["z"]) ** 2) / np.sum((test["z"] - test["z"].mean(0)) ** 2)
    return {
        "beta": beta,
        "val_objective": validation_objective,
        "decoder_sigma2": decoder_variance,
        "state_r2": float(state_r2),
        "swd_mean_decode": synthetic.sliced_wasserstein(generated_mean, test["x"], seed=seed),
        "swd_stochastic_decode": synthetic.sliced_wasserstein(generated_stochastic, test["x"], seed=seed),
        "fid_mean_decode": synthetic.frechet(generated_mean, test["x"]),
        "fid_stochastic_decode": synthetic.frechet(generated_stochastic, test["x"]),
    }
