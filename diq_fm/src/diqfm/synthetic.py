from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import nn


torch.set_num_threads(4)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def features(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    q1, q2 = q[:, 0:1], q[:, 1:2]
    return np.concatenate(
        [
            np.ones((len(q), 1)),
            q,
            np.sin(q),
            np.cos(q) - 1,
            q1 * q2,
            q1 * q1 / (1 + q1 * q1),
            q2 * q2 / (1 + q2 * q2),
        ],
        axis=1,
    )


def h_true_np(u: np.ndarray) -> np.ndarray:
    q, r = u[:, :2], u[:, 2:]
    q1, q2 = q[:, 0], q[:, 1]
    r1, r2, r3, r4 = [r[:, i] for i in range(4)]
    h1 = (
        0.72 * np.sin(q1)
        + 0.24 * q2 * r1 / (1 + np.abs(q2 * r1))
        + 0.20 * np.tanh(r2)
        + 0.10 * np.sin(r3 + r4)
    )
    h2 = (
        0.58 * (np.cos(q2) - 1)
        + 0.22 * q1 * q1 / (1 + q1 * q1)
        + 0.18 * r3 * r4 / (1 + np.abs(r3 * r4))
        + 0.12 * np.sin(q1 * r2)
    )
    return np.stack([h1, h2], axis=1)


def make_params(n_context: int = 8, seed: int = 123) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_context, endpoint=False)
    state = np.stack(
        [
            2.0 * np.cos(angles) + 0.25 * np.sin(3 * angles),
            1.5 * np.sin(angles) + 0.2 * np.cos(2 * angles),
        ],
        axis=1,
    )
    means = []
    for angle in angles:
        base = np.array([1.2 * np.cos(angle), 1.0 * np.sin(angle)])
        offsets = np.array([[-1.0, -0.4], [0.7, -0.8], [0.2, 1.1]])
        rotation = np.array(
            [
                [np.cos(0.4 * angle), -np.sin(0.4 * angle)],
                [np.sin(0.4 * angle), np.cos(0.4 * angle)],
            ]
        )
        means.append(base + offsets @ rotation.T)
    qmeans = np.asarray(means)
    weights = np.array([0.25, 0.45, 0.30])
    p = features(np.zeros((1, 2))).shape[1]
    fiber_coef = rng.normal(scale=0.22, size=(n_context, 4, p))
    fiber_coef[:, :, 1:3] += rng.normal(scale=0.35, size=(n_context, 4, 2))
    cholesky = []
    for _ in range(n_context):
        matrix = rng.normal(scale=0.15, size=(4, 4))
        covariance = matrix @ matrix.T + 0.08 * np.eye(4)
        cholesky.append(np.linalg.cholesky(covariance))
    return {
        "s": state,
        "qmeans": qmeans,
        "weights": weights,
        "B": fiber_coef,
        "L": np.asarray(cholesky),
    }


def sample(
    n_per_context: int,
    params: dict[str, np.ndarray],
    seed: int,
    paired: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_context = len(params["s"])
    n_components = len(params["weights"])
    rows: list[tuple] = []
    for context in range(n_context):
        n = n_per_context * (2 if paired else 1)
        component = rng.choice(n_components, size=n, p=params["weights"])
        q = params["qmeans"][context, component] + rng.standard_t(df=6, size=(n, 2)) * np.array([0.28, 0.22])
        q[:, 1] += 0.16 * (q[:, 0] ** 2 - np.mean(q[:, 0] ** 2))
        mean_r = features(q) @ params["B"][context].T
        r = mean_r + rng.normal(size=(n, 4)) @ params["L"][context].T
        z = params["s"][context] + 0.035 * rng.normal(size=(n, 2))
        u = np.concatenate([q, r], axis=1)
        y = z + h_true_np(u)
        x = np.concatenate([y, u], axis=1)
        if paired:
            x = x.reshape(n_per_context, 2, 8)
            z = z.reshape(n_per_context, 2, 2)
            q = q.reshape(n_per_context, 2, 2)
            r = r.reshape(n_per_context, 2, 4)
            for i in range(n_per_context):
                rows.append((context, x[i, 0], x[i, 1], z[i, 0], z[i, 1], q[i, 0], q[i, 1], r[i, 0], r[i, 1]))
        else:
            for i in range(n):
                rows.append((context, x[i], z[i], q[i], r[i]))
    rng.shuffle(rows)
    if paired:
        return {
            "c": np.array([v[0] for v in rows]),
            "xa": np.stack([v[1] for v in rows]),
            "xb": np.stack([v[2] for v in rows]),
            "za": np.stack([v[3] for v in rows]),
            "zb": np.stack([v[4] for v in rows]),
            "qa": np.stack([v[5] for v in rows]),
            "qb": np.stack([v[6] for v in rows]),
            "ra": np.stack([v[7] for v in rows]),
            "rb": np.stack([v[8] for v in rows]),
        }
    return {
        "c": np.array([v[0] for v in rows]),
        "x": np.stack([v[1] for v in rows]),
        "z": np.stack([v[2] for v in rows]),
        "q": np.stack([v[3] for v in rows]),
        "r": np.stack([v[4] for v in rows]),
    }


class ShearNet(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u) - self.net(torch.zeros_like(u))


def train_shear(
    train_pairs: dict[str, np.ndarray],
    validation_pairs: dict[str, np.ndarray],
    seed: int = 0,
    steps: int = 1600,
    batch_size: int = 512,
) -> tuple[ShearNet, list[tuple[int, float, float]]]:
    set_seed(seed)
    model = ShearNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    xa = torch.tensor(train_pairs["xa"], dtype=torch.float32)
    xb = torch.tensor(train_pairs["xb"], dtype=torch.float32)
    va = torch.tensor(validation_pairs["xa"], dtype=torch.float32)
    vb = torch.tensor(validation_pairs["xb"], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_state = None
    best_validation = float("inf")
    stale = 0
    trace: list[tuple[int, float, float]] = []
    for step in range(steps):
        index = rng.integers(0, len(xa), size=min(batch_size, len(xa)))
        a, b = xa[index], xb[index]
        za = a[:, :2] - model(a[:, 2:])
        zb = b[:, :2] - model(b[:, 2:])
        pair_loss = 0.5 * ((za - zb) ** 2).sum(1).mean()
        regularizer = 1e-6 * sum(parameter.square().mean() for parameter in model.parameters())
        loss = pair_loss + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 40 == 0:
            with torch.no_grad():
                zva = va[:, :2] - model(va[:, 2:])
                zvb = vb[:, :2] - model(vb[:, 2:])
                validation = float((0.5 * ((zva - zvb) ** 2).sum(1).mean()).item())
            trace.append((step + 1, float(pair_loss.item()), validation))
            if validation < best_validation - 1e-6:
                best_validation = validation
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 10 and step > 600:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, trace


def fit_linear_shear(train_pairs: dict[str, np.ndarray]) -> np.ndarray:
    du = train_pairs["xa"][:, 2:] - train_pairs["xb"][:, 2:]
    dy = train_pairs["xa"][:, :2] - train_pairs["xb"][:, :2]
    return np.linalg.lstsq(du, dy, rcond=1e-6)[0]


def encode(model: ShearNet, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        correction = model(torch.tensor(x[:, 2:], dtype=torch.float32)).numpy()
    return np.concatenate([x[:, :2] - correction, x[:, 2:]], axis=1)


def encode_linear(matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:, :2] - x[:, 2:] @ matrix, x[:, 2:]], axis=1)


def fit_diq_density(
    y: np.ndarray,
    context: np.ndarray,
    n_context: int,
    seed: int = 0,
    components: tuple[int, ...] = (2, 3, 4),
) -> list[tuple]:
    models = []
    for c in range(n_context):
        yc = y[context == c]
        z, q, r = yc[:, :2], yc[:, 2:4], yc[:, 4:]
        z_mean = z.mean(0)
        z_covariance = np.cov(z, rowvar=False) + 1e-4 * np.eye(2)
        best, best_bic = None, np.inf
        for k in components:
            candidate = GaussianMixture(
                k,
                covariance_type="full",
                reg_covar=1e-4,
                random_state=seed + c,
                n_init=1,
                max_iter=120,
            ).fit(q)
            bic = candidate.bic(q)
            if bic < best_bic:
                best, best_bic = candidate, bic
        design = features(q)
        coefficient = np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]), design.T @ r)
        residual = r - design @ coefficient
        residual_covariance = np.cov(residual, rowvar=False) + 1e-4 * np.eye(4)
        models.append((z_mean, z_covariance, best, coefficient, residual_covariance))
    return models


def mvn_logpdf(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, (x - mean).T)
    return -0.5 * (
        x.shape[1] * np.log(2 * np.pi)
        + 2 * np.log(np.diag(cholesky)).sum()
        + (whitened * whitened).sum(0)
    )


def diq_logpdf(models: list[tuple], y: np.ndarray, context: np.ndarray) -> np.ndarray:
    output = np.empty(len(y))
    for c, model in enumerate(models):
        index = np.flatnonzero(context == c)
        if not len(index):
            continue
        yc = y[index]
        z, q, r = yc[:, :2], yc[:, 2:4], yc[:, 4:]
        z_mean, z_covariance, q_gmm, coefficient, r_covariance = model
        output[index] = (
            mvn_logpdf(z, z_mean, z_covariance)
            + q_gmm.score_samples(q)
            + mvn_logpdf(r, features(q) @ coefficient, r_covariance)
        )
    return output


def sample_diq(models: list[tuple], context: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((len(context), 8))
    for c, model in enumerate(models):
        index = np.flatnonzero(context == c)
        n = len(index)
        if not n:
            continue
        z_mean, z_covariance, q_gmm, coefficient, r_covariance = model
        z = rng.multivariate_normal(z_mean, z_covariance, size=n)
        component = rng.choice(q_gmm.n_components, size=n, p=q_gmm.weights_)
        q = np.empty((n, 2))
        for k in range(q_gmm.n_components):
            selected = np.flatnonzero(component == k)
            if len(selected):
                q[selected] = rng.multivariate_normal(q_gmm.means_[k], q_gmm.covariances_[k], size=len(selected))
        r = features(q) @ coefficient + rng.multivariate_normal(np.zeros(4), r_covariance, size=n)
        output[index] = np.concatenate([z, q, r], axis=1)
    return output


def decode_shear(model: ShearNet, y: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        correction = model(torch.tensor(y[:, 2:], dtype=torch.float32)).numpy()
    return np.concatenate([y[:, :2] + correction, y[:, 2:]], axis=1)


def fit_full_gmms(
    x: np.ndarray,
    context: np.ndarray,
    n_context: int,
    seed: int = 0,
    components: tuple[int, ...] = (2, 3, 4),
) -> list[GaussianMixture]:
    models = []
    for c in range(n_context):
        xc = x[context == c]
        best, best_bic = None, np.inf
        for k in components:
            candidate = GaussianMixture(
                k,
                covariance_type="full",
                reg_covar=1e-4,
                random_state=seed + c,
                n_init=1,
                max_iter=120,
            ).fit(xc)
            bic = candidate.bic(xc)
            if bic < best_bic:
                best, best_bic = candidate, bic
        models.append(best)
    return models


def model_logpdf(models: list[GaussianMixture], x: np.ndarray, context: np.ndarray) -> np.ndarray:
    output = np.empty(len(x))
    for c, model in enumerate(models):
        index = np.flatnonzero(context == c)
        if len(index):
            output[index] = model.score_samples(x[index])
    return output


def sample_gmms(models: list[GaussianMixture], context: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((len(context), models[0].means_.shape[1]))
    for c, model in enumerate(models):
        index = np.flatnonzero(context == c)
        n = len(index)
        if not n:
            continue
        component = rng.choice(model.n_components, size=n, p=model.weights_)
        samples = np.empty((n, output.shape[1]))
        for k in range(model.n_components):
            selected = np.flatnonzero(component == k)
            if len(selected):
                samples[selected] = rng.multivariate_normal(model.means_[k], model.covariances_[k], size=len(selected))
        output[index] = samples
    return output


def fit_pca_mixture(
    x: np.ndarray,
    context: np.ndarray,
    n_context: int,
    latent_dim: int = 4,
    seed: int = 0,
    components: int = 3,
) -> tuple[PCA, float, list[GaussianMixture]]:
    pca = PCA(latent_dim, random_state=seed).fit(x)
    z = pca.transform(x)
    reconstruction = pca.inverse_transform(z)
    variance = float(np.mean((x - reconstruction) ** 2) + 1e-4)
    models = [
        GaussianMixture(
            components,
            covariance_type="full",
            reg_covar=1e-4,
            random_state=seed + c,
            n_init=1,
            max_iter=120,
        ).fit(z[context == c])
        for c in range(n_context)
    ]
    return pca, variance, models


def sample_pca_model(model: tuple, context: np.ndarray, seed: int = 0) -> np.ndarray:
    pca, variance, gmms = model
    z = sample_gmms(gmms, context, seed)
    rng = np.random.default_rng(seed + 999)
    return pca.inverse_transform(z) + rng.normal(scale=np.sqrt(variance), size=(len(context), pca.n_features_in_))


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    projections: int = 256,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y))
    x = x[rng.choice(len(x), n, replace=False)]
    y = y[rng.choice(len(y), n, replace=False)]
    directions = rng.normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    return float(np.sqrt(np.mean((np.sort(x @ directions, axis=0) - np.sort(y @ directions, axis=0)) ** 2)))


def frechet(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = x.mean(0), y.mean(0)
    covariance_x = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    covariance_y = np.cov(y, rowvar=False) + 1e-6 * np.eye(y.shape[1])
    root = sqrtm(covariance_x @ covariance_y).real
    value = np.sum((mean_x - mean_y) ** 2) + np.trace(covariance_x + covariance_y - 2 * root)
    return float(max(0.0, value))


def run(
    seed: int = 0,
    n_train: int = 40,
    n_validation: int = 80,
    n_test: int = 800,
    steps: int = 1000,
) -> dict[str, float]:
    params = make_params(seed=31415)
    train = sample(n_train, params, seed + 10, paired=True)
    validation = sample(n_validation, params, seed + 20, paired=True)
    test = sample(n_test, params, seed + 30, paired=False)
    chart, trace = train_shear(train, validation, seed=seed, steps=steps)
    linear = fit_linear_shear(train)
    train_x = np.concatenate([train["xa"], train["xb"]], axis=0)
    train_context = np.concatenate([train["c"], train["c"]], axis=0)
    learned_train = encode(chart, train_x)
    linear_train = encode_linear(linear, train_x)
    learned_test = encode(chart, test["x"])
    linear_test = encode_linear(linear, test["x"])
    with torch.no_grad():
        estimated_h = chart(torch.tensor(test["x"][:, 2:], dtype=torch.float32)).numpy()
    metrics = {
        "h_mse": float(np.mean((estimated_h - h_true_np(test["x"][:, 2:])) ** 2)),
        "state_mse": float(np.mean((learned_test[:, :2] - test["z"]) ** 2)),
        "state_mse_linear": float(np.mean((linear_test[:, :2] - test["z"]) ** 2)),
        "state_r2": float(1 - np.sum((learned_test[:, :2] - test["z"]) ** 2) / np.sum((test["z"] - test["z"].mean(0)) ** 2)),
        "pair_val_loss": trace[-1][2],
        "steps": trace[-1][0],
    }
    n_context = len(params["s"])
    diq = fit_diq_density(learned_train, train_context, n_context, seed)
    linear_density = fit_diq_density(linear_train, train_context, n_context, seed)
    full = fit_full_gmms(train_x, train_context, n_context, seed)
    pca = fit_pca_mixture(train_x, train_context, n_context, latent_dim=4, seed=seed)
    metrics.update(
        {
            "nll_diq": -float(diq_logpdf(diq, learned_test, test["c"]).mean()),
            "nll_linear": -float(diq_logpdf(linear_density, linear_test, test["c"]).mean()),
            "nll_full_gmm": -float(model_logpdf(full, test["x"], test["c"]).mean()),
        }
    )
    generated_diq = decode_shear(chart, sample_diq(diq, test["c"], seed + 100))
    generated_linear_coordinates = sample_diq(linear_density, test["c"], seed + 101)
    generated_linear = np.concatenate(
        [generated_linear_coordinates[:, :2] + generated_linear_coordinates[:, 2:] @ linear, generated_linear_coordinates[:, 2:]],
        axis=1,
    )
    generated_full = sample_gmms(full, test["c"], seed + 102)
    generated_pca = sample_pca_model(pca, test["c"], seed + 103)
    for name, generated in {
        "diq": generated_diq,
        "linear": generated_linear,
        "full_gmm": generated_full,
        "pca_latent": generated_pca,
    }.items():
        metrics[f"swd_{name}"] = sliced_wasserstein(generated, test["x"], seed=seed)
        metrics[f"fid_{name}"] = frechet(generated, test["x"])
    metrics["params_diq_density"] = int(
        sum((m[2].n_components - 1) + m[2].n_components * 2 + m[2].n_components * 3 + 2 + 3 + features(np.zeros((1, 2))).shape[1] * 4 + 10 for m in diq)
    )
    metrics["params_full_gmm"] = int(
        sum((m.n_components - 1) + m.n_components * 8 + m.n_components * 36 for m in full)
    )
    return metrics
