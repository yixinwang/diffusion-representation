"""Observed-data conditional models for certified transport-depth routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)
DETAIL_COUNT = 9 * 16 * 16


def paired_dequantize(images: np.ndarray, record_ids: np.ndarray, seed: int) -> np.ndarray:
    """Uniformly dequantize uint8 images with a counter keyed by record and pixel."""
    values = np.asarray(images, dtype=np.uint8)
    ids = np.asarray(record_ids, dtype=np.uint64)
    if values.ndim != 4 or values.shape[1:] != (3, 32, 32) or ids.shape != (len(values),):
        raise ValueError("expected NCHW CIFAR images and one record id per image")
    pixel = np.arange(3 * 32 * 32, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        counter = ids[:, None] * np.uint64(3 * 32 * 32) + pixel + np.uint64(seed) * np.uint64(0x9E3779B1)
        counter ^= counter >> np.uint64(30)
        counter *= np.uint64(0xBF58476D1CE4E5B9)
        counter ^= counter >> np.uint64(27)
        counter *= np.uint64(0x94D049BB133111EB)
        counter ^= counter >> np.uint64(31)
    uniform = ((counter >> np.uint64(40)).astype(np.float32) + 0.5) / float(1 << 24)
    return (values.reshape(len(values), -1).astype(np.float32) + uniform).reshape(values.shape) / 256.0


def image_haar(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB low-pass and nine ordered 2D-Haar detail channels."""
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (3, 32, 32):
        raise ValueError("expected NCHW 32x32 RGB images")
    root_two = np.float32(math.sqrt(2.0))
    low_w = (values[..., 0::2] + values[..., 1::2]) / root_two
    high_w = (values[..., 0::2] - values[..., 1::2]) / root_two
    ll = (low_w[:, :, 0::2] + low_w[:, :, 1::2]) / root_two
    hl = (low_w[:, :, 0::2] - low_w[:, :, 1::2]) / root_two
    lh = (high_w[:, :, 0::2] + high_w[:, :, 1::2]) / root_two
    hh = (high_w[:, :, 0::2] - high_w[:, :, 1::2]) / root_two
    return ll, np.concatenate((hl, lh, hh), axis=1)


def coarse_features(coarse: np.ndarray, sample: np.ndarray | None = None) -> np.ndarray:
    values = np.moveaxis(np.asarray(coarse, dtype=np.float64), 1, -1).reshape(-1, 3)
    if sample is not None:
        values = values[sample]
    products = np.column_stack((values[:, 0] * values[:, 1], values[:, 0] * values[:, 2], values[:, 1] * values[:, 2]))
    return np.column_stack((np.ones(len(values)), values, values**2, products, np.tanh(values)))


def flatten_details(detail: np.ndarray, sample: np.ndarray | None = None) -> np.ndarray:
    values = np.moveaxis(np.asarray(detail, dtype=np.float64), 1, -1).reshape(-1, 9)
    return values if sample is None else values[sample]


def fit_ridge_locations(
    coarse: np.ndarray,
    detail: np.ndarray,
    autoregressive: bool,
    ridge: float = 1e-3,
    sample: np.ndarray | None = None,
) -> list[np.ndarray]:
    base = coarse_features(coarse, sample)
    targets = flatten_details(detail, sample)
    coefficients: list[np.ndarray] = []
    for channel in range(9):
        features = base if not autoregressive or channel == 0 else np.column_stack((base, targets[:, :channel]))
        gram = features.T @ features
        penalty = ridge * np.eye(features.shape[1])
        penalty[0, 0] = 0.0
        coefficients.append(np.linalg.solve(gram + penalty, features.T @ targets[:, channel]))
    return coefficients


def predict_locations(
    coarse: np.ndarray,
    detail: np.ndarray,
    coefficients: list[np.ndarray],
    autoregressive: bool,
    sample: np.ndarray | None = None,
) -> np.ndarray:
    base = coarse_features(coarse, sample)
    targets = flatten_details(detail, sample)
    means = np.empty_like(targets)
    for channel, beta in enumerate(coefficients):
        features = base if not autoregressive or channel == 0 else np.column_stack((base, targets[:, :channel]))
        means[:, channel] = np.clip(features @ beta, -1.0, 1.0)
    return means


def coarse_energy_strata(
    coarse: np.ndarray,
    boundaries: np.ndarray | None = None,
    sample: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.moveaxis(np.asarray(coarse, dtype=np.float64), 1, -1).reshape(-1, 3)
    if sample is not None:
        rgb = rgb[sample]
    energy = np.sum(rgb**2, axis=1)
    if boundaries is None:
        boundaries = np.quantile(energy, (0.25, 0.5, 0.75))
    return np.searchsorted(boundaries, energy, side="right"), np.asarray(boundaries, dtype=float)


@dataclass(frozen=True)
class ScaleMixture:
    weights: np.ndarray
    scales: np.ndarray

    def log_prob(self, residual: np.ndarray) -> np.ndarray:
        values = np.asarray(residual, dtype=float)[:, None]
        logs = np.log(self.weights)[None, :] - np.log(self.scales)[None, :] - 0.5 * (LOG_2PI + (values / self.scales[None, :]) ** 2)
        maximum = np.max(logs, axis=1)
        return maximum + np.log(np.exp(logs - maximum[:, None]).sum(axis=1))


def fit_scale_mixture(
    residual: np.ndarray,
    components: int,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
    weight_floor: float = 1e-4,
    scale_floor: float = 0.05,
    scale_ceiling: float = 2.0,
) -> ScaleMixture:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    if values.size < components:
        raise ValueError("not enough residuals for mixture")
    rms = max(float(np.sqrt(np.mean(values**2))), scale_floor)
    scales = np.clip(rms * np.geomspace(0.5, 1.8, components), scale_floor, scale_ceiling)
    weights = np.full(components, 1.0 / components)
    previous = -np.inf
    for _ in range(max_iterations):
        logs = np.log(weights)[None, :] - np.log(scales)[None, :] - 0.5 * (values[:, None] / scales[None, :]) ** 2
        maximum = np.max(logs, axis=1)
        responsibilities = np.exp(logs - maximum[:, None])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        mass = responsibilities.sum(axis=0)
        empirical_weights = mass / len(values)
        weights = weight_floor + (1.0 - components * weight_floor) * empirical_weights
        scales = np.sqrt((responsibilities * values[:, None] ** 2).sum(axis=0) / np.maximum(mass, 1e-12))
        scales = np.clip(scales, scale_floor, scale_ceiling)
        order = np.argsort(scales)
        scales, weights = scales[order], weights[order]
        objective = float(np.sum(maximum + np.log(np.exp(logs - maximum[:, None]).sum(axis=1))))
        if np.isfinite(previous) and abs(objective - previous) <= tolerance * (1.0 + abs(previous)):
            break
        previous = objective
    return ScaleMixture(weights=weights, scales=scales)


@dataclass(frozen=True)
class ConditionalModel:
    coefficients: list[np.ndarray]
    mixtures: list[list[ScaleMixture]]
    boundaries: np.ndarray
    autoregressive: bool
    conditional_strata: bool

    def channel_log_prob(self, coarse: np.ndarray, detail: np.ndarray) -> np.ndarray:
        means = predict_locations(coarse, detail, self.coefficients, self.autoregressive)
        targets = flatten_details(detail)
        strata, _ = coarse_energy_strata(coarse, self.boundaries)
        output = np.empty_like(targets)
        for channel in range(9):
            residual = targets[:, channel] - means[:, channel]
            if not self.conditional_strata:
                output[:, channel] = self.mixtures[channel][0].log_prob(residual)
                continue
            for stratum in range(4):
                selected = strata == stratum
                output[selected, channel] = self.mixtures[channel][stratum].log_prob(residual[selected])
        return output


def fit_conditional_model(
    coarse: np.ndarray,
    detail: np.ndarray,
    components: int,
    autoregressive: bool,
    conditional_strata: bool,
    rng: np.random.Generator,
    maximum_residuals: int = 250_000,
    coefficients: list[np.ndarray] | None = None,
    boundaries: np.ndarray | None = None,
    sample: np.ndarray | None = None,
) -> tuple[ConditionalModel, str]:
    total_sites = len(coarse) * 16 * 16
    if sample is None:
        sample_size = min(maximum_residuals, total_sites)
        sample = np.sort(rng.choice(total_sites, size=sample_size, replace=False))
    else:
        sample = np.asarray(sample, dtype=np.int64)
    if coefficients is None:
        coefficients = fit_ridge_locations(coarse, detail, autoregressive, sample=sample)
    means = predict_locations(coarse, detail, coefficients, autoregressive, sample=sample)
    targets = flatten_details(detail, sample)
    strata, fitted_boundaries = coarse_energy_strata(coarse, boundaries, sample)
    if boundaries is None:
        boundaries = fitted_boundaries
    sample_hash = hashlib.sha256(sample.astype("<i8").tobytes()).hexdigest()
    mixtures: list[list[ScaleMixture]] = []
    for channel in range(9):
        residual = targets[:, channel] - means[:, channel]
        if not conditional_strata:
            mixtures.append([fit_scale_mixture(residual, components)])
            continue
        channel_mixtures = []
        for stratum in range(4):
            selected = strata == stratum
            channel_mixtures.append(fit_scale_mixture(residual[selected], components))
        mixtures.append(channel_mixtures)
    return ConditionalModel(coefficients, mixtures, boundaries, autoregressive, conditional_strata), sample_hash


def per_image_channel_log_prob(model: ConditionalModel, coarse: np.ndarray, detail: np.ndarray) -> np.ndarray:
    site_scores = model.channel_log_prob(coarse, detail).reshape(len(coarse), 16 * 16, 9)
    return site_scores.sum(axis=1)


def mixture_log_bounds(weight_floor: float = 1e-4, scale_floor: float = 0.05, scale_ceiling: float = 2.0) -> tuple[float, float]:
    residual_bound = 2.0
    lower = math.log(weight_floor) - math.log(scale_ceiling) - 0.5 * LOG_2PI - residual_bound**2 / (2.0 * scale_floor**2)
    upper = -math.log(scale_floor) - 0.5 * LOG_2PI
    return lower, upper


def empirical_bernstein_upper(values: np.ndarray, lower: float, upper: float, family_size: int, alpha: float) -> float:
    sample = np.asarray(values, dtype=float)
    if len(sample) < 2 or not lower < upper or np.any(sample < lower - 1e-10) or np.any(sample > upper + 1e-10):
        raise ValueError("invalid bounded sample")
    variance = float(np.var(sample, ddof=1))
    log_term = math.log(2.0 * family_size / alpha)
    radius = math.sqrt(2.0 * variance * log_term / len(sample)) + 7.0 * (upper - lower) * log_term / (3.0 * (len(sample) - 1))
    return float(np.mean(sample) + radius)
