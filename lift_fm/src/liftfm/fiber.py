from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri

LOG_2PI = math.log(2.0 * math.pi)


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def _det_one_shape(residual: np.ndarray, shrinkage: float = 0.02, eig_floor: float = 0.15, eig_ceiling: float = 6.0) -> np.ndarray:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, 3)
    scatter = rows.T @ rows / max(len(rows), 1)
    iso = np.trace(scatter) / 3.0
    scatter = (1.0 - shrinkage) * scatter + shrinkage * iso * np.eye(3)
    values, vectors = np.linalg.eigh((scatter + scatter.T) / 2.0)
    values = np.maximum(values, 1e-10)
    logs = np.log(values)
    # Project log spectrum to sum zero with box constraints.
    lo, hi = math.log(eig_floor), math.log(eig_ceiling)
    left, right = float(np.min(logs - hi)), float(np.max(logs - lo))
    for _ in range(100):
        mid = 0.5 * (left + right)
        total = np.sum(np.clip(logs - mid, lo, hi))
        if total > 0:
            left = mid
        else:
            right = mid
    projected = np.exp(np.clip(logs - 0.5 * (left + right), lo, hi))
    projected /= np.prod(projected) ** (1.0 / 3.0)
    shape = (vectors * projected) @ vectors.T
    return (shape + shape.T) / 2.0


def _recycle_component(normal: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(normal, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    zero = np.nextafter(0.0, 1.0)
    one = np.nextafter(1.0, 0.0)
    uniforms = np.clip(ndtr(values), zero, one)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    component = np.searchsorted(cumulative, uniforms, side="left")
    component = np.minimum(component, len(probabilities) - 1)
    previous = np.where(component == 0, 0.0, cumulative[np.maximum(component - 1, 0)])
    within = np.clip((uniforms - previous) / probabilities[component], zero, one)
    return component.astype(np.int64), ndtri(within)


@dataclass(frozen=True)
class JointGSM:
    weights: np.ndarray
    scales: np.ndarray
    shape: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        shape = np.asarray(self.shape, dtype=np.float64)
        if weights.ndim != 1 or scales.shape != weights.shape or np.any(weights <= 0) or np.any(scales <= 0):
            raise ValueError("invalid mixture")
        weights = weights / np.sum(weights)
        shape = (shape + shape.T) / 2.0
        chol = np.linalg.cholesky(shape)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "chol", chol)
        object.__setattr__(self, "log_det_chol", float(np.log(np.diag(chol)).sum()))

    def _radius(self, residual: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(residual, dtype=np.float64)
        if array.shape[-1] != 3:
            raise ValueError("last dimension must be 3")
        rows = array.reshape(-1, 3)
        white = solve_triangular(self.chol, rows.T, lower=True, check_finite=False).T
        return np.sum(white**2, axis=1), array.shape[:-1]

    def component_log_prob(self, residual: np.ndarray) -> np.ndarray:
        radius, leading = self._radius(residual)
        logs = (
            np.log(self.weights)[None, :]
            - 1.5 * LOG_2PI
            - self.log_det_chol
            - 3.0 * np.log(self.scales)[None, :]
            - 0.5 * radius[:, None] / self.scales[None, :] ** 2
        )
        return logs.reshape((*leading, len(self.weights)))

    def log_prob(self, residual: np.ndarray) -> np.ndarray:
        return _logsumexp(self.component_log_prob(residual), axis=-1)

    def marginal_log_prob(self, residual: np.ndarray) -> np.ndarray:
        array = np.asarray(residual, dtype=np.float64)
        rows = array.reshape(-1, 3)
        std = self.scales[:, None] * np.sqrt(np.diag(self.shape))[None, :]
        logs = (
            np.log(self.weights)[None, :, None]
            - 0.5 * LOG_2PI
            - np.log(std)[None, :, :]
            - 0.5 * (rows[:, None, :] / std[None, :, :]) ** 2
        )
        return _logsumexp(logs, axis=1).reshape(array.shape)

    def product_log_prob(self, residual: np.ndarray) -> np.ndarray:
        return np.sum(self.marginal_log_prob(residual), axis=-1)

    def sample_joint(self, base_normal: np.ndarray) -> np.ndarray:
        array = np.asarray(base_normal, dtype=np.float64)
        rows = array.reshape(-1, 3).copy()
        component, remapped = _recycle_component(rows[:, 0], self.weights)
        rows[:, 0] = remapped
        sampled = (rows @ self.chol.T) * self.scales[component, None]
        return sampled.reshape(array.shape)

    def sample_product(self, base_normal: np.ndarray) -> np.ndarray:
        array = np.asarray(base_normal, dtype=np.float64)
        rows = array.reshape(-1, 3)
        sampled = np.empty_like(rows)
        marginal_shape = np.sqrt(np.diag(self.shape))
        for coordinate in range(3):
            component, remapped = _recycle_component(rows[:, coordinate], self.weights)
            sampled[:, coordinate] = remapped * self.scales[component] * marginal_shape[coordinate]
        return sampled.reshape(array.shape)


def fit_joint_gsm(residual: np.ndarray, components: int = 4, max_iterations: int = 150) -> JointGSM:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, 3)
    if len(rows) < 20 * components:
        raise ValueError("insufficient residual blocks")
    shape = _det_one_shape(rows)
    template = JointGSM(np.array([1.0]), np.array([1.0]), shape)
    radius, _ = template._radius(rows)
    rms = math.sqrt(np.mean(radius) / 3.0)
    scales = np.clip(rms * np.geomspace(0.45, 1.9, components), 0.02, 3.0)
    weights = np.full(components, 1.0 / components)
    previous = -np.inf
    for _ in range(max_iterations):
        model = JointGSM(weights, scales, shape)
        component_logs = model.component_log_prob(rows)
        point = _logsumexp(component_logs, axis=1)
        responsibilities = np.exp(component_logs - point[:, None])
        mass = responsibilities.sum(axis=0)
        weights = np.maximum(mass / len(rows), 1e-4)
        weights /= weights.sum()
        variances = (responsibilities * radius[:, None]).sum(axis=0) / np.maximum(3.0 * mass, 1e-12)
        scales = np.sqrt(np.clip(variances, 0.02**2, 3.0**2))
        order = np.argsort(scales)
        scales, weights = scales[order], weights[order]
        objective = float(np.sum(point))
        if np.isfinite(previous) and objective - previous <= 1e-8 * (1.0 + abs(previous)):
            break
        previous = objective
    return JointGSM(weights, scales, shape)


def design_features(coarse_standardized: np.ndarray, labels: np.ndarray, classes: int = 10) -> np.ndarray:
    z = np.asarray(coarse_standardized, dtype=np.float64).reshape(len(coarse_standardized), -1)
    y = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(classes, dtype=np.float64)[y]
    return np.column_stack((np.ones(len(z)), z, z**2, np.tanh(z), one_hot))


@dataclass
class ConditionalBlockFiber:
    ridge: np.ndarray
    location_scale: np.ndarray
    class_models: tuple[JointGSM, ...]
    active_mean: np.ndarray
    active_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        coarse: np.ndarray,
        details: np.ndarray,
        labels: np.ndarray,
        components: int = 4,
        ridge_penalty: float = 1e-2,
    ) -> "ConditionalBlockFiber":
        z = np.asarray(coarse, dtype=np.float64)
        r = np.asarray(details, dtype=np.float64)
        active_mean = z.reshape(len(z), -1).mean(axis=0)
        active_scale = z.reshape(len(z), -1).std(axis=0)
        active_scale = np.maximum(active_scale, 1e-3)
        z_std = (z.reshape(len(z), -1) - active_mean) / active_scale
        features = design_features(z_std, labels)
        targets = r.reshape(len(r), -1)
        penalty = ridge_penalty * np.eye(features.shape[1])
        penalty[0, 0] = 0.0
        ridge = np.linalg.solve(features.T @ features + penalty, features.T @ targets)
        mean = features @ ridge
        residual = (targets - mean).reshape(r.shape)
        location_scale = np.sqrt(np.mean(residual**2, axis=(0, 3)))
        location_scale = np.maximum(location_scale, 1e-3)
        normalized = residual / location_scale[None, :, :, None]
        models = []
        for class_id in range(10):
            selected = normalized[np.asarray(labels) == class_id]
            models.append(fit_joint_gsm(selected, components=components))
        return cls(ridge, location_scale, tuple(models), active_mean, active_scale)

    def _mean(self, coarse: np.ndarray, labels: np.ndarray) -> np.ndarray:
        z = np.asarray(coarse, dtype=np.float64).reshape(len(coarse), -1)
        z_std = (z - self.active_mean) / self.active_scale
        features = design_features(z_std, labels)
        return (features @ self.ridge).reshape(len(coarse), self.location_scale.shape[0], self.location_scale.shape[1], 3)

    def residual(self, coarse: np.ndarray, details: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return (np.asarray(details, dtype=np.float64) - self._mean(coarse, labels)) / self.location_scale[None, :, :, None]

    def nll_per_coefficient(self, coarse: np.ndarray, details: np.ndarray, labels: np.ndarray, joint: bool) -> float:
        residual = self.residual(coarse, details, labels)
        total = 0.0
        count = 0
        jacobian = float(np.sum(np.log(self.location_scale)) * 3.0)
        for class_id, model in enumerate(self.class_models):
            selected = np.asarray(labels) == class_id
            if not np.any(selected):
                continue
            values = residual[selected]
            log_prob = model.log_prob(values) if joint else model.product_log_prob(values)
            total -= float(np.sum(log_prob))
            total += int(np.sum(selected)) * jacobian
            count += values.size
        return total / count

    def sample(self, coarse: np.ndarray, labels: np.ndarray, base_normal: np.ndarray, joint: bool) -> np.ndarray:
        mean = self._mean(coarse, labels)
        noise = np.asarray(base_normal, dtype=np.float64)
        if noise.shape != mean.shape:
            raise ValueError("fiber base noise shape mismatch")
        output = np.empty_like(noise)
        for class_id, model in enumerate(self.class_models):
            selected = np.asarray(labels) == class_id
            if not np.any(selected):
                continue
            sampled = model.sample_joint(noise[selected]) if joint else model.sample_product(noise[selected])
            output[selected] = sampled
        return mean + output * self.location_scale[None, :, :, None]
