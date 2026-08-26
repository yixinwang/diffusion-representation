from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def balanced_labels(n: int, classes: int = 10) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % classes


def energy_score(reference: np.ndarray, reference_labels: np.ndarray, generated: np.ndarray, generated_labels: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    y = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    scores = []
    norm = np.sqrt(x.shape[1])
    for class_id in range(10):
        xr = x[np.asarray(reference_labels) == class_id]
        yg = y[np.asarray(generated_labels) == class_id]
        if len(xr) == 0 or len(yg) < 2:
            continue
        cross = cdist(xr, yg).mean(axis=1)
        within = cdist(yg, yg)
        # V-statistic is stable and differs by O(1/m).
        scores.extend((cross - 0.5 * within.mean()).tolist())
    return float(np.mean(scores) / norm)


def sliced_wasserstein(reference: np.ndarray, generated: np.ndarray, seed: int, projections: int = 256) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    y = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(x @ direction, y @ direction) for direction in directions]))


def rbf_mmd(reference: np.ndarray, generated: np.ndarray, maximum: int = 800) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)[:maximum]
    y = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)[:maximum]
    combined = np.concatenate((x, y))
    distances = cdist(combined[: min(len(combined), 400)], combined[: min(len(combined), 400)], metric="sqeuclidean")
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 1e-8)
    kxx = np.exp(-cdist(x, x, "sqeuclidean") / (2.0 * bandwidth))
    kyy = np.exp(-cdist(y, y, "sqeuclidean") / (2.0 * bandwidth))
    kxy = np.exp(-cdist(x, y, "sqeuclidean") / (2.0 * bandwidth))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def frechet_pixel(reference: np.ndarray, generated: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    y = np.asarray(generated, dtype=np.float64).reshape(len(generated), -1)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    root = sqrtm(cx @ cy)
    if np.iscomplexobj(root):
        root = root.real
    return float(np.sum((mx - my) ** 2) + np.trace(cx + cy - 2.0 * root))


def classifier_metrics(train: np.ndarray, train_labels: np.ndarray, generated: np.ndarray, requested_labels: np.ndarray) -> dict[str, float | list[float]]:
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=3.0))
    classifier.fit(np.asarray(train).reshape(len(train), -1), train_labels)
    predicted = classifier.predict(np.asarray(generated).reshape(len(generated), -1))
    distribution = np.bincount(predicted, minlength=10) / len(predicted)
    return {
        "requested_label_accuracy": float(np.mean(predicted == requested_labels)),
        "predicted_class_distribution": distribution.tolist(),
        "predicted_class_l1_from_uniform": float(np.sum(np.abs(distribution - 0.1))),
    }


def coverage_metrics(train: np.ndarray, reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    train_flat = np.asarray(train).reshape(len(train), -1)
    ref_flat = np.asarray(reference).reshape(len(reference), -1)
    gen_flat = np.asarray(generated).reshape(len(generated), -1)
    norm = np.sqrt(ref_flat.shape[1])
    return {
        "reference_to_generated_nn": float(cdist(ref_flat, gen_flat).min(axis=1).mean() / norm),
        "generated_to_train_nn": float(cdist(gen_flat, train_flat).min(axis=1).mean() / norm),
        "generated_pairwise_mean": float(cdist(gen_flat[:500], gen_flat[:500]).mean() / norm),
    }


def all_metrics(
    train: np.ndarray,
    train_labels: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    generated: np.ndarray,
    generated_labels: np.ndarray,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "energy_score": energy_score(reference, reference_labels, generated, generated_labels),
        "swd": sliced_wasserstein(reference, generated, seed),
        "mmd2": rbf_mmd(reference, generated),
        "pixel_frechet": frechet_pixel(reference, generated),
    }
    result.update(classifier_metrics(train, train_labels, generated, generated_labels))
    result.update(coverage_metrics(train, reference, generated))
    return result
