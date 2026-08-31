"""Observed conditional RGB-block models for the frozen B1-v2 protocol.

All detail tensors in this module use the explicit layout
``[image, row, column, Haar band, color]``.  Conversion from the historical
NCHW Haar output is deliberate and lossless.  The module never loads data or
chooses a split; callers supply arrays and one common fitting-site sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Mapping

import numpy as np
from scipy.special import gammainc

from qalt.observed_routing import coarse_energy_strata, coarse_features
from qalt.rgb_block import (
    FixedShapeGSM,
    GSMFitDiagnostics,
    fit_determinant_one_shape,
    fit_fixed_shape_gsm,
    full_parent_masks,
    project_log_eigenvalues,
    water_filled_weights,
    within_band_parent_masks,
)


BAND_COUNT = 3
COLOR_COUNT = 3
DETAIL_DIMENSION = BAND_COUNT * COLOR_COUNT
STRATUM_COUNT = 4
COARSE_FEATURE_COUNT = 13
LOG_2PI = math.log(2.0 * math.pi)
PIT_GRID = np.linspace(0.05, 0.95, 19)
ARM_NAMES = (
    "b1",
    "b4",
    "b8",
    "p4",
    "z4",
    "d4",
    "b4_unconditional",
    "o4",
    "a4",
    "a8",
    "i4",
    "i8",
    "e4",
)


def detail_to_blocks(detail: np.ndarray) -> np.ndarray:
    """Convert historical ``[N, 9, H, W]`` details to explicit blocks."""

    values = np.asarray(detail)
    if values.ndim != 4 or values.shape[1] != DETAIL_DIMENSION:
        raise ValueError("expected detail shape [image, 9, row, column]")
    if not np.all(np.isfinite(values)):
        raise ValueError("detail coefficients must be finite")
    return values.reshape(len(values), BAND_COUNT, COLOR_COUNT, *values.shape[2:]).transpose(0, 3, 4, 1, 2)


def blocks_to_detail(blocks: np.ndarray) -> np.ndarray:
    """Invert :func:`detail_to_blocks` without changing coefficient order."""

    values = _as_blocks(blocks)
    return values.transpose(0, 3, 4, 1, 2).reshape(len(values), DETAIL_DIMENSION, *values.shape[1:3])


def image_haar_inverse(coarse: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Invert the exact LL/HL/LH/HH convention used by ``image_haar``."""

    coarse_values, block_values = _validate_inputs(coarse, blocks)
    detail = blocks_to_detail(block_values)
    hl, lh, hh = np.split(detail, BAND_COUNT, axis=1)
    root_two = math.sqrt(2.0)
    low_width = np.empty(
        (len(coarse_values), COLOR_COUNT, 2 * coarse_values.shape[2], coarse_values.shape[3]),
        dtype=np.float64,
    )
    high_width = np.empty_like(low_width)
    low_width[:, :, 0::2] = (coarse_values + hl) / root_two
    low_width[:, :, 1::2] = (coarse_values - hl) / root_two
    high_width[:, :, 0::2] = (lh + hh) / root_two
    high_width[:, :, 1::2] = (lh - hh) / root_two
    images = np.empty(
        (len(coarse_values), COLOR_COUNT, low_width.shape[2], 2 * low_width.shape[3]),
        dtype=np.float64,
    )
    images[..., 0::2] = (low_width + high_width) / root_two
    images[..., 1::2] = (low_width - high_width) / root_two
    return images


def _as_blocks(blocks: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 5 or values.shape[-2:] != (BAND_COUNT, COLOR_COUNT):
        raise ValueError("expected explicit [image, row, column, band, color] layout")
    if not np.all(np.isfinite(values)):
        raise ValueError("detail coefficients must be finite")
    return values


def _validate_inputs(coarse: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coarse_values = np.asarray(coarse)
    block_values = _as_blocks(blocks)
    if coarse_values.ndim != 4 or coarse_values.shape[1] != COLOR_COUNT:
        raise ValueError("expected coarse shape [image, color, row, column]")
    if not np.all(np.isfinite(coarse_values)):
        raise ValueError("coarse coefficients must be finite")
    if (
        len(coarse_values) != len(block_values)
        or coarse_values.shape[2] != block_values.shape[1]
        or coarse_values.shape[3] != block_values.shape[2]
    ):
        raise ValueError("coarse and detail spatial layouts do not match")
    return coarse_values, block_values


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def _stable_whitened(rows: np.ndarray, cholesky: np.ndarray) -> np.ndarray:
    """Forward-substitute rowwise so results do not depend on batch size."""

    values = np.asarray(rows, dtype=np.float64)
    dimension = values.shape[1]
    whitened = np.empty_like(values)
    for coordinate in range(dimension):
        numerator = values[:, coordinate].copy()
        for parent in range(coordinate):
            numerator -= cholesky[coordinate, parent] * whitened[:, parent]
        whitened[:, coordinate] = numerator / cholesky[coordinate, coordinate]
    return whitened


def _stable_component_log_prob(mixture: FixedShapeGSM, residual: np.ndarray) -> np.ndarray:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, COLOR_COUNT)
    radius = np.sum(_stable_whitened(rows, mixture.cholesky) ** 2, axis=1)
    return (
        np.log(mixture.weights)[None, :]
        - 1.5 * LOG_2PI
        - mixture.log_det_cholesky
        - COLOR_COUNT * np.log(mixture.scales)[None, :]
        - 0.5 * radius[:, None] / mixture.scales[None, :] ** 2
    )


def _stable_joint_log_prob(mixture: FixedShapeGSM, residual: np.ndarray) -> np.ndarray:
    return _logsumexp(_stable_component_log_prob(mixture, residual), axis=1)


def _stable_exact_scalar_log_prob(mixture: FixedShapeGSM, residual: np.ndarray) -> np.ndarray:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, COLOR_COUNT)
    output = np.empty_like(rows)
    log_weights = np.log(mixture.weights)
    log_scales = np.log(mixture.scales)
    for coordinate in range(COLOR_COUNT):
        if coordinate == 0:
            means = np.zeros(len(rows))
            variance_factor = float(mixture.shape[0, 0])
            posterior_logs = np.broadcast_to(log_weights, (len(rows), len(log_weights)))
        else:
            leading = mixture.shape[:coordinate, :coordinate]
            beta = np.linalg.solve(leading, mixture.shape[:coordinate, coordinate])
            means = np.sum(rows[:, :coordinate] * beta[None, :], axis=1)
            variance_factor = float(
                mixture.shape[coordinate, coordinate]
                - mixture.shape[coordinate, :coordinate] @ beta
            )
            leading_cholesky = np.linalg.cholesky(leading)
            parent_radius = np.sum(
                _stable_whitened(rows[:, :coordinate], leading_cholesky) ** 2,
                axis=1,
            )
            unnormalized = (
                log_weights[None, :]
                - coordinate * log_scales[None, :]
                - 0.5 * parent_radius[:, None] / mixture.scales[None, :] ** 2
            )
            posterior_logs = unnormalized - _logsumexp(unnormalized, axis=1)[:, None]
        deviations = rows[:, coordinate] - means
        component_logs = (
            posterior_logs
            - 0.5 * LOG_2PI
            - log_scales[None, :]
            - 0.5 * math.log(variance_factor)
            - 0.5
            * deviations[:, None] ** 2
            / (mixture.scales[None, :] ** 2 * variance_factor)
        )
        output[:, coordinate] = _logsumexp(component_logs, axis=1)
    return output


def canonical_site_sample(
    total_sites: int,
    sample: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    maximum_sites: int = 250_000,
) -> np.ndarray:
    """Return the sorted common fitting sites used by every fitted arm."""

    if total_sites < 1 or maximum_sites < 1:
        raise ValueError("site counts must be positive")
    if sample is None:
        if rng is None:
            raise ValueError("supply either a site sample or an RNG")
        size = min(total_sites, maximum_sites)
        chosen = rng.choice(total_sites, size=size, replace=False)
    else:
        chosen = np.asarray(sample, dtype=np.int64)
        if chosen.ndim != 1 or len(chosen) == 0:
            raise ValueError("sample must be a nonempty index vector")
    chosen = np.sort(chosen.astype(np.int64, copy=False))
    if chosen[0] < 0 or chosen[-1] >= total_sites or np.any(np.diff(chosen) == 0):
        raise ValueError("sample indices must be unique and in range")
    return chosen


def sample_sha256(sample: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(sample, dtype="<i8").tobytes()).hexdigest()


def declared_parent_masks(mode: str) -> tuple[tuple[int, ...], ...]:
    if mode == "coarse":
        return ((),) * DETAIL_DIMENSION
    if mode == "within_band":
        return within_band_parent_masks(BAND_COUNT, COLOR_COUNT)
    if mode == "full":
        return full_parent_masks(DETAIL_DIMENSION)
    raise ValueError("mode must be 'coarse', 'within_band', or 'full'")


@dataclass(frozen=True)
class RidgeLocation:
    """Nine ridge regressions with an explicit acyclic detail-parent mask."""

    coefficients: tuple[np.ndarray, ...]
    parent_masks: tuple[tuple[int, ...], ...]
    ridge: float

    def __post_init__(self) -> None:
        if len(self.coefficients) != DETAIL_DIMENSION or len(self.parent_masks) != DETAIL_DIMENSION:
            raise ValueError("a location model needs nine coefficients and masks")
        frozen = []
        for target, (coefficient, parents) in enumerate(zip(self.coefficients, self.parent_masks)):
            if tuple(sorted(set(parents))) != parents or any(parent < 0 or parent >= target for parent in parents):
                raise ValueError("detail parents must be unique, ordered, and strictly earlier")
            values = np.asarray(coefficient, dtype=np.float64).copy()
            if values.shape != (COARSE_FEATURE_COUNT + len(parents),) or not np.all(np.isfinite(values)):
                raise ValueError("coefficient shape does not match its declared parents")
            values.setflags(write=False)
            frozen.append(values)
        if self.ridge < 0.0:
            raise ValueError("ridge penalty must be nonnegative")
        object.__setattr__(self, "coefficients", tuple(frozen))

    def predict_flat(
        self,
        coarse: np.ndarray,
        blocks: np.ndarray,
        sample: np.ndarray | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        coarse_values, block_values = _validate_inputs(coarse, blocks)
        base = coarse_features(coarse_values, sample)
        targets = block_values.reshape(-1, DETAIL_DIMENSION)
        if sample is not None:
            targets = targets[np.asarray(sample, dtype=np.int64)]
        output = np.empty((len(targets), DETAIL_DIMENSION), dtype=np.float64)
        for target, (coefficient, parents) in enumerate(zip(self.coefficients, self.parent_masks)):
            features = base if not parents else np.column_stack((base, targets[:, parents]))
            # A rowwise reduction is independent of the image chunk size,
            # unlike BLAS implementations that may switch GEMV/GEMM kernels.
            output[:, target] = np.sum(features * coefficient[None, :], axis=1)
        if clip:
            np.clip(output, -1.0, 1.0, out=output)
        return output.reshape(-1, BAND_COUNT, COLOR_COUNT)

    def parameter_count(self) -> int:
        return sum(coefficient.size for coefficient in self.coefficients)


def fit_ridge_location(
    coarse: np.ndarray,
    blocks: np.ndarray,
    parent_masks: tuple[tuple[int, ...], ...],
    sample: np.ndarray,
    ridge: float = 1e-3,
) -> RidgeLocation:
    coarse_values, block_values = _validate_inputs(coarse, blocks)
    chosen = canonical_site_sample(len(coarse_values) * block_values.shape[1] * block_values.shape[2], sample)
    base = coarse_features(coarse_values, chosen)
    targets = block_values.reshape(-1, DETAIL_DIMENSION)[chosen]
    coefficients = []
    for target, parents in enumerate(parent_masks):
        if any(parent >= target for parent in parents):
            raise ValueError("a regression cannot use its target or a future detail")
        features = base if not parents else np.column_stack((base, targets[:, parents]))
        penalty = ridge * np.eye(features.shape[1])
        penalty[0, 0] = 0.0
        coefficients.append(np.linalg.solve(features.T @ features + penalty, features.T @ targets[:, target]))
    return RidgeLocation(tuple(coefficients), parent_masks, ridge)


@dataclass(frozen=True)
class ScalarGSM:
    """Zero-mean one-dimensional Gaussian scale mixture."""

    weights: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        scales = np.asarray(self.scales, dtype=np.float64).copy()
        if (
            weights.ndim != 1
            or scales.shape != weights.shape
            or len(weights) == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("weights must be positive and sum to one")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("scales must be finite and positive")
        weights.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scales", scales)

    def component_log_prob(self, residual: np.ndarray) -> np.ndarray:
        values = np.asarray(residual, dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError("residuals must be finite")
        return (
            np.log(self.weights)
            - 0.5 * LOG_2PI
            - np.log(self.scales)
            - 0.5 * (values[..., None] / self.scales) ** 2
        )

    def log_prob(self, residual: np.ndarray) -> np.ndarray:
        return _logsumexp(self.component_log_prob(residual), axis=-1)


@dataclass(frozen=True)
class ScalarFitDiagnostics:
    log_likelihood: tuple[float, ...]
    converged: bool
    initialization: int
    final_log_likelihoods: tuple[float, ...]


def fit_scalar_gsm(
    residual: np.ndarray,
    components: int,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    weight_floor: float = 1e-4,
    scale_floor: float = 0.05,
    scale_ceiling: float = 2.0,
) -> tuple[ScalarGSM, ScalarFitDiagnostics]:
    """Fit a scalar GSM by exact constrained EM with deterministic starts."""

    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    if len(values) < components or components < 1 or np.any(~np.isfinite(values)):
        raise ValueError("need at least one finite residual per component")
    if not 0.0 < weight_floor < 1.0 / components:
        raise ValueError("weight floor must satisfy 0 < floor < 1/K")
    if not 0.0 < scale_floor <= scale_ceiling or max_iterations < 1 or tolerance < 0.0:
        raise ValueError("invalid scalar EM controls")

    root_mean_scale = math.sqrt(float(np.mean(values**2)))
    geometric = np.clip(
        root_mean_scale * np.geomspace(0.5, 1.8, components),
        scale_floor,
        scale_ceiling,
    )
    quantiles = (np.arange(components, dtype=np.float64) + 0.5) / components
    radial = np.clip(np.sqrt(np.quantile(values**2, quantiles)), scale_floor, scale_ceiling)
    starts = [geometric]
    if components > 1 and not np.array_equal(geometric, radial):
        starts.append(radial)

    fits: list[tuple[ScalarGSM, tuple[float, ...], bool]] = []
    for start in starts:
        weights = np.full(components, 1.0 / components)
        scales = np.sort(start.astype(np.float64, copy=True))
        trace: list[float] = []
        converged = False
        for _ in range(max_iterations):
            model = ScalarGSM(weights, scales)
            component_logs = model.component_log_prob(values)
            point_logs = _logsumexp(component_logs, axis=1)
            old_likelihood = float(np.sum(point_logs))
            if not trace:
                trace.append(old_likelihood)
            responsibilities = np.exp(component_logs - point_logs[:, None])
            masses = np.sum(responsibilities, axis=0)
            new_weights = water_filled_weights(masses, weight_floor)
            variances = np.sum(responsibilities * values[:, None] ** 2, axis=0) / np.maximum(
                masses,
                np.finfo(np.float64).tiny,
            )
            new_scales = np.sqrt(np.clip(variances, scale_floor**2, scale_ceiling**2))
            order = np.argsort(new_scales)
            weights, scales = new_weights[order], new_scales[order]
            updated = ScalarGSM(weights, scales)
            new_likelihood = float(np.sum(updated.log_prob(values)))
            numerical_tolerance = 1e-10 * (1.0 + abs(old_likelihood))
            if new_likelihood < old_likelihood - numerical_tolerance:
                raise FloatingPointError("constrained scalar EM decreased observed likelihood")
            trace.append(new_likelihood)
            if new_likelihood - old_likelihood <= tolerance * (1.0 + abs(old_likelihood)):
                converged = True
                break
        fits.append((ScalarGSM(weights, scales), tuple(trace), converged))

    final_likelihoods = tuple(trace[-1] for _, trace, _ in fits)
    best = int(np.argmax(final_likelihoods))
    model, trace, converged = fits[best]
    return model, ScalarFitDiagnostics(trace, converged, best, final_likelihoods)


@dataclass(frozen=True)
class BlockConditionalModel:
    location: RidgeLocation
    mixtures: tuple[tuple[FixedShapeGSM, ...], ...]
    boundaries: np.ndarray
    conditional_strata: bool
    view: str = "joint"
    shape_dof: int = 5
    derived_from: str | None = None
    fit_diagnostics: tuple[tuple[GSMFitDiagnostics, ...], ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.mixtures) != BAND_COUNT:
            raise ValueError("a block model needs three bands")
        cells = STRATUM_COUNT if self.conditional_strata else 1
        if any(len(band) != cells for band in self.mixtures):
            raise ValueError("mixture cells do not match the stratum design")
        if self.view not in {"joint", "product", "exact_scalar"}:
            raise ValueError("unknown block-density view")
        boundaries = np.asarray(self.boundaries, dtype=np.float64).copy()
        if boundaries.shape != (STRATUM_COUNT - 1,) or np.any(np.diff(boundaries) < 0.0):
            raise ValueError("expected three ordered energy boundaries")
        boundaries.setflags(write=False)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def converged(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(
            tuple(diagnostics.converged for diagnostics in band)
            for band in self.fit_diagnostics
        )

    def fit_trace_export(self) -> dict[str, object]:
        return {
            "derived_from": self.derived_from,
            "cells": [
                [_fit_diagnostic_export(diagnostics) for diagnostics in band]
                for band in self.fit_diagnostics
            ],
        }

    def site_band_log_prob(self, coarse: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        coarse_values, block_values = _validate_inputs(coarse, blocks)
        residual = block_values.reshape(-1, BAND_COUNT, COLOR_COUNT) - self.location.predict_flat(
            coarse_values,
            block_values,
        )
        if self.conditional_strata:
            strata, _ = coarse_energy_strata(coarse_values, self.boundaries)
        else:
            strata = np.zeros(len(residual), dtype=np.int64)
        output = np.empty((len(residual), BAND_COUNT), dtype=np.float64)
        for band in range(BAND_COUNT):
            for stratum, mixture in enumerate(self.mixtures[band]):
                selected = strata == stratum
                if self.view == "joint":
                    output[selected, band] = _stable_joint_log_prob(mixture, residual[selected, band])
                elif self.view == "product":
                    output[selected, band] = mixture.product_log_prob(residual[selected, band])
                else:
                    output[selected, band] = np.sum(
                        _stable_exact_scalar_log_prob(mixture, residual[selected, band]),
                        axis=1,
                    )
        return output.reshape(len(block_values), block_values.shape[1], block_values.shape[2], BAND_COUNT)

    def parameter_count(self) -> dict[str, int]:
        components = len(self.mixtures[0][0].weights)
        cells = BAND_COUNT * (STRATUM_COUNT if self.conditional_strata else 1)
        represented = self.location.parameter_count() + cells * (2 * components - 1 + self.shape_dof)
        if self.conditional_strata:
            represented += STRATUM_COUNT - 1
        return {
            "represented": represented,
            "independently_fitted": 0 if self.derived_from is not None else represented,
            "incremental_from_source": 0 if self.derived_from is not None else represented,
        }


@dataclass(frozen=True)
class ScalarConditionalModel:
    location: RidgeLocation
    mixtures: tuple[tuple[ScalarGSM, ...], ...]
    boundaries: np.ndarray
    conditional_strata: bool
    fit_diagnostics: tuple[tuple[ScalarFitDiagnostics, ...], ...]

    def __post_init__(self) -> None:
        if len(self.mixtures) != DETAIL_DIMENSION:
            raise ValueError("a scalar model needs nine coordinate heads")
        cells = STRATUM_COUNT if self.conditional_strata else 1
        if any(len(coordinate) != cells for coordinate in self.mixtures):
            raise ValueError("mixture cells do not match the stratum design")
        boundaries = np.asarray(self.boundaries, dtype=np.float64).copy()
        if boundaries.shape != (STRATUM_COUNT - 1,) or np.any(np.diff(boundaries) < 0.0):
            raise ValueError("expected three ordered energy boundaries")
        boundaries.setflags(write=False)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def converged(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(
            tuple(diagnostics.converged for diagnostics in coordinate)
            for coordinate in self.fit_diagnostics
        )

    def fit_trace_export(self) -> dict[str, object]:
        return {
            "derived_from": None,
            "cells": [
                [_fit_diagnostic_export(diagnostics) for diagnostics in coordinate]
                for coordinate in self.fit_diagnostics
            ],
        }

    def site_coordinate_log_prob(self, coarse: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        coarse_values, block_values = _validate_inputs(coarse, blocks)
        targets = block_values.reshape(-1, DETAIL_DIMENSION)
        means = self.location.predict_flat(coarse_values, block_values).reshape(-1, DETAIL_DIMENSION)
        residual = targets - means
        if self.conditional_strata:
            strata, _ = coarse_energy_strata(coarse_values, self.boundaries)
        else:
            strata = np.zeros(len(residual), dtype=np.int64)
        output = np.empty_like(residual)
        for coordinate in range(DETAIL_DIMENSION):
            for stratum, mixture in enumerate(self.mixtures[coordinate]):
                selected = strata == stratum
                output[selected, coordinate] = mixture.log_prob(residual[selected, coordinate])
        return output.reshape(
            len(block_values),
            block_values.shape[1],
            block_values.shape[2],
            BAND_COUNT,
            COLOR_COUNT,
        )

    def site_band_log_prob(self, coarse: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        return np.sum(self.site_coordinate_log_prob(coarse, blocks), axis=-1)

    def parameter_count(self) -> dict[str, int]:
        components = len(self.mixtures[0][0].weights)
        cells = DETAIL_DIMENSION * (STRATUM_COUNT if self.conditional_strata else 1)
        represented = self.location.parameter_count() + cells * (2 * components - 1)
        if self.conditional_strata:
            represented += STRATUM_COUNT - 1
        return {
            "represented": represented,
            "independently_fitted": represented,
            "incremental_from_source": represented,
        }


def _fit_diagonal_shape(residual: np.ndarray, shrinkage: float = 0.001) -> np.ndarray:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, COLOR_COUNT)
    diagonal = np.mean(rows**2, axis=0)
    isotropic = float(np.mean(diagonal))
    regularized = (1.0 - shrinkage) * diagonal + shrinkage * isotropic
    projected = project_log_eigenvalues(regularized, 0.1, 10.0)
    return np.diag(projected)


def _fit_block_cells(
    residual: np.ndarray,
    strata: np.ndarray,
    shapes: tuple[tuple[np.ndarray, ...], ...],
    components: int,
    conditional_strata: bool,
    max_iterations: int,
    tolerance: float,
    progress: Callable[[str], None] | None = None,
    label: str = "block",
) -> tuple[tuple[tuple[FixedShapeGSM, ...], ...], tuple[tuple[GSMFitDiagnostics, ...], ...]]:
    cell_count = STRATUM_COUNT if conditional_strata else 1
    mixtures = []
    converged = []
    for band in range(BAND_COUNT):
        band_mixtures = []
        band_converged = []
        for stratum in range(cell_count):
            selected = strata == stratum if conditional_strata else np.ones(len(strata), dtype=bool)
            mixture, diagnostics = fit_fixed_shape_gsm(
                residual[selected, band],
                shapes[band][stratum],
                components,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            band_mixtures.append(mixture)
            band_converged.append(diagnostics)
            if progress is not None:
                progress(f"{label}:band={band}:stratum={stratum}:done")
        mixtures.append(tuple(band_mixtures))
        converged.append(tuple(band_converged))
    return tuple(mixtures), tuple(converged)


def _fit_shapes(
    residual: np.ndarray,
    strata: np.ndarray,
    conditional_strata: bool,
    diagonal: bool,
) -> tuple[tuple[np.ndarray, ...], ...]:
    cell_count = STRATUM_COUNT if conditional_strata else 1
    output = []
    for band in range(BAND_COUNT):
        band_shapes = []
        for stratum in range(cell_count):
            selected = strata == stratum if conditional_strata else np.ones(len(strata), dtype=bool)
            if not np.any(selected):
                raise ValueError("every fitted energy stratum must contain a site")
            fitter = _fit_diagonal_shape if diagonal else fit_determinant_one_shape
            band_shapes.append(fitter(residual[selected, band]))
        output.append(tuple(band_shapes))
    return tuple(output)


def _prepare_dense_b_fit(
    coarse: np.ndarray,
    blocks: np.ndarray,
    sample: np.ndarray | None,
    rng: np.random.Generator | None,
    maximum_sites: int,
    ridge: float,
    progress: Callable[[str], None] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    RidgeLocation,
    np.ndarray,
    tuple[tuple[np.ndarray, ...], ...],
]:
    """Prepare the common sample, location, residual, and shapes for B arms."""

    coarse_values, block_values = _validate_inputs(coarse, blocks)
    total_sites = len(coarse_values) * block_values.shape[1] * block_values.shape[2]
    chosen = canonical_site_sample(total_sites, sample, rng, maximum_sites)
    if progress is not None:
        progress(f"sample:sites={len(chosen)}:done")
    _, boundaries = coarse_energy_strata(coarse_values, sample=chosen)
    strata, _ = coarse_energy_strata(coarse_values, boundaries, chosen)

    b_location = fit_ridge_location(
        coarse_values,
        block_values,
        declared_parent_masks("coarse"),
        chosen,
        ridge,
    )
    if progress is not None:
        progress("b_location:done")
    sampled_targets = block_values.reshape(-1, BAND_COUNT, COLOR_COUNT)[chosen]
    b_residual = sampled_targets - b_location.predict_flat(coarse_values, block_values, chosen)
    dense_shapes = _fit_shapes(b_residual, strata, True, False)
    return (
        coarse_values,
        block_values,
        chosen,
        boundaries,
        strata,
        b_location,
        b_residual,
        dense_shapes,
    )


def _fit_dense_b_model(
    location: RidgeLocation,
    residual: np.ndarray,
    strata: np.ndarray,
    shapes: tuple[tuple[np.ndarray, ...], ...],
    boundaries: np.ndarray,
    components: int,
    max_iterations: int,
    tolerance: float,
    progress: Callable[[str], None] | None,
) -> BlockConditionalModel:
    """Fit one dense-shape B arm from its common prepared quantities."""

    mixtures, convergence = _fit_block_cells(
        residual,
        strata,
        shapes,
        components,
        True,
        max_iterations,
        tolerance,
        progress,
        f"b{components}",
    )
    return BlockConditionalModel(
        location,
        mixtures,
        boundaries,
        True,
        fit_diagnostics=convergence,
    )


def _fit_scalar_model(
    coarse: np.ndarray,
    blocks: np.ndarray,
    location: RidgeLocation,
    boundaries: np.ndarray,
    sample: np.ndarray,
    components: int,
    max_iterations: int,
    tolerance: float,
    progress: Callable[[str], None] | None = None,
    label: str = "scalar",
) -> ScalarConditionalModel:
    targets = blocks.reshape(-1, DETAIL_DIMENSION)[sample]
    means = location.predict_flat(coarse, blocks, sample).reshape(-1, DETAIL_DIMENSION)
    residual = targets - means
    strata, _ = coarse_energy_strata(coarse, boundaries, sample)
    mixtures = []
    convergence = []
    for coordinate in range(DETAIL_DIMENSION):
        coordinate_mixtures = []
        coordinate_convergence = []
        for stratum in range(STRATUM_COUNT):
            selected = strata == stratum
            mixture, diagnostics = fit_scalar_gsm(
                residual[selected, coordinate],
                components,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            coordinate_mixtures.append(mixture)
            coordinate_convergence.append(diagnostics)
            if progress is not None:
                progress(f"{label}:coordinate={coordinate}:stratum={stratum}:done")
        mixtures.append(tuple(coordinate_mixtures))
        convergence.append(tuple(coordinate_convergence))
    return ScalarConditionalModel(
        location,
        tuple(mixtures),
        boundaries,
        True,
        tuple(convergence),
    )


@dataclass(frozen=True)
class ObservedBlockModels:
    b1: BlockConditionalModel
    b4: BlockConditionalModel
    b8: BlockConditionalModel
    p4: BlockConditionalModel
    z4: BlockConditionalModel
    d4: BlockConditionalModel
    b4_unconditional: BlockConditionalModel
    o4: ScalarConditionalModel
    a4: ScalarConditionalModel
    a8: ScalarConditionalModel
    i4: ScalarConditionalModel
    i8: ScalarConditionalModel
    e4: BlockConditionalModel
    boundaries: np.ndarray
    sample: np.ndarray
    sample_hash: str

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.boundaries, dtype=np.float64).copy()
        sample = np.asarray(self.sample, dtype=np.int64).copy()
        boundaries.setflags(write=False)
        sample.setflags(write=False)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "sample", sample)

    def arms(self) -> dict[str, BlockConditionalModel | ScalarConditionalModel]:
        return {name: getattr(self, name) for name in ARM_NAMES}

    def parameter_counts(self) -> dict[str, dict[str, int]]:
        return {name: model.parameter_count() for name, model in self.arms().items()}

    def fit_trace_export(self) -> dict[str, dict[str, object]]:
        return {name: model.fit_trace_export() for name, model in self.arms().items()}


@dataclass(frozen=True)
class ObservedB4Fit:
    """B4 model and the common fitting-site metadata needed to reproduce it."""

    b4: BlockConditionalModel
    boundaries: np.ndarray
    sample: np.ndarray
    sample_hash: str

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.boundaries, dtype=np.float64).copy()
        sample = np.asarray(self.sample, dtype=np.int64).copy()
        boundaries.setflags(write=False)
        sample.setflags(write=False)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "sample", sample)

    def arms(self) -> dict[str, BlockConditionalModel]:
        return {"b4": self.b4}

    def parameter_counts(self) -> dict[str, dict[str, int]]:
        return {"b4": self.b4.parameter_count()}

    def fit_trace_export(self) -> dict[str, dict[str, object]]:
        return {"b4": self.b4.fit_trace_export()}


def _fit_diagnostic_export(
    diagnostics: GSMFitDiagnostics | ScalarFitDiagnostics,
) -> dict[str, object]:
    return {
        "log_likelihood": [float(value) for value in diagnostics.log_likelihood],
        "converged": bool(diagnostics.converged),
        "initialization": int(diagnostics.initialization),
        "final_log_likelihoods": [float(value) for value in diagnostics.final_log_likelihoods],
    }


def fit_observed_b4_model(
    coarse: np.ndarray,
    blocks: np.ndarray,
    sample: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    maximum_sites: int = 250_000,
    ridge: float = 1e-3,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    progress: Callable[[str], None] | None = None,
) -> ObservedB4Fit:
    """Fit only B4, using the exact common B-arm preparation and cell order."""

    (
        _,
        _,
        chosen,
        boundaries,
        strata,
        b_location,
        b_residual,
        dense_shapes,
    ) = _prepare_dense_b_fit(
        coarse,
        blocks,
        sample,
        rng,
        maximum_sites,
        ridge,
        progress,
    )
    b4 = _fit_dense_b_model(
        b_location,
        b_residual,
        strata,
        dense_shapes,
        boundaries,
        4,
        max_iterations,
        tolerance,
        progress,
    )
    if progress is not None:
        progress("b4_only:done")
    return ObservedB4Fit(b4, boundaries, chosen, sample_sha256(chosen))


def fit_observed_block_models(
    coarse: np.ndarray,
    blocks: np.ndarray,
    sample: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    maximum_sites: int = 250_000,
    ridge: float = 1e-3,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    progress: Callable[[str], None] | None = None,
) -> ObservedBlockModels:
    """Fit all frozen B/O/A/I arms from one canonical fitting-site sample."""

    (
        coarse_values,
        block_values,
        chosen,
        boundaries,
        strata,
        b_location,
        b_residual,
        dense_shapes,
    ) = _prepare_dense_b_fit(
        coarse,
        blocks,
        sample,
        rng,
        maximum_sites,
        ridge,
        progress,
    )
    sampled_targets = block_values.reshape(-1, BAND_COUNT, COLOR_COUNT)[chosen]
    coarse_masks = declared_parent_masks("coarse")

    def dense_model(components: int) -> BlockConditionalModel:
        return _fit_dense_b_model(
            b_location,
            b_residual,
            strata,
            dense_shapes,
            boundaries,
            components,
            max_iterations,
            tolerance,
            progress,
        )

    b1 = dense_model(1)
    b4 = dense_model(4)
    b8 = dense_model(8)
    p4 = BlockConditionalModel(
        b_location,
        b4.mixtures,
        boundaries,
        True,
        view="product",
        derived_from="b4",
        fit_diagnostics=b4.fit_diagnostics,
    )
    e4 = BlockConditionalModel(
        b_location,
        b4.mixtures,
        boundaries,
        True,
        view="exact_scalar",
        derived_from="b4",
        fit_diagnostics=b4.fit_diagnostics,
    )
    z4_cells = tuple(tuple(mixture.zero_correlation() for mixture in band) for band in b4.mixtures)
    z4 = BlockConditionalModel(
        b_location,
        z4_cells,
        boundaries,
        True,
        derived_from="b4",
        shape_dof=2,
        fit_diagnostics=b4.fit_diagnostics,
    )

    unconditional_strata = np.zeros(len(chosen), dtype=np.int64)
    unconditional_shapes = _fit_shapes(b_residual, unconditional_strata, False, False)
    unconditional_cells, unconditional_convergence = _fit_block_cells(
        b_residual,
        unconditional_strata,
        unconditional_shapes,
        4,
        False,
        max_iterations,
        tolerance,
        progress,
        "b4_unconditional",
    )
    b4_unconditional = BlockConditionalModel(
        b_location,
        unconditional_cells,
        boundaries,
        False,
        fit_diagnostics=unconditional_convergence,
    )

    d_location = fit_ridge_location(coarse_values, block_values, coarse_masks, chosen, ridge)
    if progress is not None:
        progress("d_location:done")
    d_residual = sampled_targets - d_location.predict_flat(coarse_values, block_values, chosen)
    diagonal_shapes = _fit_shapes(d_residual, strata, True, True)
    d4_cells, d4_convergence = _fit_block_cells(
        d_residual,
        strata,
        diagonal_shapes,
        4,
        True,
        max_iterations,
        tolerance,
        progress,
        "d4",
    )
    d4 = BlockConditionalModel(
        d_location,
        d4_cells,
        boundaries,
        True,
        shape_dof=2,
        fit_diagnostics=d4_convergence,
    )

    o_location = fit_ridge_location(coarse_values, block_values, coarse_masks, chosen, ridge)
    a_location = fit_ridge_location(
        coarse_values,
        block_values,
        declared_parent_masks("within_band"),
        chosen,
        ridge,
    )
    i_location = fit_ridge_location(
        coarse_values,
        block_values,
        declared_parent_masks("full"),
        chosen,
        ridge,
    )
    if progress is not None:
        progress("o_a_i_locations:done")
    o4 = _fit_scalar_model(
        coarse_values,
        block_values,
        o_location,
        boundaries,
        chosen,
        4,
        max_iterations,
        tolerance,
        progress,
        "o4",
    )
    a4 = _fit_scalar_model(
        coarse_values,
        block_values,
        a_location,
        boundaries,
        chosen,
        4,
        max_iterations,
        tolerance,
        progress,
        "a4",
    )
    a8 = _fit_scalar_model(
        coarse_values,
        block_values,
        a_location,
        boundaries,
        chosen,
        8,
        max_iterations,
        tolerance,
        progress,
        "a8",
    )
    i4 = _fit_scalar_model(
        coarse_values,
        block_values,
        i_location,
        boundaries,
        chosen,
        4,
        max_iterations,
        tolerance,
        progress,
        "i4",
    )
    i8 = _fit_scalar_model(
        coarse_values,
        block_values,
        i_location,
        boundaries,
        chosen,
        8,
        max_iterations,
        tolerance,
        progress,
        "i8",
    )
    if progress is not None:
        progress("all_models:done")
    return ObservedBlockModels(
        b1,
        b4,
        b8,
        p4,
        z4,
        d4,
        b4_unconditional,
        o4,
        a4,
        a8,
        i4,
        i8,
        e4,
        boundaries,
        chosen,
        sample_sha256(chosen),
    )


def site_band_log_scores(
    models: ObservedBlockModels,
    coarse: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, np.ndarray]:
    _validate_inputs(coarse, blocks)
    return {name: model.site_band_log_prob(coarse, blocks) for name, model in models.arms().items()}


def per_image_band_log_scores(
    models: ObservedBlockModels,
    coarse: np.ndarray,
    blocks: np.ndarray,
    chunk_images: int | None = None,
) -> dict[str, np.ndarray]:
    """Return each arm's log scores with shape ``[image, band]``."""

    coarse_values, block_values = _validate_inputs(coarse, blocks)
    if chunk_images is None:
        chunk_images = len(block_values)
    if chunk_images < 1:
        raise ValueError("chunk size must be positive")
    output = {name: np.empty((len(block_values), BAND_COUNT)) for name in ARM_NAMES}
    for start in range(0, len(block_values), chunk_images):
        stop = min(start + chunk_images, len(block_values))
        scores = site_band_log_scores(models, coarse_values[start:stop], block_values[start:stop])
        for name, values in scores.items():
            output[name][start:stop] = np.sum(values, axis=(1, 2))
    return output


@dataclass(frozen=True)
class BlockDiagnosticSufficientStatistics:
    pit_grid: np.ndarray
    per_image_counts: np.ndarray
    per_image_pit_leq_counts: np.ndarray
    per_image_angular_counts: np.ndarray
    per_image_angular_second_sums: np.ndarray
    per_image_angular_fourth_sums: np.ndarray
    per_image_responsibility_sums: np.ndarray
    shape_eigenvalues: np.ndarray
    scale_lower_hits: np.ndarray
    scale_upper_hits: np.ndarray
    weight_floor_hits: np.ndarray
    shape_lower_hits: np.ndarray
    shape_upper_hits: np.ndarray
    per_image_normalized_energy_count: np.ndarray
    per_image_normalized_energy_sum: np.ndarray
    per_image_normalized_energy_outer_sum: np.ndarray
    per_image_heatmap_b4_minus_i8: np.ndarray
    per_image_clipped_location_counts: Mapping[str, np.ndarray]
    per_image_finite_score_counts: Mapping[str, np.ndarray]
    b4_e4_max_abs: float
    p4_product_max_abs: float
    z4_component_marginal_max_abs: float

    @property
    def counts(self) -> np.ndarray:
        return np.sum(self.per_image_counts, axis=0)

    @property
    def pit_leq_counts(self) -> np.ndarray:
        return np.sum(self.per_image_pit_leq_counts, axis=0)

    @property
    def angular_counts(self) -> np.ndarray:
        return np.sum(self.per_image_angular_counts, axis=0)

    @property
    def angular_second_sums(self) -> np.ndarray:
        return np.sum(self.per_image_angular_second_sums, axis=0)

    @property
    def angular_fourth_sums(self) -> np.ndarray:
        return np.sum(self.per_image_angular_fourth_sums, axis=0)

    @property
    def responsibility_sums(self) -> np.ndarray:
        return np.sum(self.per_image_responsibility_sums, axis=0)

    @property
    def normalized_energy_count(self) -> int:
        return int(np.sum(self.per_image_normalized_energy_count))

    @property
    def normalized_energy_sum(self) -> np.ndarray:
        return np.sum(self.per_image_normalized_energy_sum, axis=0)

    @property
    def normalized_energy_outer_sum(self) -> np.ndarray:
        return np.sum(self.per_image_normalized_energy_outer_sum, axis=0)

    @property
    def heatmap_b4_minus_i8_sum(self) -> np.ndarray:
        return np.sum(self.per_image_heatmap_b4_minus_i8, axis=0)

    @property
    def heatmap_image_count(self) -> int:
        return len(self.per_image_heatmap_b4_minus_i8)

    @property
    def clipped_location_counts(self) -> dict[str, int]:
        return {
            name: int(np.sum(values))
            for name, values in self.per_image_clipped_location_counts.items()
        }

    @property
    def location_prediction_count(self) -> int:
        return int(np.sum(self.per_image_normalized_energy_count)) * DETAIL_DIMENSION

    @property
    def finite_score_counts(self) -> dict[str, int]:
        return {
            name: int(np.sum(values))
            for name, values in self.per_image_finite_score_counts.items()
        }

    @property
    def score_count_per_arm(self) -> int:
        return int(np.sum(self.per_image_normalized_energy_count)) * BAND_COUNT

    def pit_max_deviation(self) -> float:
        empirical = self.pit_leq_counts / self.counts[..., None]
        return float(np.max(np.abs(empirical - self.pit_grid)))

    def angular_max_deviations(self) -> tuple[float, float]:
        second = self.angular_second_sums / self.angular_counts[..., None, None]
        fourth = self.angular_fourth_sums / self.angular_counts[..., None, None]
        second_target = np.eye(COLOR_COUNT) / COLOR_COUNT
        fourth_target = np.full((COLOR_COUNT, COLOR_COUNT), 1.0 / 15.0)
        np.fill_diagonal(fourth_target, 1.0 / 5.0)
        return (
            float(np.max(np.abs(second - second_target))),
            float(np.max(np.abs(fourth - fourth_target))),
        )

    def cross_band_energy_correlation(self) -> np.ndarray:
        count = self.normalized_energy_count
        centered = self.normalized_energy_outer_sum - np.outer(
            self.normalized_energy_sum,
            self.normalized_energy_sum,
        ) / count
        covariance = centered / (count - 1)
        standard = np.sqrt(np.diag(covariance))
        return covariance / np.outer(standard, standard)


def diagnostic_sufficient_statistics(
    models: ObservedBlockModels,
    coarse: np.ndarray,
    blocks: np.ndarray,
    pit_grid: np.ndarray = PIT_GRID,
) -> BlockDiagnosticSufficientStatistics:
    """Compute frozen B4 mechanism diagnostics without using class labels."""

    coarse_values, block_values = _validate_inputs(coarse, blocks)
    grid = np.asarray(pit_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0 or np.any(np.diff(grid) <= 0.0) or grid[0] <= 0.0 or grid[-1] >= 1.0:
        raise ValueError("PIT grid must be strictly increasing inside (0,1)")
    site_count = int(np.prod(block_values.shape[:3]))
    image_count = len(block_values)
    sites_per_image = block_values.shape[1] * block_values.shape[2]
    image_index = np.repeat(np.arange(image_count), sites_per_image)
    residual = block_values.reshape(-1, BAND_COUNT, COLOR_COUNT) - models.b4.location.predict_flat(
        coarse_values,
        block_values,
    )
    strata, _ = coarse_energy_strata(coarse_values, models.boundaries)
    component_count = len(models.b4.mixtures[0][0].weights)
    cell_shape = (image_count, BAND_COUNT, STRATUM_COUNT)
    counts = np.zeros(cell_shape, dtype=np.int64)
    pit_counts = np.zeros((*cell_shape, len(grid)), dtype=np.int64)
    angular_counts = np.zeros_like(counts)
    angular_second = np.zeros((*cell_shape, COLOR_COUNT, COLOR_COUNT))
    angular_fourth = np.zeros_like(angular_second)
    responsibility_sums = np.zeros((*cell_shape, component_count))
    spectra = np.empty((BAND_COUNT, STRATUM_COUNT, COLOR_COUNT))
    scale_lower_hits = np.zeros((BAND_COUNT, STRATUM_COUNT, component_count), dtype=bool)
    scale_upper_hits = np.zeros_like(scale_lower_hits)
    weight_floor_hits = np.zeros_like(scale_lower_hits)
    shape_lower_hits = np.zeros((BAND_COUNT, STRATUM_COUNT, COLOR_COUNT), dtype=bool)
    shape_upper_hits = np.zeros_like(shape_lower_hits)
    normalized_energy = np.empty((site_count, BAND_COUNT))

    for band in range(BAND_COUNT):
        for stratum, mixture in enumerate(models.b4.mixtures[band]):
            selected = strata == stratum
            rows = residual[selected, band]
            count = int(np.sum(selected))
            if count == 0:
                raise ValueError("every diagnostic stratum must contain a site")
            selected_images = image_index[selected]
            np.add.at(counts[:, band, stratum], selected_images, 1)
            whitened = _stable_whitened(rows, mixture.cholesky)
            radius = np.sum(whitened**2, axis=1)
            pit = np.sum(
                mixture.weights[None, :]
                * gammainc(1.5, radius[:, None] / (2.0 * mixture.scales[None, :] ** 2)),
                axis=1,
            )
            np.add.at(
                pit_counts[:, band, stratum],
                selected_images,
                (pit[:, None] <= grid[None, :]).astype(np.int64),
            )
            nonzero = radius > np.finfo(np.float64).tiny
            direction = whitened[nonzero] / np.sqrt(radius[nonzero, None])
            angular_images = selected_images[nonzero]
            np.add.at(angular_counts[:, band, stratum], angular_images, 1)
            direction_outer = direction[:, :, None] * direction[:, None, :]
            np.add.at(angular_second[:, band, stratum], angular_images, direction_outer)
            squared = direction**2
            squared_outer = squared[:, :, None] * squared[:, None, :]
            np.add.at(angular_fourth[:, band, stratum], angular_images, squared_outer)
            component_logs = _stable_component_log_prob(mixture, rows)
            responsibilities = np.exp(component_logs - _logsumexp(component_logs, axis=1)[:, None])
            np.add.at(responsibility_sums[:, band, stratum], selected_images, responsibilities)
            expected_radius_scale = float(np.sum(mixture.weights * mixture.scales**2))
            normalized_energy[selected, band] = radius / (COLOR_COUNT * expected_radius_scale)
            eigenvalues = np.linalg.eigvalsh(mixture.shape)
            spectra[band, stratum] = eigenvalues
            scale_lower_hits[band, stratum] = np.isclose(mixture.scales, 0.05, atol=1e-12, rtol=0.0)
            scale_upper_hits[band, stratum] = np.isclose(mixture.scales, 2.0, atol=1e-12, rtol=0.0)
            weight_floor_hits[band, stratum] = np.isclose(mixture.weights, 1e-4, atol=1e-12, rtol=0.0)
            shape_lower_hits[band, stratum] = np.isclose(eigenvalues, 0.1, atol=1e-12, rtol=0.0)
            shape_upper_hits[band, stratum] = np.isclose(eigenvalues, 10.0, atol=1e-12, rtol=0.0)

    site_scores = site_band_log_scores(models, coarse_values, block_values)
    heatmap = (site_scores["b4"] - site_scores["i8"]).transpose(0, 3, 1, 2)
    clipped = {}
    for name, location in {
        "b": models.b4.location,
        "d": models.d4.location,
        "o": models.o4.location,
        "a": models.a8.location,
        "i": models.i8.location,
    }.items():
        raw = location.predict_flat(coarse_values, block_values, clip=False).reshape(
            image_count,
            sites_per_image,
            DETAIL_DIMENSION,
        )
        clipped[name] = np.sum(np.abs(raw) > 1.0, axis=(1, 2))
    finite = {
        name: np.sum(np.isfinite(score), axis=(1, 2, 3))
        for name, score in site_scores.items()
    }
    p4_error = 0.0
    z4_error = 0.0
    for band in range(BAND_COUNT):
        for stratum in range(STRATUM_COUNT):
            fitted = models.b4.mixtures[band][stratum]
            zero = models.z4.mixtures[band][stratum]
            selected = strata == stratum
            expected_product = np.sum(fitted.marginal_log_prob(residual[selected, band]), axis=1)
            p4_error = max(
                p4_error,
                float(np.max(np.abs(expected_product - fitted.product_log_prob(residual[selected, band])))),
            )
            fitted_variance = fitted.scales[:, None] ** 2 * np.diag(fitted.shape)[None, :]
            zero_variance = zero.scales[:, None] ** 2 * np.diag(zero.shape)[None, :]
            z4_error = max(z4_error, float(np.max(np.abs(fitted_variance - zero_variance))))
    energy_by_image = normalized_energy.reshape(image_count, sites_per_image, BAND_COUNT)
    return BlockDiagnosticSufficientStatistics(
        grid.copy(),
        counts,
        pit_counts,
        angular_counts,
        angular_second,
        angular_fourth,
        responsibility_sums,
        spectra,
        scale_lower_hits,
        scale_upper_hits,
        weight_floor_hits,
        shape_lower_hits,
        shape_upper_hits,
        np.full(image_count, sites_per_image, dtype=np.int64),
        np.sum(energy_by_image, axis=1),
        np.einsum("nsa,nsb->nab", energy_by_image, energy_by_image),
        heatmap,
        clipped,
        finite,
        float(np.max(np.abs(site_scores["b4"] - site_scores["e4"]))),
        p4_error,
        z4_error,
    )
