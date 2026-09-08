"""Complete observed-coordinate Gaussian flow from convex conditional splines.

The causal graph and parameter-sharing groups are public inputs fixed without
access to evaluation arrays. This module does not infer image structure or
claim that a two-parent factorization describes real images. All coordinates,
including roots, have fitted densities. No teacher, chart, or VAE is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.special import ndtr, ndtri

from .positive_density_spline import (
    PositiveDensitySpline, PositiveSplineFitDiagnostics, fit_positive_density_spline,
)

_LOG_2PI = float(np.log(2 * np.pi))


def _graph(parents, groups):
    parents = tuple(tuple(p) for p in parents)
    if not parents:
        raise ValueError("at least one coordinate is required")
    for coordinate, entry in enumerate(parents):
        if len(entry) > 2 or len(set(entry)) != len(entry):
            raise ValueError("each coordinate requires zero to two distinct parents")
        if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer))
               or p < 0 or p >= coordinate for p in entry):
            raise ValueError("parents must be integer indices of earlier coordinates")
    raw = np.asarray(groups)
    if raw.shape != (len(parents),) or raw.dtype.kind not in 'iu':
        raise ValueError("groups must be an integer vector with one entry per coordinate")
    if np.any(raw < 0) or set(raw.tolist()) != set(range(int(raw.max()) + 1)):
        raise ValueError("group indices must be contiguous starting at zero")
    groups = tuple(int(g) for g in raw)
    for group in set(groups):
        if len({len(parents[i]) for i, g in enumerate(groups) if g == group}) != 1:
            raise ValueError("shared conditionals must have equal context dimensions")
    return parents, groups


@dataclass(frozen=True)
class PositiveSplineFlowFitDiagnostics:
    independent_array_count: int
    dimension: int
    group_site_counts: tuple[int, ...]
    group_fits: tuple[PositiveSplineFitDiagnostics, ...]

    @property
    def per_coordinate_optimization_gap(self):
        return sum(k * fit.frank_wolfe_gap for k, fit in
                   zip(self.group_site_counts, self.group_fits)) / self.dimension

    @property
    def converged(self):
        return all(fit.converged for fit in self.group_fits)


@dataclass(frozen=True)
class PositiveSplineFlow:
    """Normalized C1 triangular transport from R^D to the open unit cube.

    Statements about C1 regularity concern the exact real-arithmetic map.
    Finite-precision Gaussian CDF saturation is rejected, never clipped.
    Methods preserve arbitrary leading batch dimensions. Returned Jacobians
    are log absolute determinants for the transformation actually requested.
    """
    parents: tuple[tuple[int, ...], ...]
    groups: tuple[int, ...]
    models: tuple[PositiveDensitySpline, ...]
    _decode_blocks: tuple[tuple[int, tuple[int, ...]], ...] = field(init=False, repr=False)

    def __post_init__(self):
        parents, groups = _graph(self.parents, self.groups)
        models = tuple(self.models)
        if len(models) != max(groups) + 1:
            raise ValueError("one fitted model is required per group")
        for i, group in enumerate(groups):
            if not isinstance(models[group], PositiveDensitySpline) or models[group].context_dimension != len(parents[i]):
                raise ValueError("model context dimension disagrees with graph")
        object.__setattr__(self, 'parents', parents)
        object.__setattr__(self, 'groups', groups)
        object.__setattr__(self, 'models', models)
        depths = []
        for entry in parents:
            depths.append(0 if not entry else 1 + max(depths[p] for p in entry))
        blocks = {}
        for i, (depth, group) in enumerate(zip(depths, groups)):
            blocks.setdefault((depth, group), []).append(i)
        object.__setattr__(self, '_decode_blocks', tuple(
            (group, tuple(sites)) for (depth, group), sites in sorted(blocks.items())))

    @property
    def dimension(self):
        return len(self.parents)

    @classmethod
    def fit(cls, data, parents, groups=None, **fit_options):
        """Fit shared group losses by averaging observed sites and whole arrays.

        Input rows, not pooled sites, are the independent units for theory.
        Sharing assumes identical conditionals within each group; it can cause
        misspecification. Every root is learned from the same observed data.
        This method has no validation/test or graph-selection input.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or min(data.shape) < 1 or not np.all(np.isfinite(data)):
            raise ValueError("training data must be a nonempty finite [n,D] array")
        if np.any((data <= 0) | (data >= 1)):
            raise ValueError("training observations must lie in the open unit cube")
        if groups is None:
            groups = tuple(range(data.shape[1]))
        parents, groups = _graph(parents, groups)
        if len(parents) != data.shape[1]:
            raise ValueError("graph and data dimensions disagree")
        models, diagnostics, site_counts = [], [], []
        for group in range(max(groups) + 1):
            sites = [i for i, g in enumerate(groups) if g == group]
            response = np.concatenate([data[:, i] for i in sites])
            contexts = (np.concatenate([data[:, parents[i]] for i in sites], axis=0)
                        if parents[sites[0]] else None)
            model, diagnostic = fit_positive_density_spline(contexts, response, **fit_options)
            models.append(model)
            diagnostics.append(diagnostic)
            site_counts.append(len(sites))
        return cls(parents, groups, tuple(models)), PositiveSplineFlowFitDiagnostics(
            len(data), data.shape[1], tuple(site_counts), tuple(diagnostics))

    def _rows(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.dimension or not np.all(np.isfinite(values)):
            raise ValueError("values must be finite with final axis equal to dimension")
        return values.reshape(-1, self.dimension), values.shape[:-1]

    def _context(self, rows, index):
        return rows[:, self.parents[index]] if self.parents[index] else None

    def log_prob(self, values):
        rows, shape = self._rows(values)
        inside = np.all((rows > 0) & (rows < 1), axis=1)
        safe = np.where(inside[:, None], rows, .5)
        result = np.zeros(len(rows))
        for i, group in enumerate(self.groups):
            result += self.models[group].log_prob(self._context(safe, i), safe[:, i])
        return np.where(inside, result, -np.inf).reshape(shape)

    def encode(self, values):
        rows, shape = self._rows(values)
        if np.any((rows <= 0) | (rows >= 1)):
            raise ValueError("observations must lie in the open unit cube")
        uniforms = np.empty_like(rows)
        log_density = np.zeros(len(rows))
        for i, group in enumerate(self.groups):
            context = self._context(rows, i)
            uniforms[:, i] = self.models[group].cdf(context, rows[:, i])
            log_density += self.models[group].log_prob(context, rows[:, i])
        if np.any((uniforms <= 0) | (uniforms >= 1)):
            raise FloatingPointError("conditional CDF reached a finite-precision boundary")
        gaussian = ndtri(uniforms)
        log_base = -.5 * (self.dimension * _LOG_2PI + np.sum(gaussian**2, axis=1))
        return gaussian.reshape(shape + (self.dimension,)), (log_density-log_base).reshape(shape)

    def decode(self, values):
        gaussian, shape = self._rows(values)
        uniforms = ndtr(gaussian)
        if np.any((uniforms <= 0) | (uniforms >= 1)):
            raise FloatingPointError("Gaussian CDF reached a finite-precision boundary")
        rows = np.empty_like(gaussian)
        log_density = np.zeros(len(rows))
        # Coordinates at the same graph depth have no mutual dependency.
        # Batch them within each shared model. The 65,536-row cap bounds spline
        # query workspace; it does not alter the graph, densities, or uniforms.
        for group, sites in self._decode_blocks:
            count = len(sites)
            total = len(rows) * count
            parent_indices = np.asarray([self.parents[i] for i in sites], dtype=int)
            for begin in range(0, total, 65536):
                flat = np.arange(begin, min(total, begin + 65536))
                batch, site_offset = np.divmod(flat, count)
                coordinate = np.asarray(sites)[site_offset]
                context = (rows[batch[:, None], parent_indices[site_offset]]
                           if parent_indices.shape[1] else None)
                response = self.models[group].icdf(context, uniforms[batch, coordinate])
                rows[batch, coordinate] = response
                contribution = self.models[group].log_prob(context, response)
                np.add.at(log_density, batch, contribution)
        if np.any((rows <= 0) | (rows >= 1)):
            raise FloatingPointError("generated coordinates reached a finite-precision boundary")
        log_base = -.5 * (self.dimension * _LOG_2PI + np.sum(gaussian**2, axis=1))
        return rows.reshape(shape + (self.dimension,)), (log_base-log_density).reshape(shape)


__all__ = ['PositiveSplineFlow', 'PositiveSplineFlowFitDiagnostics']
