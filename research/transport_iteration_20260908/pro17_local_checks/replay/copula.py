"""Pro17: fixed, full-dimensional triangular copula stress model.

No native data, model downloads, PSC calls, latent labels, or adaptive design.
All learned estimators receive only the observed Gaussian-marginal coordinates.
The source-order grouping is public and identical for every local-flow method.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, ndtri

RHO = 0.9

def psi(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(x, dtype=np.float64) - 1.0

def inverse_conditional_cdf(e: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Invert F(v)=(1-a)*v+a*v*v, without clipping or repair."""
    e, a = np.broadcast_arrays(np.asarray(e, float), np.asarray(a, float))
    if np.any(~np.isfinite(e)) or np.any(~np.isfinite(a)):
        raise ValueError('Nonfinite input')
    if np.any((e < 0) | (e > 1)) or np.any(np.abs(a) >= 1):
        raise ValueError('Require e in [0,1] and |a|<1')
    d = (1.0-a)**2 + 4.0*a*e
    if np.any(d <= 0):
        raise ArithmeticError('Nonpositive discriminant; no repair permitted')
    v = 2.0*e / ((1.0-a) + np.sqrt(d))
    if np.any(~np.isfinite(v)) or np.any(v < -1e-14) or np.any(v > 1+1e-14):
        raise ArithmeticError('Invalid inverse; no repair permitted')
    return v

def conditional_cdf(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (1.0-a)*v + a*v*v

def cell_indices(c: np.ndarray, bins: int) -> np.ndarray:
    c = np.asarray(c, float)
    if bins < 1 or np.any(c < 0) or np.any(c >= 1):
        raise ValueError('Require positive bins and c in [0,1)')
    return np.floor(c*bins).astype(int)

def true_response(c: np.ndarray, amplitude: float) -> np.ndarray:
    """Simulator/evaluator only. NEVER called by fit()."""
    return amplitude*np.sin(2.0*np.pi*np.asarray(c))

@dataclass
class Model:
    theta: np.ndarray  # [bins, 1] for sharing; [bins, groups] otherwise
    anchors: int
    groups: int
    rho: float = RHO

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=np.float64)
        if self.theta.ndim != 2 or self.theta.shape[1] not in (1, self.groups):
            raise ValueError('Invalid response shape')
        if not 0 < self.rho < 1 or np.any(np.abs(self.theta) > self.rho):
            raise ValueError('Response outside positivity constraint')

    @property
    def dim(self) -> int:
        return 1 + self.anchors + 3*self.groups

    def response(self, c: np.ndarray) -> np.ndarray:
        return self.theta[cell_indices(c, self.theta.shape[0])]

    def decode(self, noise: np.ndarray) -> np.ndarray:
        """All D independent Gaussian coordinates are retained; root/anchors unchanged."""
        noise = np.asarray(noise, dtype=float)
        if noise.ndim != 2 or noise.shape[1] != self.dim:
            raise ValueError('Wrong full-dimensional input shape')
        c = ndtr(noise[:, 0])
        u = ndtr(noise[:, 1+self.anchors:]).reshape(-1, self.groups, 3)
        a = self.response(c)*psi(u[..., 0])*psi(u[..., 1])
        v = u.copy()
        v[..., 2] = inverse_conditional_cdf(u[..., 2], a)
        if np.any((v <= 0) | (v >= 1)):
            raise ArithmeticError('Finite Gaussian roundtrip reached endpoint; no clipping')
        out = noise.copy()
        out[:, 1+self.anchors:] = ndtri(v).reshape(len(noise), -1)
        return out

    def encode(self, observed: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed, float)
        c, v = read_observations(observed, self.anchors, self.groups)
        a = self.response(c)*psi(v[..., 0])*psi(v[..., 1])
        u = v.copy()
        u[..., 2] = conditional_cdf(v[..., 2], a)
        if np.any((u <= 0) | (u >= 1)):
            raise ArithmeticError('Inverse reached endpoint; no clipping')
        out = observed.copy()
        out[:, 1+self.anchors:] = ndtri(u).reshape(len(observed), -1)
        return out

    def log_prob(self, observed: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed, float)
        c, v = read_observations(observed, self.anchors, self.groups)
        x = np.prod(psi(v), axis=-1)
        return -0.5*np.sum(observed**2+math.log(2*math.pi), axis=1) + np.sum(
            np.log1p(self.response(c)*x), axis=1)


def read_observations(observed: np.ndarray, anchors: int, groups: int):
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 1+anchors+3*groups:
        raise ValueError('Wrong observed dimension')
    if np.any(~np.isfinite(observed)):
        raise ValueError('Nonfinite observed data')
    c = ndtr(observed[:, 0])
    v = ndtr(observed[:, 1+anchors:]).reshape(-1, groups, 3)
    return c, v


def fit(observed: np.ndarray, bins: int, anchors: int, groups: int,
        tied: bool, rho: float = RHO) -> Model:
    """Efficient bounded moment estimator. No theta, simulator seed, or true signs input.

    Projection is a declared statistical estimator, NOT numerical inverse repair.
    """
    c, v = read_observations(observed, anchors, groups)
    cells = cell_indices(c, bins)
    statistic = 27.0*np.prod(psi(v), axis=-1)
    theta = np.empty((bins, 1 if tied else groups))
    for b in range(bins):
        rows = statistic[cells == b]
        if len(rows) == 0:
            raise ValueError('Empty context cell; stop rather than adapt bins')
        theta[b] = rows.mean() if tied else rows.mean(axis=0)
    return Model(np.clip(theta, -rho, rho), anchors, groups, rho)


def compile_equal_information_copy(model: Model) -> Model:
    """Conventional tied triangular flow with exactly the same fitted map and cost.

    This is an equivalence control, NOT an independently trained victory claim.
    """
    return Model(model.theta.copy(), model.anchors, model.groups, model.rho)


def generate_observed(rng: np.random.Generator, n: int, bins: int, anchors: int,
                      groups: int, amplitude: float, signs: np.ndarray):
    """Balanced TRAIN-only synthetic design. Returns observed data, not innovations."""
    if n % bins:
        raise ValueError('Balanced design requires n divisible by bins')
    signs = np.asarray(signs, float)
    if signs.shape != (groups,) or np.any(np.abs(signs) != 1):
        raise ValueError('Expected one +/-1 simulator sign per group')
    c = (np.repeat(np.arange(bins), n//bins)+rng.random(n))/bins
    rng.shuffle(c)
    anchor_u = rng.random((n, anchors))
    u = rng.random((n, groups, 3))
    a = true_response(c, amplitude)[:, None]*signs[None, :]*psi(u[..., 0])*psi(u[..., 1])
    v = u.copy()
    v[..., 2] = inverse_conditional_cdf(u[..., 2], a)
    observed_u = np.concatenate((c[:, None], anchor_u, v.reshape(n, -1)), axis=1)
    if np.any((observed_u <= 0) | (observed_u >= 1)):
        raise ArithmeticError('Uniform endpoint encountered; do not redraw')
    observed = ndtri(observed_u)
    return observed


def copula_kl(t: np.ndarray, e: np.ndarray, terms: int = 200) -> np.ndarray:
    """KL(p_t || p_e) using an absolutely convergent analytic series.

    p_t(v)=1+t*prod(2v_i-1), v in [0,1]^3. For |t|,|e|<=rho,
    a conservative per-triple tail bound is 3*rho**(2*(terms+1))/(1-rho**2).
    Small negative roundoff is NOT silently clipped.
    """
    t, e = np.broadcast_arrays(np.asarray(t, float), np.asarray(e, float))
    if np.any(np.abs(t) >= 1) or np.any(np.abs(e) >= 1):
        raise ValueError('Parameters must have absolute value <1')
    out = np.zeros_like(t)
    t2, e2 = t*t, e*e
    tp, ep, eo = t2.copy(), e2.copy(), e.copy()
    for m in range(1, terms+1):
        n = 2*m
        out += (tp/(n*(n-1)) + ep/n - t*eo/(n-1))/(n+1)**3
        tp *= t2
        ep *= e2
        eo *= e2
    return out


def expected_kl(model: Model, amplitude: float, signs: np.ndarray, nodes: int = 64) -> float:
    """Analytic toy population evaluation; no real dataset outcomes."""
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    bins = len(model.theta)
    total = 0.0
    for b in range(bins):
        c = (b+(roots+1)/2)/bins
        t = true_response(c, amplitude)[:, None]*np.asarray(signs)[None, :]
        kl = copula_kl(t, model.response(c))
        total += float(np.sum(weights*np.sum(kl, axis=1)))/(2*bins)
    return total


def oracle_cell_means(bins: int, amplitude: float) -> np.ndarray:
    """Evaluation diagnostic only. Not accessible to the estimator."""
    edges = np.arange(bins+1)/bins
    return amplitude*bins/(2*np.pi)*(np.cos(2*np.pi*edges[:-1])-np.cos(2*np.pi*edges[1:]))
