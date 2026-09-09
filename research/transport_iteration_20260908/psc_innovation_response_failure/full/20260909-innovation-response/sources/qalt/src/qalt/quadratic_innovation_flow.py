"""Known-context quadratic conditional flow fitted from whole observed arrays.

This is an established separable generalized-FGM-type family. Coordinate zero
has a known uniform law and h(c)=cos(2*pi*c) is fixed public structure. Given
that context, residual coordinates are conditionally independent with a shared
parameter. No learned context, teacher innovations, or VAE is supplied. An
identical conditional stochastic decoder with the same parameter ties exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, ndtri

_LOG_2PI = math.log(2*math.pi)


def _rho(value):
    if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError('rho must be a real scalar strictly between zero and one')
    value=float(value)
    if not math.isfinite(value) or not 0<value<1:
        raise ValueError('rho must be finite and strictly between zero and one')
    return value


def _array(value):
    if np.iscomplexobj(value):raise ValueError('real observations or source required')
    result=np.asarray(value,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('finite observations or source required')
    return result


@dataclass(frozen=True)
class QuadraticInnovationFitDiagnostics:
    independent_array_count: int
    dimension: int
    residuals_per_array: int
    raw_moment_estimate: float
    clipped_moment_estimate: float
    parameter_was_clipped: bool
    cluster_statistic_sample_variance: float | None
    observed_scalar_count: int
    estimator: str = 'mean of whole-array statistics 6*cos(2*pi*C)*mean(2*R-1)'
    residuals_assumed_independent_statistical_units: bool = False


@dataclass(frozen=True)
class QuadraticInnovationFlow:
    """Full-D Gaussian transport with one observed context and D-1 innovations.

    ``fit(data, rho=.65)`` returns ``(model, diagnostics)``. Each input row is
    one complete observed array; no teacher parameter or source is accepted.
    ``encode`` and ``decode`` preserve arbitrary leading batch dimensions and
    return ``(values, log_abs_det)`` for the requested transformation.
    ``log_prob`` returns minus infinity outside the open cube. Nonfinite
    inputs are rejected. Floating-point CDF saturation is rejected explicitly.
    """
    theta: float
    dimension: int
    rho: float = .65

    def __post_init__(self):
        rho=_rho(self.rho)
        if isinstance(self.dimension,(bool,np.bool_)) or not isinstance(self.dimension,(int,np.integer)) or self.dimension<2:
            raise ValueError('dimension must be an integer at least two')
        if isinstance(self.theta,(bool,np.bool_)) or not np.isscalar(self.theta) or np.iscomplexobj(self.theta):
            raise ValueError('theta must be a real scalar')
        theta=float(self.theta)
        if not math.isfinite(theta) or abs(theta)>rho:
            raise ValueError('theta must lie inside the declared parameter interval')
        object.__setattr__(self,'rho',rho)
        object.__setattr__(self,'theta',theta)
        object.__setattr__(self,'dimension',int(self.dimension))

    @classmethod
    def fit(cls, observations, *, rho=.65):
        rho=_rho(rho)
        data=_array(observations)
        if data.ndim!=2 or data.shape[0]<1 or data.shape[1]<2:
            raise ValueError('fitting requires nonempty observed [n,D] arrays with D>=2')
        if np.any((data<=0)|(data>=1)):
            raise ValueError('fitting observations must lie strictly inside the unit cube')
        cluster=6*np.cos(2*np.pi*data[:,0])*np.mean(2*data[:,1:]-1,axis=1)
        raw=float(cluster.mean())
        theta=float(np.clip(raw,-rho,rho))
        model=cls(theta=theta,dimension=data.shape[1],rho=rho)
        diagnostics=QuadraticInnovationFitDiagnostics(
            independent_array_count=len(data),dimension=model.dimension,
            residuals_per_array=model.dimension-1,raw_moment_estimate=raw,
            clipped_moment_estimate=theta,parameter_was_clipped=theta!=raw,
            cluster_statistic_sample_variance=float(cluster.var(ddof=1)) if len(data)>1 else None,
            observed_scalar_count=data.size)
        return model,diagnostics

    def _rows(self,values):
        array=_array(values)
        if array.ndim<1 or array.shape[-1]!=self.dimension or array.size==0:
            raise ValueError('expected configured full dimension and nonempty leading batches')
        return array.reshape(-1,self.dimension),array.shape[:-1]

    def _log_density(self,rows):
        amplitude=self.theta*np.cos(2*np.pi*rows[:,0,None])
        return np.log1p(amplitude*(2*rows[:,1:]-1)).sum(axis=1)

    def log_prob(self,values):
        rows,shape=self._rows(values)
        inside=np.all((rows>0)&(rows<1),axis=1)
        result=np.full(len(rows),-np.inf)
        result[inside]=self._log_density(rows[inside])
        return result.reshape(shape)

    def decode(self,source):
        gaussian,shape=self._rows(source)
        uniforms=ndtr(gaussian)
        if np.any((uniforms<=0)|(uniforms>=1)):
            raise FloatingPointError('Gaussian CDF reached a finite-precision boundary')
        rows=uniforms.copy()
        amplitude=self.theta*np.cos(2*np.pi*rows[:,0,None])
        u=uniforms[:,1:]
        discriminant=(1-u)*(1-amplitude)**2+u*(1+amplitude)**2
        rows[:,1:]=2*u/(1-amplitude+np.sqrt(discriminant))
        if not np.isfinite(rows).all() or np.any((rows<=0)|(rows>=1)):
            raise FloatingPointError('quadratic inverse reached a finite-precision boundary')
        log_base=-.5*(self.dimension*_LOG_2PI+np.sum(gaussian**2,axis=1))
        return rows.reshape(shape+(self.dimension,)),(log_base-self._log_density(rows)).reshape(shape)

    def encode(self,observations):
        rows,shape=self._rows(observations)
        if np.any((rows<=0)|(rows>=1)):
            raise ValueError('encoding observations must lie strictly inside the unit cube')
        uniforms=rows.copy()
        amplitude=self.theta*np.cos(2*np.pi*rows[:,0,None])
        residual=rows[:,1:]
        uniforms[:,1:]=residual+amplitude*residual*(residual-1)
        if np.any((uniforms<=0)|(uniforms>=1)):
            raise FloatingPointError('conditional CDF reached a finite-precision boundary')
        gaussian=ndtri(uniforms)
        if not np.isfinite(gaussian).all():raise FloatingPointError('Gaussian quantile is nonfinite')
        log_base=-.5*(self.dimension*_LOG_2PI+np.sum(gaussian**2,axis=1))
        return gaussian.reshape(shape+(self.dimension,)),(self._log_density(rows)-log_base).reshape(shape)


__all__=['QuadraticInnovationFlow','QuadraticInnovationFitDiagnostics']
