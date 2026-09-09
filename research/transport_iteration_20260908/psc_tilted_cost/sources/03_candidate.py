"""Work-only equivalent central/tail tilt transport and cached-time Heun.

No fitted parameters, measurements of performance, data loaders, or production
integration. The branch selects a numerical algorithm, never clips/truncates z.
SciPy binary64 functions are numerical implementations, not interval enclosures.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr,ndtri,ndtri_exp,log_ndtr


def _inputs(z,e):
    z,e=np.broadcast_arrays(np.asarray(z,dtype=np.float64),np.asarray(e,dtype=np.float64))
    if not np.isfinite(z).all() or not np.isfinite(e).all() or np.any(np.abs(e)>.6):
        raise ValueError('finite sources and e in [-.6,.6] required')
    return z,e


def quantile_transport(z,e):
    z,e=_inputs(z,e)
    result=np.empty_like(z)
    central=np.abs(z)<=5
    for mask,use_log in ((central,False),(~central,True)):
        value,tilt=z[mask],e[mask]
        sign=np.where(value<=0,-1.,1.)
        base=1+sign*tilt
        if use_log:
            lp=log_ndtr(-np.abs(value));p=np.exp(lp)
        else:p=ndtr(-np.abs(value))
        discriminant=base*base-4*sign*tilt*p
        if np.any(discriminant<=0) or not np.isfinite(discriminant).all():
            raise FloatingPointError('invalid quadratic discriminant')
        denominator=base+np.sqrt(discriminant)
        if use_log:
            result[mask]=-sign*ndtri_exp(np.log(2.)+lp-np.log(denominator))
        else:
            probability=2*p/denominator
            if np.any((probability<=0)|(probability>=1)):
                raise FloatingPointError('central inverse probability outside open interval')
            result[mask]=-sign*ndtri(probability)
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite transport result')
    return result


def inverse_transport(r,e):
    r,e=_inputs(r,e)
    sign=np.where(r<=0,-1.,1.)
    lp=log_ndtr(-np.abs(r));p=np.exp(lp)
    # lower F=(1-e)p+ep²; upper survival=(1+e)p-ep².
    result=-sign*ndtri_exp(lp+np.log1p(sign*e*(1-p)))
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite inverse result')
    return result


def log_density(r,e):
    r,e=_inputs(r,e)
    return -.5*r*r-.5*math.log(2*math.pi)+np.log1p(e*(2*ndtr(r)-1))


def forward_logdet(z,e):
    z,e=_inputs(z,e);r=quantile_transport(z,e)
    return .5*(r-z)*(r+z)-np.log1p(e*(2*ndtr(r)-1))


@dataclass(frozen=True)
class TimeConstants:
    t: float
    k: float
    normal_coefficient: float


def time_constants(t):
    if not 0<=t<=1:raise ValueError('time outside [0,1]')
    a=1-t;s2=a*a+t*t;d=math.sqrt(2*a*a+t*t)
    return TimeConstants(t,t/d,2*a/(s2*d*math.sqrt(2*math.pi)))


def cached_field(y,e,constants):
    value=constants.k*y
    return e*constants.normal_coefficient*np.exp(-.5*value*value)/(1+e*(2*ndtr(value)-1))


def heun(z,e,nfe):
    if nfe not in (4,8,16,32,64):raise ValueError('fixed actual-call grid required')
    z,e=_inputs(z,e);y=z.copy();h=2/nfe
    cache=tuple(time_constants(i*h) for i in range(nfe//2+1))
    calls=0
    for i in range(nfe//2):
        first=cached_field(y,e,cache[i]);calls+=1
        second=cached_field(y+h*first,e,cache[i+1]);calls+=1
        y+=h/2*(first+second)
    if calls!=nfe or not np.isfinite(y).all():raise FloatingPointError('invalid complete Heun evaluation')
    return y,{'field_calls':calls,'cached_time_points':len(cache),'endpoint_zero_field_called':True}


def endpoint_heun(z,e,nfe):
    """Same Heun map, with analytic first/final fields and no final predictor.

    There are nfe equivalent mathematical stages, but only nfe-2 nontrivial
    exp/CDF field-kernel calls. Time-cache construction remains inside this call.
    This algorithmic specialization never changes step sizes or coefficients.
    """
    if nfe not in (4,8,16,32,64):raise ValueError('fixed mathematical-stage grid required')
    z,e=_inputs(z,e);y=z.copy();h=2/nfe
    cache=tuple(time_constants(i*h) for i in range(nfe//2+1))
    calls=0
    for i in range(nfe//2):
        if i==0:first=e/math.sqrt(math.pi)
        else:
            first=cached_field(y,e,cache[i]);calls+=1
        if i==nfe//2-1:
            # w(1,any finite state;e)=0, so no predictor allocation is needed.
            y+=h/2*first
        else:
            second=cached_field(y+h*first,e,cache[i+1]);calls+=1
            y+=h/2*(first+second)
    if calls!=nfe-2 or not np.isfinite(y).all():raise FloatingPointError('invalid specialized Heun evaluation')
    return y,{'equivalent_mathematical_stages':nfe,'nontrivial_field_kernel_calls':calls,
              'analytic_endpoint_stages':2,'final_predictor_allocated':False,
              'cached_time_points':len(cache)}
