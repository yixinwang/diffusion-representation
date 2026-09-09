"""Standalone Pro10 scalar-kernel review. No repository writes or scheduler calls.

Reference formulas inspected at e7ee7938a9b146aa5b167a1176e53f99f9ccd4bd,
research/transport_iteration_20260908/pro8_artifacts/positive_class_reference.py.
This file is a new implementation, NOT a byte-identical repository snapshot.
Float64 only. The hybrid threshold 8 is fixed before local timings.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, ndtri, ndtri_exp, log_ndtr

NFES = (4, 8, 16, 32, 64)
CENTRAL = 8.0

def _inputs(z, e):
    z, e = np.broadcast_arrays(np.asarray(z, dtype=np.float64),
                               np.asarray(e, dtype=np.float64))
    if not np.isfinite(z).all() or not np.isfinite(e).all():
        raise ValueError('finite inputs required')
    if np.any(np.abs(e) > 0.6):
        raise ValueError('this reviewed implementation is scoped to |e| <= 0.6')
    return z, e

def quantile_log_reference(z, e):
    """Original all-log single-branch formula, with the common scoped validator."""
    z, e = _inputs(z, e)
    lp = log_ndtr(-np.abs(z)); p = np.exp(lp)
    sign = np.where(z <= 0, -1., 1.)
    base = 1 + sign*e
    denom = base + np.sqrt(base*base - 4*sign*e*p)
    return -sign*ndtri_exp(np.log(2.) + lp - np.log(denom))

def _q_region(z, e, logarithmic):
    sign = np.where(z <= 0, -1., 1.)
    a = sign*e; base = 1 + a
    if logarithmic:
        lp = log_ndtr(-np.abs(z)); p = np.exp(lp)
        den = base + np.sqrt(base*base - 4*a*p)
        out = -sign*ndtri_exp(np.log(2.) + lp - np.log(den))
    else:
        p = ndtr(-np.abs(z))
        den = base + np.sqrt(base*base - 4*a*p)
        out = -sign*ndtri((2*p)/den)
    return out

def quantile_hybrid(z, e):
    """Same real map F_e^-1(Phi(z)). No clipping, truncation or resampling.

    Tail entries alone execute the logarithmic kernel. Inputs beyond finite
    special-function representability fail explicitly, not by clipping.
    No assertion of exactness for the distribution of rounded float outputs.
    """
    z, e = _inputs(z, e)
    mask = np.abs(z) <= CENTRAL
    if bool(np.all(mask)):
        out = _q_region(z, e, False)
    elif not bool(np.any(mask)):
        out = _q_region(z, e, True)
    else:
        out = np.empty(z.shape, dtype=np.float64)
        out[mask] = _q_region(z[mask], e[mask], False)
        out[~mask] = _q_region(z[~mask], e[~mask], True)
    if not np.isfinite(out).all():
        raise FloatingPointError('quantile outside finite kernel representability')
    return out

def inverse_log_reference(r, e):
    r, e = _inputs(r, e)
    lp = log_ndtr(-np.abs(r)); p = np.exp(lp)
    lo = ndtri_exp(lp + np.log(1-e+e*p))
    hi = -ndtri_exp(lp + np.log(1+e-e*p))
    return np.where(r <= 0, lo, hi)

def _inverse_region(r, e, logarithmic):
    sign = np.where(r <= 0, -1., 1.)
    if logarithmic:
        lp = log_ndtr(-np.abs(r)); p = np.exp(lp)
        return -sign*ndtri_exp(lp + np.log1p(sign*e*(1-p)))
    p = ndtr(-np.abs(r))
    return -sign*ndtri(p*(1+sign*e*(1-p)))

def inverse_hybrid(r, e):
    r, e = _inputs(r, e); mask = np.abs(r) <= CENTRAL
    if bool(np.all(mask)):
        out = _inverse_region(r, e, False)
    elif not bool(np.any(mask)):
        out = _inverse_region(r, e, True)
    else:
        out = np.empty(r.shape, dtype=np.float64)
        out[mask] = _inverse_region(r[mask], e[mask], False)
        out[~mask] = _inverse_region(r[~mask], e[~mask], True)
    if not np.isfinite(out).all():
        raise FloatingPointError('inverse outside finite kernel representability')
    return out

def field_reference(y, t, e):
    a=1-t; s2=a*a+t*t; d=np.sqrt(2*a*a+t*t); k=t/d
    value=k*y
    return 2*e*a/(s2*d)*np.exp(-value*value/2)/np.sqrt(2*np.pi)/(1+e*(2*ndtr(value)-1))

def heun_reference(z, e, nfe):
    if nfe not in NFES: raise ValueError('fixed grid required')
    y=np.array(z,dtype=np.float64,copy=True); h=2/nfe
    for i in range(nfe//2):
        v=field_reference(y,i*h,e)
        y+=h/2*(v+field_reference(y+h*v,(i+1)*h,e))
    return y

@dataclass(frozen=True)
class HeunPlan:
    """Immutable time-only cache. Charge construction to setup, not each batch."""
    nfe: int
    h: float
    coefficients: tuple[tuple[float, float], ...]
    @staticmethod
    def build(nfe):
        if nfe not in NFES: raise ValueError('fixed grid required')
        h=2/nfe; coefficients=[]
        for i in range(nfe//2+1):
            t=i*h; a=1-t; s2=a*a+t*t; d=math.sqrt(2*a*a+t*t)
            coefficients.append((t/d, 2*a/(s2*d)/math.sqrt(2*math.pi)))
        return HeunPlan(nfe,h,tuple(coefficients))

def _field_cached(y, e, coefficient):
    k, A = coefficient; q=k*y
    return (A*e)*np.exp(-q*q/2)/(1+e*(2*ndtr(q)-1))

def heun_cached(z, e, plan):
    """Identical real-arithmetic uniform-Heun map; N-2 nonlinear kernels.

    w(0,y,e)=e/sqrt(pi); w(1,y,e)=0. Repeated interior times have
    distinct states and their velocities are NOT reused.
    No validation asymmetry: caller performs common full-source validation.
    """
    if not isinstance(plan, HeunPlan): raise TypeError('HeunPlan required')
    y=np.array(z,dtype=np.float64,copy=True); h=plan.h; steps=plan.nfe//2
    for i in range(steps):
        v=(e/math.sqrt(math.pi) if i==0 else
           _field_cached(y,e,plan.coefficients[i]))
        if i+1==steps:
            y += (h/2)*v
        else:
            pred = y+h*v
            v2 = _field_cached(pred,e,plan.coefficients[i+1])
            y += (h/2)*(v+v2)
    return y

def heun_with_jacobian(z, e, plan):
    """Evaluation-only exact analytic derivative of the real Heun formula.
    Not used or charged to generation timings.
    """
    y=np.array(z,dtype=np.float64,copy=True); J=np.ones_like(y)
    h=plan.h; steps=plan.nfe//2
    def fd(y,coefficient):
        k,A=coefficient; q=k*y
        ph=np.exp(-q*q/2)/math.sqrt(2*math.pi)
        den=1+e*(2*ndtr(q)-1)
        w=(A*e)*np.exp(-q*q/2)/den
        wy=-k*w*(q+2*e*ph/den)
        return w,wy
    for i in range(steps):
        v,dy=(e/math.sqrt(math.pi),0.) if i==0 else fd(y,plan.coefficients[i])
        if i+1==steps:
            fac=1+(h/2)*dy; y += (h/2)*v
        else:
            v2,dy2=fd(y+h*v,plan.coefficients[i+1])
            fac=1+(h/2)*(dy+dy2*(1+h*dy))
            y += (h/2)*(v+v2)
        if np.any(fac<=0): raise FloatingPointError('nonpositive step Jacobian')
        J *= fac
    return y,J
