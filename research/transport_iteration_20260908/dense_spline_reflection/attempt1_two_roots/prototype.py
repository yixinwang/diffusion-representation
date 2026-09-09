# SCRATCH PROTOTYPE ONLY: exact reflected inverse, no native replay or production repair.
# Derived from preserved original_dense_spline.py; exact-real map is unchanged.
"""Additive dense rational-quadratic spline with device-side validity status.

Same Neural Spline Flows equations and identity tails as qalt.spline; no model
uses this implementation by default. Unlike the reference, this evaluates fixed
shapes without boolean indexing or tensor-to-Python validation synchronization.
The caller MUST aggregate returned validity with logical AND across every layer
and reject the whole batch if false, before accepting outputs, gradients, losses,
or samples. Ignoring validity is an incorrect use of this interface.

No profiling claim is made: extra all-coordinate arithmetic, workspace, compiler
startup/recompilation, and the final aggregated host check must all be charged.
Dense evaluation may be slower, especially on all-tail batches. Provenance:
Durkan et al., Neural Spline Flows (NeurIPS 2019), section3.1 and appendixA;
https://papers.nips.cc/paper_files/paper/2019/file/7ac71d433f282034e088473244df8c02-Paper.pdf
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def dense_spline_kernel(inputs, raw_w, raw_h, raw_d, *, inverse=False,
                        tail_bound=3., min_bin_width=1e-3,
                        min_bin_height=1e-3, min_derivative=1e-3):
    """Unchecked static metadata; returns (values, logdet, scalar_bool_valid).

    Use dense_rational_quadratic_spline to validate shape/dtype/configuration.
    This kernel contains no data-dependent Python branch or host scalar read.
    Only out-of-domain evaluation arguments are clamped; interior roots, values,
    and discriminants are never clipped to pretend success. Invalid intermediate
    divisions/square roots use placeholders for numerical containment and always
    invalidate the batch. Their returned values have no accepted interpretation.
    """
    k=raw_w.shape[-1]
    inside=(inputs > -tail_bound)&(inputs < tail_bound)
    valid=torch.isfinite(inputs).all() & torch.isfinite(raw_w).all() & torch.isfinite(raw_h).all() & torch.isfinite(raw_d).all()
    # Inactive tails use identity parameters, avoiding unused extreme arithmetic
    # and preserving zero parameter gradients through torch.where.
    rw=torch.where(inside[...,None],raw_w,torch.zeros_like(raw_w))
    rh=torch.where(inside[...,None],raw_h,torch.zeros_like(raw_h))
    rd=torch.where(inside[...,None],raw_d,torch.zeros_like(raw_d))
    x=inputs.clamp(-tail_bound,tail_bound)
    widths=min_bin_width+(1-k*min_bin_width)*torch.softmax(rw,dim=-1)
    heights=min_bin_height+(1-k*min_bin_height)*torch.softmax(rh,dim=-1)
    def knots(f):
        return torch.cat((torch.full_like(f[...,:1],-tail_bound),
            -tail_bound+2*tail_bound*torch.cumsum(f,dim=-1)[...,:-1],
            torch.full_like(f[...,:1],tail_bound)),dim=-1)
    xk,yk=knots(widths),knots(heights)
    widths=xk[...,1:]-xk[...,:-1];heights=yk[...,1:]-yk[...,:-1]
    good_bins=(torch.isfinite(widths)&(widths>0)&torch.isfinite(heights)&(heights>0)).all(-1)
    valid=valid & ((~inside)|good_bins).all()
    derivatives=torch.cat((torch.ones_like(widths[...,:1]),
        min_derivative+F.softplus(rd+math.log(math.expm1(1-min_derivative))),
        torch.ones_like(widths[...,:1])),dim=-1)
    search=yk if inverse else xk
    index=(x[...,None]>=search[...,1:-1]).sum(-1,keepdim=True)
    def take(a):return a.gather(-1,index).squeeze(-1)
    xl,yl=take(xk),take(yk)
    xr,yr=take(xk[...,1:]),take(yk[...,1:])
    width,height=take(widths),take(heights)
    dl,dr=take(derivatives),take(derivatives[...,1:])
    # Invalid bins are contained, never reported valid.
    width=torch.where(torch.isfinite(width)&(width>0),width,torch.ones_like(width))
    height=torch.where(torch.isfinite(height)&(height>0),height,torch.ones_like(height))
    delta=height/width
    curvature=dl+dr-2*delta
    if inverse:
        def inverse_fraction(eta, left_derivative, right_derivative):
            good_eta=torch.isfinite(eta)&(eta>=0)&(eta<=1)
            b=left_derivative-eta*curvature
            # Algebraically a=delta-b; avoids delta-left+eta*curvature.
            a=delta-b
            c=-delta*eta
            disc=(left_derivative*(1-eta)-right_derivative*eta).square()+4*delta.square()*eta*(1-eta)
            good_disc=torch.isfinite(disc)&(disc>=0)
            root=torch.sqrt(torch.where(good_disc,disc,torch.ones_like(disc)))
            negative=b<0
            den=torch.where(negative,2*a,b+root)
            good_den=torch.isfinite(den)&(den!=0)
            safe_den=torch.where(good_den,den,torch.ones_like(den))
            fraction=torch.where(negative,-b+root,-2*c)/safe_den
            return fraction,good_eta&good_disc&good_den&torch.isfinite(fraction)
        left,left_valid=inverse_fraction((x-yl)/height,dl,dr)
        right,right_valid=inverse_fraction((yr-x)/height,dr,dl)
        use_left=left<=.5
        short=torch.where(use_left,left,right)
        theta=torch.where(use_left,short,1-short)
        complement=torch.where(use_left,1-short,short)
        chosen_valid=torch.where(use_left,left_valid,right_valid)
        valid=valid&((~inside)|chosen_valid).all()
    else:
        theta=(x-xl)/width
        complement=1-theta
    # Root leaving its bin is a numerical failure, not a clipped inverse.
    good_theta=torch.isfinite(theta)&(theta>=0)&(theta<=1)&torch.isfinite(complement)&(complement>=0)&(complement<=1)
    valid=valid & ((~inside)|good_theta).all()
    cross=theta*complement
    denominator=delta+curvature*cross
    numerator=dr*theta.square()+2*delta*cross+dl*complement.square()
    good_terms=torch.isfinite(delta)&(delta>0)&torch.isfinite(denominator)&(denominator>0)&torch.isfinite(numerator)&(numerator>0)
    valid=valid & ((~inside)|good_terms).all()
    safe_delta=torch.where(good_terms,delta,torch.ones_like(delta))
    safe_den=torch.where(good_terms,denominator,torch.ones_like(denominator))
    safe_num=torch.where(good_terms,numerator,torch.ones_like(numerator))
    forward_ld=2*torch.log(safe_delta)+torch.log(safe_num)-2*torch.log(safe_den)
    mapped=torch.where(use_left,xl+theta*width,xr-complement*width) if inverse else yl+height*(delta*theta.square()+dl*cross)/safe_den
    ld=-forward_ld if inverse else forward_ld
    valid=valid & ((~inside)|(torch.isfinite(mapped)&torch.isfinite(ld))).all()
    return torch.where(inside,mapped,inputs),torch.where(inside,ld,torch.zeros_like(inputs)),valid


def dense_rational_quadratic_spline(inputs, unnormalized_widths,
        unnormalized_heights, unnormalized_derivatives, *, inverse=False,
        tail_bound=3., min_bin_width=1e-3,min_bin_height=1e-3,min_derivative=1e-3):
    """Static-validation wrapper; numeric validity stays a device scalar tensor.

    Input S; widths/heights S+(K,); derivatives S+(K-1,). No broadcasting.
    All-zero parameters initialize identity; endpoints and tails are identity.
    One caller-level ``bool(torch.stack(statuses).all())`` is required after a
    whole batch transformation. It must raise on false before backward/use.
    """
    params=(unnormalized_widths,unnormalized_heights,unnormalized_derivatives)
    if not isinstance(inputs,torch.Tensor) or inputs.dtype not in (torch.float32,torch.float64):
        raise ValueError('input must be float32/float64 tensor')
    if any(not isinstance(p,torch.Tensor) or p.dtype!=inputs.dtype or p.device!=inputs.device for p in params):
        raise ValueError('model parameters must share tensor dtype/device')
    if params[0].ndim!=inputs.ndim+1:
        raise ValueError('width tensor needs one appended bin dimension')
    k=params[0].shape[-1]
    if k<2 or params[0].shape!=inputs.shape+(k,) or params[1].shape!=inputs.shape+(k,) or params[2].shape!=inputs.shape+(k-1,):
        raise ValueError('parameter shapes must exactly match input and bins')
    if not isinstance(inverse,bool) or not math.isfinite(tail_bound) or tail_bound<=0:
        raise ValueError('invalid direction or tail bound')
    if not(0<min_bin_width<1/k and 0<min_bin_height<1/k and 0<min_derivative<1):
        raise ValueError('invalid minimum spline parameters')
    return dense_spline_kernel(inputs,*params,inverse=inverse,tail_bound=tail_bound,
        min_bin_width=min_bin_width,min_bin_height=min_bin_height,min_derivative=min_derivative)


__all__=['dense_rational_quadratic_spline','dense_spline_kernel']
