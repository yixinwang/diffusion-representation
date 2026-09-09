"""Opt-in dense-kernel backend for the unchanged conditional spline flow.

Inherits precisely the original parameters, masks, and state-dict names. Dense
arithmetic returns device validity, combined across layers and rejected once at
the end, before a loss/output can be accepted or differentiated. Original model
defaults are untouched. Compilation, clones/workspace, first-call setup and full
conditioner cost must be measured separately; scalar kernel results do not prove
a full-flow acceleration. No full-model speed claim is made here.
"""
from __future__ import annotations
import torch
from .global_conditional_spline import GlobalConditionalSplineDecoder
from .dense_spline import dense_spline_kernel


class DenseGlobalConditionalSplineDecoder(GlobalConditionalSplineDecoder):
    """Identical flow with explicit backend='dense_eager' or 'compiled'.

    The compiled option uses torch.compile(fullgraph=True,dynamic=False) and the
    default Inductor backend. Compilation is opt-in and can fail; no fallback is
    provided. There is no data-dependent host synchronization inside the dense
    spline layers. Inherited input checks precede the transform, and a single
    final validity check rejects the entire batch before returning.
    """
    def __init__(self, residual_channels, context_channels, size, layers=4,
                 width=32, bins=8, attention_heads=4, *, backend='dense_eager'):
        if backend not in ('dense_eager','compiled'):
            raise ValueError('backend must be dense_eager or compiled')
        super().__init__(residual_channels,context_channels,size,layers,width,bins,attention_heads)
        self.backend=backend
        self._spline = (dense_spline_kernel if backend=='dense_eager' else
                        torch.compile(dense_spline_kernel,fullgraph=True,dynamic=False))

    def _transform(self,x,coarse,inverse):
        self._validate(x,coarse)
        total=x.new_zeros(x.shape[0])
        valid=torch.ones((),dtype=torch.bool,device=x.device)
        for layer in reversed(self.layers) if inverse else self.layers:
            fixed=torch.where(layer.mask,x,torch.zeros_like(x))
            raw=layer.conditioner(fixed,x.new_zeros(x.shape[0]),coarse)
            raw=raw.reshape(x.shape[0],layer.channels,3*layer.bins+1,*x.shape[2:]).permute(0,1,3,4,2)
            valid=valid & torch.isfinite(raw).all()
            shift,log_scale=raw[...,0],2*torch.tanh(raw[...,1])
            w=raw[...,2:2+layer.bins]
            h=raw[...,2+layer.bins:2+2*layer.bins]
            d=raw[...,2+2*layer.bins:]
            if inverse:
                value,ld,layer_valid=self._spline(x,w,h,d,inverse=True)
                value=(value-shift)*torch.exp(-log_scale)
                ld=ld-log_scale
            else:
                value=x*torch.exp(log_scale)+shift
                valid=valid & torch.isfinite(value).all()
                value,ld,layer_valid=self._spline(value,w,h,d)
                ld=ld+log_scale
            # Check all computed coordinates, including inactive masked entries;
            # masking must not hide an arithmetic failure.
            valid=valid & layer_valid & torch.isfinite(value).all() & torch.isfinite(ld).all()
            x=torch.where(layer.mask,x,value)
            increment=torch.where(layer.mask,torch.zeros_like(ld),ld).flatten(1).sum(1)
            total=total+increment
        valid=valid & torch.isfinite(x).all() & torch.isfinite(total).all()
        if not bool(valid):
            raise FloatingPointError('dense conditional spline batch failed aggregated numerical validation')
        return x,total


__all__=['DenseGlobalConditionalSplineDecoder']
