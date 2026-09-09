"""Globally attending full-dimensional FM with a fixed pixel permutation.

Default RGB 32 arrays become 48 channels on an 8x8 token grid by pixel unshuffle.
All 3072 coordinates remain: no fitted analysis, latent bottleneck, coarse prior,
external context, or decoder is used. The established conditional velocity has
one identically zero context channel solely for API compatibility; its input
weights are counted even though this channel carries no information.

Ordinary independent-source and variance-normalized Gaussian-reference paths
share exactly the same velocity architecture. Neither finite-Heun sampler has
an exact likelihood supplied here. Global attention costs quadratically in the
number of packed spatial tokens; packing itself costs linear coordinate work.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .flow_matching import heun_integrate
from .learned_latent_flow_matching import GlobalConditionalVelocity
from .gaussian_reference_flow_matching import gaussian_reference_loss


class GlobalFullFlowMatching(nn.Module):
    def __init__(self, channels=3, size=32, packing_factor=4, width=124,
                 attention_heads=4, path='ordinary'):
        super().__init__()
        values = (channels,size,packing_factor,width,attention_heads)
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in values):
            raise ValueError('dimensions must be positive integers')
        if size % packing_factor or width % attention_heads:
            raise ValueError('size/packing and width/heads must divide exactly')
        if path not in ('ordinary','gaussian_reference'):
            raise ValueError('unknown probability path')
        self.channels,self.size,self.packing_factor = channels,size,packing_factor
        self.packed_channels,self.packed_size = channels*packing_factor**2,size//packing_factor
        self.dimension,self.path = channels*size*size,path
        self.velocity = GlobalConditionalVelocity(self.packed_channels,1,self.packed_size,width,attention_heads)

    def _validate(self,x,shape):
        p=next(self.parameters())
        if x.ndim != 4 or len(x)<1 or x.shape[1:] != shape:
            raise ValueError('expected configured nonempty NCHW array')
        if x.dtype not in (torch.float32,torch.float64) or x.dtype != p.dtype or x.device != p.device or not bool(torch.isfinite(x).all()):
            raise ValueError('expected finite model-matched float array')

    def pack(self,x):
        """Fixed permutation; per-example log absolute determinant is zero."""
        self._validate(x,(self.channels,self.size,self.size))
        return F.pixel_unshuffle(x,self.packing_factor)

    def unpack(self,x):
        self._validate(x,(self.packed_channels,self.packed_size,self.packed_size))
        return F.pixel_shuffle(x,self.packing_factor)

    def _field(self,x,t,context=None):
        if context is not None:
            raise ValueError('unconditional model accepts no external context')
        zero=x.new_zeros((len(x),1,self.packed_size,self.packed_size))
        return self.velocity(x,t,zero)

    def training_loss(self,x,generator=None):
        target=self.pack(x)
        if self.path == 'gaussian_reference':
            return gaussian_reference_loss(self._field,target,generator=generator)
        source=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
        t=torch.rand(len(target),device=target.device,dtype=target.dtype,generator=generator)
        tb=t[:,None,None,None]
        prediction=self._field((1-tb)*source+tb*target,t)
        return (prediction-(target-source)).square().mean()

    @torch.no_grad()
    def sample_from_gaussian(self,z,steps=16):
        """Flat source follows observed pixel order; 2*steps field evaluations."""
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError('source must retain every full-dimensional coordinate')
        source=self.pack(z.reshape(-1,self.channels,self.size,self.size))
        result=heun_integrate(self._field,source,steps=steps)
        return self.unpack(result)

    @property
    def parameter_counts(self):
        total=sum(p.numel() for p in self.parameters())
        return {'total':total,'zero_context_input_weights':self.velocity.input.out_channels*9}


__all__=['GlobalFullFlowMatching']
