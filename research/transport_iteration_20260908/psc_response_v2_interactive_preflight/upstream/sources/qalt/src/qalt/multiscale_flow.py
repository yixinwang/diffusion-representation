"""Complete Gaussian-source multiscale spline flow; no novelty claim.

Haar conditional factorization follows Wavelet Flow (Yu et al., 2020).
Elementwise transforms follow Neural Spline Flows (Durkan et al., 2019).
All coarse and detail laws are learned; generation accepts no observed context.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .spline import rational_quadratic_spline


def haar_split(x):
    if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError('expected NCHW with even spatial dimensions')
    a,b,c,d=x[...,::2,::2],x[...,::2,1::2],x[...,1::2,::2],x[...,1::2,1::2]
    return (a+b+c+d)/2, torch.cat(((a+b-c-d)/2,(a-b+c-d)/2,(a-b-c+d)/2),dim=1)


def haar_merge(coarse,detail):
    if detail.shape[1] != 3*coarse.shape[1] or detail.shape[0] != coarse.shape[0] or detail.shape[2:] != coarse.shape[2:]:
        raise ValueError('incompatible coarse/detail shapes')
    h,v,d=detail.chunk(3,dim=1)
    out=coarse.new_empty((coarse.shape[0],coarse.shape[1],2*coarse.shape[2],2*coarse.shape[3]))
    out[...,::2,::2]=(coarse+h+v+d)/2
    out[...,::2,1::2]=(coarse+h-v-d)/2
    out[...,1::2,::2]=(coarse-h+v-d)/2
    out[...,1::2,1::2]=(coarse-h-v+d)/2
    return out


class SplineCoupling(nn.Module):
    def __init__(self,channels,context_channels=0,width=32,bins=8,parity=0):
        super().__init__()
        self.channels,self.context_channels,self.bins,self.parity=channels,context_channels,bins,parity
        self.net=nn.Sequential(nn.Conv2d(channels+context_channels,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,channels*(3*bins-1),1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,context=None,inverse=False):
        if x.ndim!=4 or x.shape[1]!=self.channels: raise ValueError('wrong coupling shape')
        rows=torch.arange(x.shape[2],device=x.device)[:,None]
        cols=torch.arange(x.shape[3],device=x.device)[None,:]
        channel=torch.arange(self.channels,device=x.device)[:,None,None]
        mask=((rows+cols+channel+self.parity)%2==0)[None]
        unchanged=x*mask
        if self.context_channels:
            if context is None or context.shape!=(x.shape[0],self.context_channels,*x.shape[2:]): raise ValueError('wrong context shape')
            net_input=torch.cat((unchanged,context),dim=1)
        else:
            if context is not None: raise ValueError('unconditional coupling received context')
            net_input=unchanged
        params=self.net(net_input).reshape(x.shape[0],self.channels,3*self.bins-1,*x.shape[2:]).permute(0,1,3,4,2)
        y,ld=rational_quadratic_spline(x,params[...,:self.bins],params[...,self.bins:2*self.bins],params[...,2*self.bins:],inverse=inverse)
        return torch.where(mask,x,y),torch.where(mask,torch.zeros_like(ld),ld).flatten(1).sum(1)


class CouplingStack(nn.Module):
    def __init__(self,channels,context_channels,layers,width,bins):
        super().__init__()
        if layers<2:raise ValueError('at least two alternating coupling layers required')
        self.location=nn.Parameter(torch.zeros(1,channels,1,1))
        self.raw_log_scale=nn.Parameter(torch.zeros(1,channels,1,1))
        self.layers=nn.ModuleList([SplineCoupling(channels,context_channels,width,bins,i%2) for i in range(layers)])

    def forward(self,x,context=None,inverse=False):
        log_scale=3*torch.tanh(self.raw_log_scale)
        affine_ld=log_scale.sum()*x.shape[2]*x.shape[3]
        total=x.new_zeros(x.shape[0])
        if not inverse:
            x=(x-self.location)*torch.exp(-log_scale);total=total-affine_ld
        for layer in (reversed(self.layers) if inverse else self.layers):
            x,ld=layer(x,context,inverse);total=total+ld
        if inverse:
            x=x*torch.exp(log_scale)+self.location;total=total+affine_ld
        return x,total


class MultiscaleSplineFlow(nn.Module):
    """NCHW bijection, returning one full-dimensional flattened Gaussian code.

For unit_interval=True, input must be strictly inside (0,1); logit is exact,
with no clipping. Otherwise the model is a density on all real tensors.
Fixed Haar is supplied equally to baselines; no generating chart is learned
from hidden synthetic coordinates. This class contains no data loading.
"""
    def __init__(self,channels=3,size=32,levels=2,coarse_layers=6,detail_layers=4,width=32,bins=8,unit_interval=False):
        super().__init__()
        if levels<1 or size%2**levels:raise ValueError('invalid pyramid size')
        self.channels,self.size,self.levels,self.unit_interval=channels,size,levels,unit_interval
        self.dimension=channels*size*size
        self.coarse_size=size//2**levels
        self.coarse_flow=CouplingStack(channels,0,coarse_layers,width,bins)
        self.detail_flows=nn.ModuleList([CouplingStack(3*channels,channels,detail_layers,width,bins) for _ in range(levels)])
        self.block_shapes=[(channels,self.coarse_size,self.coarse_size)]+[(3*channels,self.coarse_size*2**i,self.coarse_size*2**i) for i in range(levels)]
        assert sum(math.prod(s) for s in self.block_shapes)==self.dimension

    def encode(self,x):
        if x.ndim!=4 or x.shape[1:]!=(self.channels,self.size,self.size): raise ValueError('wrong input shape')
        total=x.new_zeros(x.shape[0])
        if self.unit_interval:
            if not bool(torch.all((x>0)&(x<1))):raise ValueError('unit interval input must be strictly interior')
            total=(-torch.log(x)-torch.log1p(-x)).flatten(1).sum(1)
            x=torch.log(x)-torch.log1p(-x)
        details=[];coarse=x
        for _ in range(self.levels):
            coarse,detail=haar_split(coarse);details.append(detail)
        z,ld=self.coarse_flow(coarse);total=total+ld;blocks=[z.flatten(1)]
        for flow,detail in zip(self.detail_flows,reversed(details)):
            z,ld=flow(detail,coarse);total=total+ld;blocks.append(z.flatten(1))
            coarse=haar_merge(coarse,detail)
        return torch.cat(blocks,dim=1),total

    def decode(self,z):
        if z.ndim!=2 or z.shape[1]!=self.dimension:raise ValueError('source must contain every Gaussian coordinate')
        blocks=[p.reshape(z.shape[0],*s) for p,s in zip(z.split([math.prod(s) for s in self.block_shapes],dim=1),self.block_shapes)]
        coarse,total=self.coarse_flow(blocks[0],inverse=True)
        for flow,source in zip(self.detail_flows,blocks[1:]):
            detail,ld=flow(source,coarse,inverse=True);total=total+ld
            coarse=haar_merge(coarse,detail)
        if self.unit_interval:
            total=total+(-torch.nn.functional.softplus(coarse)-torch.nn.functional.softplus(-coarse)).flatten(1).sum(1)
            coarse=torch.sigmoid(coarse)
        return coarse,total

    def log_prob(self,x):
        z,ld=self.encode(x)
        return -.5*(z.square()+math.log(2*math.pi)).sum(1)+ld

    def forward(self,x):return self.log_prob(x)

    def sample_from_gaussian(self,z):return self.decode(z)[0]

    def representation(self,x):return self.encode(x)[0]


class CopiedStochasticLatentDecoder(nn.Module):
    """Equality control: identical learned coarse prior and stochastic decoder.

This wrapper shares model weights and calls the identical computation. It
establishes a class-containment tie, not an independently trained baseline.
Both training and inference costs inherit those of the wrapped flow.
"""
    def __init__(self,flow):super().__init__();self.flow=flow
    def sample_from_gaussian(self,z):return self.flow.sample_from_gaussian(z)
    def log_prob(self,x):return self.flow.log_prob(x)
