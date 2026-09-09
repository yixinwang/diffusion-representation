"""Exact learned-analysis/coarse/conditional-residual flow on full logits.

Standard triangular flow composition, with no novelty or quality guarantee.
All Gaussian coordinates are retained; fixed residual packing is a permutation.
The coarse decoder uses one identically zero compatibility context channel, whose
weights remain explicitly counted. There are no FM parameters, ODE solvers,
sigmoid/clipping operations, or random draws in this component's transformations.
"""
from __future__ import annotations
from itertools import chain
import math
import torch
from torch import nn
from .multiscale_flow import CouplingStack, MultiscaleSplineFlow
from .learned_latent_flow_matching import pack_residuals, unpack_residuals
from .global_conditional_spline import GlobalConditionalSplineDecoder


class GlobalInnovationFlow(nn.Module):
    """z=(zc,zr) -> c=T_C(zc), r=T_R(zr;c) -> logits=A^-1(c,r).

    ``encode`` and ``decode`` are differentiable and return per-example log
    absolute determinants. ``log_prob`` is the complete density on logits,
    including both learned priors and all retained analysis transforms. An outer
    pixel/logit Jacobian, if desired, is the caller's separately specified chart.
    """
    def __init__(self, channels=3, size=32, levels=2, pre_layers=2,
                 analysis_coarse_layers=6, analysis_detail_layers=4,
                 coarse_layers=4, residual_layers=4, width=32, bins=8,
                 attention_heads=4):
        super().__init__()
        vals=(channels,size,levels,pre_layers,analysis_coarse_layers,
              analysis_detail_layers,coarse_layers,residual_layers,width,bins,attention_heads)
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in vals):
            raise ValueError('configuration must contain positive integers')
        if size % 2**levels or (size//2**levels)%2 or size//2**levels<2:
            raise ValueError('analysis requires an even coarse grid of size at least two')
        self.channels,self.size,self.levels=channels,size,levels
        self.coarse_size=size//2**levels
        self.dimension=channels*size*size
        self.latent_dimension=channels*self.coarse_size**2
        self.residual_dimension=self.dimension-self.latent_dimension
        self.packed_residual_channels=channels*(4**levels-1)
        # Reuse the exact retained analysis modules, without constructing FM heads.
        self.pre_analysis=CouplingStack(channels,0,pre_layers,width,bins)
        self.analysis=MultiscaleSplineFlow(channels,size,levels,analysis_coarse_layers,
                                           analysis_detail_layers,width,bins,unit_interval=False)
        self.coarse_decoder=GlobalConditionalSplineDecoder(channels,1,self.coarse_size,
            layers=coarse_layers,width=width,bins=bins,attention_heads=attention_heads)
        self.residual_decoder=GlobalConditionalSplineDecoder(self.packed_residual_channels,
            channels,self.coarse_size,layers=residual_layers,width=width,bins=bins,
            attention_heads=attention_heads)
        self.register_buffer('_analysis_frozen',torch.tensor(False))
        self.register_load_state_dict_post_hook(self._restore_analysis)

    def _analysis_parameters(self):
        return chain(self.pre_analysis.parameters(),self.analysis.parameters())

    def _restore_analysis(self,module,incompatible_keys):
        frozen=bool(self._analysis_frozen)
        for p in self._analysis_parameters():
            p.requires_grad_(not frozen)
            if frozen:p.grad=None
        self.pre_analysis.train(self.training and not frozen)
        self.analysis.train(self.training and not frozen)

    def freeze_analysis(self):
        self._analysis_frozen.fill_(True)
        self._restore_analysis(self,None)

    def train(self,mode=True):
        super().train(mode)
        if bool(self._analysis_frozen):
            self.pre_analysis.eval();self.analysis.eval()
        return self

    def _validate(self,value,source=False):
        shape=(self.dimension,) if source else (self.channels,self.size,self.size)
        p=next(self.parameters())
        if value.ndim!=len(shape)+1 or len(value)<1 or value.shape[1:]!=shape:
            raise ValueError('full configured batch shape required')
        if value.dtype not in (torch.float32,torch.float64) or value.dtype!=p.dtype or value.device!=p.device or not bool(torch.isfinite(value).all()):
            raise ValueError('finite model-matched float input required')

    def _split(self,code):
        blocks=[v.reshape(len(code),*shape) for v,shape in zip(
            code.split([math.prod(s) for s in self.analysis.block_shapes],dim=1),self.analysis.block_shapes)]
        return blocks[0],pack_residuals(blocks[1:])

    def _join(self,coarse,residual):
        blocks=unpack_residuals(residual,self.channels,self.levels)
        return torch.cat([coarse.flatten(1)]+[b.flatten(1) for b in blocks],dim=1)

    def _zero(self,coarse):
        return coarse.new_zeros((len(coarse),1,self.coarse_size,self.coarse_size))

    def encode(self,logits):
        self._validate(logits)
        mixed,pre_ld=self.pre_analysis(logits)
        code,analysis_ld=self.analysis.encode(mixed)
        coarse,residual=self._split(code)
        zr,residual_ld=self.residual_decoder.encode(residual,coarse)
        zc,coarse_ld=self.coarse_decoder.encode(coarse,self._zero(coarse))
        z=torch.cat((zc.flatten(1),zr.flatten(1)),dim=1)
        return z,pre_ld+analysis_ld+residual_ld+coarse_ld

    def decode(self,source):
        self._validate(source,source=True)
        zc=source[:,:self.latent_dimension].reshape(-1,self.channels,self.coarse_size,self.coarse_size)
        zr=source[:,self.latent_dimension:].reshape(-1,self.packed_residual_channels,self.coarse_size,self.coarse_size)
        coarse,coarse_ld=self.coarse_decoder.decode(zc,self._zero(zc))
        residual,residual_ld=self.residual_decoder.decode(zr,coarse)
        mixed,analysis_ld=self.analysis.decode(self._join(coarse,residual))
        logits,pre_ld=self.pre_analysis(mixed,inverse=True)
        return logits,coarse_ld+residual_ld+analysis_ld+pre_ld

    def log_prob(self,logits):
        source,ld=self.encode(logits)
        return -.5*(source.square()+math.log(2*math.pi)).sum(1)+ld

    def sample_from_gaussian(self,source):
        return self.decode(source)[0]

    @property
    def parameter_counts(self):
        count=lambda values:sum(p.numel() for p in values)
        a=count(self._analysis_parameters());c=count(self.coarse_decoder.parameters());r=count(self.residual_decoder.parameters())
        zero=sum(layer.conditioner.input.out_channels*9 for layer in self.coarse_decoder.layers)
        return {'analysis':a,'coarse':c,'residual':r,'total':a+c+r,
                'coarse_zero_context_input_weights':zero}


__all__=['GlobalInnovationFlow']
