"""Deterministic RGB codec and ordinary Gaussian straight-path FM prototypes.

No VAE, extra decoder noise, likelihood claim, pretrained weights or data loader.
Default codec maps 3x32x32 RGB in [-1,1] to 16x8x8. The lossy latent model
uses 1024 Gaussian coordinates; the pixel model uses all 3072. This difference
is explicit and is not full-dimensional equivalence. All codec work is charged
by a future experiment. Components alone establish no quality/competence claim.

SSIM uses an 11x11 normalized Gaussian (sigma1.5), valid convolution, population
moments, C1=.01^2/C2=.03^2 on RGB mapped to [0,1]. Thus no artificial image
padding enters SSIM. Time embedding has 32 sine and 32 cosine channels with
frequencies exp(-log(10000)*j/31), j=0..31, arguments t*frequency (no2pi).
Straight paths and conditional squared velocity regression follow Lipman et al.,
Flow Matching for Generative Modeling, https://arxiv.org/abs/2210.02747.
Heun uses32 steps/64 field calls by default, including both endpoint calls.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive(value, name):
    if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')


def _finite(*values):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError('nonfinite numerical value')


def _tensor(x, shape, module):
    p=next(module.parameters())
    if not isinstance(x,torch.Tensor) or x.ndim!=len(shape)+1 or x.shape[0]<1 or tuple(x.shape[1:])!=tuple(shape):
        raise ValueError('tensor must have a nonempty batch and configured shape')
    if x.dtype not in (torch.float32,torch.float64) or x.dtype!=p.dtype or x.device!=p.device:
        raise ValueError('model and input must share float32/float64 dtype and device')
    _finite(x)


class _CodecResidual(nn.Module):
    def __init__(self,width,groups):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(groups,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1,bias=True),
            nn.GroupNorm(groups,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1,bias=True))
    def forward(self,x):
        y=x+self.net(x);_finite(y);return y


class RGBCodec(nn.Module):
    """Deterministic codec. Input bounds are checked; decoder uses a declared tanh."""
    def __init__(self,size=32,latent_channels=16,width=64,groups=32,blocks=2):
        super().__init__()
        for k,v in locals().copy().items():
            if k in ('size','latent_channels','width','groups','blocks'):_positive(v,k)
        if size%4 or size<12 or width%groups:raise ValueError('size must be >=12 and divisible by4; groups must divide width')
        self.size,self.latent_channels=size,latent_channels;self.latent_size=size//4
        def residual(w):return [_CodecResidual(w,groups) for _ in range(blocks)]
        self.encoder=nn.Sequential(nn.Conv2d(3,width,3,padding=1,bias=True),*residual(width),
            nn.Conv2d(width,2*width,4,stride=2,padding=1,bias=True),*residual(2*width),
            nn.Conv2d(2*width,2*width,4,stride=2,padding=1,bias=True),*residual(2*width),
            nn.Conv2d(2*width,latent_channels,1,bias=True))
        self.decoder=nn.Sequential(nn.Conv2d(latent_channels,2*width,3,padding=1,bias=True),*residual(2*width),
            nn.ConvTranspose2d(2*width,2*width,4,stride=2,padding=1,bias=True),*residual(2*width),
            nn.ConvTranspose2d(2*width,width,4,stride=2,padding=1,bias=True),*residual(width),
            nn.Conv2d(width,3,3,padding=1,bias=True))
        grid=torch.arange(11,dtype=torch.float64)-5
        gaussian=torch.exp(-grid.square()/(2*1.5**2));gaussian=gaussian/gaussian.sum()
        self.register_buffer('ssim_window',(gaussian[:,None]*gaussian[None,:])[None,None].repeat(3,1,1,1).float())
    @property
    def latent_shape(self):return (self.latent_channels,self.latent_size,self.latent_size)
    def encode(self,x):
        _tensor(x,(3,self.size,self.size),self)
        if bool((x.abs()>1).any()):raise ValueError('RGB codec input must lie in [-1,1]')
        z=self.encoder(x);_finite(z);return z
    def decode(self,z):
        _tensor(z,self.latent_shape,self)
        raw=self.decoder(z);_finite(raw)
        return torch.tanh(raw)
    def forward(self,x):return self.decode(self.encode(x))
    def reconstruction_loss(self,x):
        z=self.encode(x);y=self.decode(z)
        a,b=(x+1)*.5,(y+1)*.5;window=self.ssim_window.to(dtype=x.dtype)
        conv=lambda v:F.conv2d(v,window,groups=3,padding=0)
        ma,mb=conv(a),conv(b)
        va,vb=conv(a*a)-ma*ma,conv(b*b)-mb*mb;cov=conv(a*b)-ma*mb
        ssim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
        loss=(x-y).abs().mean()+.2*(1-ssim.mean())+1e-4*z.square().mean()
        _finite(ssim,loss);return loss


class _VelocityBlock(nn.Module):
    def __init__(self,width,groups,dilation):
        super().__init__()
        self.norm1=nn.GroupNorm(groups,width);self.norm2=nn.GroupNorm(groups,width)
        self.conv1=nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,bias=True)
        self.conv2=nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,bias=True)
        self.film=nn.Linear(256,2*width,bias=True)
    def forward(self,x,time):
        h=self.conv1(F.silu(self.norm1(x)));scale,shift=self.film(time).chunk(2,dim=1)
        h=self.norm2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        _finite(h,scale,shift)
        y=x+self.conv2(F.silu(h));_finite(y);return y


class ContinuousVelocity(nn.Module):
    """Convolutional FiLM velocity, without observed conditioning or hidden noise."""
    def __init__(self,channels,size,width=128,blocks=12,groups=32):
        super().__init__()
        for k,v in [('channels',channels),('size',size),('width',width),('blocks',blocks),('groups',groups)]:_positive(v,k)
        if width%groups:raise ValueError('groups must divide width')
        self.channels,self.size=channels,size
        self.register_buffer('time_frequencies',torch.exp(-math.log(10000)*torch.arange(32,dtype=torch.float64)/31).float())
        self.time_mlp=nn.Sequential(nn.Linear(64,256,bias=True),nn.SiLU(),nn.Linear(256,256,bias=True),nn.SiLU())
        self.input=nn.Conv2d(channels,width,3,padding=1,bias=True)
        self.blocks=nn.ModuleList([_VelocityBlock(width,groups,(1,2,4)[i%3]) for i in range(blocks)])
        self.output=nn.Sequential(nn.GroupNorm(groups,width),nn.SiLU(),nn.Conv2d(width,channels,3,padding=1,bias=True))
    def forward(self,x,t):
        _tensor(x,(self.channels,self.size,self.size),self)
        if not isinstance(t,torch.Tensor) or t.shape!=(len(x),) or t.dtype!=x.dtype or t.device!=x.device:raise ValueError('time must be a matching batch vector')
        _finite(t)
        if bool(((t<0)|(t>1)).any()):raise ValueError('time outside[0,1]')
        phase=t[:,None]*self.time_frequencies[None,:];embedding=self.time_mlp(torch.cat((phase.sin(),phase.cos()),dim=1));_finite(embedding)
        h=self.input(x);_finite(h)
        for block in self.blocks:h=block(h,embedding)
        y=self.output(h);_finite(y);return y


def _loss(field,target,generator=None):
    _tensor(target,(field.channels,field.size,field.size),field)
    z=torch.randn(target.shape,dtype=target.dtype,device=target.device,generator=generator)
    t=torch.rand(len(target),dtype=target.dtype,device=target.device,generator=generator)
    w=t[:,None,None,None];xt=(1-w)*z+w*target
    loss=(field(xt,t)-(target-z)).square().mean();_finite(loss);return loss


def _heun(field,z,steps):
    _positive(steps,'steps')
    _tensor(z,(field.channels,field.size,field.size),field)
    x=z;dt=1./steps
    for i in range(steps):
        t=x.new_full((len(x),),i/steps);v=field(x,t);predictor=x+dt*v;_finite(predictor)
        v1=field(predictor,x.new_full((len(x),),(i+1)/steps));x=x+(.5*dt)*(v+v1);_finite(x)
    return x


class PixelFlowMatching(nn.Module):
    """Ordinary pixel-space FM; samples are unbounded, with no output clipping."""
    def __init__(self,size=32,width=128,blocks=12,groups=32):
        super().__init__();self.size=size;self.field=ContinuousVelocity(3,size,width,blocks,groups)
        self.register_buffer('_stage',torch.tensor(0,dtype=torch.int64))
    @property
    def source_dimension(self):return 3*self.size*self.size
    def set_stage(self,stage):
        if stage not in ('flow','inference'):raise ValueError('unknown stage')
        self._stage.fill_(0 if stage=='flow' else 1)
        self.requires_grad_(stage=='flow');self.train(stage=='flow')
        if stage=='inference':
            for p in self.parameters():p.grad=None
        return self
    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs)
        if int(self._stage) not in (0,1):raise ValueError('invalid serialized stage')
        self.set_stage('flow' if int(self._stage)==0 else 'inference');return result
    def training_loss(self,x,generator=None):
        if self._stage.item()!=0:raise ValueError('flow training stage required')
        return _loss(self.field,x,generator)
    @torch.no_grad()
    def sample_from_gaussian(self,z,steps=32):
        if self._stage.item()!=1:raise ValueError('inference stage required')
        _tensor(z,(self.source_dimension,),self)
        return _heun(self.field,z.reshape(len(z),3,self.size,self.size),steps)


class RGBCodecFlowMatching(nn.Module):
    """Explicit codec -> normalization -> flow -> inference stages.

    update_normalization accepts only the caller's declared FIT images; this API
    cannot authenticate a data split. It encodes under no_grad, pooling spatial
    sites for per-channel population moments without claiming site independence.
    Welford combination is accumulated in float64 and frozen to model dtype.
    Moment buffers retain float64 across model dtype conversions. freeze_normalization rejects
    fewer than2 scalar observations/channel or any population std<1e-3.
    State-dict load restores stages and parameter freezing as well as moment state.
    Source arrays are N x source_dimension; sampling accepts no observed image.
    """
    STAGES=('codec','normalization','flow','inference')
    def __init__(self,size=32,latent_channels=16,codec_width=64,field_width=128,
                 codec_blocks=2,field_blocks=8,groups=32):
        super().__init__()
        self.codec=RGBCodec(size,latent_channels,codec_width,groups,codec_blocks)
        self.field=ContinuousVelocity(latent_channels,size//4,field_width,field_blocks,groups)
        self.register_buffer('_stage',torch.tensor(0,dtype=torch.int64))
        self.register_buffer('moment_count',torch.tensor(0,dtype=torch.int64))
        self.register_buffer('moment_mean',torch.zeros(latent_channels,dtype=torch.float64))
        self.register_buffer('moment_m2',torch.zeros(latent_channels,dtype=torch.float64))
        self.register_buffer('latent_mean',torch.zeros(latent_channels))
        self.register_buffer('latent_std',torch.ones(latent_channels))
        self.register_buffer('normalization_frozen',torch.tensor(False))
        self.set_stage('codec')
    def _apply(self,fn,recurse=True):
        # Keep Welford precision even when the surrounding module is cast to float32.
        mean,m2=self.moment_mean,self.moment_m2
        result=super()._apply(fn,recurse=recurse)
        self.moment_mean=mean.to(device=self.latent_mean.device,dtype=torch.float64)
        self.moment_m2=m2.to(device=self.latent_mean.device,dtype=torch.float64)
        return result
    @property
    def source_dimension(self):return math.prod(self.codec.latent_shape)
    def _freeze_flags(self):
        stage=self.STAGES[int(self._stage)]
        self.codec.requires_grad_(stage=='codec');self.field.requires_grad_(stage=='flow')
        self.codec.train(stage=='codec');self.field.train(stage=='flow')
        for p in self.parameters():
            if not p.requires_grad:p.grad=None
    def set_stage(self,stage):
        if stage not in self.STAGES:raise ValueError('unknown stage')
        if stage in ('flow','inference') and not bool(self.normalization_frozen):raise ValueError('frozen FIT normalization required')
        if stage=='codec' and bool(self.normalization_frozen):raise ValueError('reset normalization before changing codec')
        if stage=='normalization' and bool(self.normalization_frozen):raise ValueError('normalization already frozen')
        self._stage.fill_(self.STAGES.index(stage));self._freeze_flags();return self
    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs)
        if not 0<=int(self._stage)<len(self.STAGES):raise ValueError('invalid serialized stage')
        if bool(self.normalization_frozen):
            _finite(self.latent_mean,self.latent_std)
            if bool((self.latent_std<1e-3).any()):raise ValueError('invalid serialized normalization')
        elif int(self._stage)>=2:raise ValueError('serialized stage requires frozen normalization')
        self._freeze_flags();return result
    @torch.no_grad()
    def reset_normalization(self):
        self.moment_count.zero_();self.moment_mean.zero_();self.moment_m2.zero_()
        self.latent_mean.zero_();self.latent_std.fill_(1);self.normalization_frozen.fill_(False)
        self._stage.zero_();self._freeze_flags()
    @torch.no_grad()
    def update_normalization(self,fit_images):
        if int(self._stage)!=1 or bool(self.normalization_frozen):raise ValueError('normalization accumulation stage required')
        if self.moment_mean.dtype!=torch.float64 or self.moment_m2.dtype!=torch.float64:raise ValueError('moment accumulation requires float64 buffers')
        z=self.codec.encode(fit_images).double().permute(1,0,2,3).reshape(self.codec.latent_channels,-1)
        n=z.shape[1];mean=z.mean(1);m2=(z-mean[:,None]).square().sum(1)
        old=int(self.moment_count);total=old+n;delta=mean-self.moment_mean
        merged_mean=self.moment_mean+delta*(n/total)
        merged_m2=self.moment_m2+m2+delta.square()*(old*n/total);_finite(merged_mean,merged_m2)
        self.moment_mean.copy_(merged_mean);self.moment_m2.copy_(merged_m2);self.moment_count.fill_(total)
    @torch.no_grad()
    def freeze_normalization(self):
        if int(self._stage)!=1 or bool(self.normalization_frozen) or int(self.moment_count)<2:raise ValueError('insufficient unfrozen FIT moments')
        std=(self.moment_m2/int(self.moment_count)).sqrt();_finite(std,self.moment_mean)
        if bool((std<1e-3).any()):raise ValueError('degenerate latent std below1e-3')
        self.latent_mean.copy_(self.moment_mean);self.latent_std.copy_(std);_finite(self.latent_mean,self.latent_std)
        if bool((self.latent_std<1e-3).any()):raise ValueError('model-dtype latent std below1e-3')
        self.normalization_frozen.fill_(True);self.set_stage('flow')
    @torch.no_grad()
    def encode_normalized(self,x):
        if not bool(self.normalization_frozen):raise ValueError('frozen FIT normalization required')
        z=self.codec.encode(x)
        result=(z-self.latent_mean[None,:,None,None])/self.latent_std[None,:,None,None]
        _finite(result);return result
    def training_loss_from_latents(self,normalized_latents,generator=None):
        """Train on a caller-owned frozen FIT cache; encoder cost is separate."""
        if int(self._stage)!=2 or not bool(self.normalization_frozen):raise ValueError('normalized flow training stage required')
        _tensor(normalized_latents,self.codec.latent_shape,self.field)
        return _loss(self.field,normalized_latents.detach(),generator)
    def training_loss(self,x,generator=None):
        stage=int(self._stage)
        if stage==0:return self.codec.reconstruction_loss(x)
        if stage!=2 or not bool(self.normalization_frozen):raise ValueError('codec or normalized flow training stage required')
        return self.training_loss_from_latents(self.encode_normalized(x),generator)
    @torch.no_grad()
    def sample_from_gaussian(self,z,steps=32):
        if int(self._stage)!=3 or not bool(self.normalization_frozen):raise ValueError('normalized inference stage required')
        _tensor(z,(self.source_dimension,),self)
        y=_heun(self.field,z.reshape(len(z),*self.codec.latent_shape),steps)
        raw=y*self.latent_std[None,:,None,None]+self.latent_mean[None,:,None,None];_finite(raw)
        return self.codec.decode(raw)


__all__=['RGBCodec','ContinuousVelocity','PixelFlowMatching','RGBCodecFlowMatching']
