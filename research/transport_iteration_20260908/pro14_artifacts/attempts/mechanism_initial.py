"""Standalone fabricated Pro14 check. No qalt imports, datasets, fits, or jobs.

The coupling head is the proposed small module. The surrounding scalar/SPD and
triangular flow are INDEPENDENT FABRICATED algebra, not a native implementation
or a copy/re-execution of the frozen repository. Run checks.py for the receipt.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn


def finite(*values: Tensor) -> None:
    if not all(bool(torch.isfinite(x).all()) for x in values):
        raise FloatingPointError('nonfinite input, intermediate, or output')


class ResponseHead(nn.Module):
    """Identical active parameters in innovation and prefix-only controls.

    Actual configuration: rank=16, summary_dim=16, width=32, 2,624 parameters.
    A zero output layer nests the old block in exact arithmetic. The prefix
    control uses [h,tanh(h),tanh(h)^2]; the candidate uses
    [h,tanh(u),tanh(u)^2]. No ignored or padded trainable parameter tensors.
    """
    def __init__(self, rank: int = 16, width: int = 32,
                 mode: str = 'innovation'):
        super().__init__()
        if rank < 1 or width < 1 or mode not in ('innovation', 'prefix'):
            raise ValueError('invalid rank, width or mode')
        self.rank, self.mode = rank, mode
        self.input = nn.Linear(3*rank, width)
        self.output = nn.Linear(width, 2*rank)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, h: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        if h.shape != u.shape or h.shape[-1] != self.rank:
            raise ValueError('h and u must have common (batch, rank) shape')
        t = torch.tanh(u if self.mode == 'innovation' else h)
        features = torch.cat((h, t, t.square()), -1)
        hidden = torch.tanh(self.input(features))
        out = self.output(hidden)
        finite(h, u, features, hidden, out)
        return out.chunk(2, -1)


class InnovationResponse(nn.Module):
    """F(u,w)=(u, m(h,u)+exp(s(h,u))*w), using an existing frame's rows.

    Anchors are deterministic evenly spaced active-block positions. Actual
    n=720, r=16 anchors are 0,45,...,675; all 720 inputs are preserved.
    """
    def __init__(self, dimension: int = 720, rank: int = 16,
                 width: int = 32, mode: str = 'innovation'):
        super().__init__()
        if not 0 < rank < dimension or dimension % rank:
            raise ValueError('require 0<rank<dimension, dimension divisible by rank')
        self.dimension, self.rank = dimension, rank
        a = torch.arange(rank) * (dimension // rank)
        mask = torch.ones(dimension, dtype=torch.bool)
        mask[a] = False
        self.register_buffer('anchors', a)
        self.register_buffer('followers', torch.arange(dimension)[mask])
        self.head = ResponseHead(rank, width, mode)

    def forward(self, x: Tensor, h: Tensor, frame: Tensor,
                inverse: bool = False) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError('configured full block required')
        if frame.shape != (self.dimension, self.rank):
            raise ValueError('wrong existing frame shape')
        finite(x, h, frame)
        u = x[:, self.anchors]
        a, b = self.head(h, u)
        uf = frame[self.followers]
        mean = a @ uf.T
        raw_scale = b @ uf.T
        scale = math.log(2.) * torch.tanh(raw_scale)
        factor = torch.exp(-scale if inverse else scale)
        # Finite checks include intermediates, not only selected/finished values.
        centered = x[:, self.followers] - mean if inverse else x[:, self.followers]
        transformed = centered * factor if inverse else mean + centered * factor
        finite(mean, raw_scale, scale, factor, centered, transformed)
        y = x.index_copy(1, self.followers, transformed)
        ld = (-scale if inverse else scale).sum(-1)
        finite(y, ld)
        return y, ld


def scalar(x: Tensor, raw: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
    """Independent integrated-linear-derivative algebra, [-4,4], identity tails.

    Not a byte copy or numerical certification of qalt's implementation.
    """
    finite(x, raw)
    k, bound = raw.shape[-1]+1, 4.
    dx = 2*bound/k
    inner = .1 + .9*(k-1)*torch.softmax(raw, -1)
    heights = torch.cat((torch.ones_like(inner[..., :1]), inner,
                         torch.ones_like(inner[..., :1])), -1)
    areas = .5*dx*(heights[..., :-1]+heights[..., 1:])
    knots = torch.cat((torch.zeros_like(areas[..., :1]), torch.cumsum(areas, -1)), -1)-bound
    active = (x > -bound) & (x < bound)
    t = torch.where(active, x, torch.zeros_like(x))
    ix = ((t[..., None] >= knots[..., 1:-1]).sum(-1) if inverse else
          torch.floor((t+bound)/dx).long().clamp(0, k-1))
    at = lambda a, j: a.gather(-1, j[..., None]).squeeze(-1)
    h0, h1 = at(heights, ix), at(heights, ix+1)
    y0 = at(knots, ix)
    x0 = -bound+dx*ix
    slope = (h1-h0)/dx
    if inverse:
        d = t-y0
        disc = h0.square()+2*slope*d
        if not bool((disc > 0).all()):
            raise FloatingPointError('invalid discriminant')
        off = 2*d/(h0+torch.sqrt(disc))
        y = x0+off
    else:
        off = t-x0
        y = y0+off*(h0+.5*slope*off)
    deriv = h0+slope*off
    if not bool((deriv > 0).all()):
        raise FloatingPointError('nonpositive derivative')
    ld = (-1 if inverse else 1)*torch.log(deriv)
    y, ld = torch.where(active, y, x), torch.where(active, ld, torch.zeros_like(ld))
    finite(heights, knots, off, y, ld)
    return y, ld.sum(-1)


def spd(x: Tensor, u: Tensor, alpha: Tensor, inverse: bool = False):
    a = -alpha if inverse else alpha
    y = x + ((x@u)*torch.expm1(a))@u.T
    finite(x, u, alpha, y)
    return y, a.sum(-1)


class FabricatedBlock(nn.Module):
    def __init__(self, n: int, r: int, mode='innovation'):
        super().__init__()
        self.n, self.r = n, r
        self.response = InnovationResponse(n, r, mode=mode)
        self.register_buffer('frame', torch.linalg.qr(torch.randn(n,r), mode='reduced')[0])
        self.register_buffer('head_weights', .07*torch.randn(r,n*9))
        self.register_buffer('alpha_weights', .15*torch.randn(r,r))
        with torch.no_grad():
            self.response.head.output.weight.normal_(0,.12)
            self.response.head.output.bias.normal_(0,.07)

    def base(self, x, h, inverse=False):
        raw = (h@self.head_weights).reshape(len(h),self.n,9)
        mean = 4*torch.tanh(raw[...,0]); ell = math.log(2)*torch.tanh(raw[...,1])
        alpha = math.log(2)*torch.tanh(h@self.alpha_weights)
        if inverse:
            v, a = spd((x-mean)*torch.exp(-ell),self.frame,alpha,True)
            v, b = scalar(v,raw[...,2:],True)
            return v,a+b-ell.sum(-1)
        v, a = scalar(x,raw[...,2:])
        v, b = spd(v,self.frame,alpha)
        return mean+torch.exp(ell)*v,a+b+ell.sum(-1)

    def forward(self, x, h, inverse=False):
        if inverse:
            v, a = self.base(x,h,True)
            v, b = self.response(v,h,self.frame,True)
        else:
            v, a = self.response(x,h,self.frame)
            v, b = self.base(v,h)
        return v,a+b


class FabricatedFlow(nn.Module):
    """Toy exact root + four full blocks + orthogonal analysis; no native model.

    The root and preceding generated blocks feed each conditional context. This
    is sufficient to test triangular Jacobians INCLUDING input gradients through
    the context, but does not test the native analysis/CNN/attention code.
    """
    def __init__(self, root=192, n=720, r=16, blocks=4, mode='innovation'):
        super().__init__()
        self.root, self.n, self.r, self.dimension = root,n,r,root+n*blocks
        if self.dimension % 2: raise ValueError('even total dimension required')
        self.layers = nn.ModuleList([FabricatedBlock(n,r,mode) for _ in range(blocks)])
        self.register_buffer('root_scale', torch.linspace(-.1,.1,root))
        self.register_buffer('root_shift', .1*torch.cos(torch.arange(root).float()))

    def context(self, prefix):
        m = prefix.mean(-1,keepdim=True)
        e = prefix.square().mean(-1,keepdim=True)
        freq = torch.arange(1,self.r+1,dtype=prefix.dtype,device=prefix.device)[None]
        return .3*torch.sin(m*freq)+.1*torch.tanh(e/freq)

    def rotate(self,x,inverse=False):
        x=x.reshape(len(x),-1,2);a,b=x[...,0],x[...,1]
        c,s=math.cos(.23),math.sin(.23)*(-1 if inverse else 1)
        return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(1)

    def forward(self, x, inverse=False):
        finite(x)
        if x.shape[1] != self.dimension: raise ValueError('all coordinates required')
        if inverse: x=self.rotate(x,True)
        c = x[:,:self.root]
        if inverse:
            zc=(c-self.root_shift)*torch.exp(-self.root_scale)
            out=[zc];prefix=c;ld=-self.root_scale.sum().expand(len(x))
        else:
            c=self.root_shift+torch.exp(self.root_scale)*c
            out=[c];prefix=c;ld=self.root_scale.sum().expand(len(x))
        for i,layer in enumerate(self.layers):
            current=x[:,self.root+i*self.n:self.root+(i+1)*self.n]
            value,inc=layer(current,self.context(prefix),inverse)
            out.append(value);ld=ld+inc
            prefix=torch.cat((prefix,current if inverse else value),-1)
        y=torch.cat(out,-1)
        if not inverse:y=self.rotate(y)
        finite(y,ld)
        return y,ld

    def log_prob(self,x):
        z,ld=self(x,True)
        return -.5*(z.square()+math.log(2*math.pi)).sum(-1)+ld
