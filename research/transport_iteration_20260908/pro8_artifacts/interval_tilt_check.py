"""Outward-rounded scalar normal-CDF-tilt Heun audit, no data or repository access.

Numerical enclosure assumptions: IEEE-754 binary64 correctly rounded +,-,*,/,
sqrt, and numpy.nextafter. Phi/pdf are enclosed using finite Taylor polynomials
and explicit remainder bounds on |x| <= 1, NOT scipy special-function accuracy.
This is a checkable interval certificate under those arithmetic assumptions,
not a formal proof-assistant certificate.
"""
import json, math, time
import numpy as np

def down(x): return np.nextafter(np.asarray(x,dtype=np.float64),-np.inf)
def up(x): return np.nextafter(np.asarray(x,dtype=np.float64),np.inf)
class I:
    def __init__(self,lo,hi=None):
        self.lo=np.asarray(lo,dtype=np.float64)
        self.hi=np.asarray(lo if hi is None else hi,dtype=np.float64)
    @staticmethod
    def cast(x): return x if isinstance(x,I) else I(x)
    def __add__(self,o):
        o=I.cast(o);return I(down(self.lo+o.lo),up(self.hi+o.hi))
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,o):return self+-I.cast(o)
    def __rsub__(self,o):return I.cast(o)+-self
    def __mul__(self,o):
        o=I.cast(o)
        v=np.stack(np.broadcast_arrays(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi))
        return I(down(v.min(axis=0)),up(v.max(axis=0)))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=I.cast(o)
        if np.any((o.lo<=0)&(o.hi>=0)):raise ArithmeticError('interval division by zero')
        return self*I(down(1/o.hi),up(1/o.lo))
    def __rtruediv__(self,o):return I.cast(o)/self
    def sqrt(self):
        if np.any(self.lo<0):raise ArithmeticError('sqrt negative interval')
        return I(down(np.sqrt(self.lo)),up(np.sqrt(self.hi)))
    def square(self):
        lo=np.minimum(self.lo*self.lo,self.hi*self.hi)
        lo=np.where((self.lo<=0)&(self.hi>=0),0.,lo)
        return I(np.maximum(0.,down(lo)),up(np.maximum(self.lo*self.lo,self.hi*self.hi)))

# These decimal brackets contain 1/sqrt(2*pi); nextafter also encloses decimal
# -> binary64 conversion. The decimal inequality is independently reproducible.
C=I(down(float('0.39894228040143267793')),up(float('0.39894228040143267795')))
def ratio(a,b):return I(a)/I(b)

def pdf_cdf(x):
    if np.any(x.lo < -1) or np.any(x.hi>1):raise ArithmeticError('Taylor domain escaped')
    v=-x.square()/2
    # 19 terms (j=0..18); all integer denominators <2**53? 18! is not,
    # so bracket the conversion explicitly before taking reciprocals.
    def coef(j,integrated):
        den=math.factorial(j)*(2*j+1 if integrated else 1)
        return 1/I(down(float(den)),up(float(den)))
    ep=coef(18,False); ip=coef(18,True)
    for j in range(17,-1,-1):
        ep=coef(j,False)+v*ep;ip=coef(j,True)+v*ip
    # Alternating-series remainders for |v|<=1/2 and |x|<=1 are <=1e-22.
    # Actual next terms are <1.57e-23 (exp) and <4.1e-25 (integral).
    rem=I(-1e-22,1e-22)
    return C*(ep+rem), .5+C*x*(ip+rem)

def field(y,t,e):
    a=1-t;s2=a*a+t*t;d=I(2*a*a+t*t).sqrt();k=t/d
    phi,Phi=pdf_cdf(k*y)
    return (2*e*a/(s2*d))*phi/(1+e*(2*Phi-1))

def check(nfe,cells=8192):
    if nfe not in (4,8,16,32,64):raise ValueError('audited NFE grid only')
    # Dyadic outer interval contains [0.4,0.6]; clamp endpoints with outward
    # decimal brackets. Grid values are just boundaries, so rounding cannot
    # create gaps between adjacent intervals.
    edges=np.linspace(down(.4),up(.6),cells+1)
    e=I(edges[:-1],edges[1:]);y=I(np.zeros(cells));h=2/nfe
    for j in range(nfe//2):
        v=field(y,j*h,e)
        y=y+h/2*(v+field(y+h*v,(j+1)*h,e))
    _,u=pdf_cdf(y)
    p=u+e*(u.square()-u)
    lo=float(np.min(down(.5-p.hi))); hi=float(np.max(up(.5-p.lo)))
    return {'nfe':nfe,'parameter_cells':cells,'cdf_defect_lower':lo,
            'cdf_defect_upper':hi,'positive_everywhere':bool(lo>0),
            'joint_KL_lower_D3072_residual2880':float((I(2*2880)*I(lo).square()).lo) if lo>0 else 0.,
            'arithmetic_scope':'outward binary64 interval audit, explicit Taylor remainders; not proof-assistant verified'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=8192);p.add_argument('--output');p.add_argument('--nfe',type=int,nargs='*',default=[4,8,16,32,64]);a=p.parse_args()
    t=time.perf_counter(); out={'checks':[check(n,a.cells) for n in a.nfe]};out['seconds']=time.perf_counter()-t
    text=json.dumps(out,indent=2);print(text)
    if a.output:open(a.output,'w').write(text+'\n')
