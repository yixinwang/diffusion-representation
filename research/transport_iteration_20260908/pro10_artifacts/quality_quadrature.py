"""Local deterministic quadrature diagnostic, NOT an outward interval certificate.

KL(P_e || Law(H_N(Z))) = E_Z[g(ell)], where ell=log(p_e(H) H'/phi(Z)),
g(x)=exp(x)*x-expm1(x). A proved analytic tail bound is returned separately;
resolution agreement of the compact quadrature is not a proof of its error.
"""
import json, math
import numpy as np
from scipy.special import roots_legendre, ndtr
from pro10_kernels import HeunPlan, heun_with_jacobian, NFES

def positive_integrand(ell):
    out=np.empty_like(ell); small=np.abs(ell)<.01
    x=ell[small]
    # Sum (n-1)/n! x^n, n=2..12; discarded term is negligible on |x|<.01.
    p=np.full_like(x,11/math.factorial(12))
    for n in range(11,1,-1):p=(n-1)/math.factorial(n)+x*p
    out[small]=x*x*p
    x=ell[~small];out[~small]=np.exp(x)*x-np.expm1(x)
    return out

def tail_upper(cut=12.):
    # Uniform over e in [.4,.6], N in NFES. |H(z)-z|<=1.7,
    # |log H'(z)|<=1.1, hence |ell(z)|<=1.7|z|+3.5.
    B,A=1.7,3.5; x=cut-B
    ph=math.exp(-x*x/2)/math.sqrt(2*math.pi)
    return 2880*(2*math.exp(A+B*B/2)*(B*ph+(B*B+A+1)*ndtr(-x))+2*ndtr(-cut))

def quadrature(nfe,nz=512,ne=32,cut=12.):
    uz,wz=roots_legendre(nz); ue,we=roots_legendre(ne)
    z=np.broadcast_to((cut*uz)[None,:],(ne,nz)).copy()
    e=(.5+.1*ue)[:,None]
    y,J=heun_with_jacobian(z,e,HeunPlan.build(nfe))
    ell=-(y-z)*(y+z)/2+np.log1p(e*(2*ndtr(y)-1))+np.log(J)
    v=positive_integrand(ell)
    phi=np.exp(-z*z/2)/math.sqrt(2*math.pi)
    return float(2880*np.sum(.5*we[:,None]*cut*wz[None,:]*phi*v))

def run():
    results={str(n):{str(nz):quadrature(n,nz) for nz in (256,512,1024)} for n in NFES}
    return {'scope':'LOCAL floating quadrature; NOT certified full-KL bounds',
            'direction':'KL(true conditional law || finite-Heun law), correct recovered catalog',
            'units':'nats/full D3072 array','uniform_e':[.4,.6],
            'cut':12,'analytic_tail_upper_full_array':tail_upper(), 'values':results}
if __name__=='__main__':
    r=run(); text=json.dumps(r,indent=2); print(text)
    open('local_quality.json','w').write(text+'\n')
