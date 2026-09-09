"""Independent ordinary float quadrature; NOT interval certification."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
from scipy.special import ndtr,roots_hermitenorm,roots_legendre
ROOT=Path(__file__).resolve().parent
REF=ROOT/'pro8_artifacts/positive_class_reference.py'
EXPECTED='38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b'
assert hashlib.sha256(REF.read_bytes()).hexdigest()==EXPECTED
spec=importlib.util.spec_from_file_location('pinned_pro8',REF);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

def field_and_derivative(y,t,e):
    left=1-t;s2=left*left+t*t;d=np.sqrt(2*left*left+t*t);k=t/d
    u=k*y;phi=np.exp(-u*u/2)/np.sqrt(2*np.pi);den=1+e*(2*ndtr(u)-1)
    value=(2*e*left/(s2*d))*phi/den
    derivative=-k*value*(u+2*e*phi/den)
    return value,derivative

def heun_with_logjac(z,e,calls):
    y=np.broadcast_arrays(z,e)[0].copy();logj=np.zeros_like(y);h=2/calls
    for step in range(calls//2):
        w,a=field_and_derivative(y,step*h,e)
        predicted=y+h*w
        v,b=field_and_derivative(predicted,(step+1)*h,e)
        increment=h/2*(a+b*(1+h*a))
        if np.any(increment<=-1):raise ArithmeticError('nonpositive Heun step Jacobian')
        y=y+h/2*(w+v);logj+=np.log1p(increment)
    return y,logj

def nonnegative_integrand(ell):
    # exp(ell)*ell-expm1(ell); use its Taylor series near zero.
    out=np.exp(ell)*ell-np.expm1(ell)
    near=np.abs(ell)<.02
    x=ell[near];series=np.zeros_like(x)
    for n in range(12,1,-1):series=series*x+(n-1)/math.factorial(n)
    out[near]=x*x*series
    return out

def compute():
    results=[]
    reported={4:.2237247,8:.03206291,16:.002146502,32:.000137748,64:.00000870961}
    for nz,ne in ((128,32),(256,64),(512,128)):
        z,wz=roots_hermitenorm(nz);v,we=roots_legendre(ne);e=.5+.1*v
        weight=wz[:,None]/np.sqrt(2*np.pi)*(we[None,:]/2)
        table={}
        for calls in (4,8,16,32,64):
            y,logj=heun_with_logjac(z[:,None],e[None,:],calls)
            ell=-(y-z[:,None])*(y+z[:,None])/2+np.log1p(e[None,:]*(2*ndtr(y)-1))+logj
            integrand=nonnegative_integrand(ell)
            value=float(2880*np.sum(weight*integrand))
            table[calls]={'joint_forward_kl':value,'normalization_residual':float(np.sum(weight*np.expm1(ell))),
              'minimum_integrand':float(integrand.min()),'difference_from_reported':value-reported[calls],
              'max_grid_displacement':float(np.max(np.abs(y-z[:,None]))),'max_grid_abs_logjac':float(np.max(np.abs(logj)))}
        results.append({'normal_nodes':nz,'tilt_nodes':ne,'values':table})
    z=np.linspace(-9,9,81)[:,None];e=np.array([.4,.5,.6])[None,:]
    parity={}
    for calls in (4,8,16,32,64):
        y,logj=heun_with_logjac(z,e,calls);orig=reference.heun(np.broadcast_to(z,y.shape).copy(),e,calls)
        step=1e-5;hi=reference.heun(np.broadcast_to(z+step,y.shape).copy(),e,calls);lo=reference.heun(np.broadcast_to(z-step,y.shape).copy(),e,calls)
        parity[calls]={'endpoint_max_error':float(np.max(np.abs(y-orig))),
          'finite_difference_jacobian_max_error':float(np.max(np.abs(np.exp(logj)-(hi-lo)/(2*step))))}
    gamma_error=math.exp(-32*192/72);rho=1-9/16*(1-math.exp(-2880*.1**2*.5**2/(24*1.6)));index_error=15*rho**256
    report={'interval_certified':False,'purpose':'independent forward-KL quadrature only; no quality eligibility or runtime measurement',
     'reference_sha256':EXPECTED,'resolutions':results,'pinned_reference_parity':parity,
     'wrong_learning_contribution_bound':17280*index_error+17408*gamma_error,
     'uniform_scalar_wrong_model_kl_bound':1.7*math.sqrt(2/math.pi)+1.7**2/2+math.log(1.6)+1.1,
     'numpy':np.__version__}
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new JSON path; refuses overwrite')
    args=parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    report=compute()
    text=json.dumps(report,indent=2)+'\n'
    if args.output is not None:
        with args.output.open('x') as handle:handle.write(text)
    print(text,end='')
