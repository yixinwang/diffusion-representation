"""Local deterministic/fabricated checks, no fits and no native inputs."""
import json, math, platform, sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri, ndtr
import scipy
import pro13 as p


def main():
    rng=np.random.default_rng(1309001)
    report={'scope':'standalone fabricated/deterministic; not PSC or interval certification',
            'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}}
    report['exact_moments']={'E_psi':'0','E_psi_squared':str(F(3,2)*F(2,3)),
        'E_psi_cubed':'0','E_psi_fourth':str(F(3,2)**2*F(3,5)),
        'squared_L1_psi':str(F(3,2)*F(3,4)**2)}
    integrals=[quad(lambda x: float(p.psi(x))**j,0,1,points=p.KNOTS[1:-1],epsabs=2e-13)[0] for j in range(1,5)]
    assert np.allclose(integrals,[0,1,0,1.35],atol=1e-12)
    report['quadrature_moments']=integrals
    cov_u=quad(lambda u: (u-.5)*float(p.psi(u)),0,1,points=p.KNOTS[1:-1],epsabs=2e-13)[0]
    cov_z=quad(lambda u: ndtri(u)*float(p.psi(u)),0,1,points=p.KNOTS[1:-1],epsabs=2e-13)[0]
    report['covariance_factor_integrals']={'ordinary':cov_u,'gaussianized':cov_z}
    assert abs(cov_u)<1e-12 and abs(cov_z)<1e-12
    report['C1_claim_counterexample']={'psi_left_derivative_at_1_8':0.,'psi_right_derivative_at_1_8':-8*p.A,
        'density_C1':False,'conditional_CDF_C1_in_response':True,'joint_transport_globally_C1':False}
    checks={}
    for dtype in [np.float64,np.float32]:
        alpha=np.concatenate([rng.uniform(-p.KAPPA*p.A,p.KAPPA*p.A,100000),np.array([-p.KAPPA*p.A,0,p.KAPPA*p.A])])
        q=rng.uniform(0,1,len(alpha))
        knots=p.KNOTS[None,:]+alpha[-3:,None]*p.PRIMITIVES[None,:]
        bb=np.concatenate([knots.ravel(),np.nextafter(knots.astype(dtype),dtype(0)).ravel(),np.nextafter(knots.astype(dtype),dtype(1)).ravel()])
        aa=np.tile(np.repeat(alpha[-3:],6),3)
        q=np.concatenate([q,np.clip(bb,0,1)]).astype(dtype); alpha=np.concatenate([alpha,aa]).astype(dtype)
        v,di=p.icdf(q,alpha,dtype,diagnostics=True)
        # Evaluate rounded output in float64 independently of the low precision arithmetic.
        err=np.max(abs(p.cdf(v.astype(float),alpha.astype(float))-q.astype(float)))
        di['float64_recomputed_residual']=float(err)
        di['unit_interval_error_bound_from_residual']=float(err/.325)
        di['cases']=len(q)
        assert err < (2e-14 if dtype==np.float64 else 2e-6)
        checks[np.dtype(dtype).name]=di
    report['inverse_checks']=checks
    z=np.array([-1000.,-100.,-40.,-12.,-9.,-8.,-3.,-1.,0.,1.,3.,8.,9.,12.,40.,100.,1000.])
    par,zz,tt=np.meshgrid(np.array([-40.,-1.,0.,1.,40.]),z,np.array([-.45,-1e-12,0,1e-12,.45]),indexing='ij')
    out=p.gaussian_child_decode(par,zz,tt)
    back=p.gaussian_child_encode(par,out,tt)
    assert np.all(np.isfinite(out))
    te=float(np.max(abs(back-zz)))
    assert te<2e-9
    report['log_tail_checks']={'cases':out.size,'max_abs_gaussian_roundtrip_error':te,
        'largest_absolute_source':1000.,'naive_ndtr_9':float(ndtr(9.)),
        'naive_ndtri_ndtr_9_is_infinite':bool(np.isinf(ndtri(ndtr(9.))))}
    # Exact bit-count parity against direct sign correlation, including non-byte length.
    x=rng.uniform(size=(137,18)); s=np.where((x<=.25)|(x>=.75),1,-1)
    packed=np.packbits(s>0,axis=0,bitorder='little').T.copy()
    codes=[int.from_bytes(row.tobytes(),'little') for row in packed]
    ints=np.array([[137-2*(ci^cj).bit_count() for cj in codes] for ci in codes])
    assert np.array_equal(ints,s.T@s)
    report['packed_integer_parity_max_error']=int(np.max(abs(ints-s.T@s)))
    # Independent segmented quadrature versus entropy series and KL sandwich.
    gx,gw=np.polynomial.legendre.leggauss(32)
    basis=np.r_[-p.A,gx*p.A,p.A]
    weights=np.r_[.25,gw/4,.25]
    prod=basis[:,None]*basis[None,:]; ww=weights[:,None]*weights[None,:]
    diffs=[]; minratio=float('inf'); maxratio=0.
    for theta in np.linspace(-.45,.45,19):
        for eta in np.linspace(-.45,.45,17):
            kl=float(np.sum(ww*(1+theta*prod)*(np.log1p(theta*prod)-np.log1p(eta*prod))))
            series=float(p.copula_entropy(theta)-p.copula_cross_log(theta,eta))
            diffs.append(abs(kl-series))
            if abs(theta-eta)>1e-8:
                ratio=kl/(theta-eta)**2; minratio=min(minratio,ratio); maxratio=max(maxratio,ratio)
                assert .5-1e-9 <=ratio<=p.bounds()['B_kappa']+1e-9
    report['KL_checks']={'quadrature_series_max_error':max(diffs),'minimum_ratio':minratio,'maximum_ratio':maxratio,'cases':323}
    report['bounds']=p.bounds()
    assert report['bounds']['joint_KL'] < .57
    assert report['bounds']['fixed_chart_product_floor']==88.19999999999999 or abs(report['bounds']['fixed_chart_product_floor']-88.2)<1e-12
    Path('math_checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__': main()
