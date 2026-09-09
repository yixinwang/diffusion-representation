#!/usr/bin/env python3
"""CPU mathematical checks and synthetic observation-only fitting.

This is NOT the proposed matched-budget global-FM falsifier, a native image
experiment, or a PSC run. Every response coordinate has its own Gaussian source.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr, ndtri
from scipy.stats import binom
from numpy.polynomial.legendre import leggauss
from model import (feature,primitive,conditional_cdf,conditional_pdf,inverse_cdf,
                   grid_cdf,grid_pdf,root_inverse,root_cdf,root_logpdf,
                   FittedModel,fit,entropy_F,entropy_F_prime,pair_kl,bound)

HERE=Path(__file__).resolve().parent

def ahash(x):
    x=np.ascontiguousarray(x)
    return {'shape':list(x.shape),'dtype':str(x.dtype),
            'sha256':hashlib.sha256(memoryview(x).cast('B')).hexdigest()}

def env():
    return {'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'platform':platform.platform(),
            'openblas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (HERE/'model.py',HERE/'run_checks.py')}}

def write(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def math_checks():
    checks=[]
    def record(name,value,tol):
        value=float(value); assert value<=tol,(name,value,tol)
        checks.append({'name':name,'maximum_error_or_violation':value,'tolerance':tol,'passed':True})
    x,w=leggauss(128); x=(x+1)/2; w=w/2; f=feature(x)
    record('feature_zero_mean',abs(w@f),1e-13)
    record('feature_unit_second_moment',abs(w@(f*f)-1),1e-13)
    record('feature_zero_third_moment',abs(w@(f**3)),1e-13)
    h=f[:,None]*f[None,:]; ww=w[:,None]*w[None,:]
    q=1+.43*h
    record('copula_normalization',abs(np.sum(ww*q)-1),1e-13)
    record('uniform_marginals',np.max(np.abs(q@w-1)),1e-13)
    record('raw_covariance_blindness',abs(w@((x-.5)*f)),1e-13)
    record('gaussian_chart_covariance_blindness',abs(w@(ndtri(x)*f)),1e-13)
    theta=np.linspace(-.45,.45,31)
    errs=[abs(float(entropy_F(t))-np.sum(ww*(1+t*h)*np.log1p(t*h))) for t in theta]
    record('entropy_series_against_2d_quadrature',max(errs),1e-12)
    mx=0.; low=0.; hi=0.; B=1/(2*(1-4*.45**2))
    for t in theta:
        for e in theta:
            kl=float(pair_kl(t,e)); direct=np.sum(ww*(1+t*h)*(np.log1p(t*h)-np.log1p(e*h)))
            mx=max(mx,abs(kl-direct)); low=max(low,.5*(t-e)**2-kl); hi=max(hi,kl-B*(t-e)**2)
    record('Bregman_KL_against_quadrature',mx,2e-12)
    record('KL_quadratic_lower_bound',low,1e-13)
    record('KL_quadratic_upper_bound',hi,1e-13)
    # Four-dimensional wrong-matching check; no Monte Carlo approximation.
    y,v=leggauss(20); y=(y+1)/2; v=v/2
    z=np.stack(np.meshgrid(y,y,y,y,indexing='ij'),axis=-1).reshape(-1,4)
    wz=np.prod(np.stack(np.meshgrid(v,v,v,v,indexing='ij'),axis=-1),axis=-1).ravel()
    fx=feature(z); t=.2; e=.17
    p=(1+t*fx[:,0]*fx[:,1])*(1+t*fx[:,2]*fx[:,3])
    qwrong=(1+e*fx[:,0]*fx[:,2])*(1+e*fx[:,1]*fx[:,3])
    val=np.sum(wz*p*np.log(p/qwrong))
    analytic=2*float(entropy_F(t))-2*float(entropy_F(e)-e*entropy_F_prime(e))
    record('wrong_matching_4d_KL',abs(val-analytic),2e-10)
    rng=np.random.default_rng(1100001)
    u=rng.uniform(1e-8,1-1e-8,4096); s=rng.uniform(1e-8,1-1e-8,4096)
    t=rng.uniform(-.45,.45,4096)
    inv=inverse_cdf(s,u,t,32,True)
    record('continuous_grid_inverse_CDF',np.max(np.abs(grid_cdf(inv,u,t,32)-s)),2e-14)
    record('inverse_original_CDF_diagnostic',np.max(np.abs(conditional_cdf(inv,u,t)-s)),2e-14)
    hsize=2.**-12
    gp=grid_pdf(inv,u,t,12); op=conditional_pdf(inv,u,t)
    record('grid_log_density_uniform_bound',np.max(np.abs(np.log(op/gp)))-2*math.pi*.45*hsize/(1-2*.45),1e-13)
    mx=0.
    for n in (1,2,7,32,100):
        ks=np.arange(n+1)
        for p0 in (.0001,.01,.1,.3,.9,1.):
            pmf=binom.pmf(ks,n,p0)
            lhs=np.sum(pmf[1:]/ks[1:])
            rhs=1/((n+1)*p0)+3/((n+1)*(n+2)*p0*p0)
            mx=max(mx,lhs-rhs)
    record('binomial_inverse_count_bound',mx,1e-13)
    p=np.array([.02,.18,.31,.49]); n=41; ks=np.arange(n+1)
    risk=sum(pj*np.sum(binom.pmf(ks,n,pj)*np.log(pj*(n+len(p))/(ks+1))) for pj in p)
    record('add_one_root_KL_bound',risk-(len(p)-1)/(n+1),1e-13)
    probs=rng.dirichlet(np.ones(8),size=4)
    r=rng.uniform(1e-6,1-1e-6,(100,4)); roots=root_inverse(r,probs)
    record('root_histogram_quantile_roundtrip',np.max(np.abs(root_cdf(roots,probs)-r)),1e-13)
    # Dense Jacobian checks that every input coordinate is retained.
    pairs=[np.arange(4,12).reshape(-1,2),np.arange(12,20).reshape(-1,2)]
    model=FittedModel(probs,pairs,np.full((2,32),.31),[True,True],8,10,.35,.45,[])
    z0=rng.normal(size=(1,20)); yy=model.sample(z0,32); step=2e-6
    jac=np.empty((20,20))
    for j in range(20):
        dz=np.zeros_like(z0);dz[0,j]=step
        jac[:,j]=((model.sample(z0+dz,32)-model.sample(z0-dz,32))/(2*step))[0]
    sign,ld=np.linalg.slogdet(jac); assert sign!=0 and np.linalg.matrix_rank(jac)==20
    logphi=float(np.sum(-.5*z0*z0-.5*math.log(2*math.pi)))
    record('full_D_dense_Jacobian_density',abs(float(model.logpdf(yy,32)[0])-(logphi-ld)),2e-6)
    b=bound(); assert b['joint_KL_upper']<.62 and b['conditional_product_KL_lower']>88
    record('D3072_joint_bound_numeric',abs(b['joint_KL_upper']-.61910165657),2e-10)
    out={'status':'passed','count':len(checks),'checks':checks,'bound_D3072':b,
         'environment':env(),'interpretation':'Analytic and numerical checks; not a matched global-FM experiment.'}
    write('math_checks.json',out);print(json.dumps({'checks':len(checks),'bound':b},indent=2))

class Fixture:
    """Evaluator-side unknown-law generator, never passed into fit()."""
    def __init__(self,seed,C,G,size,offset=.35,amp=.075):
        self.seed=seed;self.C=C;self.G=G;self.size=size;self.D=C+G*size
        self.offset=offset;self.amp=amp
        rng=np.random.default_rng(seed)
        p=rng.uniform(.6,1.4,(C,8));p/=p.sum(axis=1,keepdims=True);p[0]=1/8
        self.root_prob=p
        self.pairs=[]
        for g in range(G):
            perm=rng.permutation(size)+C+g*size
            self.pairs.append(np.sort(perm.reshape(-1,2),axis=1))
        self.phase=rng.uniform(0,2*math.pi,G)
        self.sign=rng.choice([-1.,1.],G)
    def theta(self,c):
        return self.sign[None,:]*(self.offset+self.amp*np.sin(2*math.pi*np.asarray(c)[:,None]+self.phase[None,:]))
    def sample(self,z):
        u=ndtr(z)
        if np.any((u<=0)|(u>=1)):raise FloatingPointError('source CDF saturation')
        out=u.copy();out[:,:self.C]=root_inverse(u[:,:self.C],self.root_prob)
        th=self.theta(out[:,0])
        for b,pairs in enumerate(self.pairs):
            i,j=pairs.T
            out[:,j]=inverse_cdf(u[:,j],u[:,i],th[:,b,None],44,True)
        return out
    def evaluate(self,model,nodes=16):
        # Independent numerical evaluation of the ungridded fitted density.
        # Source generator used 2**44 CDF interpolation; its density discrepancy
        # is bounded separately and is negligible at the printed precision.
        K=model.bins; q,w=leggauss(nodes)
        c=((np.arange(K)[:,None]+(q[None,:]+1)/2)/K).ravel()
        wt=np.tile(w/(2*K),K)
        true=self.theta(c); est=model.context(c)
        root=float(np.sum(self.root_prob*np.log(self.root_prob/model.root_prob)))
        paired=0.;product=0.;pred=0.;overlap=[]
        for g in range(self.G):
            edges=set(map(tuple,np.sort(self.pairs[g],axis=1)))
            hit=sum(tuple(x) in edges for x in np.sort(model.pairings[g],axis=1));m=len(edges)
            overlap.append(int(hit))
            t=true[:,g];e=est[:,g]
            conditional=m*entropy_F(t)-m*entropy_F(e)-(hit*t-m*e)*entropy_F_prime(e)
            paired+=float(wt@conditional)
            product+=float(wt@(m*entropy_F(t)))
            pred+=float(wt@(hit*(t-e)**2+(m-hit)*(t*t+e*e)))
        return {'root_KL':root,'conditional_KL':paired,'joint_KL':root+paired,
                'conditional_product_floor':product,'product_same_fitted_root_KL':root+product,
                'functional_prediction_excess_sum_one_per_pair':pred,
                'masked_feature_gain_per_coordinate':(sum(float(wt@(true[:,g]**2)) for g in range(self.G))/self.G-pred/(self.G*self.size/2)),
                'correct_edges_per_group':overlap,'total_true_edges':self.G*self.size//2}


def learning_check(seed,C=192,G=4,size=720,N=4000,ns=2000):
    raw={'seed':seed,'status':'completed_standalone_synthetic_check','N_fit':N,
         'N_structure':ns,'N_context':N-ns,'C':C,'G':G,'group_size':size,
         'D':C+G*size,'environment':env(),'worlds':{},
         'not_run':['global FM training/tuning','native image loading','PSC','repository edits','matched-budget falsifier'],
         'banks':'Gaussian and observed full training banks regenerated from recorded seeds; hashes retained, bulk banks not packaged.'}
    for name,offset in (('positive',.35),('zero_mean_change',0.)):
        law=Fixture(seed,C,G,size,offset=offset)
        t0=time.perf_counter();z=np.random.default_rng(seed+10000000).standard_normal((N,law.D))
        zh=ahash(z);obs=law.sample(z);oh=ahash(obs);del z
        generation=time.perf_counter()-t0
        t0=time.perf_counter();model=fit(obs,root_dim=C,groups=G,n_structure=ns)
        fit_seconds=time.perf_counter()-t0
        result=law.evaluate(model,16); check=law.evaluate(model,24)
        discrepancy=max(abs(result[k]-check[k]) for k in ('root_KL','conditional_KL','joint_KL','conditional_product_floor','functional_prediction_excess_sum_one_per_pair'))
        assert discrepancy<1e-9
        # A genuinely fitted constant-context ablation, not a teacher parameter.
        static=FittedModel(model.root_prob.copy(),[p.copy() for p in model.pairings],
                           model.theta.copy(),list(model.structural_pass),size,ns,.35,.45,[])
        for g,pairs in enumerate(model.pairings):
            if model.structural_pass[g]:
                t=(feature(obs[ns:,pairs[:,0]])*feature(obs[ns:,pairs[:,1]])).mean()
                static.theta[g]=np.clip(t,-.45,.45)
        static_result=law.evaluate(static,16)
        # Fresh full-D random-coordinate bank, separate from fitting.
        zz=np.random.default_rng(seed+20000000).standard_normal((64,law.D))
        generated=model.sample(zz,32); encoded=model.encode_uniform(generated,32)
        roundtrip=float(np.max(np.abs(encoded-ndtr(zz))))
        assert roundtrip<2e-13 and np.isfinite(model.logpdf(generated,32)).all()
        # Independent analytic-copy implementation uses exactly the same density.
        direct=root_logpdf(generated[:,:C],model.root_prob)
        th=model.context(generated[:,0])
        for g,pairs in enumerate(model.pairings):
            direct+=np.log(grid_pdf(generated[:,pairs[:,1]],generated[:,pairs[:,0]],th[:,g,None],32)).sum(axis=1)
        copy_error=float(np.max(np.abs(direct-model.logpdf(generated,32))))
        assert copy_error==0.
        source_grid_error=(G*size/2)*2*math.pi*.45/(2.**44*(1-2*.45))
        raw['worlds'][name]={'offset':offset,'amplitude':.075,'structural_pass':model.structural_pass,
             'graph_diagnostics':model.score_diagnostics,'evaluation':result,
             'constant_context_ablation':static_result,'quadrature_refinement_max_difference':discrepancy,
             'Gaussian_source_seed':seed+10000000,'Gaussian_source':zh,'observations':oh,
             'fresh_roundtrip_source_seed':seed+20000000,'fresh_roundtrip_source':ahash(zz),
             'generated':ahash(generated),'maximum_uniform_source_roundtrip_error':roundtrip,
             'analytic_copy_logdensity_max_error':copy_error,'truth_generator_continuous_grid_logdensity_budget':source_grid_error,
             'CPU_times_seconds':{'source_and_fixture_generation':generation,'observation_only_fit':fit_seconds},
             'timing_scope':'one CPU run per law, including no global-FM control; not a comparative speed claim'}
        np.savez_compressed(HERE/f'fit_{name}_{seed}.npz',root_prob=model.root_prob,
                            theta=model.theta,pairings=np.stack(model.pairings),
                            source_check=zz,generated_check=generated,
                            evaluator_root_prob=law.root_prob,evaluator_pairings=np.stack(law.pairs),
                            evaluator_phase=law.phase,evaluator_sign=law.sign)
        del obs
        print(name,result,'seconds',fit_seconds,flush=True)
    write(f'learning_seed_{seed}.json',raw)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--math',action='store_true');ap.add_argument('--seed',type=int)
    ap.add_argument('--small',action='store_true');args=ap.parse_args()
    if args.math:math_checks()
    if args.seed is not None:
        learning_check(args.seed,C=16,size=20) if args.small else learning_check(args.seed)
    if not args.math and args.seed is None:ap.error('select --math or --seed')
