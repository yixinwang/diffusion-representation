"""Standalone local numerical checks and full D3072 output timings.
No training, native data, repository execution, PSC connection or submission.
These are new source banks, not a replay of the PSC stored observation hashes.
"""
import argparse, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr, expit, log_ndtr
import mpmath as mp
import pro10_kernels as K

SEEDS=(2026090901,2026090902,2026090903)
D,C,M=3072,192,2880

def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def dictionary():
    a=np.random.default_rng(77180).normal(size=(192,16))
    q,_=np.linalg.qr(a);return q.T

def haar_inverse(c,d):
    lh,hl,hh=np.split(d,3,axis=1)
    out=np.empty((len(c),3,2*c.shape[2],2*c.shape[3]),dtype=np.float64)
    out[:,:,0::2,0::2]=(c+lh+hl+hh)/2
    out[:,:,0::2,1::2]=(c-lh+hl-hh)/2
    out[:,:,1::2,0::2]=(c+lh-hl-hh)/2
    out[:,:,1::2,1::2]=(c-lh-hl+hh)/2
    return out

def decode(c,r):
    x=haar_inverse(c.reshape(-1,3,8,8),r[:,:576].reshape(-1,9,8,8))
    return expit(haar_inverse(x,r[:,576:].reshape(-1,9,16,16)))

def haar_forward(x):
    a,b,c,d=x[:,:,0::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,0::2],x[:,:,1::2,1::2]
    return (a+b+c+d)/2,np.concatenate(((a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),axis=1)

def encode(x):
    if np.any((x<=0)|(x>=1)):raise FloatingPointError('saturated outer sigmoid')
    c,r2=haar_forward(np.log(x)-np.log1p(-x));c,r1=haar_forward(c)
    return c.reshape(len(x),-1),np.concatenate((r1.reshape(len(x),-1),r2.reshape(len(x),-1)),axis=1)

def generate(z, O, kind, plans, nfe=None):
    if z.dtype != np.float64 or z.ndim!=2 or z.shape[1]!=D or not np.isfinite(z).all():
        raise ValueError('finite full-D float64 source required')
    quant=K.quantile_log_reference if kind in ('exact_log','heun_legacy') else K.quantile_hybrid
    c=quant(z[:,:C],.5)
    e=.5+.1*(2*ndtr(z[:,:C]@O[7])-1)
    if kind in ('exact_log','exact_hybrid'):
        r=quant(z[:,C:],e[:,None])
    elif kind=='heun_legacy':
        r=K.heun_reference(z[:,C:],e[:,None],nfe)
    else:
        r=K.heun_cached(z[:,C:],e[:,None],plans[nfe])
    out=decode(c,r)
    if not np.isfinite(out).all():raise FloatingPointError('invalid full output')
    return out

def high_precision_reference(z,e,guess):
    # Independent log-CDF root solve at 90 decimal digits, not scipy inversion.
    with mp.workdps(90):
        z=mp.mpf(float(z));e=mp.mpf(float(e));s=1 if z>0 else -1
        p=mp.erfc(abs(z)/mp.sqrt(2))/2;b=1+s*e
        q=2*p/(b+mp.sqrt(b*b-4*s*e*p));ell=mp.log(q)
        f=lambda x:mp.log(mp.erfc(-x/mp.sqrt(2))/2)-ell
        x0=mp.mpf(float(-s*guess))
        x=mp.findroot(f,(x0-mp.mpf('.01'),x0+mp.mpf('.01')),tol=mp.mpf('1e-75'))
        return float(-s*x)

def checks():
    near=[]
    for b in (-8.,0.,8.):near += [np.nextafter(b,-np.inf),b,np.nextafter(b,np.inf)]
    points=np.unique(np.r_[np.linspace(-50,50,4001),near])
    es=np.unique(np.r_[np.linspace(-.6,.6,81),-.5,.4,.5,.6])
    z,e=np.broadcast_arrays(points[:,None],es[None,:])
    q=K.quantile_hybrid(z,e);qref=K.quantile_log_reference(z,e)
    rt=K.inverse_hybrid(q,e)
    source_log=log_ndtr(-np.abs(z))
    logcdf=log_ndtr(q)+np.log1p(-e*(1-ndtr(q)))
    logsurv=log_ndtr(-q)+np.log1p(e*ndtr(q))
    actual=np.where(z<=0,logcdf,logsurv)
    result={'grid_shape':list(z.shape),'fwd_max_difference_vs_log':float(np.max(abs(q-qref))),
            'inverse_roundtrip_max':float(np.max(abs(rt-z))),
            'log_probability_identity_max':float(np.max(abs(actual-source_log))),
            'largest_downward_step':float(max(0.,-np.diff(q,axis=0).min()))}
    hp=[]
    for ev in (-.6,-.5,0.,.4,.5,.6):
        for zv in [-50.,-30.,-12.,-8.000000000000002,-8.,-7.999999999999999,-1.,-1e-15,0.,1e-15,1.,7.999999999999999,8.,8.000000000000002,12.,30.,50.]:
            a=float(K.quantile_hybrid(zv,ev));b=high_precision_reference(zv,ev,a)
            hp.append({'z':zv,'e':ev,'value':a,'mp90':b,'abs_error':abs(a-b)})
    result['high_precision_checks']=hp
    result['high_precision_max_error']=max(x['abs_error'] for x in hp)
    rng=np.random.default_rng(2026091101)
    za=rng.normal(size=(32,3072));ea=.4+.2*rng.random((32,1))
    za[0,:7]=[-50,-12,-8,0,8,12,50]
    result['heun_parity']={};result['jacobian_finite_difference']={}
    for n in K.NFES:
        plan=K.HeunPlan.build(n)
        a=K.heun_reference(za,ea,n);b=K.heun_cached(za,ea,plan)
        result['heun_parity'][str(n)]=float(np.max(abs(a-b)))
        zz=np.linspace(-5,5,1001);step=1e-5
        yy,j=K.heun_with_jacobian(zz,.55,plan)
        fd=(K.heun_cached(zz+step,.55,plan)-K.heun_cached(zz-step,.55,plan))/(2*step)
        result['jacobian_finite_difference'][str(n)]=float(np.max(abs(fd-j)))
    O=dictionary();plans={n:K.HeunPlan.build(n) for n in K.NFES}
    result['full_pipeline']=[]
    for seed in SEEDS:
        r=np.random.default_rng(seed);z0=r.normal(size=(32,D))
        x=generate(z0,O,'exact_hybrid',plans);c,res=encode(x)
        u=K.inverse_hybrid(c,.5);ee=.5+.1*(2*ndtr(u@O[7])-1)
        v=K.inverse_hybrid(res,ee[:,None]);back=np.concatenate((u,v),axis=1)
        old=generate(z0,O,'exact_log',plans)
        result['full_pipeline'].append({'seed':seed,'roundtrip':float(np.max(abs(back-z0))),
            'max_output_difference':float(np.max(abs(x-old))),
            'copy_bitwise':bool(np.array_equal(x,generate(z0,O,'exact_hybrid',plans)))})
    result['gaussian_control_heun_identity']=all(np.array_equal(K.heun_cached(za,0.,plans[n]),za) for n in K.NFES)
    failures=[]
    for zbad,ebad in ((np.nan,.5),(np.inf,.5),(0.,.61),(0.,np.nan),(1e308,.5)):
        try:
            with np.errstate(all='ignore'):K.quantile_hybrid(zbad,ebad)
        except (ValueError,FloatingPointError):failures.append(True)
        else:failures.append(False)
    result['invalid_or_unrepresentable_inputs_rejected']=all(failures)
    gates=[result['high_precision_max_error']<=2e-13,result['inverse_roundtrip_max']<=1e-12,
        result['log_probability_identity_max']<=1e-11,result['largest_downward_step']<=1e-13,
        max(result['heun_parity'].values())<=2e-13,
        max(result['jacobian_finite_difference'].values())<=1e-8,
        all(c['roundtrip']<=1e-9 and c['copy_bitwise'] for c in result['full_pipeline']),
        result['gaussian_control_heun_identity'],result['invalid_or_unrepresentable_inputs_rejected']]
    result['all_declared_numerical_gates']=all(gates)
    return result

def benchmark(repeats=9):
    O=dictionary();t=time.perf_counter();plans={n:K.HeunPlan.build(n) for n in K.NFES};setup=time.perf_counter()-t
    labels=['exact_log','exact_hybrid']+[f'legacy_{n}' for n in K.NFES]+[f'cached_{n}' for n in K.NFES]
    records=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for batch in (1,64):
            z=rng.normal(size=(batch,D));times={x:[] for x in labels}
            def sample(label):
                if label.startswith('legacy_'):return generate(z,O,'heun_legacy',plans,int(label.split('_')[1]))
                if label.startswith('cached_'):return generate(z,O,'heun_cached',plans,int(label.split('_')[1]))
                return generate(z,O,label,plans)
            for label in labels:
                for _ in range(3):sample(label)
            order=[]
            for _ in range(repeats):
                sequence=list(rng.permutation(labels));order.append(sequence)
                for label in sequence:
                    t=time.perf_counter_ns();out=sample(label);elapsed=(time.perf_counter_ns()-t)*1e-9
                    times[label].append(elapsed)
                    with open('local_timing_live.jsonl','a') as live:
                        live.write(json.dumps({'seed':seed,'batch':batch,'label':label,'elapsed':elapsed})+'\n')
                        live.flush()
            records.append({'seed':seed,'batch':batch,'source_sha256':array_hash(z),'raw_seconds':times,
                'order':order,'median_ms':{k:float(np.median(v)*1000) for k,v in times.items()},
                'p95_ms':{k:float(np.quantile(v,.95)*1000) for k,v in times.items()}})
    return {'scope':'LOCAL standalone CPU, one process; NOT PSC or replicated process timing. Fixed gamma=.5,j=7; NO fitting performed.',
        'source_stream':'new local Gaussian banks; not byte-identical PSC replay',
        'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
        'threads':{x:os.environ.get(x) for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'dimensions':[D,C,M],'dictionary_sha256':array_hash(O),'setup_all_five_plans_seconds':setup,
        'measurement':'full root, summary, residual, Haar and sigmoid; RNG excluded for all; public numerical checks charged',
        'memory':'not independently measured here; required for proposed PSC protocol',
        'warmups_per_arm':3,'repeats':repeats,'records':records}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['check','bench'],required=True);a=p.parse_args()
    result=checks() if a.mode=='check' else benchmark()
    path=Path('local_numerical.json' if a.mode=='check' else 'local_timing.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.mode=='check':
        print(json.dumps({k:v for k,v in result.items() if k!='high_precision_checks'},indent=2))
    else:
        for r in result['records']:print(r['seed'],r['batch'],r['median_ms'])
    if a.mode=='check' and not result['all_declared_numerical_gates']:raise SystemExit('numerical gate failure; outputs preserved')
