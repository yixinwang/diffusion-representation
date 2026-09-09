"""Independent 80-digit residual checks, plus learned-state normalization audit.
mpmath is a high-precision cross-check, not an outward-rounded interval proof.
"""
import json,math
from pathlib import Path
import numpy as np
import mpmath as mp
import pro13 as p
mp.mp.dps=80

def reference_primitive(v):
    # Integrate each linear piece from its endpoints, independent of hardcoded H.
    a=mp.sqrt(mp.mpf(3)/2)
    knots=list(map(mp.mpf,['0','.125','.375','.625','.875','1']))
    vals=[a,a,-a,-a,a,a]
    total=mp.mpf(0)
    for j in range(5):
        width=min(max(v-knots[j],mp.mpf(0)),knots[j+1]-knots[j])
        slope=(vals[j+1]-vals[j])/(knots[j+1]-knots[j])
        total+=vals[j]*width+slope*width*width/2
    return total

def main():
    rng=np.random.default_rng(1309333)
    report={'qualification':'80-digit independent arithmetic, NOT interval certification'}
    for dtype in [np.float64,np.float32]:
        alpha=np.array([-p.KAPPA*p.A,0.,p.KAPPA*p.A]+rng.uniform(-p.KAPPA*p.A,p.KAPPA*p.A,125).tolist(),dtype=dtype)
        bounds=p.KNOTS[None,:]+alpha[:,None].astype(float)*p.PRIMITIVES[None,:]
        qq=bounds.astype(dtype)
        qs=np.concatenate([qq,np.nextafter(qq,dtype(0)),np.nextafter(qq,dtype(1))],axis=1)
        qs=np.clip(qs,0,1);aa=np.broadcast_to(alpha[:,None],qs.shape)
        out=p.icdf(qs,aa,dtype)
        worst=mp.mpf(0);worst_error_bound=mp.mpf(0)
        for q,a,v in zip(qs.ravel(),aa.ravel(),out.ravel()):
            q=mp.mpf(float(q));a=mp.mpf(float(a));v=mp.mpf(float(v))
            residual=abs(v+a*reference_primitive(v)-q)
            minimum=1-abs(a)*mp.sqrt(mp.mpf(3)/2)
            worst=max(worst,residual);worst_error_bound=max(worst_error_bound,residual/minimum)
        report[np.dtype(dtype).name]=dict(cases=qs.size,independent_mp_residual=str(worst),
            residual_over_actual_min_density=str(worst_error_bound))
    root=[]
    for path in sorted(Path('standalone_results').glob('*_packed_state.json')):
        s=json.loads(path.read_text());N=s['n_fit'];k=s['root_bins'];prob=np.asarray(s['root_probabilities'])
        counts=np.rint(prob*(N+k)-1).astype(int)
        assert np.all(counts>=0) and np.all(counts.sum(axis=1)==N)
        assert np.array_equal((counts+1)/(N+k),prob)
        root.append(dict(state=path.name,exact_integer_counts_recoverable=True,
            maximum_floating_probability_sum_error=float(np.max(abs(prob.sum(axis=1)-1)))))
    report['root_normalization']=root
    # More useful normal-tail errors separated from the deliberately extreme 1000 case.
    z=np.array([-1000.,-100.,-40.,-12.,-9.,-8.,-3.,-1.,0.,1.,3.,8.,9.,12.,40.,100.,1000.])
    pp,zz,tt=np.meshgrid([-40.,-1.,0.,1.,40.],z,[-.45,-1e-12,0,1e-12,.45],indexing='ij')
    y=p.gaussian_child_decode(pp,zz,tt);back=p.gaussian_child_encode(pp,y,tt)
    report['gaussian_roundtrip_by_abs_source_cap']={str(k):float(np.max(abs(back-zz)[abs(zz)<=k])) for k in [8,12,40,100,1000]}
    Path('precision_addendum.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
