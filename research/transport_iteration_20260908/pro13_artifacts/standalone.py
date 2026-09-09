"""New Pro13 fabricated development fits; NOT replication of Pro11 artifacts.
No CLI data paths or native loaders. Generator truths never enter learner calls.
Source arrays are generated in memory and hashed, not retained as payload files.
"""
from __future__ import annotations
import hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits,threadpool_info
import pro13 as p


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def fixture(seed):
    r=np.random.default_rng(seed)
    probs=.5/8+.5*r.dirichlet(np.ones(8),size=192)
    pairs=[]
    for g in range(4):
        mat=r.permutation(720).reshape(-1,2)
        pairs.append([tuple(sorted(e)) for e in mat.tolist()])
    phase=np.array([0.,.7,1.4,2.1])
    left=np.arange(8)/8; right=left+1/8
    mean_sine=np.sum(probs[0,None,:]*8*(np.cos(2*np.pi*left[None,:]+phase[:,None])-np.cos(2*np.pi*right[None,:]+phase[:,None]))/(2*np.pi),axis=1)
    return dict(root_probabilities=probs.tolist(),pairs=pairs,phase=phase.tolist(),mean_sine=mean_sine.tolist())


def theta(c,f,law):
    v=np.sin(2*np.pi*np.asarray(c)[...,None]+np.asarray(f['phase']))
    if law=='positive':
        return np.array([.4,-.4,.395,-.395])+np.array([.04,.04,.035,.035])*v
    if law=='centered_zero_mean':
        return (.5/(2*np.pi))*(v-np.asarray(f['mean_sine']))
    raise ValueError('unknown fabricated law')


def generate(f,law,seed,n=4000):
    rng=np.random.default_rng(seed)
    z=rng.standard_normal((n,3072)); zh=ahash(z); x=ndtr(z)
    rp=np.asarray(f['root_probabilities'])
    for j in range(192):
        cum=np.r_[0.,np.cumsum(rp[j])]; q=x[:,j].copy()
        cells=np.searchsorted(cum[1:-1],q,side='right')
        x[:,j]=(cells+(q-cum[cells])/rp[j,cells])/8
    th=theta(x[:,0],f,law)
    for g,e in enumerate(f['pairs']):
        ij=np.asarray(e)+192+g*720
        x[:,ij[:,1]]=p.icdf(x[:,ij[:,1]],th[:,g,None]*p.psi(x[:,ij[:,0]]))
    return x,dict(gaussian_source_sha256=zh,observed_sha256=ahash(x),shape=list(x.shape),dtype=str(x.dtype))


def evaluate(state,f,law):
    # Deterministic integration, split at all 32 fitted context cells.
    # Not interval-certified. Entropy series separately checked by quadrature.
    nodes,weights=np.polynomial.legendre.leggauss(24)
    c=((np.arange(32)[:,None]+(nodes+1)/2)/32).ravel()
    w=np.tile(weights/64,32)
    rp=np.asarray(f['root_probabilities']); w*=8*rp[0,np.minimum((c*8).astype(int),7)]
    tv=theta(c,f,law)
    th=np.asarray(state['theta']); cb=np.minimum((c*32).astype(int),31)
    entropy=360*np.sum(w[:,None]*p.copula_entropy(tv))
    cross=0.;correct=[];theta_risks=[]
    for g,edges in enumerate(state['pairs']):
        truth=set(map(tuple,f['pairs'][g])); number=sum(tuple(e) in truth for e in edges)
        correct.append(number)
        eta=th[g,cb]
        cross+=number*np.sum(w*p.copula_cross_log(tv[:,g],eta))
        cross+=(len(edges)-number)*np.sum(w*p.copula_cross_log(np.zeros_like(c),eta))
        theta_risks.append(float(np.sum(w*(tv[:,g]-eta)**2)))
    fitted=np.asarray(state['root_probabilities'])
    root=float(np.sum(rp*np.log(rp/fitted)))
    # Only assert this formula for correct recovered blocks or explicit product fallback.
    assert all(k==len(state['pairs'][g]) for g,k in enumerate(correct))
    return dict(root_KL=root,conditional_KL=float(entropy-cross),joint_KL=float(root+entropy-cross),
        ideal_product_conditional_KL=float(entropy),correct_pairs=correct,
        theta_L2_risk_by_group=theta_risks,
        random_half_mask_hidden_coordinate_psi_over_A_excess_MSE=float(np.mean(theta_risks)/3),
        true_theta_mean=(w[:,None]*tv).sum(axis=0).tolist(),
        true_theta_second_moment=(w[:,None]*tv**2).sum(axis=0).tolist(),
        qualification='deterministic quadrature/96-term entropy series, not an interval certificate')


def strip_timing(s):
    return {k:v for k,v in s.items() if k not in ['timings','graph_diagnostics','method']}


def main():
    out=Path('standalone_results');out.mkdir(exist_ok=True)
    records=[]
    for rep in range(3):
        seed=1309100+rep
        f=fixture(seed)
        (out/f'fixture_{rep}.json').write_text(json.dumps(f,indent=2)+'\n')
        for law in ['positive','centered_zero_mean']:
            x,hashes=generate(f,law,seed+10000)
            # Both calls see exactly x and public configuration, no f, law, noise, or truth matching.
            with threadpool_limits(limits=1):
                packed=p.fit(x,method='packed')
                dense=p.fit(x,method='dense')
            path=out/f'{law}_{rep}_packed_state.json'
            path.write_text(json.dumps(packed,indent=2,allow_nan=False)+'\n')
            (out/f'{law}_{rep}_dense_state.json').write_text(json.dumps(dense,indent=2,allow_nan=False)+'\n')
            parity=(strip_timing(packed)==strip_timing(dense))
            # Serialized exact-copy control: same fitted parameters and same Gaussian inputs.
            copied=json.loads(path.read_text())
            z=np.random.default_rng(seed+20000).standard_normal((64,3072))
            y=p.decode(packed,z); yc=p.decode(copied,z)
            assert np.array_equal(y,yc)
            assert np.array_equal(p.log_prob(packed,y),p.log_prob(copied,yc))
            assert np.all(np.isfinite(p.log_prob(packed,x[:8])))
            timings=[]
            for _ in range(7):
                t=time.perf_counter();p.decode(packed,z);timings.append(time.perf_counter()-t)
            record=dict(rep=rep,law=law,fixture_seed=seed,array_seed=seed+10000,hashes=hashes,
                packed=packed['timings'],dense=dense['timings'],packed_graph_diagnostics=packed['graph_diagnostics'],
                fitted_parameter_parity=parity,packed_quality=evaluate(packed,f,law),dense_quality=evaluate(dense,f,law),
                exact_copy_bitwise_equal=True,decode_batch64_raw_seconds=timings,
                decode_gaussian_sha256=ahash(z),decode_output_sha256=ahash(y),
                speed_ratio_dense_over_packed_graph=dense['timings']['graph_seconds']/packed['timings']['graph_seconds'],
                speed_ratio_dense_over_packed_fit=dense['timings']['total_fit_seconds']/packed['timings']['total_fit_seconds'])
            records.append(record)
            (out/f'{law}_{rep}_record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            print(law,rep,record['packed_quality']['joint_KL'],record['packed_quality']['correct_pairs'],
                  record['speed_ratio_dense_over_packed_graph'],flush=True)
    summary=dict(scope='new fabricated development arrays only; no PSC, native, or FM fits; not Pro11 replication',
        versions=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threadpools=threadpool_info()),
        timing_qualification='Each graph fit timed once per fixture, one BLAS thread. Not an optimized cross-hardware frontier. Full decode seven raw batch64 repetitions after one parity call. Source generation and evaluator cost excluded from training, as specified.',
        records=records)
    Path('standalone_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
