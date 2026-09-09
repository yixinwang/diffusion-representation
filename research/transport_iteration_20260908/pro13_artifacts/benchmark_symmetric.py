"""Stronger dense comparison: one-triangle BLAS SYRK; same observed samples.
Does not refit theta or generate/inspect any new development law.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg.blas import dsyrk
from threadpoolctl import threadpool_limits
import pro13 as p
import packed_kernel as ck
import standalone as s


def symmetric_graph(block):
    x=p._unit(block);n,d=x.shape;f=p.psi(x)
    upper=dsyrk(alpha=1/n,a=f.T,lower=0,trans=0)
    ii,jj=np.where(np.triu(np.abs(upper)>=.175,1))
    deg=np.bincount(np.r_[ii,jj],minlength=d)
    return list(zip(ii.tolist(),jj.tolist())) if np.all(deg==1) else []


def main():
    ck.load_kernel();rows=[]
    for rep in range(3):
        f=json.loads(Path(f'standalone_results/fixture_{rep}.json').read_text())
        old=json.loads(Path(f'standalone_results/positive_{rep}_record.json').read_text())
        x,hashes=s.generate(f,'positive',old['array_seed']);assert hashes==old['hashes']
        for threads in [1,4]:
            with threadpool_limits(limits=threads):
                raw={k:[] for k in ['compiled','syrk','gemm']}
                for order in [('compiled','syrk','gemm'),('syrk','gemm','compiled'),('gemm','compiled','syrk')]:
                    for method in order:
                        start=time.perf_counter();edges=[]
                        for g in range(4):
                            b=x[:2000,192+720*g:192+720*(g+1)]
                            if method=='compiled':e=ck.compiled_graph(b)[0]
                            elif method=='syrk':e=symmetric_graph(b)
                            else:e=p.dense_graph(b)[0]
                            edges.append(e)
                        raw[method].append(time.perf_counter()-start)
                        expected=json.loads(Path(f'standalone_results/positive_{rep}_packed_state.json').read_text())['pairs']
                        assert json.dumps(edges)==json.dumps(expected)
                rows.append(dict(rep=rep,available_BLAS_threads=threads,raw_seconds=raw,
                    ratio_syrk_to_compiled=float(np.median(raw['syrk'])/np.median(raw['compiled'])),
                    ratio_gemm_to_compiled=float(np.median(raw['gemm'])/np.median(raw['compiled']))))
    report=dict(scope='local fabricated positive cases only, graph including validation/feature/packing; not PSC, no GPU',
        symmetric_gram_FMAs_including_diagonal=4*720*721//2*2000,rows=rows)
    Path('symmetric_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
