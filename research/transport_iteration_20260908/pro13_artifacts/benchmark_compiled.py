"""Same algorithm, compiled implementation; retain Python-negative timings.
Regenerates ONLY our own fabricated development arrays and verifies their
previously recorded byte hashes before using them. Never Pro11/native arrays.
"""
import json,time,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import pro13 as p
import packed_kernel as ck
import standalone as s


def main():
    # Compile outside warm timing but report the cost explicitly.
    begin=time.perf_counter();ck.load_kernel();first_load=time.perf_counter()-begin
    Path('compiled_build_record.json').write_text(json.dumps(dict(ck.BUILD_RECORD,first_load_seconds=first_load),indent=2)+'\n')
    # Independent parity including sizes not divisible by 8, 30, or 64.
    rng=np.random.default_rng(1309222)
    tests=[]
    for n in [1,7,30,63,64,65,137,2000]:
        a=rng.uniform(size=(n,18)); e,di=p.packed_graph(a);ec,dic=ck.compiled_graph(a)
        assert e==ec
        assert all(di[k]==dic[k] for k in ['accepted','threshold_edges','zero_degree','multiple_degree'])
        tests.append(n)
    rows=[]
    for rep in range(3):
        f=json.loads(Path(f'standalone_results/fixture_{rep}.json').read_text())
        for law in ['positive','centered_zero_mean']:
            old=json.loads(Path(f'standalone_results/{law}_{rep}_record.json').read_text())
            x,hashes=s.generate(f,law,old['array_seed'])
            assert hashes==old['hashes']
            oldstate=json.loads(Path(f'standalone_results/{law}_{rep}_packed_state.json').read_text())
            with threadpool_limits(limits=1):
                fit=p.fit(x,method='packed_c')
                assert json.dumps(s.strip_timing(fit),sort_keys=True)==json.dumps(s.strip_timing(oldstate),sort_keys=True)
                times={'packed_c':[],'dense':[],'packed_python':[]}
                # Alternating fixed method order controls one-sided warmup/order bias.
                orders=[['packed_c','dense','packed_python'],['dense','packed_python','packed_c'],['packed_python','packed_c','dense']]
                for order in orders:
                    for method in order:
                        t=time.perf_counter()
                        for g in range(4):
                            b=x[:2000,192+g*720:192+(g+1)*720]
                            if method=='packed_c':ck.compiled_graph(b)
                            elif method=='dense':p.dense_graph(b)
                            else:p.packed_graph(b)
                        times[method].append(time.perf_counter()-t)
            Path(f'standalone_results/{law}_{rep}_compiled_state.json').write_text(json.dumps(fit,indent=2)+'\n')
            row=dict(law=law,rep=rep,observed_hash_verified=True,fitted_parameter_exact_parity=True,
                raw_graph_seconds=times,compiled_fit_timings=fit['timings'],
                dense_to_compiled_graph_ratio=float(np.median(times['dense'])/np.median(times['packed_c'])),
                python_to_compiled_graph_ratio=float(np.median(times['packed_python'])/np.median(times['packed_c'])))
            rows.append(row)
            print(law,rep,row['dense_to_compiled_graph_ratio'],fit['timings'],flush=True)
    report=dict(scope='fabricated-only local kernel/full-fit timing, not PSC/FM; original Python losses retained',
        parity_sizes=tests,compilation_first_load_seconds=first_load,compile_included_in_warm_timings=False,
        cold_fit_rule='Add compilation_first_load_seconds to first compiled-fit total; one-time common imports not timed.',
        rows=rows)
    Path('compiled_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
