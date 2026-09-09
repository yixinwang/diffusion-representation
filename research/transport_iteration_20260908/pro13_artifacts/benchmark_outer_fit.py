"""Outer-call fit timing, including the initial whole-array validation scan.
Compilation is separately recorded; fabricated input generation is not training.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import pro13 as p
import packed_kernel as ck
import standalone as s

def main():
    t=time.perf_counter();ck.load_kernel();compile_load=time.perf_counter()-t
    rows=[]
    for rep in range(3):
        f=json.loads(Path(f'standalone_results/fixture_{rep}.json').read_text())
        old=json.loads(Path(f'standalone_results/positive_{rep}_record.json').read_text())
        x,hashes=s.generate(f,'positive',old['array_seed']);assert hashes==old['hashes']
        expected=json.loads(Path(f'standalone_results/positive_{rep}_packed_state.json').read_text())
        expected_json=json.dumps(s.strip_timing(expected),sort_keys=True)
        raw={k:[] for k in ['packed_c','dense']}
        with threadpool_limits(limits=1):
            for order in [('packed_c','dense'),('dense','packed_c'),('packed_c','dense')]:
                for method in order:
                    start=time.perf_counter();st=p.fit(x,method=method);elapsed=time.perf_counter()-start
                    raw[method].append(elapsed)
                    assert json.dumps(s.strip_timing(st),sort_keys=True)==expected_json
        rows.append(dict(rep=rep,raw_seconds=raw,fitted_parameters_verified=True,
            fitted_parameter_sha256=hashlib.sha256(expected_json.encode()).hexdigest(),
            dense_over_compiled_ratio=float(np.median(raw['dense'])/np.median(raw['packed_c']))))
    report=dict(scope='outer-call positive-law fabricated fits, local one-thread, no serialization or input-generation time',
        compilation_load_seconds=compile_load,compilation_in_warm_rows=False,build_record=ck.BUILD_RECORD,rows=rows)
    Path('outer_fit_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
