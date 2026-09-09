"""Exploratory local four-call comparison after full-grid harness timeouts.
All earlier partial timings are retained. This is NOT a full-frontier benchmark.
"""
import json,time,platform,os
from pathlib import Path
import numpy as np
import scipy
from run_local import dictionary, generate, SEEDS, array_hash
from pro10_kernels import HeunPlan
O=dictionary();plans={4:HeunPlan.build(4)}
labels=['exact_log','exact_hybrid','heun_legacy','heun_cached']
records=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    for batch in (1,64):
        z=rng.normal(size=(batch,3072));raw={l:[] for l in labels};order=[]
        for label in labels:
            for _ in range(3):generate(z,O,label,plans,4)
        for i in range(9):
            perm=list(rng.permutation(labels));order.append(perm)
            for label in perm:
                t=time.perf_counter();out=generate(z,O,label,plans,4);dt=time.perf_counter()-t
                raw[label].append(dt)
                with open('local_focused_live.jsonl','a') as f:
                    f.write(json.dumps(dict(seed=seed,batch=batch,repeat=i,label=label,seconds=dt))+'\n')
        row=dict(seed=seed,batch=batch,source_sha256=array_hash(z),raw_seconds=raw,order=order,
                 median_ms={k:float(np.median(v)*1000) for k,v in raw.items()})
        records.append(row)
        Path('local_focused_timing.json').write_text(json.dumps(dict(
            scope='LOCAL exploratory CPU, one completed process, four-call comparison only; NO fitting, NO PSC, no full-frontier claim',
            python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            records=records),indent=2)+'\n')
        print(seed,batch,row['median_ms'],flush=True)
