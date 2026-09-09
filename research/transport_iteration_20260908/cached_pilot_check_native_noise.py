import subprocess,shlex,json,time
from pathlib import Path
script='''import numpy as np,torch,json,hashlib
from pathlib import Path
p=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-cached-pilot');r={}
for seed in [77201,77202,77203]:
 source=np.load(p/f'seed_{seed}/common_gaussian.npy',allow_pickle=False);replay=torch.randn(2000,3072,generator=torch.Generator().manual_seed(seed+300)).numpy()
 r[str(seed)]={'generation_source_exact':bool(np.array_equal(source,replay)),'numerical_sources':{}}
 for arm in ['S','M','J','RQS']:
  with np.load(p/f'seed_{seed}/{arm}/numerical.npz',allow_pickle=False) as f:
   r[str(seed)]['numerical_sources'][arm]=bool(np.array_equal(f['source'],torch.randn(8,3072,generator=torch.Generator().manual_seed(seed+99)).numpy()))
assert all(v['generation_source_exact'] and all(v['numerical_sources'].values()) for v in r.values())
print(json.dumps({'status':'same_environment_registered_source_replay_pass','torch':torch.__version__,'results':r}))
''';start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(script)],capture_output=True,text=True,timeout=300);d={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:d={'timeout':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
d['elapsed_seconds']=time.monotonic()-start;Path('work/psc-cached-pilot/native-noise-check.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
