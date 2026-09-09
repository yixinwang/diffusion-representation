import sys,json,hashlib,subprocess
from pathlib import Path
repo=Path('/ocean/projects/mth250006p/ywang26/diffusion-cached-pilot-20260909');results=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-cached-pilot');out=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-cached-repair-audit');rev='1f6be303c8b93ba45ac63c43e7c5971fa38a0082';identity=json.loads((results/'source_identity.json').read_text());assert identity['commit']==rev and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==rev
for name,h in identity['sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h
out.mkdir(exist_ok=False)
try:
 sys.path.insert(0,str(repo/'qalt/src'))
 import numpy as np
 from qalt.observed_flow_data import load_observed_flow_data
 data=load_observed_flow_data(Path('/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py'))
 assert data.ledger==json.loads((results/'data_ledger.json').read_text())
 value_hash=hashlib.sha256(np.ascontiguousarray(data.repair).tobytes()).hexdigest();id_hash=hashlib.sha256(np.asarray(data.repair_ids,dtype='<i8').tobytes()).hexdigest()
 for seed in [77201,77202,77203]:
  pair=json.loads((results/f'seed_{seed}/pair_identity.json').read_text());assert pair['repair_values_sha256']==value_hash and pair['repair_ids_sha256']==id_hash
 np.save(out/'repair.npy',data.repair);np.save(out/'repair_ids.npy',data.repair_ids)
 result={'status':'strict_canonical_repair_identity_pass','source_commit':rev,'repair_values_sha256':value_hash,'repair_ids_sha256':id_hash,'shape':list(data.repair.shape),'dtype':str(data.repair.dtype),'official_test_read':False,'discovery_exported':False,'fit_exported':False,'payload_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['repair.npy','repair_ids.npy']}}
except BaseException as e:result={'status':'failure','error':repr(e)}
with (out/'status.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
