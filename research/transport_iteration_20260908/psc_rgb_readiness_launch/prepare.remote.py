import os,json,subprocess,hashlib,traceback,time
from pathlib import Path
CONFIG={'source_commit': '84c0cad383be64d6632a5fa61e5d1bc4b605ca5f', 'source_sha256': {'qalt/src/qalt/__init__.py': 'a54aca828ea497a35a82724573d32fbba7d029370e9b66f0c170666b076cbdaf', 'qalt/src/qalt/core.py': '4f7893896da3d4330837ca41537c361ca2f65d2e5a17428524620601aec25fd6', 'qalt/src/qalt/rgb_codec_flow_matching.py': '48445678a60ab3ec031e64523c6f642f0ae6b3bc826847c7b458b6185fb6a1c4', 'qalt/tests/test_rgb_codec_flow_matching.py': 'b13e6d73044aed7357187da93a6cc53f56a63b382cb69cf314901dc8e6c49644', 'qalt/experiments/rgb_codec_readiness/run.py': '67063c491c9aa66df8aa72c401dbde362845fa0dceaefe94a3f6cd1e69666b1b', 'qalt/experiments/rgb_codec_readiness/PROTOCOL.md': '7af58f59a218428ddfa172545e2552b23b347f849dbb8ef0b347b0d20035af5f', 'qalt/experiments/rgb_codec_readiness/run.sh': '0749017c4fb60c9daf29e41e10995d8586f6950277d9ba737811d8b3856d3941'}, 'checkout': '/ocean/projects/mth260022p/ywang26/diffusion-rgb-readiness-20260909-84c0cad3', 'staging': '/ocean/projects/mth260022p/ywang26/rgb-readiness-launch-20260909-84c0cad3', 'result_root': '/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-rgb-readiness-84c0cad3', 'new_project': '/ocean/projects/mth260022p/ywang26', 'original_read_only_checkout': '/ocean/projects/mth250006p/ywang26/diffusion-representation', 'remote': 'https://github.com/yixinwang/diffusion-representation.git'}
stage=Path(CONFIG['staging']);stage.mkdir(parents=True,exist_ok=False)
started=time.time();env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GIT_OPTIONAL_LOCKS':'0','TMPDIR':str(stage/'tmp'),'XDG_CACHE_HOME':str(stage/'cache')}
Path(env['TMPDIR']).mkdir();Path(env['XDG_CACHE_HOME']).mkdir()
def call(args,**kw):return subprocess.check_output(args,env=env,timeout=300,**kw)
original=CONFIG['original_read_only_checkout'];before=None;head=None
try:
 head=call(['git','-C',original,'rev-parse','HEAD']);before=call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])
 repo=Path(CONFIG['checkout']);assert not repo.exists()
 with (stage/'clone.log').open('wb') as log:
  subprocess.run(['git','clone','--depth=1','--filter=blob:none','--no-checkout',CONFIG['remote'],str(repo)],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'fetch','--depth=1','origin',CONFIG['source_commit']],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'sparse-checkout','set','qalt'],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'checkout','--detach',CONFIG['source_commit']],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
 assert call(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==CONFIG['source_commit']
 for rel,sha in CONFIG['source_sha256'].items():
  raw=(repo/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
  assert raw==call(['git','-C',str(repo),'show',CONFIG['source_commit']+':'+rel])
 assert call(['git','-C',original,'rev-parse','HEAD'])==head and call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])==before
 result={'status':'prepared_no_allocation','config':CONFIG,'original_head':head.decode().strip(),'original_status_sha256':hashlib.sha256(before).hexdigest(),'original_unchanged':True,'elapsed_seconds':time.time()-started}
except BaseException as exc:
 result={'status':'preparation_failed_no_allocation','config':CONFIG,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started}
 if before is not None:
  result['original_head']=head.decode().strip();result['original_status_sha256']=hashlib.sha256(before).hexdigest()
  try:result['original_unchanged']=call(['git','-C',original,'rev-parse','HEAD'])==head and call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])==before
  except BaseException as check:result['original_verification_error']=repr(check)
(stage/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='prepared_no_allocation':raise SystemExit(1)
