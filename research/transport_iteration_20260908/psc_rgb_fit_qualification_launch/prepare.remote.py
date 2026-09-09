import os,json,subprocess,hashlib,traceback,time
from pathlib import Path
CONFIG={'source_commit': '6ae48a91706c97e65c833f94bd51a7b54f74c653', 'source_sha256': {'qalt/src/qalt/__init__.py': 'a54aca828ea497a35a82724573d32fbba7d029370e9b66f0c170666b076cbdaf', 'qalt/src/qalt/core.py': '4f7893896da3d4330837ca41537c361ca2f65d2e5a17428524620601aec25fd6', 'qalt/src/qalt/rgb_codec_flow_matching.py': '48445678a60ab3ec031e64523c6f642f0ae6b3bc826847c7b458b6185fb6a1c4', 'qalt/src/qalt/data_integrity.py': '8e950adfc29135dd1dedf60bb4a9b1886c8e4f88383c3bca14aa9e795ab1eabf', 'qalt/src/qalt/observed_flow_data.py': '6b97cef92d50cd24b288706c9121f3f1341416fd3f06ffd1d2bafeaac92273c3', 'qalt/data/observed_manifest_v1.json': '8c0ae7a0c11a61876bf1be39ebe85c800afabe19969bfec770a6181a205b9995', 'qalt/experiments/rgb_codec_fit_qualification/run.py': '7c1c7039c2b038ab45d77c441533aa8b112b8697edcad51c69b5c9fce58fa0e3', 'qalt/experiments/rgb_codec_fit_qualification/PROTOCOL.md': '37e0ff91be689e41b33f08b683cb19f38cba4fd37abd77ec74caa26c35a5c602', 'qalt/experiments/rgb_codec_fit_qualification/run.sh': 'b82d314c995438db1c4bf838027a7210de0f46072d576fa52f9bd43b3c7d8fc6', 'qalt/tests/test_rgb_codec_fit_qualification.py': '718ba26a90bcead95663ffe3a3a846b40d7c4a5ea36d548f89c64fc1be501da1', 'qalt/tests/test_rgb_codec_flow_matching.py': 'b13e6d73044aed7357187da93a6cc53f56a63b382cb69cf314901dc8e6c49644'}, 'checkout': '/ocean/projects/mth260022p/ywang26/diffusion-rgb-fit-qualification-20260909-6ae48a91', 'staging': '/ocean/projects/mth260022p/ywang26/rgb-fit-qualification-launch-20260909-6ae48a91', 'result_root': '/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-rgb-fit-qualification-6ae48a91', 'new_project': '/ocean/projects/mth260022p/ywang26', 'original_read_only_checkout': '/ocean/projects/mth250006p/ywang26/diffusion-representation', 'remote': 'https://github.com/yixinwang/diffusion-representation.git'}
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
