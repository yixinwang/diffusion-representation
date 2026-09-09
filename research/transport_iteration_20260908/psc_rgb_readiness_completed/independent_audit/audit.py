"""Authenticate data-free readiness outputs; no canonical observations or fitting."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
REV='84c0cad383be64d6632a5fa61e5d1bc4b605ca5f'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main(root,repo):
 s=read(root/'status.json');assert s['status']=='completed_data_free_readiness'
 assert s['scope']=='DATA_FREE_COMPONENT_READINESS_ONLY' and s['real_data_accessed'] is False
 for name,h in s['payload_sha256'].items():assert sha(root/name)==h,name
 identity=read(root/'source_identity.json');assert identity['commit']==REV and len(identity['sha256'])==7
 for name,h in identity['sha256'].items():
  raw=(root/'source'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h
  assert raw==subprocess.check_output(['git','-C',str(repo),'show',REV+':'+name])
 assert s['counts']=={'codec_encoder':1728272,'codec_decoder':1744643,'latent_field':3011472,'latent_pipeline_total':6484387,'pixel_total':4427395}
 inputs=np.load(root/'synthetic_inputs.npy');assert inputs.shape==(256,3,32,32) and inputs.dtype==np.float32 and np.isfinite(inputs).all() and np.max(np.abs(inputs))<=1
 result={'source_payload_verified':True,'payload_count':len(s['payload_sha256']),'scope':'Synthetic component timing/numerical evidence only, not real-data quality or a trained baseline','training':{},'sampling':{}}
 for label,count in [('codec_train',3472915),('latent_train',3011472),('pixel_train',4427395)]:
  r=read(root/(label+'.json'));assert len(r['warmup'])==5 and len(r['timed'])==20 and r['intended_parameter_elements']==count
  for x in r['warmup']+r['timed']+[r['batch125_one_step']]:assert np.isfinite(x['loss']) and 0<x['seconds']<3600
  values=[x['seconds'] for x in r['timed']]
  assert np.isclose(np.median(values),r['batch32_p50_seconds'],rtol=1e-12) and np.isclose(np.quantile(values,.9),r['batch32_p90_seconds'],rtol=1e-12)
  result['training'][label]={'batch32_p50_seconds':r['batch32_p50_seconds'],'batch32_p90_seconds':r['batch32_p90_seconds'],'batch125_seconds':r['batch125_one_step']['seconds'],'batch32_memory':r['batch32_memory'],'batch125_memory':r['batch125_one_step']['memory']}
 cache=np.load(root/'synthetic_normalized_cache.npy');assert cache.shape==(256,16,8,8) and cache.dtype==np.float32 and np.isfinite(cache).all()
 channels=cache.transpose(1,0,2,3).reshape(16,-1).astype(float);mean_error=float(np.abs(channels.mean(1)).max());variance_error=float(np.abs(channels.var(1)-1).max())
 assert mean_error<5e-6 and variance_error<5e-6
 result['saved_cache_normalization']={'mean_error':mean_error,'population_variance_error':variance_error}
 for label,dimension in [('latent',1024),('pixel',3072)]:
  r=read(root/(label+'_sampling.json'));assert r['steps']==32 and r['field_calls']==64 and r['includes_decoder']==(label=='latent')
  for batch in (1,32):
   source=np.load(root/f'{label}_source_b{batch}.npy');assert source.shape==(batch,dimension) and source.dtype==np.float32 and np.isfinite(source).all()
   outputs=[np.load(root/f'{label}_synthetic_output_b{batch}_iteration{i}.npy') for i in range(4)]
   assert all(x.shape==(batch,3,32,32) and np.isfinite(x).all() for x in outputs)
   assert all(np.array_equal(outputs[0],x) for x in outputs[1:])
   assert np.array_equal(outputs[-1],np.load(root/f'{label}_synthetic_output_b{batch}.npy'))
   t=r['batches'][str(batch)];assert len(t['timed_seconds'])==3 and all(np.isfinite(x) and x>0 for x in [t['warmup_seconds']]+t['timed_seconds'])
   if label=='latent':assert np.max(np.abs(outputs[0]))<=1
  assert np.array_equal(np.load(root/(label+'_reloaded_synthetic_output.npy')),np.load(root/f'{label}_synthetic_output_b1.npy'))
  ck=s[label+'_checkpoint'];assert ck['bytes']==(root/(label+'_synthetic_checkpoint.pt')).stat().st_size and ck['sha256']==sha(root/(label+'_synthetic_checkpoint.pt'))
  result['sampling'][label]=r
 result['status']='PASS independent saved-output audit';return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('repo',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.write_text(json.dumps(main(a.root,a.repo),indent=2)+'\n')
