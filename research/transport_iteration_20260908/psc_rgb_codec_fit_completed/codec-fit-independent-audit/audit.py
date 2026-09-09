"""Independent NumPy/SciPy saved-reconstruction audit. No model or data loader.

Requires authenticated complete output and explicit terminal SHA/commit. Does not
regenerate targets, rerun a codec, or infer frozen weights from metric arrays.
"""
import argparse,hashlib,json,math,subprocess,traceback
from pathlib import Path
import numpy as np
from scipy.ndimage import correlate1d
MSE_ATOL=2e-12
SSIM_ATOL=2e-10
RTOL=1e-10

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(test,message):
 if not test:raise ValueError(message)
def equal_numeric(a,b,atol,rtol=RTOL):
 a,b=np.asarray(a),np.asarray(b)
 require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite or incompatible metric arrays')
 require(np.allclose(a,b,atol=atol,rtol=rtol),'numeric mismatch')
def metrics(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 require(x.ndim==4 and x.shape==y.shape and x.shape[1]==3 and min(x.shape[2:])>=11,'matching RGB shape required')
 require(np.isfinite(x).all() and np.isfinite(y).all(),'nonfinite image')
 require(((x>=0)&(x<=1)&(y>=0)&(y<=1)).all(),'unit RGB range required')
 index=np.arange(-5,6,dtype=np.float64);g=np.exp(-index**2/4.5);g/=g.sum()
 def smooth(v):
  # Separable filtering; crop discards every value that touches padding.
  return correlate1d(correlate1d(v,g,axis=-1,mode='constant',cval=0),g,axis=-2,mode='constant',cval=0)[...,5:-5,5:-5]
 mx,my=smooth(x),smooth(y);vx=smooth(x*x)-mx*mx;vy=smooth(y*y)-my*my;cov=smooth(x*y)-mx*my
 score=((2*mx*my+.0001)*(2*cov+.0009))/((mx*mx+my*my+.0001)*(vx+vy+.0009))
 mse=np.mean((x-y)**2,axis=(1,2,3));ssim=np.mean(score,axis=(1,2,3))
 require(np.isfinite(mse).all() and np.isfinite(ssim).all(),'nonfinite derived metrics')
 return mse,ssim

def audit(root,expected_status_sha,expected_commit,repo=None):
 require(digest(root/'status.json')==expected_status_sha,'terminal status SHA mismatch')
 status=read(root/'status.json');inventory=status['payload_sha256']
 require(status['status']=='completed_FIT_only_qualification','failed/incomplete run is not a complete reconstruction bank')
 def checked(rel):
  require(rel in inventory,'accessed file absent from terminal manifest: '+rel)
  path=root/rel;require(path.resolve().is_relative_to(root.resolve()),'path escapes result root')
  require(digest(path)==inventory[rel],'payload hash mismatch: '+rel);return path
 for rel in inventory:checked(rel)
 identity=read(checked('source_identity.json'));require(identity['commit']==expected_commit and len(expected_commit)==40,'source commit mismatch')
 for rel,h in identity['sha256'].items():
  path=checked('source/'+rel);require(digest(path)==h,'source snapshot mismatch')
  if repo is not None:require(path.read_bytes()==subprocess.check_output(['git','-C',str(repo),'show',expected_commit+':'+rel]),'source Git blob mismatch')
 frozen=read(checked('FIT_FROZEN.json'));require(frozen['codec_frozen'] is True,'codec not reported frozen')
 require(digest(checked('fit_frozen.pt'))==frozen['checkpoint_sha256'],'frozen checkpoint mismatch')
 fit=frozen['fit'];require(fit['status']=='frozen' and type(fit['updates']) is int and fit['updates']>0,'invalid fit freeze completion')
 require(fit['updates']==status['fit']['updates'],'fit update count mismatch')
 require([r['nominal_seconds'] for r in fit['checkpoints']]==[60.,150.,300.,450.,600.],'checkpoint schedule mismatch')
 previous=0
 for r in fit['checkpoints']:
  require(digest(checked(r['path']))==r['sha256'],'scheduled checkpoint SHA mismatch')
  require(type(r['updates']) is int and previous<=r['updates']<=fit['updates'] and math.isfinite(r['actual_seconds']) and r['actual_seconds']>=r['nominal_seconds'],'invalid checkpoint timing/update record');previous=r['updates']
 binding=read(checked('stage_input_binding.json'));cache_binding=read(checked('cache_input_binding.json'))
 require(binding['ordered_rows']==4000 and binding['inplace_input_changes'] is False,'input binding mismatch')
 for key,name in [('fit_ids_file_sha256','fit_ids.npy'),('rounded_target_file_sha256','fit_rounded_evaluation_targets.npy'),('input_identity_sha256','input_identity.json')]:
  require(digest(checked(name))==binding[key]==cache_binding[key],'bound input file mismatch')
 for key,name in [('normalized_cache_sha256','fit_normalized_cache.npy'),('normalization_state_sha256','normalized_state.pt')]:require(digest(checked(name))==cache_binding[key],'cache provenance mismatch')
 ids=np.load(checked('fit_ids.npy'),allow_pickle=False);require(ids.shape==(4000,) and ids.dtype.kind in 'iu' and len(np.unique(ids))==4000 and np.all(np.diff(ids)>0),'ordered unique FIT IDs required')
 target=np.load(checked('fit_rounded_evaluation_targets.npy'),mmap_mode='r',allow_pickle=False);recon=np.load(checked('fit_reconstructions.npy'),mmap_mode='r',allow_pickle=False)
 require(target.shape==recon.shape==(4000,3,32,32) and target.dtype==recon.dtype==np.float32,'complete float32 banks required')
 require(read(checked('evaluation_progress.json'))['completed_rows']==4000,'partial reconstruction bank')
 a=[];b=[]
 for first in range(0,4000,32):
  mse,ssim=metrics(target[first:first+32],recon[first:first+32]);a.append(mse);b.append(ssim)
 mse=np.concatenate(a);ssim=np.concatenate(b);saved=np.load(checked('fit_per_image_metrics.npz'),allow_pickle=False)
 require(set(saved.files)=={'mse','ssim'} and saved['mse'].shape==saved['ssim'].shape==(4000,),'metric bank schema mismatch')
 equal_numeric(mse,saved['mse'],MSE_ATOL);equal_numeric(ssim,saved['ssim'],SSIM_ATOL)
 for name,value in [('first16_true.npy',target[:16]),('first16_reconstructed.npy',recon[:16])]:require(np.array_equal(np.load(checked(name),allow_pickle=False),value),'first16 bank mismatch')
 mean_mse=float(mse.mean());psnr=math.inf if mean_mse==0 else -10*math.log10(mean_mse);mean_ssim=float(ssim.mean())
 std=np.load(checked('latent_std.npy'),allow_pickle=False);mean=np.load(checked('latent_mean.npy'),allow_pickle=False)
 require(std.shape==mean.shape==(16,) and np.isfinite(std).all() and np.isfinite(mean).all(),'invalid normalization vectors')
 cache=np.load(checked('fit_normalized_cache.npy'),mmap_mode='r',allow_pickle=False);require(cache.shape==(4000,16,8,8) and cache.dtype==np.float32 and np.isfinite(cache).all(),'invalid complete cache')
 q=status['qualification'];reported=q['global_FIT_PSNR']
 if math.isinf(psnr):require(reported=='infinity','zero MSE PSNR convention mismatch')
 else:equal_numeric(psnr,reported,2e-10)
 equal_numeric(mean_ssim,q['mean_FIT_SSIM'],SSIM_ATOL)
 gates={'FIT_global_PSNR_ge26':psnr>=26.,'FIT_mean_SSIM_ge085':mean_ssim>=.85,'latent_std_finite_ge001':bool((std>=1e-3).all())}
 require(gates==q['gates'] and all(gates.values())==q['engineering_gates_pass'],'recomputed gates mismatch')
 return {'status':'VERIFIED_SAVED_FIT_RECONSTRUCTION_METRICS','gate_pass':all(gates.values()),'gates':gates,'mean_MSE':mean_mse,'global_PSNR':'infinity' if math.isinf(psnr) else psnr,'mean_SSIM':mean_ssim,'max_per_image_MSE_error':float(np.max(np.abs(mse-saved['mse']))),'max_per_image_SSIM_error':float(np.max(np.abs(ssim-saved['ssim']))),'tolerances':{'mse_atol':MSE_ATOL,'ssim_atol':SSIM_ATOL,'rtol':RTOL},'status_sha256':expected_status_sha,'source_commit':expected_commit,'source_git_verified':repo is not None,'payload_count':len(inventory),'scope':'Independent separable valid Gaussian population SSIM and global-mean-MSE PSNR on authenticated saved rounded targets/reconstructions. FIT-only, not generalization or generation. Frozen-state and shared-stage-input assertions source-reviewed/hash-bound, not inferred from arrays. Cache finiteness is checked; encoder/cache recomputation is not performed.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--expected-status-sha',required=True);p.add_argument('--expected-commit',required=True);p.add_argument('--repo',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 # Exclusive directory preserves either success or failure without overwriting.
 args.output.mkdir(parents=True,exist_ok=False)
 try:result=audit(args.root,args.expected_status_sha,args.expected_commit,args.repo)
 except BaseException as error:
  (args.output/'failure.json').write_text(json.dumps({'error':repr(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
 (args.output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
