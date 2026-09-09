"""Prospective FIT-only RGB codec reconstruction qualification; no generation claim."""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'qalt/src'))
from qalt.rgb_codec_flow_matching import RGBCodecFlowMatching
from qalt.observed_flow_data import load_observed_flow_data
SEED=79201
FIT_SECONDS=600.
CHECKPOINT_SECONDS=(60.,150.,300.,450.,600.)
DATA_ROOT=Path('/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py')
FILES=('qalt/src/qalt/__init__.py','qalt/src/qalt/core.py','qalt/src/qalt/rgb_codec_flow_matching.py',
 'qalt/src/qalt/data_integrity.py','qalt/src/qalt/observed_flow_data.py','qalt/data/observed_manifest_v1.json',
 'qalt/experiments/rgb_codec_fit_qualification/run.py','qalt/experiments/rgb_codec_fit_qualification/PROTOCOL.md',
 'qalt/experiments/rgb_codec_fit_qualification/run.sh','qalt/tests/test_rgb_codec_fit_qualification.py','qalt/tests/test_rgb_codec_flow_matching.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sync(device):
 if device.type=='cuda':torch.cuda.synchronize(device)
def finite(*values):
 if not all(bool(torch.isfinite(v).all()) for v in values):raise FloatingPointError('nonfinite numerical value')
def source_guard(commit,out):
 if len(commit)!=40 or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit:raise ValueError('expected full frozen HEAD required')
 hashes={}
 for rel in FILES:
  raw=(ROOT/rel).read_bytes()
  if raw!=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT):raise ValueError('source not frozen: '+rel)
  target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);hashes[rel]=sha(target)
 for name,m in list(sys.modules.items()):
  if name=='qalt' or name.startswith('qalt.'):
   if str(Path(m.__file__).resolve().relative_to(ROOT)) not in FILES:raise ValueError('unfrozen imported module')
 atomic(out/'source_identity.json',{'commit':commit,'sha256':hashes})
def learning_rate(elapsed):
 # Wall-clock warmup from2e-5 to2e-4 over30s, then cosine to2e-5 at600s.
 x=min(max(float(elapsed),0.),FIT_SECONDS)
 if x<=.05*FIT_SECONDS:return 2e-5+(2e-4-2e-5)*x/(.05*FIT_SECONDS)
 return 2e-5+.5*(2e-4-2e-5)*(1+math.cos(math.pi*(x-.05*FIT_SECONDS)/(.95*FIT_SECONDS)))
def save_checkpoint(path,model,optimizer,rng,updates,elapsed,nominal):
 record={'model':model.state_dict(),'optimizer':None if optimizer is None else optimizer.state_dict(),
  'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
  'updates':updates,'stage_elapsed_seconds':elapsed,'nominal_stage_seconds':nominal,'seed':SEED,'stage':int(model._stage)}
 tmp=path.with_suffix('.tmp');torch.save(record,tmp);os.replace(tmp,path)
def fit_codec(model,x,ids,out,state):
 device=next(model.parameters()).device
 if int(model._stage)!=0:raise ValueError('fresh codec stage required')
 parameters=list(model.codec.parameters());optimizer_start=time.perf_counter();optimizer=torch.optim.AdamW(parameters,lr=2e-4,betas=(.9,.99),weight_decay=1e-4)
 optimizer_constructor_seconds=time.perf_counter()-optimizer_start
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
 rng=np.random.default_rng(SEED+1);updates=0;next_checkpoint=0;sync(device);start=time.perf_counter()
 record={'checkpoints':[],'stage_seconds':FIT_SECONDS,'updates':0,'optimizer_constructor_seconds':optimizer_constructor_seconds}
 try:
  save_checkpoint(out/'initial.pt',model,optimizer,rng,0,time.perf_counter()-start,0.)
  with (out/'fit_steps.jsonl').open('x') as ledger:
   while time.perf_counter()-start<FIT_SECONDS:
    elapsed=time.perf_counter()-start;index=rng.integers(0,len(ids),size=32);lr=learning_rate(elapsed)
    for group in optimizer.param_groups:group['lr']=lr
    draw={'step':updates+1,'indices':index.tolist(),'fit_ids':np.asarray(ids[index]).tolist(),'lr':lr,'draw_stage_seconds':elapsed}
    state.update(phase='codec_fit',draw=draw,completed_updates=updates)
    ledger.write(json.dumps({'event':'draw',**draw})+'\n');ledger.flush()
    optimizer.zero_grad(set_to_none=True);loss=model.training_loss(x[torch.as_tensor(index,device=device)])
    if loss.ndim!=0:raise ValueError('codec loss must be scalar')
    finite(loss);loss.backward()
    if any(p.grad is None for p in parameters):raise RuntimeError('missing codec gradient')
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):raise FloatingPointError('nonfinite codec gradient')
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise FloatingPointError('nonfinite updated codec parameter')
    sync(device);updates+=1;now=time.perf_counter()-start
    ledger.write(json.dumps({'event':'completed','step':updates,'loss':float(loss.detach()),'stage_seconds':now})+'\n');ledger.flush()
    while next_checkpoint<len(CHECKPOINT_SECONDS) and now>=CHECKPOINT_SECONDS[next_checkpoint]:
     nominal=CHECKPOINT_SECONDS[next_checkpoint];path=out/f'checkpoint_{int(nominal):03d}.pt'
     save_checkpoint(path,model,optimizer,rng,updates,now,nominal)
     record['checkpoints'].append({'nominal_seconds':nominal,'actual_seconds':now,'updates':updates,'path':path.name,'sha256':sha(path)})
     next_checkpoint+=1;now=time.perf_counter()-start
    record.update(updates=updates,elapsed_seconds=now);atomic(out/'fit_progress.json',record)
  if updates==0:raise RuntimeError('zero-update codec fit')
  # I/O may consume a checkpoint threshold after the final update. Preserve that state.
  while next_checkpoint<len(CHECKPOINT_SECONDS):
   nominal=CHECKPOINT_SECONDS[next_checkpoint];now=time.perf_counter()-start;path=out/f'checkpoint_{int(nominal):03d}.pt'
   save_checkpoint(path,model,optimizer,rng,updates,now,nominal)
   record['checkpoints'].append({'nominal_seconds':nominal,'actual_seconds':now,'updates':updates,'path':path.name,'sha256':sha(path)});next_checkpoint+=1
  model.set_stage('normalization');model.eval()
  save_checkpoint(out/'fit_frozen.pt',model,optimizer,rng,updates,time.perf_counter()-start,600.)
  final_checkpoint_sha256=sha(out/'fit_frozen.pt');elapsed=time.perf_counter()-start
  record.update(updates=updates,elapsed_seconds=elapsed,overrun_seconds=max(0.,elapsed-FIT_SECONDS),status='frozen',clock_boundary='through final checkpoint hash; receipt/accounting serialization excluded from stage and charged to standalone wall')
  atomic(out/'FIT_FROZEN.json',{'checkpoint_sha256':final_checkpoint_sha256,'codec_frozen':not any(p.requires_grad for p in model.codec.parameters()),'fit':record})
  if device.type=='cuda':record.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
  atomic(out/'fit_accounting.json',record)
  return record
 except BaseException as error:
  original={'error':repr(error),'traceback':traceback.format_exc(),'completed_updates':updates,'optimizer_constructor_seconds':optimizer_constructor_seconds}
  try:save_checkpoint(out/'failed_fit.pt',model,optimizer,rng,updates,time.perf_counter()-start,None)
  except BaseException as secondary:original['checkpoint_save_error']=repr(secondary)
  try:atomic(out/'fit_failure.json',original)
  except BaseException as secondary:print('Failed to persist fit_failure: '+repr(secondary),file=sys.stderr)
  raise

def require_frozen(model,out):
 receipt=out/'FIT_FROZEN.json'
 if not receipt.exists() or any(p.requires_grad for p in model.codec.parameters()):raise ValueError('fit must freeze before reconstruction evaluation')
 if json.loads(receipt.read_text())['checkpoint_sha256']!=sha(out/'fit_frozen.pt'):raise ValueError('frozen checkpoint changed')

def reconstruction_metrics(target,reconstruction):
 """Per-image MSE and valid11x11 sigma1.5 SSIM on [0,1], no clipping."""
 if target.shape!=reconstruction.shape or target.ndim!=4 or target.shape[1]!=3 or min(target.shape[2:])<11:raise ValueError('matching RGB image batches>=11 required')
 finite(target,reconstruction)
 if bool(((target<0)|(target>1)|(reconstruction<0)|(reconstruction>1)).any()):raise ValueError('metric RGB range must be[0,1]')
 x=target.double();y=reconstruction.double();grid=torch.arange(11,device=x.device,dtype=x.dtype)-5;g=torch.exp(-grid.square()/(2*1.5**2));g=g/g.sum();w=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
 conv=lambda z:F.conv2d(z,w,groups=3)
 a,b=conv(x),conv(y);vx,vy=conv(x*x)-a*a,conv(y*y)-b*b;cov=conv(x*y)-a*b
 ssim=((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(vx+vy+.03**2));mse=(x-y).square().flatten(1).mean(1);ssim=ssim.flatten(1).mean(1);finite(mse,ssim);return mse,ssim

def global_psnr(mse):
 values=np.asarray(mse,dtype=np.float64)
 if values.size==0 or not np.isfinite(values).all() or (values<0).any():raise ValueError('finite nonnegative MSE required')
 mean=float(values.mean());return float('inf') if mean==0 else -10*math.log10(mean)

@torch.no_grad()
def normalize_and_evaluate(model,x,out,state):
 require_frozen(model,out);device=x.device;start=time.perf_counter();raw=np.empty((len(x),16,8,8),dtype=np.float32)
 state['phase']='FIT_normalization'
 for first in range(0,len(x),64):
  batch=x[first:first+64];model.update_normalization(batch);raw[first:first+len(batch)]=model.codec.encode(batch).cpu().numpy()
 # Two encoder passes/chunk (moments API plus cache) are real, fully charged work.
 model.freeze_normalization();np.save(out/'latent_mean.npy',model.latent_mean.cpu().numpy());np.save(out/'latent_std.npy',model.latent_std.cpu().numpy())
 raw=(raw-model.latent_mean.cpu().numpy()[None,:,None,None])/model.latent_std.cpu().numpy()[None,:,None,None]
 if not np.isfinite(raw).all():raise FloatingPointError('nonfinite normalized cache')
 np.save(out/'fit_normalized_cache.npy',raw);torch.save(model.state_dict(),out/'normalized_state.pt');model.set_stage('inference')
 normalization_seconds=time.perf_counter()-start;start=time.perf_counter();state['phase']='FIT_reconstruction_evaluation'
 reconstructed=np.lib.format.open_memmap(out/'fit_reconstructions.npy',mode='w+',dtype=np.float32,shape=(len(x),3,32,32));mses=[];ssims=[]
 for first in range(0,len(x),64):
  batch=x[first:first+64];pred=(model.codec(batch)+1)*.5;truth=(batch+1)*.5;mse,ssim=reconstruction_metrics(truth,pred)
  reconstructed[first:first+len(batch)]=pred.cpu().numpy();reconstructed.flush();mses.extend(mse.cpu().tolist());ssims.extend(ssim.cpu().tolist())
  state['evaluated_rows']=first+len(batch);atomic(out/'evaluation_progress.json',{'completed_rows':state['evaluated_rows']})
 np.savez(out/'fit_per_image_metrics.npz',mse=np.asarray(mses),ssim=np.asarray(ssims));np.save(out/'first16_true.npy',((x[:16]+1)*.5).cpu().numpy());np.save(out/'first16_reconstructed.npy',np.array(reconstructed[:16]))
 psnr=global_psnr(mses);mean_ssim=float(np.mean(ssims));std=model.latent_std.cpu().numpy();gates={'FIT_global_PSNR_ge26':psnr>=26.,'FIT_mean_SSIM_ge085':mean_ssim>=.85,'latent_std_finite_ge001':bool(np.isfinite(std).all() and (std>=1e-3).all())}
 return {'global_FIT_PSNR':psnr if math.isfinite(psnr) else 'infinity','mean_FIT_SSIM':mean_ssim,'gates':gates,'engineering_gates_pass':all(gates.values()),'normalization_cache_seconds':normalization_seconds,'FIT_evaluation_seconds':time.perf_counter()-start,'normalization_encoder_passes':2}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-commit',required=True);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();state={};report={'scope':'FIT_ONLY_CODEC_ENGINEERING_QUALIFICATION_NOT_GENERALIZATION','seed':SEED,'repair_physically_constructed_by_loader':True,'repair_used_for_statistics_or_evaluation':False};model=None
 try:
  source_guard(args.expected_commit,out)
  if not torch.cuda.is_available():raise RuntimeError('CUDA required')
  gpu=torch.cuda.get_device_properties(0)
  if 'V100' not in gpu.name or gpu.total_memory<30*1024**3:raise RuntimeError('V10032 required')
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  def stop(signum,frame):raise TimeoutError('allocation termination; preserve state, no retry')
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  atomic(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':gpu.name,'job':os.environ.get('SLURM_JOB_ID')})
  with (out/'preflight.txt').open('w') as f:subprocess.run([sys.executable,'-m','pytest','-q','qalt/tests/test_rgb_codec_fit_qualification.py','qalt/tests/test_rgb_codec_flow_matching.py'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'qalt/src')},stdout=f,stderr=subprocess.STDOUT,check=True)
  state['phase']='canonical_fit_loading';t=time.perf_counter();data=load_observed_flow_data(DATA_ROOT)
  if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture'] or data.fit.shape!=(4000,3,32,32):raise ValueError('canonical4000FIT required')
  atomic(out/'data_ledger.json',data.ledger);np.save(out/'fit_ids.npy',data.fit_ids);ids=data.fit_ids.copy()
  atomic(out/'input_identity.json',{'fit_float64_sha256':hashlib.sha256(np.ascontiguousarray(data.fit).tobytes()).hexdigest(),'fit_ids_sha256':sha(out/'fit_ids.npy'),'input_transform':'2*float32(FIT[0,1])-1; metrics compare resulting rounded float32 RGB'})
  x=(2*torch.as_tensor(np.array(data.fit,dtype=np.float32))-1).cuda();del data;np.save(out/'fit_rounded_evaluation_targets.npy',((x+1)*.5).cpu().numpy());report['loader_input_seconds']=time.perf_counter()-t
  state['phase']='fresh_initialization';t=time.perf_counter();model=RGBCodecFlowMatching().cuda();sync(x.device);report['initialization_seconds']=time.perf_counter()-t
  report['parameter_counts']={name:sum(p.numel() for p in module.parameters()) for name,module in [('full_model',model),('codec',model.codec),('field',model.field)]};report['field_untrained']=True
  binding={'fit_ids_file_sha256':sha(out/'fit_ids.npy'),'rounded_target_file_sha256':sha(out/'fit_rounded_evaluation_targets.npy'),'input_identity_sha256':sha(out/'input_identity.json'),'ordered_rows':4000,'shared_tensor_roles':['codec_training','FIT_normalization','normalized_cache','FIT_reconstruction_evaluation'],'inplace_input_changes':False}
  atomic(out/'stage_input_binding.json',binding)
  report['fit']=fit_codec(model,x,ids,out,state)
  if not np.array_equal(((x+1)*.5).cpu().numpy(),np.load(out/'fit_rounded_evaluation_targets.npy')) or not np.array_equal(ids,np.load(out/'fit_ids.npy')):raise ValueError('FIT tensor or ordered IDs changed during training')
  report['qualification']=normalize_and_evaluate(model,x,out,state)
  atomic(out/'cache_input_binding.json',{**binding,'normalized_cache_sha256':sha(out/'fit_normalized_cache.npy'),'normalization_state_sha256':sha(out/'normalized_state.pt')});report['status']='completed_FIT_only_qualification'
 except BaseException as error:
  report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),state=state)
  if model is not None:
   try:torch.save(model.state_dict(),out/'failure_model_state.pt')
   except BaseException as secondary:report['failure_checkpoint_save_error']=repr(secondary)
  raise
 finally:
  if torch.cuda.is_available():
   report['gpu_peak_allocated_bytes_since_fit_reset']=torch.cuda.max_memory_allocated();report['gpu_peak_reserved_bytes_since_fit_reset']=torch.cuda.max_memory_reserved()
  report['cpu_peak_rss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  report['payload_sha256']={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='status.json'}
  report['full_standalone_wall_seconds']=time.perf_counter()-started
  report['wall_clock_boundary']='through payload hashing; excludes final status serialization and process teardown'
  try:atomic(out/'status.json',report)
  except BaseException as secondary:
   if sys.exc_info()[0] is not None and report.get('status')=='failed':print('Secondary status save failure: '+repr(secondary)+'; original: '+report.get('traceback',''),file=sys.stderr)
   else:raise
if __name__=='__main__':main()
