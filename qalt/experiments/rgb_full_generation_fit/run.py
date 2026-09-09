"""Fresh baseline fitting only. No generation, reconstruction/repair evaluation."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'qalt/src'))
from qalt.rgb_codec_flow_matching import RGBCodecFlowMatching,PixelFlowMatching
from qalt.observed_flow_data import load_observed_flow_data
UTILITY=ROOT/'qalt/experiments/observed_flow_pilot/run_shared.py'
spec=importlib.util.spec_from_file_location('rgb_full_fit_native_utility',UTILITY);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
SEEDS=(78201,78202,78203)
FIELD_SECONDS=1800.;CODEC_SECONDS=600.
PIXEL_POINTS=(360.,780.,1200.,1800.);LATENT_POINTS=(180.,600.,1200.,1800.);CODEC_POINTS=(60.,150.,300.,450.,600.)
DATA_ROOT=Path('/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py')
def sources():
 return sorted(set(list((ROOT/'qalt/src/qalt').glob('*.py'))+[UTILITY,ROOT/'qalt/data/observed_manifest_v1.json',ROOT/'qalt/tests/test_rgb_full_generation_fit.py',ROOT/'qalt/tests/test_rgb_codec_flow_matching.py']+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.md'))+list(Path(__file__).parent.glob('*.sh'))))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sync(device):
 if device.type=='cuda':torch.cuda.synchronize(device)
def finite(t):
 if not bool(torch.isfinite(t).all()):raise FloatingPointError('nonfinite numerical value')
def source_guard(commit,out):
 if len(commit)!=40 or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit:raise ValueError('exact frozen HEAD required')
 hashes={}
 for p in sources():
  rel=str(p.relative_to(ROOT));raw=p.read_bytes()
  if raw!=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT):raise ValueError('unfrozen source '+rel)
  q=out/'source'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);hashes[rel]=sha(q)
 for name,module in list(sys.modules.items()):
  if name=='qalt' or name.startswith('qalt.'):
   if str(Path(module.__file__).resolve().relative_to(ROOT)) not in hashes:raise ValueError('unfrozen import '+name)
 if Path(native.__file__).resolve()!=UTILITY:raise ValueError('native transform import mismatch')
 atomic(out/'source_identity.json',{'commit':commit,'sha256':hashes})
def learning_rate(elapsed,budget):
 t=min(max(float(elapsed),0.),budget);warm=.05*budget
 return 2e-5+1.8e-4*t/warm if t<=warm else 2e-5+9e-5*(1+math.cos(math.pi*(t-warm)/(budget-warm)))
def model_spec(family,seed):
 if family not in ('pixel','latent') or seed not in SEEDS:raise ValueError('registered family and seed required')
 return {'family':family,'seed':seed,'constructor':'PixelFlowMatching' if family=='pixel' else 'RGBCodecFlowMatching','constructor_kwargs':{},'source_dimension':3072 if family=='pixel' else 1024,'source_distribution':'independent standard Gaussian','output_chart':'native_logits_no_normalization' if family=='pixel' else 'codec_RGB_minus1_plus1','field_width':128,'field_blocks':12 if family=='pixel' else 8,'fresh_initialization':True,'checkpoint79201_reuse':False,'streams':{'initialization':seed,'FIT_index_PCG64_reset_each_stage':seed+31,'CUDA_field_noise_and_time':seed+32}}
def make_model(family,seed):
 model_spec(family,seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 return PixelFlowMatching() if family=='pixel' else RGBCodecFlowMatching()
def checkpoint(path,model,optimizer,index_rng,path_rng,metadata,updates,elapsed,nominal):
 v={'architecture':metadata,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'index_rng':index_rng.bit_generator.state,'path_rng':None if path_rng is None else path_rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'updates':updates,'stage_elapsed_seconds':elapsed,'nominal_stage_seconds':nominal}
 tmp=path.with_suffix('.tmp');torch.save(v,tmp);os.replace(tmp,path)
def fit_stage(model,targets,ids,out,name,budget,points,seed,loss_callback,metadata,standalone_start,state,path_noise):
 """No update begins after deadline; preserve and charge final update/IO overrun."""
 device=targets.device;parameters=[p for p in model.parameters() if p.requires_grad]
 if not parameters:raise ValueError('no trainable parameters')
 before=time.perf_counter();optimizer=torch.optim.AdamW(parameters,lr=2e-4,betas=(.9,.99),weight_decay=1e-4);optimizer_seconds=time.perf_counter()-before
 index_rng=np.random.default_rng(seed+31);path_rng=torch.Generator(device=device).manual_seed(seed+32) if path_noise else None
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
 sync(device);start=time.perf_counter();updates=0;next_point=0;receipts=[]
 def save(nominal):
  path=out/(name+'_initial.pt' if nominal==0 else name+f'_{int(nominal):04d}.pt')
  elapsed=time.perf_counter()-start;checkpoint(path,model,optimizer,index_rng,path_rng,metadata,updates,elapsed,nominal);digest=sha(path)
  row={'path':path.name,'sha256':digest,'nominal_seconds':nominal,'updates':updates,'stage_seconds_through_hash':time.perf_counter()-start,'standalone_prefix_seconds_through_hash':time.perf_counter()-standalone_start}
  receipts.append(row);atomic(out/(name+'_checkpoint_receipts.json'),receipts)
 try:
  save(0)
  with (out/(name+'_steps.jsonl')).open('x') as ledger:
   while time.perf_counter()-start<budget:
    elapsed=time.perf_counter()-start;index=index_rng.integers(0,len(ids),size=32);lr=learning_rate(elapsed,budget)
    for group in optimizer.param_groups:group['lr']=lr
    draw={'event':'draw','step':updates+1,'stage_seconds':elapsed,'indices':index.tolist(),'fit_ids':np.asarray(ids[index]).tolist(),'lr':lr};ledger.write(json.dumps(draw)+'\n');ledger.flush();state.update(stage=name,draw=draw,updates=updates)
    optimizer.zero_grad(set_to_none=True);loss=loss_callback(targets[torch.as_tensor(index,device=device)],path_rng)
    if loss.ndim!=0:raise ValueError('scalar loss required')
    finite(loss);loss.backward()
    if any(p.grad is None for p in parameters):raise RuntimeError('missing intended gradient')
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise FloatingPointError('nonfinite updated parameter')
    sync(device);updates+=1;elapsed=time.perf_counter()-start
    ledger.write(json.dumps({'event':'completed','step':updates,'stage_seconds':elapsed,'loss':float(loss.detach())})+'\n');ledger.flush()
    while next_point<len(points) and time.perf_counter()-start>=points[next_point]:save(points[next_point]);next_point+=1
    atomic(out/(name+'_progress.json'),{'updates':updates,'stage_seconds':time.perf_counter()-start,'budget':budget})
  if not updates:raise RuntimeError('zero-update stage')
  while next_point<len(points):save(points[next_point]);next_point+=1
  elapsed=time.perf_counter()-start
  report={'updates':updates,'elapsed_seconds':elapsed,'overrun_seconds':max(0.,elapsed-budget),'budget_seconds':budget,'optimizer_constructor_seconds':optimizer_seconds,'checkpoints':receipts,'standalone_prefix_seconds':time.perf_counter()-standalone_start,'trainable_parameters':sum(p.numel() for p in parameters),'clock_boundary':'through final checkpoint hash and checkpoint receipt; final accounting write charged standalone'}
  if device.type=='cuda':report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
  atomic(out/(name+'_accounting.json'),report);return report
 except BaseException as error:
  original={'error':repr(error),'traceback':traceback.format_exc(),'state':state}
  try:checkpoint(out/(name+'_failed.pt'),model,optimizer,index_rng,path_rng,metadata,updates,time.perf_counter()-start,None)
  except BaseException as secondary:original['checkpoint_error']=repr(secondary)
  try:atomic(out/(name+'_failure.json'),original)
  except BaseException as secondary:print(repr(secondary)+'\n'+original['traceback'],file=sys.stderr)
  raise

@torch.no_grad()
def freeze_codec_and_cache(model,rgb,out,state):
 model.set_stage('normalization');model.eval();state['stage']='FIT_normalization_cache';start=time.perf_counter()
 raw=np.empty((len(rgb),16,8,8),dtype=np.float32)
 for first in range(0,len(rgb),64):
  x=rgb[first:first+64];model.update_normalization(x);raw[first:first+len(x)]=model.codec.encode(x).cpu().numpy()
 model.freeze_normalization();mean=model.latent_mean.cpu().numpy();std=model.latent_std.cpu().numpy();raw=(raw-mean[None,:,None,None])/std[None,:,None,None]
 if not np.isfinite(raw).all():raise FloatingPointError('nonfinite normalized cache')
 np.save(out/'fit_normalized_cache.npy',raw);np.save(out/'latent_mean.npy',mean);np.save(out/'latent_std.npy',std);torch.save(model.state_dict(),out/'codec_normalized_state.pt')
 if any(p.requires_grad for p in model.codec.parameters()) or not all(p.requires_grad for p in model.field.parameters()):raise RuntimeError('codec must freeze; field only trains')
 return torch.from_numpy(raw).to(rgb.device),{'seconds':time.perf_counter()-start,'encoder_passes_per_chunk':2,'cache_sha256':sha(out/'fit_normalized_cache.npy'),'state_sha256':sha(out/'codec_normalized_state.pt'),'FIT_only':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=('pixel','latent'),required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--expected-commit',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);main_started=time.perf_counter();started=int(os.environ.get('RGB_PROCESS_START_NS','0'))/1e9;state={};model=None
 report={'scope':'FIT_ONLY_FULL_GENERATION_BASELINE_NO_EVALUATION','family':args.family,'seed':args.seed,'repair_physically_constructed_by_loader':True,'repair_used':False,'global_all_fit_barrier_created':False}
 try:
  if not 0<started<=main_started:raise ValueError('run.sh pre-import monotonic process clock required')
  report['pre_main_import_startup_seconds']=main_started-started
  source_guard(args.expected_commit,out)
  if not torch.cuda.is_available():raise RuntimeError('CUDA required')
  gpu=torch.cuda.get_device_properties(0)
  if 'V100' not in gpu.name or gpu.total_memory<30*1024**3:raise RuntimeError('V10032 required')
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  def stop(signum,frame):raise TimeoutError('one-hour allocation ended; no extension/retry')
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
  atomic(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':gpu.name,'job':os.environ.get('SLURM_JOB_ID')})
  with (out/'preflight.txt').open('w') as f:subprocess.run([sys.executable,'-m','pytest','-q','qalt/tests/test_rgb_full_generation_fit.py','qalt/tests/test_rgb_codec_flow_matching.py'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'qalt/src')},stdout=f,stderr=subprocess.STDOUT,check=True)
  state['stage']='canonical_FIT_loading';t=time.perf_counter();data=load_observed_flow_data(DATA_ROOT)
  if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture'] or data.fit.shape!=(4000,3,32,32):raise ValueError('strict canonical4000FIT required')
  ids=data.fit_ids.copy();np.save(out/'fit_ids.npy',ids);atomic(out/'data_ledger.json',data.ledger)
  original_sha=hashlib.sha256(np.ascontiguousarray(data.fit).tobytes()).hexdigest()
  targets=native.logit_inputs(data.fit)[0] if args.family=='pixel' else 2*torch.as_tensor(np.array(data.fit,dtype=np.float32))-1
  np.save(out/'fit_model_targets.npy',targets.numpy());targets=targets.cuda();del data
  binding={'original_FIT_float64_sha256':original_sha,'fit_ids_sha256':sha(out/'fit_ids.npy'),'model_targets_sha256':sha(out/'fit_model_targets.npy'),'transform':'native.logit_inputs(FIT)[0] exact; no normalization' if args.family=='pixel' else '2*float32(FIT)-1','rows':4000};atomic(out/'input_binding.json',binding);report['loader_target_seconds']=time.perf_counter()-t
  state['stage']='fresh_initialization';t=time.perf_counter();model=make_model(args.family,args.seed).cuda();sync(targets.device);report['initialization_seconds']=time.perf_counter()-t;metadata=model_spec(args.family,args.seed)
  report['parameter_counts']={'complete':sum(p.numel() for p in model.parameters()),'field':sum(p.numel() for p in model.field.parameters()),'codec':sum(p.numel() for p in model.codec.parameters()) if args.family=='latent' else 0};metadata['parameter_counts']=report['parameter_counts'];metadata['input_binding']=binding;metadata['source_commit']=args.expected_commit
  report['stages']={}
  if args.family=='latent':
   report['stages']['codec']=fit_stage(model,targets,ids,out,'codec',CODEC_SECONDS,CODEC_POINTS,args.seed,lambda x,g:model.training_loss(x),metadata,started,state,False)
   targets,report['normalization_cache']=freeze_codec_and_cache(model,targets,out,state);metadata['normalization_cache']=report['normalization_cache']
   callback=model.training_loss_from_latents;points=LATENT_POINTS
  else:callback=model.training_loss;points=PIXEL_POINTS
  report['stages']['field']=fit_stage(model,targets,ids,out,'field',FIELD_SECONDS,points,args.seed,callback,metadata,started,state,True)
  model.set_stage('inference');state['stage']='final_freeze';path=out/'pipeline_final.pt';torch.save({'architecture':metadata,'model':model.state_dict()},path)
  atomic(out/'FIT_COMPLETE.json',{'family':args.family,'seed':args.seed,'source_commit':args.expected_commit,'final_checkpoint':path.name,'sha256':sha(path),'selected_field_checkpoints':report['stages']['field']['checkpoints'][1:],'input_binding':binding,'standalone_seconds_through_final_hash':time.perf_counter()-started,'single_fit_only_not_global_barrier':True});report['status']='completed_single_fit_no_evaluation'
 except BaseException as error:
  report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),state=state)
  if model is not None:
   try:torch.save(model.state_dict(),out/'failure_model_state.pt')
   except BaseException as secondary:report['failure_save_error']=repr(secondary)
  raise
 finally:
  report['cpu_peak_rss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  report['payload_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='status.json'};report['main_wall_seconds']=time.perf_counter()-main_started;report['full_standalone_wall_seconds']=time.perf_counter()-started if started>0 else None;report['clock_boundary']='stdlib bootstrap before heavy imports through payload hashes; excludes bootstrap interpreter startup, final status write/teardown; terminal worker/allocation accounting also required'
  try:atomic(out/'status.json',report)
  except BaseException as secondary:
   if report.get('status')=='failed':print(repr(secondary)+'\n'+report['traceback'],file=sys.stderr)
   else:raise
if __name__=='__main__':main()
