"""Data-free, bounded GPU component readiness; never imports an image loader."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'qalt/src'))
from qalt.rgb_codec_flow_matching import RGBCodecFlowMatching,PixelFlowMatching
SEED=2026090920
FILES=('qalt/src/qalt/__init__.py','qalt/src/qalt/core.py','qalt/src/qalt/rgb_codec_flow_matching.py','qalt/tests/test_rgb_codec_flow_matching.py','qalt/experiments/rgb_codec_readiness/run.py','qalt/experiments/rgb_codec_readiness/PROTOCOL.md','qalt/experiments/rgb_codec_readiness/run.sh')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,value):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sync():torch.cuda.synchronize()
def timed(fn):
 sync();start=time.perf_counter();result=fn();sync();return result,time.perf_counter()-start
def counts(m):return sum(p.numel() for p in m.parameters())
def memory():
 try:return {'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved(),'resident_allocated':torch.cuda.memory_allocated(),'resident_reserved':torch.cuda.memory_reserved()}
 except BaseException as error:return {'memory_query_error':repr(error)}
def step(loss_function,parameters,optimizer):
 optimizer.zero_grad(set_to_none=True);loss=loss_function()
 if loss.ndim or not bool(torch.isfinite(loss)):raise FloatingPointError('invalid scalar loss')
 loss.backward()
 if any(p.grad is None for p in parameters):raise RuntimeError('missing intended gradient')
 if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):raise FloatingPointError('nonfinite gradient')
 optimizer.step()
 if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise FloatingPointError('nonfinite updated parameter')
 return float(loss.detach())
def train_profile(model,loss32,loss125,out,state,label):
 parameters=[p for p in model.parameters() if p.requires_grad];assert parameters
 model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
 optimizer=torch.optim.AdamW(parameters,lr=2e-4,betas=(.9,.99),weight_decay=1e-4)
 record={'scope':'synthetic numerical/timing screen, no quality','intended_parameter_tensors':len(parameters),'intended_parameter_elements':sum(p.numel() for p in parameters),'warmup':[],'timed':[]}
 for i in range(25):
  state.update(stage=label,iteration=i)
  try:loss,seconds=timed(lambda:step(loss32,parameters,optimizer))
  except BaseException as error:
   record.update(failed_iteration=i,error=repr(error),failure_memory=memory());atomic(out/(label+'.json'),record);raise
  record['warmup' if i<5 else 'timed'].append({'loss':loss,'seconds':seconds})
  atomic(out/(label+'.json'),record)
 record['batch32_memory']=memory();atomic(out/(label+'.json'),record);torch.cuda.reset_peak_memory_stats()
 state.update(stage=label+'_batch125',iteration=0)
 try:loss,seconds=timed(lambda:step(loss125,parameters,optimizer))
 except BaseException as error:
  record['batch125_failure']={'error':repr(error),'memory':memory()};atomic(out/(label+'.json'),record);raise
 record['batch125_one_step']={'loss':loss,'seconds':seconds,'memory':memory()}
 values=[x['seconds'] for x in record['timed']];record.update(batch32_p50_seconds=float(np.median(values)),batch32_p90_seconds=float(np.quantile(values,.9)))
 atomic(out/(label+'.json'),record);del optimizer;model.zero_grad(set_to_none=True)
 return record
def checkpoint(model,out,label):
 def save():
  torch.save(model.state_dict(),out/(label+'.pt'))
  with (out/(label+'.pt')).open('rb') as f:os.fsync(f.fileno())
 _,seconds=timed(save)
 return {'seconds':seconds,'bytes':(out/(label+'.pt')).stat().st_size,'sha256':sha(out/(label+'.pt'))}
def sample_profile(model,out,state,label,generator):
 model.set_stage('inference');result={'steps':32,'field_calls':64,'includes_decoder':label=='latent','encoder_retained_on_gpu':label=='latent','memory_scope':'whole resident model plus shared synthetic inputs; not deployment-minimal memory','batches':{}}
 for batch in (1,32):
  source=torch.randn(batch,model.source_dimension,generator=generator).cuda();np.save(out/f'{label}_source_b{batch}.npy',source.cpu().numpy())
  torch.cuda.reset_peak_memory_stats();timings=[]
  for i in range(4):
   state.update(stage=label+'_sampling',batch=batch,iteration=i)
   try:values,seconds=timed(lambda:model.sample_from_gaussian(source,steps=32))
   except BaseException as error:
    result['failure']={'batch':batch,'iteration':i,'previous_seconds':timings,'error':repr(error),'memory':memory()};atomic(out/(label+'_sampling.json'),result);raise
   assert values.shape==(batch,3,32,32) and bool(torch.isfinite(values).all())
   timings.append(seconds)
   saved=values.cpu().numpy();np.save(out/f'{label}_synthetic_output_b{batch}_iteration{i}.npy',saved)
   if i==3:np.save(out/f'{label}_synthetic_output_b{batch}.npy',saved)
   del values,saved
   result['progress']={'batch':batch,'completed_iterations':i+1,'seconds':timings};atomic(out/(label+'_sampling.json'),result)
  result['batches'][str(batch)]={'warmup_seconds':timings[0],'timed_seconds':timings[1:],'memory':memory()}
  atomic(out/(label+'_sampling.json'),result)
 return result

def reload_check(factory,out,label):
 state=torch.load(out/(label+'_synthetic_checkpoint.pt'),map_location='cpu',weights_only=True)
 clone=factory().cuda();clone.load_state_dict(state);clone.set_stage('inference')
 source=torch.as_tensor(np.load(out/f'{label}_source_b1.npy'),device='cuda')
 actual,seconds=timed(lambda:clone.sample_from_gaussian(source,steps=32))
 expected=np.load(out/f'{label}_synthetic_output_b1.npy')
 copied=actual.cpu().numpy();np.save(out/(label+'_reloaded_synthetic_output.npy'),copied)
 assert np.array_equal(copied,expected),'same-device reload must reproduce synthetic output exactly'
 del clone,actual,state;torch.cuda.empty_cache()
 return {'exact':True,'batch':1,'seconds_sampling_only':seconds,'scope':'separate numerical qualification, excluded from warmed sampling timings'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-commit',required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();state={};report={'scope':'DATA_FREE_COMPONENT_READINESS_ONLY','real_data_accessed':False,'seed':SEED};model=None
 try:
  assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==a.expected_commit
  identity={}
  for name in FILES:
   raw=(ROOT/name).read_bytes();assert raw==subprocess.check_output(['git','-C',str(ROOT),'show',a.expected_commit+':'+name])
   target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);identity[name]=sha(target)
  atomic(out/'source_identity.json',{'commit':a.expected_commit,'sha256':identity})
  assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
  device=torch.cuda.get_device_properties(0);assert 'V100' in device.name and device.total_memory>=30*1024**3,'typed V100-32 required'
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  def stop(signum,frame):raise TimeoutError('allocation termination; no retry')
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
  torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);g=torch.Generator().manual_seed(SEED+1);training_rng=torch.Generator(device='cuda').manual_seed(SEED+2)
  report['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'host':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),'threads':torch.get_num_threads(),'tf32':False,'amp':False}
  atomic(out/'environment.json',report['environment'])
  state['stage']='preflight'
  with (out/'preflight.txt').open('w') as log:subprocess.run([sys.executable,'-m','pytest','-q','qalt/tests/test_rgb_codec_flow_matching.py'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'qalt/src')},stdout=log,stderr=subprocess.STDOUT,check=True)
  images=2*torch.rand(256,3,32,32,generator=g)-1;np.save(out/'synthetic_inputs.npy',images.numpy());x32=images[:32].cuda();x125=images[:125].cuda()
  state['stage']='codec_initialization';model,seconds=timed(lambda:RGBCodecFlowMatching().cuda())
  assert counts(model.codec.encoder)==1728272 and counts(model.codec.decoder)==1744643 and counts(model.field)==3011472 and counts(model)==6484387
  report['latent_initialization_seconds']=seconds;report['counts']={'codec_encoder':counts(model.codec.encoder),'codec_decoder':counts(model.codec.decoder),'latent_field':counts(model.field),'latent_pipeline_total':counts(model)}
  train_profile(model,lambda:model.training_loss(x32),lambda:model.training_loss(x125),out,state,'codec_train')
  model.set_stage('normalization');state['stage']='synthetic_normalization'
  def normalize():
   for start in range(0,256,32):model.update_normalization(images[start:start+32].cuda())
   model.freeze_normalization()
  _,report['synthetic_normalization_seconds']=timed(normalize)
  state['stage']='synthetic_cache'
  def cache():
   values=[]
   for start in range(0,256,32):values.append(model.encode_normalized(images[start:start+32].cuda()).cpu())
   z=torch.cat(values);np.save(out/'synthetic_normalized_cache.npy',z.numpy())
   with (out/'synthetic_normalized_cache.npy').open('rb') as f:os.fsync(f.fileno())
   return z.cuda()
  z,seconds=timed(cache);report['synthetic_cache']={'count':256,'seconds_including_encode_transfer_write':seconds,'bytes':(out/'synthetic_normalized_cache.npy').stat().st_size,'sha256':sha(out/'synthetic_normalized_cache.npy')}
  train_profile(model,lambda:model.training_loss_from_latents(z[:32],training_rng),lambda:model.training_loss_from_latents(z[:125],training_rng),out,state,'latent_train')
  report['latent_checkpoint']=checkpoint(model,out,'latent_synthetic_checkpoint');report['latent_sampling']=sample_profile(model,out,state,'latent',g)
  report['latent_reload']=reload_check(RGBCodecFlowMatching,out,'latent')
  model.cpu();del model,z;model=None;torch.cuda.empty_cache()
  state['stage']='pixel_initialization';model,seconds=timed(lambda:PixelFlowMatching().cuda());assert counts(model)==4427395
  report['pixel_initialization_seconds']=seconds;report['counts']['pixel_total']=counts(model)
  train_profile(model,lambda:model.training_loss(x32,training_rng),lambda:model.training_loss(x125,training_rng),out,state,'pixel_train')
  report['pixel_checkpoint']=checkpoint(model,out,'pixel_synthetic_checkpoint');report['pixel_sampling']=sample_profile(model,out,state,'pixel',g)
  report['pixel_reload']=reload_check(PixelFlowMatching,out,'pixel')
  report.update(status='completed_data_free_readiness',elapsed_seconds=time.perf_counter()-started,cpu_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 except BaseException as error:
  report.update(status='failed',state=state,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
  if torch.cuda.is_available():report['failure_memory']=memory()
  if model is not None:
   try:torch.save(model.state_dict(),out/'failed_synthetic_checkpoint.pt')
   except BaseException as second:report['checkpoint_error']=repr(second)
  atomic(out/'status.json',report);raise
 finally:
  report['payload_sha256']={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='status.json'}
  atomic(out/'status.json',report)
if __name__=='__main__':main()
