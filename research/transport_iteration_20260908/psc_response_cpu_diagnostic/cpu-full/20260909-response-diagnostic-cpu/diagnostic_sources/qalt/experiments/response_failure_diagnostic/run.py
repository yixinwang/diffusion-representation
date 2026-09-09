"""Read-only failed RQS forward capture; no fitting or quality evaluation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, signal, subprocess, sys, traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
REFERENCE='168efc227e361a86a2a6aa6952786e5d0e13e30f'
FAILURE_ROOT=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-innovation-response')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def captured_call(function,*args,**kwargs):
    """Capture actual unchanged Python function locals at return, including masks."""
    if sys.gettrace() is not None:raise RuntimeError('existing tracing forbidden')
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is function.__code__:
            if event=='return':captured.update({k:v.detach().cpu().clone() for k,v in frame.f_locals.items() if isinstance(v,torch.Tensor)})
            return trace
        return None
    sys.settrace(trace)
    try:result=function(*args,**kwargs)
    finally:sys.settrace(None)
    return result,captured

def recover_draw(ids,updates,checkpoint_rng,expected_hash):
    rng=np.random.default_rng(78231);h=hashlib.sha256()
    for _ in range(updates+1):
        index=rng.integers(0,len(ids),size=32);h.update(np.asarray(ids[index],dtype='<i8').tobytes())
    if rng.bit_generator.state!=checkpoint_rng or h.hexdigest()!=expected_hash:raise ValueError('failed draw provenance mismatch')
    return index

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expected-commit',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);out=args.output;record={}
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if len(args.expected_commit)!=40 or head!=args.expected_commit:raise ValueError('diagnostic HEAD mismatch')
        hashes={}
        diagnostic_files=list(Path(__file__).parent.iterdir())+[ROOT/'qalt/tests/test_response_failure_diagnostic.py']
        for p in sorted(diagnostic_files):
            if p.suffix not in ['.py','.md','.slurm']:continue
            rel=p.relative_to(ROOT);raw=p.read_bytes()
            if raw!=subprocess.check_output(['git','show',f'{head}:{rel}'],cwd=ROOT):raise ValueError('unfrozen diagnostic source')
            dest=out/'diagnostic_sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);hashes[str(rel)]=digest(p)
        if digest(FAILURE_ROOT/'status.json')!='015c5b732d1d5de7b4ca6a55f8b4ebca428bffbc1488f52ba0675b856ec1f588':raise ValueError('pinned failure status mismatch')
        identity=json.loads((FAILURE_ROOT/'source_identity.json').read_text());status=json.loads((FAILURE_ROOT/'status.json').read_text())
        if identity['commit']!=REFERENCE or status['state']!={'phase':'RQS_prefix','seed':78201}:raise ValueError('wrong failure identity')
        # Verify every archived source against its original Git blob before import.
        source=FAILURE_ROOT/'sources'
        for rel,h in identity['sha256'].items():
            p=source/rel
            if digest(p)!=h or p.read_bytes()!=subprocess.check_output(['git','show',f'{REFERENCE}:{rel}'],cwd=ROOT):raise ValueError('reference source mismatch')
        for rel,h in status['payload_sha256'].items():
            if digest(FAILURE_ROOT/rel)!=h:raise ValueError('failure payload mismatch')
        write(out/'source_identity.json',{'diagnostic_commit':head,'diagnostic_sha256':hashes,'reference_commit':REFERENCE,'reference_sha256':identity['sha256'],'failure_status_sha256':digest(FAILURE_ROOT/'status.json'),'verified_failure_payload_count':len(status['payload_sha256'])})
        if any(k=='qalt' or k.startswith('qalt.') for k in sys.modules):raise ValueError('qalt must not be imported before reference guard')
        sys.path.insert(0,str(source/'qalt/src'))
        spec=importlib.util.spec_from_file_location('frozen_response_runner',source/'qalt/experiments/innovation_response_pilot/run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        for name,module in list(sys.modules.items()):
            if name=='qalt' or name.startswith('qalt.'):
                if not Path(module.__file__).resolve().is_relative_to(source.resolve()):raise ValueError('reference import escaped archived sources')
        if torch.__version__!='2.10.0+cu128':raise ValueError('same PSC Torch build required')
        if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def stop(signum,frame):raise TimeoutError('diagnostic scheduler warning')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
        failed=FAILURE_ROOT/'seed_78201/RQS_prefix_failed.pt';ck=torch.load(failed,map_location='cpu',weights_only=True);progress=json.loads((FAILURE_ROOT/'seed_78201/RQS_prefix_progress.json').read_text())
        saved_ids=np.load(FAILURE_ROOT/'record_ids.npz')['fit'];index=recover_draw(saved_ids,progress['updates'],ck['rng'],progress['record_order_sha256'])
        np.savez(out/'batch_identity.npz',fit_subset_indices=index,canonical_record_ids=saved_ids[index],original_chunk_starts=np.array(sorted(set((index//64*64).tolist()))))
        # Frozen strict loader: selected repair is loaded as historically, never evaluated.
        data=runner.load_observed_flow_data(runner.DATA_ROOT)
        if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture'] or not np.array_equal(data.fit_ids,saved_ids):raise ValueError('canonical fit identity mismatch')
        original=json.loads((FAILURE_ROOT/'fitting_input_identity.json').read_text());logits,outer=runner.utility.logit_inputs(data.fit)
        if hashlib.sha256(np.ascontiguousarray(data.fit).tobytes()).hexdigest()!=original['observations_sha256'] or hashlib.sha256(logits.numpy().tobytes()).hexdigest()!=original['logits_sha256']:raise ValueError('fit tensor identity mismatch')
        model=runner.restore_model(ck).to(args.device)
        runner.activate(model,'residual_decoder')  # original prefix modes; no optimizer is created
        before=runner.state_fingerprint(model)
        write(out/'module_modes.json',{name:module.training for name,module in model.named_modules()})
        # Reproduce original 64-row A calls, preserving global cache position.
        # Compute only chunks containing the failed 32 positions; each call still has
        # its original 64 neighbours, hence no change to convolution batch shape.
        coarse={};residual={};ald={}
        with torch.no_grad():
            for first in sorted(set((index//64*64).tolist())):
                x=logits[first:first+64].to(args.device)
                c,r,a=runner.analysis_parts(model,x)
                for pos in index[(index>=first)&(index<first+64)]:coarse[int(pos)]=c[int(pos)-first];residual[int(pos)]=r[int(pos)-first];ald[int(pos)]=a[int(pos)-first]
            c=torch.stack([coarse[int(i)] for i in index]);r=torch.stack([residual[int(i)] for i in index]);a=torch.stack([ald[int(i)] for i in index])
            torch.save({'logits':logits[index].cpu(),'coarse':c.cpu(),'residual':r.cpu(),'analysis_ld':a.cpu()},out/'actual_batch.pt')
            decoder=model.residual_decoder;kernel=decoder._spline;layers=list(reversed(decoder.layers));captures=[];handles=[]
            for number,layer in enumerate(layers):
                def hook(module,inputs,output,number=number,layer=layer):
                    torch.save({'conditioner_raw':output.detach().cpu(),'fixed':inputs[0].detach().cpu(),'context':inputs[2].detach().cpu(),'mask':layer.mask.detach().cpu()},out/f'layer_{number}_conditioner.pt')
                handles.append(layer.conditioner.register_forward_hook(hook))
            def observed_kernel(*a,**kw):
                number=len(captures);result,locals_=captured_call(kernel,*a,**kw)
                torch.save(locals_,out/f'layer_{number}_kernel_locals.pt')
                inside=locals_['inside'];summary={'layer_in_encode_order':number,'valid':bool(result[2]),'scalar_count':inside.numel(),'inside_count':int(inside.sum()),'invalid_masks':{}}
                for name,value in locals_.items():
                    if name.startswith('good_') and value.dtype==torch.bool and value.shape==inside.shape:
                        bad=inside&~value;summary['invalid_masks'][name]={'count':int(bad.sum()),'coordinates':bad.nonzero().tolist()}
                captures.append(summary);write(out/'layers.json',captures)
                return result
            decoder._spline=observed_kernel
            try:
                z,ld=decoder.encode(r,c);torch.save({'z':z.cpu(),'ld':ld.cpu()},out/'encoded.pt');record['decoder_outcome']='returned_valid'
            except FloatingPointError as error:record['decoder_outcome']='numerical_guard_raised';record['decoder_exception']=str(error)
            finally:
                decoder._spline=kernel
                for handle in handles:handle.remove()
        if runner.state_fingerprint(model)!=before or digest(failed)!=status['payload_sha256']['seed_78201/RQS_prefix_failed.pt']:raise ValueError('model/checkpoint mutated')
        record.update(status='completed_diagnostic',device=args.device,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),host=platform.node(),gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,successful_updates=progress['updates'],failed_draw=progress['updates']+1,optimizer_steps=sorted(set(int(v['step'].item()) for v in ck['optimizer']['state'].values())),state_unchanged=True,repair_evaluated=False,training_performed=False,layers=captures)
    except BaseException as error:
        record.update(status='diagnostic_failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        record['payload_sha256']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'};write(out/'status.json',record)

if __name__=='__main__':main()
