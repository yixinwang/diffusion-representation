"""Complete3072-coordinate diagnostic only; no optimization or quality evaluation."""
import hashlib,json,math,traceback
from pathlib import Path
import torch


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def finite(t):return bool(torch.isfinite(t).all())
def snapshot(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def unchanged(model,before):return all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
def saved_gradients(model):return {name:None if p.grad is None else p.grad.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
def grads_ok(values):return bool(values) and all(v is not None and finite(v) for v in values.values())

def qualify(runner,checkpoint,kernel,logits,out,device,checkpoint_sha256):
    out=Path(out);out.mkdir(exist_ok=False);report={'scope':'full_model_'+device,'source_dimension':3072,'training_performed':False,'repair_evaluated':False,'optimizer_created':False,'optimizer_steps':0,'checkpoint_sha256':checkpoint_sha256,'source_seed':78300}
    def restore():
        model=runner.restore_model(checkpoint).to(device)
        model.residual_decoder._spline=kernel
        runner.freeze(model)
        return model
    try:
        model=restore();before=snapshot(model)
        source=torch.randn(8,3072,generator=torch.Generator(device='cpu').manual_seed(78300),dtype=torch.float32).to(device)
        observed=logits.detach().to(device)
        if observed.shape!=(32,3,32,32):raise ValueError('exact32 observed logits required')
        partial={'source':source.cpu(),'observed_logits':observed.cpu()}
        def preserve(**values):
            partial.update({k:v.detach().cpu() for k,v in values.items()});torch.save(partial,out/'partial_numerical_bank.pt')
        preserve()
        with torch.no_grad():
            generated,dld=model.decode(source);preserve(generated_logits=generated,source_decode_ld=dld)
            recovered,eld=model.encode(generated);preserve(recovered_source=recovered,source_encode_ld=eld)
            observed_source,oeld=model.encode(observed);preserve(observed_encoded_source=observed_source,observed_encode_ld=oeld)
            observed_recovered,odld=model.decode(observed_source);preserve(observed_recovered_logits=observed_recovered,observed_decode_ld=odld)
            reloaded=restore();copy_source,copy_ld=reloaded.decode(source);preserve(exact_reload_logits=copy_source,exact_reload_ld=copy_ld)
            exact_reload=torch.equal(generated,copy_source) and torch.equal(dld,copy_ld)
        bank={'source':source.cpu(),'generated_logits':generated.cpu(),'recovered_source':recovered.cpu(),'source_decode_ld':dld.cpu(),'source_encode_ld':eld.cpu(),'observed_logits':observed.cpu(),'observed_encoded_source':observed_source.cpu(),'observed_recovered_logits':observed_recovered.cpu(),'observed_encode_ld':oeld.cpu(),'observed_decode_ld':odld.cpu(),'exact_reload_logits':copy_source.cpu(),'exact_reload_ld':copy_ld.cpu()}
        torch.save(bank,out/'numerical_bank.pt');torch.save(source.cpu(),out/'source_bank.pt')
        report.update(source_bank_sha256=file_sha(out/'source_bank.pt'),numerical_bank_sha256=file_sha(out/'numerical_bank.pt'),source_values_sha256=tensor_sha(source),finite=all(finite(v) for v in bank.values()),source_roundtrip_max_abs=float((source-recovered).abs().max()),source_logdet_cancellation_max_abs=float((dld+eld).abs().max()),observed_roundtrip_max_abs=float((observed-observed_recovered).abs().max()),observed_logdet_cancellation_max_abs=float((oeld+odld).abs().max()),exact_reload=exact_reload,numerical_model_state_unchanged=unchanged(model,before))
        invalid={};invalid_arrays={}
        for name,method,value in [('source_nan',model.decode,source.clone()),('observed_nan',model.encode,observed.clone()),('source_shape',model.decode,source[:,:-1]),('observed_shape',model.encode,observed[:,:,:,:-1]),('source_dtype',model.decode,source.double()),('observed_dtype',model.encode,observed.double())]:
            if name.endswith('_nan'):value.flatten()[0]=float('nan')
            invalid_arrays[name]=value.detach().cpu()
            try:
                with torch.no_grad():method(value)
                invalid[name]=False
            except (ValueError,FloatingPointError):invalid[name]=True
        torch.save(invalid_arrays,out/'invalid_input_bank.pt')
        report['invalid_input_checks']=invalid;report['invalid_inputs_rejected']=all(invalid.values())
        # Configure precisely the original joint branch: A and residual active,
        # root parameter weights frozen, root context/input graph still live.
        joint=restore();runner.activate(joint,'residual_decoder');joint.unfreeze_analysis()
        expected_analysis_ids={id(p) for p in joint._analysis_parameters()}
        expected_residual_ids={id(p) for p in joint.residual_decoder.parameters()}
        expected_analysis_names={name for name,p in joint.named_parameters() if id(p) in expected_analysis_ids}
        expected_residual_names={name for name,p in joint.named_parameters() if id(p) in expected_residual_ids}
        actual_trainable_names={name for name,p in joint.named_parameters() if p.requires_grad}
        if actual_trainable_names!=expected_analysis_names|expected_residual_names:raise ValueError('joint trainable set differs from exact A plus residual parameter union')
        report['expected_analysis_parameter_names']=sorted(expected_analysis_names)
        report['expected_residual_parameter_names']=sorted(expected_residual_names)
        report['actual_trainable_parameter_names']=sorted(actual_trainable_names)
        joint_before=snapshot(joint);root_before=snapshot(joint.coarse_decoder)
        input_logits=observed.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            coarse,residual,a_ld=runner.analysis_parts(joint,input_logits)
            coarse.retain_grad();residual.retain_grad()
            # Identity clone separates root-input derivative from the residual
            # context derivative without changing any forward floating value.
            root_input=coarse.clone();root_input.retain_grad()
            zr,rld=joint.residual_decoder.encode(residual,coarse)
            zc,cld=joint.coarse_decoder.encode(root_input,joint._zero(root_input))
            z=torch.cat((zc.flatten(1),zr.flatten(1)),dim=1)
            ld=a_ld+rld+cld
            loss=(.5*(z.square()+math.log(2*math.pi)).sum(1)-ld).mean()/3072
            torch.save({'logits':input_logits.detach().cpu(),'coarse':coarse.detach().cpu(),'residual':residual.detach().cpu(),'root_input':root_input.detach().cpu(),'encoded':z.detach().cpu(),'analysis_ld':a_ld.detach().cpu(),'residual_ld':rld.detach().cpu(),'coarse_ld':cld.detach().cpu()},out/'joint_forward.pt')
            report['joint_forward_finite']=all(finite(t) for t in (input_logits,coarse,residual,root_input,zr,rld,zc,cld,z,ld,a_ld,loss))
            report['joint_scalar_loss_finite']=finite(loss)
            report['finite']=report['finite'] and report['joint_forward_finite']
            if not report['joint_forward_finite']:raise FloatingPointError('nonfinite complete joint forward/loss; no backward accepted')
            loss.backward()
        gradients=saved_gradients(joint);torch.save(gradients,out/'joint_parameter_gradients.pt')
        inputs={k:None if v.grad is None else v.grad.detach().cpu() for k,v in [('logits',input_logits),('coarse',coarse),('residual',residual),('fixed_root_input',root_input)]};torch.save(inputs,out/'joint_input_gradients.pt')
        analysis={k:v for k,v in gradients.items() if k.startswith(('pre_analysis.','analysis.'))};residual_grad={k:v for k,v in gradients.items() if k.startswith('residual_decoder.')}
        root_frozen=all(not p.requires_grad and p.grad is None for p in joint.coarse_decoder.parameters())
        report.update(joint_gradients_finite_and_present=grads_ok(gradients),analysis_gradients_finite_and_present=grads_ok(analysis),residual_parameter_gradients_finite_and_present=grads_ok(residual_grad),residual_input_gradients_finite_and_present=grads_ok({'residual':inputs['residual']}),coarse_input_gradients_finite_and_present=grads_ok({'coarse':inputs['coarse']}),fixed_root_input_gradients_finite_and_present=grads_ok({'root':inputs['fixed_root_input']}),observed_input_gradients_finite_and_present=grads_ok({'logits':inputs['logits']}),root_parameters_frozen=root_frozen,root_parameters_unchanged=unchanged(joint.coarse_decoder,root_before),joint_model_state_unchanged=unchanged(joint,joint_before),joint_parameter_gradient_count=len(gradients),analysis_parameter_gradient_count=len(analysis),residual_parameter_gradient_count=len(residual_grad),gradient_nonzero={k:v is not None and bool((v!=0).any()) for k,v in inputs.items()})
        report['state_unchanged']=report['numerical_model_state_unchanged'] and report['joint_model_state_unchanged'] and report['root_parameters_unchanged']
        report['roundtrip_gate']=report['source_roundtrip_max_abs']<=1e-3 and report['observed_roundtrip_max_abs']<=1e-3
        report['logdet_gate']=report['source_logdet_cancellation_max_abs']<=1e-2 and report['observed_logdet_cancellation_max_abs']<=1e-2
        report['status']='captured'
    except Exception as error:
        if 'joint' in locals():torch.save(saved_gradients(joint),out/'partial_joint_parameter_gradients.pt')
        report.update(status='qualification_failed',error=repr(error),traceback=traceback.format_exc())
    report['payload_sha256']={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()}
    report=json_safe(report)
    (out/'status.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
