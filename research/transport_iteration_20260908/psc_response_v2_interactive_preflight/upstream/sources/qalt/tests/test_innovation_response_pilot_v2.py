"""Fabricated protocol/fork tests only; no canonical data or full training."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PATH=Path(__file__).parents[1]/'experiments/innovation_response_pilot_v2/run.py'
spec=importlib.util.spec_from_file_location('response_pilot_v2_tests_runner',PATH)
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)

@pytest.fixture(scope='module')
def template():
    torch.manual_seed(91);model=run.prepared_model();model.freeze_analysis();run.freeze(model);return model


def test_seven_arm_counts_modes_and_initial_state(template):
    assert run.SEEDS==(78201,78202,78203)
    assert len(run.ARMS)==7 and len(set(run.ARMS))==7
    models={k:run.make_family(template,k,9) for k in ('P','I','RQS','S42')}
    assert run.state_fingerprint(models['P'])==run.state_fingerprint(models['I'])
    assert run.model_spec(models['P'])['response_mode']=='prefix'
    assert run.model_spec(models['I'])['response_mode']=='innovation'
    assert run.model_spec(models['RQS'])['backend']=='dense_reflected'
    for k,m in models.items():assert m.parameter_counts['total']==run.EXPECTED_COUNTS[k]
    assert models['P'].parameter_counts['total']<=models['S42'].parameter_counts['total']<=1.05*models['P'].parameter_counts['total']


@pytest.mark.parametrize('family',['P','I','RQS','S42'])
def test_checkpoint_mode_and_tensor_roundtrip(template,family,tmp_path):
    model=run.make_family(template,family,19)
    if family in ('P','I'):
        with torch.no_grad():
            for response in model.residual_decoder.responses:response.output.weight.normal_(0,.02)
    if family=='RQS':
        with torch.no_grad():
            for layer in model.residual_decoder.layers:
                layer.conditioner.output.weight.normal_(0,.01)
                layer.conditioner.output.bias.normal_(0,.01)
    path=tmp_path/'frozen.pt';run.checkpoint(path,model)
    record=torch.load(path,weights_only=True);restored=run.restore_model(record)
    assert run.model_spec(restored)==record['architecture']
    assert run.state_fingerprint(restored)==run.state_fingerprint(model)
    source=torch.randn(1,3072,generator=torch.Generator().manual_seed(21))
    with torch.no_grad():
        for item in (model,restored):
            if hasattr(item.residual_decoder,'prepare_inference'):item.residual_decoder.prepare_inference()
        left,ld=model.decode(source);right,rd=restored.decode(source)
    assert torch.equal(left,right) and torch.equal(ld,rd)
    bad=copy.deepcopy(record);bad['architecture']['backend']='compiled'
    with pytest.raises(ValueError):run.restore_model(bad)


@pytest.mark.parametrize('family',['P','I','RQS'])
def test_family_fork_preserves_adam_rng_without_aliasing(family):
    torch.manual_seed(44);model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(123);x=torch.tensor(rng.normal(size=(4,3)),dtype=torch.float32)
    model(x).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    fork,state,rng_state=run.fork_training_state(model,optimizer,rng)
    original=run.optimizer_fingerprint(optimizer.state_dict())
    assert run.optimizer_fingerprint(state)==original
    restored=torch.optim.Adam(fork.parameters(),lr=.001);restored.load_state_dict(state)
    assert run.optimizer_fingerprint(restored.state_dict())==original
    left=np.random.default_rng();left.bit_generator.state=copy.deepcopy(rng_state)
    right=np.random.default_rng();right.bit_generator.state=copy.deepcopy(rng_state)
    assert np.array_equal(left.integers(0,4000,32),right.integers(0,4000,32))
    before=copy.deepcopy(fork.state_dict())
    with torch.no_grad():model.weight.add_(1)
    assert all(torch.equal(v,fork.state_dict()[k]) for k,v in before.items())
    next(iter(optimizer.state.values()))['exp_avg'].add_(1)
    assert run.optimizer_fingerprint(state)==original


def reports():
    return {k:{'residual_nll':1.,'complete_nll':1.,'covariance_error':1.,'kid':.2,
        'gradient_means':[1.,1.],'repair_gradient_means':[1.,1.],'energy_mean':.1} for k in run.ARMS}


def test_material_gate_and_each_control_are_separate():
    r=reports();r['I_joint']['kid']=.19
    assert run.engineering_gates(r)['I_joint_KID_material_5pct_both_RQS']
    r['RQS_joint']['kid']=.19
    assert not run.engineering_gates(r)['I_joint_KID_material_5pct_both_RQS']
    r['P_joint']['kid']=.18
    assert not run.engineering_gates(r)['I_joint_kid_better_P_joint']
    r['RQS_joint']['kid']=-.01;r['I_joint']['kid']=-.02
    assert not run.engineering_gates(r)['I_joint_KID_material_5pct_both_RQS']
    with pytest.raises(ValueError):run.engineering_gates({k:v for k,v in r.items() if k!='P_joint'})


def test_freeze_receipt_rejects_missing_seed_before_repair(tmp_path):
    with pytest.raises(ValueError):run.write_fit_freeze_receipt(tmp_path,{})
    assert not (tmp_path/'ALL_FITS_FROZEN.json').exists()


def test_shared_root_gradients_live_for_joint_mode(template):
    model=run.make_family(template,'I',15);run.activate(model,'residual_decoder');model.unfreeze_analysis()
    assert all(p.requires_grad for p in model._analysis_parameters())
    assert not any(p.requires_grad for p in model.coarse_decoder.parameters())
    # Fixed-root parameters do not stop gradients to its input.
    coarse=torch.randn(1,3,8,8,requires_grad=True)
    value,ld=model.coarse_decoder.encode(coarse,model._zero(coarse))
    (.5*value.square().sum()-ld.sum()).backward()
    assert coarse.grad is not None and torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum()>0


def test_protocol_preserves_fit_before_repair_and_explicit_backend():
    text=PATH.read_text()
    assert text.index('write_fit_freeze_receipt(args.output,all_models)')<text.index('repair_logits,outer=utility.logit_inputs(data.repair)')
    assert "backend='dense_reflected'" in text
    assert "out/(prefix_name+'.pt')" in text
    assert "optimizer.load_state_dict(fork_optimizer)" in text


def test_pending_qualification_fails_before_data_or_assets(tmp_path,monkeypatch):
    pending=tmp_path/'pending.json';pending.write_text('{"qualified": false}')
    monkeypatch.setattr(run,'QUALIFICATION',pending)
    monkeypatch.setattr(run,'source_guard',lambda *a:None)
    def forbidden(*args,**kwargs):raise AssertionError('pending qualification crossed access boundary')
    monkeypatch.setattr(run,'load_observed_flow_data',forbidden)
    monkeypatch.setattr(run.evaluator,'validate_artifacts',forbidden)
    monkeypatch.setattr(sys,'argv',['run.py','--expected-commit','0'*40,'--output',str(tmp_path/'out'),'--inception-source','unused','--weights','unused'])
    with pytest.raises(ValueError,match='qualification is pending'):run.main()
    import json
    failure=json.loads((tmp_path/'out/failure.json').read_text())
    assert 'qualification is pending' in failure['error']
    assert failure['state']['phase']=='gpu_qualification'


def test_strict_qualification_and_frozen_receipt(tmp_path,monkeypatch):
    import hashlib,json
    qpath=tmp_path/'qualification.json';receiptpath=tmp_path/'receipt.json'
    monkeypatch.setattr(run,'QUALIFICATION',qpath);monkeypatch.setattr(run,'QUALIFICATION_RECEIPT',receiptpath)
    receipt={'kernel_sha256':run.digest(run.KERNEL_PATH),'diagnostic_source_commit':'a'*40,'diagnostic_job_id':'123','original_failure_job_id':'45582364','device':'cuda','original_failure_reproduced':True,'candidate_valid':True,'candidate_gradients_finite':True,'grad_enabled':True,'state_unchanged':True,'training_performed':False,'repair_evaluated':False}
    for scope in ('conditional_decoder','full_model_cpu','full_model_cuda'):
        evidence={'source_dimension':2880 if scope=='conditional_decoder' else 3072,
            'source_roundtrip_max_abs':1e-5,'source_logdet_cancellation_max_abs':1e-4,
            'observed_roundtrip_max_abs':1e-5,'observed_logdet_cancellation_max_abs':1e-4}
        for key in ('finite','roundtrip_gate','logdet_gate','exact_reload','joint_gradients_finite_and_present',
            'analysis_gradients_finite_and_present','residual_input_gradients_finite_and_present',
            'coarse_input_gradients_finite_and_present','fixed_root_input_gradients_finite_and_present',
            'root_parameters_frozen','state_unchanged','invalid_inputs_rejected'):evidence[key]=True
        for key in ('source_bank_sha256','numerical_bank_sha256','checkpoint_sha256'):evidence[key]='b'*64
        receipt[scope]=evidence
    receiptpath.write_text(json.dumps(receipt));q={k:receipt[k] for k in ('kernel_sha256','diagnostic_source_commit','diagnostic_job_id','original_failure_job_id')};q.update(schema=1,qualified=True,backend='dense_reflected',independent_review_verified=True,diagnostic_receipt_sha256=run.digest(receiptpath))
    qpath.write_text(json.dumps(q));assert run.validate_qualification()==q
    for field,value in [('schema',True),('qualified',1),('qualified','true'),('independent_review_verified',1),('kernel_sha256','0'*64),('diagnostic_source_commit','a'*7),('diagnostic_receipt_sha256','0'*64)]:
        bad=dict(q);bad[field]=value;qpath.write_text(json.dumps(bad))
        with pytest.raises(ValueError):run.validate_qualification()
    for field,value in [('source_roundtrip_max_abs',.00101),('source_logdet_cancellation_max_abs',.01001),
        ('observed_roundtrip_max_abs',float('nan')),('observed_logdet_cancellation_max_abs',float('inf')),
        ('source_roundtrip_max_abs',True),('source_dimension',2880),('exact_reload',False),
        ('joint_gradients_finite_and_present',1),('numerical_bank_sha256','bad')]:
        bad=copy.deepcopy(receipt);bad['full_model_cuda'][field]=value;receiptpath.write_text(json.dumps(bad))
        q['diagnostic_receipt_sha256']=run.digest(receiptpath);qpath.write_text(json.dumps(q))
        with pytest.raises(ValueError):run.validate_qualification()
    qpath.write_text(json.dumps(q));receipt['candidate_gradients_finite']=1;receiptpath.write_text(json.dumps(receipt));q['diagnostic_receipt_sha256']=run.digest(receiptpath);qpath.write_text(json.dumps(q))
    with pytest.raises(ValueError):run.validate_qualification()


def test_reflected_checkpoint_rejects_old_backend(template,tmp_path):
    model=run.make_family(template,'RQS',19);path=tmp_path/'model.pt';run.checkpoint(path,model)
    record=torch.load(path,weights_only=True);record['architecture']['backend']='dense_eager'
    with pytest.raises(ValueError):run.restore_model(record)
    names={p.name for p in run.sources()}
    assert {'qualification.json','qualification_receipt.json','reflected_dense_spline.py'}<=names


def test_all_numerics_admission_preserves_partial_and_blocks_quality(tmp_path,monkeypatch):
    class FakeModel:
        residual_decoder=object()
        def cuda(self):return self
        def cpu(self):return self
    inventory={seed:({arm:FakeModel() for arm in run.ARMS},None) for seed in run.SEEDS}
    monkeypatch.setattr(run,'model_spec',lambda model:{'family':'P'})
    monkeypatch.setattr(run,'ARMS',('P_frozen','P_joint'))
    inventory={seed:({arm:FakeModel() for arm in run.ARMS},None) for seed in run.SEEDS}
    with pytest.raises(ValueError,match='freeze'):run.admit_all_numerics(tmp_path,inventory,{}, {})
    (tmp_path/'ALL_FITS_FROZEN.json').write_text('{}')
    for seed in run.SEEDS:(tmp_path/f'seed_{seed}').mkdir()
    calls=[]
    def gate(model,out,seed):
        calls.append((seed,out.name))
        (out/'numerical.npz').write_bytes(b'fabricated retained numerical bank')
        (out/'numerical.json').write_text('{}')
        if len(calls)==2:raise FloatingPointError('fabricated failed gate')
    monkeypatch.setattr(run,'numerical_gate',gate)
    with pytest.raises(FloatingPointError):run.admit_all_numerics(tmp_path,inventory,{}, {})
    assert not (tmp_path/'ALL_NUMERICS_ADMITTED.json').exists()
    assert (tmp_path/'numerical_admission_progress.json').exists()
    assert len(list(tmp_path.rglob('numerical.npz')))==2
    with pytest.raises(ValueError,match='admission'):
        run.evaluate_seed(run.SEEDS[0],{},None,None,None,None,tmp_path/f'seed_{run.SEEDS[0]}',None,{}, {})


def test_all_numerics_precede_any_repair_or_extractor():
    import ast
    tree=ast.parse(PATH.read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    text=ast.unparse(main)
    assert text.index('write_fit_freeze_receipt(')<text.index('admit_all_numerics(')<text.index('utility.logit_inputs(data.repair)')<text.index('evaluator.make_extractor(')
    evaluate=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate_seed')
    assert 'numerical_gate(' not in ast.unparse(evaluate)
