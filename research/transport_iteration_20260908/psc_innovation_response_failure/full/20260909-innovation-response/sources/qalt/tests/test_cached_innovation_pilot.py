"""Fabricated runner checks; no canonical data, fitting study or GPU timing."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PATH=Path(__file__).resolve().parents[1]/'experiments/cached_innovation_pilot/run.py'
spec=importlib.util.spec_from_file_location('cached_innovation_pilot_test_runner',PATH)
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)


def test_shape_only_parameter_match_counts_active_scalar_weights():
    torch.manual_seed(9)
    model=run.prepared_model();width,total=run.scalar_width(model)
    scalar=run.CachedGlobalInnovationDecoder(45,3,8,width=width,rank=16,use_mixer=False)
    actual=model.parameter_counts['total']-model.parameter_counts['residual']+sum(p.numel() for p in scalar.parameters())
    assert actual==total
    assert model.parameter_counts['total']<=actual<=1.05*model.parameter_counts['total']
    assert not list(scalar.frames) and all(not list(c.eigen_head.parameters()) for c in scalar.conditioners)


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.sum()
    @staticmethod
    def backward(ctx,g):return g.new_full((1,),float('inf'))


def test_gradient_failure_cannot_update_parameters():
    p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.Adam([p],lr=.1)
    with pytest.raises(FloatingPointError,match='optimizer not stepped'):
        run.finite_step(BadGradient.apply(p),[p],optimizer)
    assert torch.equal(p,torch.ones(1)) and not optimizer.state


def test_two_update_progress_and_failed_state_preservation(tmp_path,monkeypatch):
    model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    clock=[0.];monkeypatch.setattr(run.time,'perf_counter',lambda:clock[0])
    def loss(index):
        clock[0]+=1
        return model(torch.ones(len(index),1)).square().mean()
    report=run.train_until(model,optimizer,np.random.default_rng(2),loss,np.arange(4),2.,tmp_path,'dummy',{})
    assert report['updates']==2 and report['status']=='completed'
    assert (tmp_path/'dummy_progress.json').is_file()
    clock[0]=0
    def invalid(index):clock[0]+=1;return model.weight.sum()*float('nan')
    with pytest.raises(FloatingPointError):
        run.train_until(model,optimizer,np.random.default_rng(2),invalid,np.arange(4),2.,tmp_path,'invalid',{})
    saved=torch.load(tmp_path/'invalid_failed.pt',weights_only=False)
    assert saved['optimizer'] is not None and saved['rng'] is not None
    with pytest.raises(RuntimeError,match='zero-update'):
        run.train_until(model,optimizer,np.random.default_rng(2),loss,np.arange(4),-1.,tmp_path,'empty',{})


def test_pair_energy_and_seedwise_gates_are_fixed():
    repair=np.zeros((2,3,32,32));fake=np.stack([np.zeros_like(repair[0]),np.ones_like(repair[0]),np.ones_like(repair[0]),np.ones_like(repair[0])])
    report,energy,desc=run.sample_metrics(fake,repair)
    np.testing.assert_allclose(energy,[0,1])
    assert report['energy_mean']==.5 and desc.shape==(4,6)
    def arm(nll,kid,cov):return dict(residual_nll=nll,complete_nll=nll,kid=kid,covariance_error=cov,gradient_means=[1,1],repair_gradient_means=[1,1],energy_mean=.1)
    arms={'S':arm(3,3,2),'M':arm(2,2,1),'J':arm(1,1,1),'RQS':arm(2,2,1)}
    assert all(run.engineering_gates(arms).values())
    arms['J']['kid']=2
    assert not run.engineering_gates(arms)['J_KID_better_RQS']
    arms['J']['energy_mean']=.1006
    assert not run.engineering_gates(arms)['J_energy_not_worse_M']


def test_failure_payload_keeps_previous_outputs_and_model(tmp_path):
    (tmp_path/'earlier.npy').write_bytes(b'preserved')
    model=torch.nn.Linear(1,1)
    try:raise RuntimeError('fabricated')
    except RuntimeError as error:run.fail(tmp_path,{'phase':'mock'}, {'model':model},error)
    import json
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['status']=='failed' and (tmp_path/'failed_model.pt').is_file()
    assert status['payload_sha256']['earlier.npy']==hashlib.sha256(b'preserved').hexdigest()


def test_failed_generation_preserves_only_completed_prefix(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(run,'numerical_gate',lambda *a:None)
    monkeypatch.setattr(run.evaluator,'extract',lambda *a:np.zeros((2,8)))
    monkeypatch.setattr(run,'raw_generate',lambda model,z,kind:torch.full((len(z),3,32,32),float('inf')))
    class Fake:
        def __init__(self):self.residual_decoder=object()
        def cuda(self):return self
    data=SimpleNamespace(repair=np.zeros((2,3,32,32)),repair_ids=np.arange(2))
    state={}
    with pytest.raises(FloatingPointError,match='logits saved before sigmoid'):
        run.evaluate_seed(7,{'S':Fake()},None,data,None,None,tmp_path,None,{},state)
    import json
    progress=json.loads((tmp_path/'S/generation_progress.json').read_text())
    assert progress['completed_rows']==0 and progress['logits_written_rows']==64
    assert state['arm']=='S' and state['chunk_start']==0 and state['completed_rows']==0
    logits=np.load(tmp_path/'S/logits.npy',mmap_mode='r')
    assert np.isinf(logits[:64]).all()
    assert progress['status']=='logits_saved_pending_validation'
