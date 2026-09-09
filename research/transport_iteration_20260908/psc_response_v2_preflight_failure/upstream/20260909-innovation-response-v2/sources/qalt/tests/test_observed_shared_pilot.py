"""Numerical and scoring safeguards for the prospective operational pilot."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

path = Path(__file__).resolve().parents[1]/'experiments/observed_flow_pilot/run_shared.py'
spec = importlib.util.spec_from_file_location('observed_shared_test', path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_energy_independent_pair_assignment_and_saved_sources(tmp_path):
    repair = np.stack([np.full((3,32,32),v) for v in [.1,.3,.8]])
    def sampler(z):
        return torch.sigmoid(z[:,0,None,None,None]).expand(-1,3,32,32)
    first = pilot.evaluate_pairs(sampler,repair,dimension=3072,device='cpu',
                                batch_size=4,artifact_directory=tmp_path/'first')
    second = pilot.evaluate_pairs(sampler,repair,dimension=3072,device='cpu',
                                 batch_size=4,artifact_directory=tmp_path/'second')
    expected=[]
    for a,b in zip(sorted((tmp_path/'first').glob('chunk*.npz')),sorted((tmp_path/'second').glob('chunk*.npz'))):
        with np.load(a) as saved, np.load(b) as copied:
            assert np.array_equal(saved['gaussian'],copied['gaussian'])
            values=saved['generated'][:,0,0,0].reshape(-1,2)
            targets=repair[saved['repair_positions'],0,0,0]
            expected.extend(.5*(abs(values[:,0]-targets)+abs(values[:,1]-targets)-abs(values[:,0]-values[:,1])))
    np.testing.assert_allclose(first['per_image_energy'],expected,rtol=1e-13,atol=1e-13)
    assert pilot.paired_energy(first,second)['first_minus_second']==0
    assert first['generated_count']==6
    with pytest.raises(ValueError):
        pilot.paired_energy(first,{**second,'per_image_energy':[0.]})
    with pytest.raises(ValueError):
        pilot.paired_energy(first,{**second,'source_stream_sha256':'different'})


def test_nonfinite_logits_cannot_be_hidden_by_sigmoid():
    class BrokenAnalysis:
        dimension=4
        latent_dimension=1
        channels=1
        packed_residual_channels=3
        coarse_size=1
        def _join_code(self,c,r): return torch.cat((c.flatten(1),r.flatten(1)),1)
        def decode_analysis(self,z):
            return torch.full((len(z),1,2,2),float('inf')),torch.zeros(len(z))
    # Sigmoid(infinity) itself is finite; reject the earlier numerical failure.
    assert torch.isfinite(torch.sigmoid(torch.tensor(float('inf'))))
    with pytest.raises(FloatingPointError,match='before sigmoid'):
        pilot.generate(BrokenAnalysis(),None,torch.zeros(2,4),kind='analysis_only')


def test_failed_training_preserves_report_without_parameter_update(tmp_path, monkeypatch):
    # Exercise the nonfinite-loss branch independently of first-optimizer import
    # time on a cold cluster filesystem. This changes only the test clock.
    monkeypatch.setattr(pilot, "time", SimpleNamespace(perf_counter=lambda: 0.))
    p=torch.nn.Parameter(torch.tensor(1.))
    report=tmp_path/'failure.json'
    with pytest.raises(FloatingPointError,match='loss'):
        pilot.train_stage([p],lambda i,g:p*float('nan'),sample_count=3,
            record_ids=np.arange(3),device='cpu',seconds=1,batch_size=2,progress_path=report)
    saved=json.loads(report.read_text())
    assert saved['status']=='failed' and saved['updates']==0
    assert p.item()==1 and saved['elapsed_seconds']>=0


def test_shared_logit_keeps_near_boundary_input_finite():
    x=np.array([np.nextafter(1.,0.),np.nextafter(0.,1.)],dtype=np.float64).reshape(1,1,1,2)
    logits,jac=pilot.logit_inputs(x)
    assert logits.dtype==torch.float32 and torch.isfinite(logits).all() and np.isfinite(jac).all()
    with pytest.raises(ValueError):pilot.logit_inputs(x.astype(np.float32))


def test_optimizer_startup_exhausting_cap_fails_without_evaluating_loss(tmp_path, monkeypatch):
    clock = iter([0., 2., 3.])
    monkeypatch.setattr(pilot, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    parameter = torch.nn.Parameter(torch.tensor(1.))
    def forbidden_loss(index, generator):
        raise AssertionError("expired stage must not evaluate its loss")
    report = tmp_path/'expired.json'
    with pytest.raises(RuntimeError, match='zero updates'):
        pilot.train_stage([parameter], forbidden_loss, sample_count=3,
            record_ids=np.arange(3), device='cpu', seconds=1, batch_size=2, progress_path=report)
    saved = json.loads(report.read_text())
    assert saved['status']=='failed' and saved['updates']==0 and saved['elapsed_seconds']==3.
    assert parameter.item()==1.
