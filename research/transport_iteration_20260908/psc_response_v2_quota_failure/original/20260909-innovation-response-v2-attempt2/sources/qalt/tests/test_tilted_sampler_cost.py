"""Fabricated numerical tests; no timing study, fitting, datasets, or scheduler."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'qalt/experiments/tilted_sampler_cost'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


runner=load('tilt_cost_runner_test',HERE/'run.py')
candidate=load('tilt_cost_candidate_test',HERE/'candidate.py')
reference=load('tilt_cost_reference_test',ROOT/runner.REFERENCE)


def test_source_pins_and_archived_parameter_provenance():
    for kind,path in [('candidate',HERE/'candidate.py'),('reference',ROOT/runner.REFERENCE),('parameters',ROOT/runner.PARAMETERS)]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==runner.PINS[kind]
    assert runner.SEEDS==(2026090901,2026090902,2026090903)
    assert len(runner.ARMS)==17
    assert all((ROOT/name).is_file() for name in runner.SOURCE_FILES)


def test_central_boundary_extremes_inverse_density_and_nominal_calls():
    z=np.array([-50.,-12.,np.nextafter(-5,-np.inf),-5.,np.nextafter(-5,np.inf),0.,
                np.nextafter(5,-np.inf),5.,np.nextafter(5,np.inf),12.,50.])[:,None]
    e=np.linspace(-.6,.6,25)[None,:]
    result=candidate.quantile_transport(z,e)
    np.testing.assert_allclose(result,reference.quantile_transport(z,e),rtol=0,atol=1e-12)
    np.testing.assert_allclose(candidate.inverse_transport(result,e),np.broadcast_to(z,result.shape),rtol=0,atol=2e-12)
    identity=candidate.log_density(result,e)+candidate.forward_logdet(z,e)+.5*z*z+.5*np.log(2*np.pi)
    assert np.max(np.abs(identity))<1e-12
    for n in runner.NFES:
        value,counts=candidate.heun(z,e,n)
        expected=reference.heun(np.broadcast_to(z,value.shape),e,n)
        np.testing.assert_allclose(value,expected,rtol=0,atol=1e-12)
        assert counts['field_calls']==n and counts['endpoint_zero_field_called']
        specialized,info=candidate.endpoint_heun(z,e,n)
        np.testing.assert_allclose(specialized,expected,rtol=0,atol=1e-12)
        assert info['equivalent_mathematical_stages']==n
        assert info['nontrivial_field_kernel_calls']==n-2
        assert info['analytic_endpoint_stages']==2 and not info['final_predictor_allocated']


def test_external_guard_rejects_wrong_head_before_snapshot(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'a'*40)
    with pytest.raises(ValueError,match='HEAD'):runner.source_guard('b'*40,tmp_path)
    assert not list(tmp_path.iterdir())


def test_numerical_only_full_pipeline_with_simulated_git_guard(monkeypatch,tmp_path):
    # This simulates committed bytes solely for testing the uncommitted new code.
    # It is not an external source freeze and cannot authorize timing/submission.
    commit='c'*40
    def fake_git(command,**kwargs):
        if command[1:]==['rev-parse','HEAD']:return commit+'\n'
        return (ROOT/command[-1].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(runner.subprocess,'check_output',fake_git)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name,'1')
    runner.run(SimpleNamespace(mode='numerical-only',expected_commit=commit,output=tmp_path),{})
    assert (tmp_path/'GATES_PASSED.json').exists()
    assert len(list(tmp_path.glob('output_*.npy')))==102
    assert len(list(tmp_path.glob('source_*.npy')))==6
    assert not list(tmp_path.glob('*timings*'))
    assert len(list((tmp_path/'sources').iterdir()))==len(runner.SOURCE_FILES)
