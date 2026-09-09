import importlib.util
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch

p=Path(__file__).resolve().parents[1]/'experiments/response_failure_diagnostic/run.py'
spec=importlib.util.spec_from_file_location('response_diagnostic_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_capture_unchanged_output_and_locals():
    def kernel(x):
        good_theta=x>=0
        mapped=x.square()
        return mapped,good_theta.all()
    x=torch.tensor([-1.,2.]);result,local=m.captured_call(kernel,x)
    assert torch.equal(result[0],kernel(x)[0])
    assert torch.equal(local['good_theta'],torch.tensor([False,True]))
    assert torch.equal(local['mapped'],x.square())
    assert m.sys.gettrace() is None

def test_capture_removes_trace_on_failure():
    def fail(x):raise ValueError('fabricated')
    with pytest.raises(ValueError):m.captured_call(fail,torch.ones(1))
    assert m.sys.gettrace() is None

def test_failed_draw_provenance():
    ids=np.arange(4000);rng=np.random.default_rng(78231);h=hashlib.sha256()
    for _ in range(4):
        index=rng.integers(0,len(ids),size=32);h.update(np.asarray(ids[index],dtype='<i8').tobytes())
    assert np.array_equal(m.recover_draw(ids,3,rng.bit_generator.state,h.hexdigest()),index)
    with pytest.raises(ValueError):m.recover_draw(ids,2,rng.bit_generator.state,h.hexdigest())
