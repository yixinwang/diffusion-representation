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

def test_candidate_backward_does_not_mutate_original(tmp_path):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.shift=torch.nn.Parameter(torch.tensor(.2));self._spline=None
        def encode(self,x,c):return x-self.shift,x.new_zeros(len(x))
        def decode(self,z,c):return z+self.shift,z.new_zeros(len(z))
    decoder=Toy();before=decoder.shift.detach().clone();x=torch.tensor([[.1,.2],[.3,.4]])
    report=m.compare_candidate(decoder,None,x,x,tmp_path)
    assert report['state_unchanged'] and report['gradients_finite_and_present']
    assert report['roundtrip_gate'] and report['logdet_gate']
    assert decoder.shift.grad is None and torch.equal(before,decoder.shift)

def test_complete_fabricated_full_model(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from qalt.reflected_dense_spline import reflected_dense_spline_kernel
    helper_path=p.parent/'full_model.py'
    hs=importlib.util.spec_from_file_location('full_diagnostic_test',helper_path);helper=importlib.util.module_from_spec(hs);hs.loader.exec_module(helper)
    runner_path=p.parent.parent/'innovation_response_pilot/run.py'
    rs=importlib.util.spec_from_file_location('full_runner_test',runner_path);runner=importlib.util.module_from_spec(rs);rs.loader.exec_module(runner)
    torch.set_num_threads(1);torch.manual_seed(83)
    model=runner.make_family(runner.prepared_model(),'RQS',83)
    ck={'architecture':runner.model_spec(model),'model':model.state_dict()}
    logits=torch.randn(32,3,32,32)*.3
    report=helper.qualify(runner,ck,reflected_dense_spline_kernel,logits,tmp_path/'full','cpu','0'*64)
    assert report['status']=='captured',report
    for flag in ['finite','roundtrip_gate','logdet_gate','exact_reload','joint_gradients_finite_and_present','analysis_gradients_finite_and_present','residual_input_gradients_finite_and_present','coarse_input_gradients_finite_and_present','fixed_root_input_gradients_finite_and_present','root_parameters_frozen','state_unchanged','invalid_inputs_rejected']:
        assert report[flag],flag
    assert report['source_dimension']==3072 and report['source_seed']==78300

def test_full_report_nonfinite_metrics_preserved_as_null():
    hs=importlib.util.spec_from_file_location('full_diagnostic_safe_test',p.parent/'full_model.py');helper=importlib.util.module_from_spec(hs);hs.loader.exec_module(helper)
    assert helper.json_safe({'error':float('inf'),'nested':[float('nan'),1.]})=={'error':None,'nested':[None,1.]}
