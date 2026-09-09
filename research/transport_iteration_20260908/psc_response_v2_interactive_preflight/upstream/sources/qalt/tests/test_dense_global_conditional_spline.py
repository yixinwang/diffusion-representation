import copy
import pytest

torch=pytest.importorskip('torch')
from qalt.global_conditional_spline import GlobalConditionalSplineDecoder
from qalt.dense_global_conditional_spline import DenseGlobalConditionalSplineDecoder


def models():
    reference=GlobalConditionalSplineDecoder(2,1,2,layers=2,width=4,bins=4,attention_heads=1).double()
    generator=torch.Generator().manual_seed(634)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(.025*torch.randn(parameter.shape,generator=generator,dtype=parameter.dtype))
    dense=DenseGlobalConditionalSplineDecoder(2,1,2,layers=2,width=4,bins=4,attention_heads=1).double()
    dense.load_state_dict(copy.deepcopy(reference.state_dict()),strict=True)
    return reference,dense


@pytest.mark.parametrize('inverse',[False,True])
def test_nonidentity_values_logdet_and_all_gradients(inverse):
    original,dense=models()
    source=torch.linspace(-.9,.8,16,dtype=torch.float64).reshape(2,2,2,2)
    coarse=torch.linspace(-.2,.4,8,dtype=torch.float64).reshape(2,1,2,2)
    inputs=[(source.clone().requires_grad_(),coarse.clone().requires_grad_()) for _ in range(2)]
    results=[];grads=[]
    for model,(x,c) in zip((original,dense),inputs):
        value,ld=model.encode(x,c) if inverse else model.decode(x,c)
        results.append((value,ld))
        grads.append(torch.autograd.grad((value.square().sum()+.17*ld.sum()),(x,c,*model.parameters())))
    for a,b in zip(*results):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    for a,b in zip(*grads):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    assert not torch.equal(results[0][0],source)


def test_roundtrip_dense_jacobian_state_and_full_source():
    original,dense=models()
    source=torch.linspace(-.9,.8,8,dtype=torch.float64).reshape(1,2,2,2)
    coarse=torch.linspace(-.2,.4,4,dtype=torch.float64).reshape(1,1,2,2)
    x,ld=dense.decode(source,coarse);back,ild=dense.encode(x,coarse)
    torch.testing.assert_close(back,source,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=1e-10,rtol=0)
    jac=torch.autograd.functional.jacobian(lambda z:dense.decode(z.reshape_as(source),coarse)[0].flatten(),source.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=1e-10,rtol=1e-10)
    assert dense.state_dict().keys()==original.state_dict().keys()
    assert dense.parameter_counts==original.parameter_counts
    original.load_state_dict(dense.state_dict(),strict=True)
    torch.testing.assert_close(original.log_prob(x,coarse),dense.log_prob(x,coarse),atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('inverse',[False,True])
def test_invalid_intermediate_rejected_before_return(inverse):
    _,dense=models()
    with torch.no_grad():dense.layers[0].conditioner.output.bias[0]=float('inf')
    x=torch.zeros(1,2,2,2,dtype=torch.float64);coarse=torch.zeros(1,1,2,2,dtype=torch.float64)
    with pytest.raises(FloatingPointError,match='aggregated'):
        dense.encode(x,coarse) if inverse else dense.decode(x,coarse)


def test_compiler_option_cpu_graph_capture_only(monkeypatch):
    compile_original=torch.compile;requests=[]
    def fake_compiler(function,**kwargs):
        requests.append(kwargs)
        return compile_original(function,backend='eager',**kwargs)
    monkeypatch.setattr(torch,'compile',fake_compiler)
    model=DenseGlobalConditionalSplineDecoder(2,1,2,layers=2,width=4,bins=4,
        attention_heads=1,backend='compiled').double()
    assert requests==[{'fullgraph':True,'dynamic':False}]
    x=torch.zeros(1,2,2,2,dtype=torch.float64);c=torch.zeros(1,1,2,2,dtype=torch.float64)
    torch.testing.assert_close(model.decode(x,c)[0],x,atol=1e-14,rtol=0)
    # This only verifies CPU Dynamo graph capture, not Inductor/GPU performance.
