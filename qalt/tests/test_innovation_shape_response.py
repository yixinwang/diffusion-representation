import math
import pytest
import torch
from qalt.innovation_response import InnovationResponse, InnovationResponseFlow
from qalt.innovation_shape_response import SinhArcsinhResponse, sinh_arcsinh_shape, promote_location_scale_flow


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('inverse', [False, True])
def test_identity_values_logdet_and_live_analytic_gradients(dtype, inverse):
    z=torch.tensor([-12.,-2.,-.1,0.,.7,3.,32.],dtype=dtype)
    t=torch.zeros_like(z,requires_grad=True);k=torch.zeros_like(z,requires_grad=True)
    y=sinh_arcsinh_shape(z,t,k,inverse)
    assert y.valid and torch.equal(y.value,z) and torch.equal(y.logdet,torch.zeros_like(z))
    gt,gk=torch.autograd.grad(y.value.sum(),(t,k),retain_graph=True)
    lt,lk=torch.autograd.grad(y.logdet.sum(),(t,k))
    r=torch.hypot(torch.ones_like(z),z);v=torch.asinh(z);sign=-1 if inverse else 1
    for actual,expected in [(gt,sign*r),(gk,-sign*v*r),(lt,sign*z/r),(lk,sign*(-1-v*z/r))]:
        torch.testing.assert_close(actual,expected,atol=1e-5 if dtype==torch.float32 else 2e-14,rtol=2e-7 if dtype==torch.float32 else 2e-15)
    assert gt.abs().sum()>0 and gk.abs().sum()>0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nonzero_direct_float64_reference_and_inverse(dtype):
    z=torch.linspace(-32,32,97,dtype=dtype)
    tau=torch.linspace(-.95,.95,len(z),dtype=dtype)
    k=torch.linspace(-math.log(2),math.log(2),len(z),dtype=dtype)
    y=sinh_arcsinh_shape(z,tau,k);back=sinh_arcsinh_shape(y.value,tau,k,True)
    arg=(torch.asinh(z.double())+tau.double())*torch.exp(-k.double())
    ref=torch.sinh(arg)
    ld=-k.double()+torch.log(torch.cosh(arg))-torch.log(torch.hypot(torch.ones_like(z.double()),z.double()))
    assert y.valid and back.valid
    torch.testing.assert_close(y.value.double(),ref,rtol=8e-8 if dtype==torch.float32 else 2e-13,atol=3e-12)
    torch.testing.assert_close(y.logdet.double(),ld,rtol=1e-7 if dtype==torch.float32 else 2e-12,atol=1e-7 if dtype==torch.float32 else 2e-13)
    # Extreme bounded shapes expand |z|=32 to ~13k; inverse cancellation
    # has a measured ~1e-9 absolute error in float64 (not an interval bound).
    torch.testing.assert_close(back.value,z,rtol=2e-7 if dtype==torch.float32 else 0,atol=3e-6 if dtype==torch.float32 else 3e-9)
    torch.testing.assert_close(y.logdet+back.logdet,torch.zeros_like(z),atol=5e-7 if dtype==torch.float32 else 3e-11,rtol=0)


def test_high_precision_bounded_shape_reference():
    mp=pytest.importorskip('mpmath');mp.mp.dps=80
    for z in (-12.,-.4,0.,.3,12.):
        for tau,k in ((.8,math.log(2)),(-.8,-math.log(2)),(1e-8,-1e-8)):
            x=torch.tensor([z],dtype=torch.double)
            y=sinh_arcsinh_shape(x,torch.full_like(x,tau),torch.full_like(x,k))
            v=(mp.asinh(z)+mp.mpf(tau))*mp.exp(-mp.mpf(k))
            ref=float(mp.sinh(v));ld=float(-mp.mpf(k)+mp.log(mp.cosh(v))-.5*mp.log1p(mp.mpf(z)**2))
            assert y.valid
            assert abs(y.value.item()-ref)<=2e-12*max(1,abs(ref))
            assert abs(y.logdet.item()-ld)<=3e-13


def test_extreme_intermediate_failure_is_not_clipped_or_hidden():
    x=torch.tensor([1e308],dtype=torch.double)
    y=sinh_arcsinh_shape(x,torch.zeros_like(x),torch.full_like(x,-math.log(2)))
    assert not y.valid and not torch.isfinite(y.value).all()
    response=SinhArcsinhResponse(4,2).float()
    with torch.no_grad():response.output.bias[4:]=torch.finfo(torch.float32).max
    result=response(torch.zeros(1,4),torch.zeros(1,16),torch.ones(4,2))
    assert not result.valid  # tanh must not conceal raw shape projection overflow
    with pytest.raises(ValueError):sinh_arcsinh_shape(x.float(),x,x)
    bad=response(torch.full((1,4),float('nan')),torch.zeros(1,16),torch.ones(4,2))
    assert not bad.valid


def tiny(mode='innovation',dtype=torch.double):
    torch.manual_seed(2311)
    return InnovationResponseFlow(channels=1,size=8,levels=1,pre_layers=2,
        analysis_coarse_layers=2,analysis_detail_layers=2,coarse_layers=2,residual_layers=2,
        width=4,bins=4,attention_heads=1,innovation_rank=2,innovation_channel_embedding=2,
        response_mode=mode).to(dtype)


@pytest.mark.parametrize('mode',['innovation','prefix'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_promoted_location_scale_exact_nesting_and_state_preservation(mode,dtype):
    old=tiny(mode,dtype)
    with torch.no_grad():
        for r in old.residual_decoder.responses:r.output.weight.normal_(0,.1);r.output.bias.normal_(0,.1)
    original={k:v.clone() for k,v in old.state_dict().items()};new=promote_location_scale_flow(old)
    z=torch.randn(2,64,dtype=dtype)*.3
    for inverse in (False,True):
        a=old.encode(z.reshape(2,1,8,8)) if inverse else old.decode(z)
        b=new.encode(z.reshape(2,1,8,8)) if inverse else new.decode(z)
        for av,bv in zip(a,b):assert torch.equal(av,bv)
    assert all(torch.equal(old.state_dict()[k],v) for k,v in original.items())
    for r in new.residual_decoder.responses:
        assert r.mode==mode and r.output.weight[2*r.rank:].count_nonzero()==0
    with pytest.raises(ValueError):promote_location_scale_flow(new)


def test_default_counts_equal_controls():
    old=InnovationResponseFlow();new=promote_location_scale_flow(old)
    p=promote_location_scale_flow(InnovationResponseFlow(response_mode='prefix'))
    assert new.parameter_counts['total']==p.parameter_counts['total']==543344
    assert old.parameter_counts['total']==539120


def test_full_nonzero_shape_inverse_jacobian_and_live_gradients():
    model=promote_location_scale_flow(tiny())
    with torch.no_grad():
        for r in model.residual_decoder.responses:r.output.weight.normal_(0,.1);r.output.bias.normal_(0,.04)
    z=torch.randn(1,64,dtype=torch.double)*.2
    x,ld=model.decode(z);back,ild=model.encode(x)
    torch.testing.assert_close(back,z,atol=4e-11,rtol=0)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=4e-11,rtol=0)
    jac=torch.autograd.functional.jacobian(lambda v:model.decode(v.reshape(1,-1))[0].flatten(),z.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=4e-11,rtol=0)
    for p in model.coarse_decoder.parameters():p.requires_grad_(False)
    (-model.log_prob(x.detach()).mean()).backward()
    for r in model.residual_decoder.responses:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())
        assert r.output.weight.grad[2*r.rank:].abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model._analysis_parameters())
    assert all(p.grad is None for p in model.coarse_decoder.parameters())
