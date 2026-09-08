import pytest
import torch
from qalt.multiscale_flow import haar_split,haar_merge,MultiscaleSplineFlow,CopiedStochasticLatentDecoder


def make_flow(unit_interval=False,size=4,levels=1):
    torch.manual_seed(128)
    flow=MultiscaleSplineFlow(channels=1,size=size,levels=levels,coarse_layers=2,detail_layers=2,width=5,bins=4,unit_interval=unit_interval).double()
    # Nonidentity test: exercise conditioner-dependent triangular Jacobian.
    with torch.no_grad():
        for name,p in flow.named_parameters():
            if 'net.4' in name:p.normal_(0,.07)
        for stack in [flow.coarse_flow,*flow.detail_flows]:
            stack.location.normal_(0,.1);stack.raw_log_scale.normal_(0,.05)
    return flow


def test_haar_orthonormal_and_roundtrip():
    torch.manual_seed(31);x=torch.randn(3,2,8,8,dtype=torch.float64)
    c,d=haar_split(x)
    torch.testing.assert_close(haar_merge(c,d),x,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(c.square().sum()+d.square().sum(),x.square().sum())


@pytest.mark.parametrize('unit_interval',[False,True])
def test_full_roundtrip_and_logdet(unit_interval):
    flow=make_flow(unit_interval,size=8,levels=2)
    z=torch.randn(3,64,dtype=torch.float64)
    x,ld=flow.decode(z);zz,ild=flow.encode(x)
    torch.testing.assert_close(zz,z,atol=2e-9,rtol=2e-9)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=2e-9,rtol=0)
    assert sum(__import__('math').prod(s) for s in flow.block_shapes)==64


def test_dense_autograd_jacobian_for_complete_flow():
    flow=make_flow(size=2)
    z=torch.tensor([.1,-.3,.7,-.4],dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(lambda u:flow.decode(u[None])[0].flatten(),z)
    numerical=torch.linalg.slogdet(jac)[1]
    analytic=flow.decode(z[None])[1][0]
    torch.testing.assert_close(analytic,numerical,atol=1e-9,rtol=1e-9)


def test_latent_copy_exact_and_gradients():
    flow=make_flow();copy=CopiedStochasticLatentDecoder(flow)
    z=torch.randn(5,16,dtype=torch.float64)
    assert torch.equal(flow.sample_from_gaussian(z),copy.sample_from_gaussian(z))
    x=torch.randn(5,1,4,4,dtype=torch.float64)
    assert torch.equal(flow.log_prob(x),copy.log_prob(x))
    loss=-flow.log_prob(x).mean();loss.backward()
    gradients=[p.grad for p in flow.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0


def test_reject_lost_dimensions_and_boundary_clipping():
    flow=make_flow(True)
    with pytest.raises(ValueError):flow.decode(torch.zeros(2,4,dtype=torch.float64))
    with pytest.raises(ValueError):flow.encode(torch.zeros(2,1,4,4,dtype=torch.float64))
