import pytest

torch = pytest.importorskip('torch')
from qalt.gaussian_reference_flow_matching import (
    gaussian_reference_path, gaussian_reference_loss, GaussianReferenceResidualFlowMatching)
from qalt.learned_latent_flow_matching import GlobalConditionalVelocity


def test_path_derivative_endpoints_and_original_velocity_equivalence():
    rng = torch.Generator().manual_seed(97)
    z, r = [torch.randn(5, 2, 2, 2, dtype=torch.float64, generator=rng) for _ in range(2)]
    t = torch.linspace(.1, .9, 5, dtype=torch.float64)
    y, w, weight = gaussian_reference_path(z, r, t)
    eps = 1e-6
    difference = (gaussian_reference_path(z,r,t+eps)[0]-gaussian_reference_path(z,r,t-eps)[0])/(2*eps)
    torch.testing.assert_close(w, difference, atol=5e-10, rtol=5e-10)
    torch.testing.assert_close(gaussian_reference_path(z,r,torch.zeros_like(t))[0], z, atol=0,rtol=0)
    torch.testing.assert_close(gaussian_reference_path(z,r,torch.ones_like(t))[0], r, atol=0,rtol=0)
    prediction = .3*y+.7
    loss = gaussian_reference_loss(lambda value,time,context: .3*value+.7, r, source=z, time=t)
    tb = t[:,None,None,None]
    x = (1-tb)*z+tb*r
    original_velocity = (2*tb-1)/weight*x+weight.sqrt()*prediction
    torch.testing.assert_close(loss, (original_velocity-(r-z)).square().mean(), atol=2e-15,rtol=2e-15)


def test_gaussian_covariance_zero_by_exact_linear_coefficients():
    # Independent N(0,I) input covariance: Cov(Y,W)=sum of coefficient products.
    # This checks the exact Gaussian conditional-mean identity without Monte Carlo.
    t = torch.linspace(0,1,101,dtype=torch.float64)
    zero = torch.zeros(101,1,dtype=torch.float64)
    one = torch.ones_like(zero)
    yz,wz,_ = gaussian_reference_path(one,zero,t)
    yr,wr,_ = gaussian_reference_path(zero,one,t)
    torch.testing.assert_close(yz.square()+yr.square(),one,atol=4e-16,rtol=4e-16)
    torch.testing.assert_close(yz*wz+yr*wr,zero,atol=5e-16,rtol=0)
    assert bool((wz.square()+wr.square()>0).all()) # raw target noise remains


def test_zero_field_identity_context_and_no_sampling_randomness():
    model = GaussianReferenceResidualFlowMatching(2,1,2,width=4,attention_heads=1).double()
    z = torch.randn(3,2,2,2,dtype=torch.float64)
    coarse = torch.randn(3,1,2,2,dtype=torch.float64)
    state = torch.random.get_rng_state().clone()
    for steps in (1,2,8):
        assert torch.equal(model.sample_from_gaussian(z,coarse,steps),z)
    assert torch.equal(torch.random.get_rng_state(),state)
    seen = []
    handle = model.velocity.register_forward_pre_hook(lambda module,args: seen.append(args[2]))
    model.sample_from_gaussian(z,coarse,steps=2)
    handle.remove()
    assert len(seen)==4 and all(context is coarse for context in seen)
    same_network = GlobalConditionalVelocity(2,1,2,4,1)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in same_network.parameters())


def test_training_gradients_reproducible_and_invalid_source_rejected():
    model = GaussianReferenceResidualFlowMatching(2,1,2,width=4,attention_heads=1).double()
    residual = torch.arange(24,dtype=torch.float64).reshape(3,2,2,2)/12
    context = torch.ones(3,1,2,2,dtype=torch.float64)
    first = model.training_loss(residual,context,torch.Generator().manual_seed(7))
    second = model.training_loss(residual,context,torch.Generator().manual_seed(7))
    assert torch.equal(first,second)
    first.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    with pytest.raises(ValueError):
        model.sample_from_gaussian(residual.flatten(1),context)
    with pytest.raises(ValueError):
        gaussian_reference_path(residual,residual,torch.full((3,),1.1,dtype=torch.float64))
