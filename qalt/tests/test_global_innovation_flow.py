import copy
import math
import pytest

torch=pytest.importorskip('torch')
from qalt.global_innovation_flow import GlobalInnovationFlow


def tiny():
    return GlobalInnovationFlow(channels=1,size=4,levels=1,pre_layers=2,
        analysis_coarse_layers=2,analysis_detail_layers=2,coarse_layers=2,
        residual_layers=2,width=4,bins=4,attention_heads=1).double()


def nonidentity(model):
    # Perturb all parameters, including conditioning and both generative priors.
    generator=torch.Generator().manual_seed(541)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.015*torch.randn(parameter.shape,generator=generator,dtype=parameter.dtype))
    return model


def test_nonidentity_inverse_and_independent_dense_jacobian():
    model=nonidentity(tiny())
    z=torch.linspace(-.8,.9,16,dtype=torch.float64).reshape(1,16)
    x,ld=model.decode(z);back,ild=model.encode(x)
    torch.testing.assert_close(back,z,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),rtol=0,atol=1e-8)
    jac=torch.autograd.functional.jacobian(lambda v:model.decode(v[None])[0].flatten(),z[0])
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],rtol=1e-8,atol=1e-8)
    _,zero_ld=tiny().decode(z)
    assert abs(float((ld-zero_ld).detach()))>1e-5


def test_complete_density_identity_gradient_and_no_sampling_randomness():
    model=nonidentity(tiny())
    z=torch.linspace(-1,1,32,dtype=torch.float64).reshape(2,16)
    state=torch.random.get_rng_state().clone()
    x,ld=model.decode(z)
    assert torch.equal(model.sample_from_gaussian(z),x)
    assert torch.equal(torch.random.get_rng_state(),state)
    expected=-.5*(z.square()+math.log(2*math.pi)).sum(1)-ld
    torch.testing.assert_close(model.log_prob(x),expected,rtol=1e-9,atol=1e-9)
    (-model.log_prob(x.detach()).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.parameter_counts['total']==sum(p.numel() for p in model.parameters())
    assert not any('velocity' in key or 'coarse_prior' in key for key in model.state_dict())
    with pytest.raises(ValueError):model.decode(z[:,:4])


def test_state_roundtrip_preserves_freeze_and_complete_copy():
    original=nonidentity(tiny());original.freeze_analysis();original.train()
    restored=tiny();restored.load_state_dict(copy.deepcopy(original.state_dict()));restored.train()
    assert all(not p.requires_grad for p in restored._analysis_parameters())
    assert not restored.pre_analysis.training and not restored.analysis.training
    assert all(p.requires_grad for p in restored.coarse_decoder.parameters())
    z=torch.arange(16,dtype=torch.float64).reshape(1,16)/20
    assert torch.equal(original.sample_from_gaussian(z),restored.sample_from_gaussian(z))
    original.eval();restored.eval()
    assert torch.equal(original.log_prob(original.decode(z)[0]),restored.log_prob(restored.decode(z)[0]))
