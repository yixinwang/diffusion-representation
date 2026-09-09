import math
import pytest
import torch

from qalt.global_conditional_spline import GlobalConditionalSplineDecoder
from qalt.learned_latent_flow_matching import pack_residuals, unpack_residuals


def model(channels=3, size=2, layers=4):
    torch.manual_seed(27)
    result = GlobalConditionalSplineDecoder(channels, 1, size, layers=layers, width=8, bins=4, attention_heads=2).double()
    with torch.no_grad():
        for layer in result.layers:
            layer.conditioner.output.weight.normal_(0, .025)
            layer.conditioner.output.bias.normal_(0, .025)
    return result


def test_inverse_and_dense_jacobian_with_affine_tails():
    flow=model()
    coarse=torch.randn(1,1,2,2,dtype=torch.double)
    z=torch.linspace(-5,5,12,dtype=torch.double).reshape(1,3,2,2)
    y,ld=flow.decode(z,coarse)
    recovered,ild=flow.encode(y,coarse)
    torch.testing.assert_close(recovered,z,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),rtol=0,atol=1e-9)
    jac=torch.autograd.functional.jacobian(lambda v:flow.decode(v.reshape_as(z),coarse)[0].flatten(),z.flatten())
    sign,actual=torch.linalg.slogdet(jac)
    assert sign.item()==1
    torch.testing.assert_close(ld[0],actual,rtol=1e-8,atol=1e-9)
    # Independent central differences also check derivatives through conditioners.
    eps=1e-5;eye=torch.eye(z.numel(),dtype=z.dtype)*eps
    finite=torch.stack([(flow.decode((z.flatten()+d).reshape_as(z),coarse)[0]-flow.decode((z.flatten()-d).reshape_as(z),coarse)[0]).flatten()/(2*eps) for d in eye],dim=1)
    torch.testing.assert_close(jac,finite,rtol=2e-5,atol=2e-6)
    assert bool((jac.norm(dim=0)>0).all())
    expected=-.5*(z.square()+math.log(2*math.pi)).flatten(1).sum(1)-ld
    torch.testing.assert_close(flow.log_prob(y,coarse),expected,rtol=1e-9,atol=1e-9)


def test_global_distant_cross_scale_and_coarse_influence():
    flow=model(channels=15,size=8)
    z=torch.randn(1,15,8,8,dtype=torch.double,requires_grad=True)
    coarse=torch.randn(1,1,8,8,dtype=torch.double,requires_grad=True)
    layer=flow.layers[0]
    # Channel 0 belongs to deepest details; channel 14 to the finer packed scale.
    assert not layer.mask[0,0,0,1] and layer.mask[0,14,7,7]
    y,_=layer(z,coarse)
    dz,dc=torch.autograd.grad(y[0,0,0,1],(z,coarse))
    assert abs(dz[0,14,7,7].item())>1e-12
    assert abs(dc[0,0,7,7].item())>1e-12
    # Far positions lie outside the two local convolutions' receptive field;
    # attention is necessary for these direct within-layer dependencies.
    assert all(m.dropout==0 for m in flow.modules() if isinstance(m,torch.nn.MultiheadAttention))


def test_conditioner_never_reads_transformed_coordinates():
    flow=model();layer=flow.layers[0]
    z=torch.randn(2,3,2,2,dtype=torch.double)
    coarse=torch.randn(2,1,2,2,dtype=torch.double)
    changed=torch.where(layer.mask,z,z+100)
    def params(x):return layer.conditioner(torch.where(layer.mask,x,torch.zeros_like(x)),x.new_zeros(len(x)),coarse)
    torch.testing.assert_close(params(z),params(changed),rtol=0,atol=0)
    y,_=layer(z,coarse)
    torch.testing.assert_close(y[layer.mask.expand_as(z)],z[layer.mask.expand_as(z)],rtol=0,atol=0)


def test_packed_shapes_all_source_coordinates_and_conditional_sampling():
    flow=model(channels=15)
    blocks=(torch.randn(2,3,2,2,dtype=torch.double),torch.randn(2,3,4,4,dtype=torch.double))
    z=pack_residuals(blocks);coarse=torch.randn(2,1,2,2,dtype=torch.double)
    y=flow.sample_from_gaussian(z,coarse)
    assert sum(v[0].numel() for v in unpack_residuals(y,1,2))==flow.dimension==60
    torch.testing.assert_close(flow.encode(y,coarse)[0],z,rtol=1e-8,atol=1e-8)
    # Sampling uses supplied residual noise and coarse context deterministically.
    torch.manual_seed(111);first=flow.sample_from_gaussian(z,coarse)
    torch.manual_seed(222);torch.testing.assert_close(first,flow.sample_from_gaussian(z,coarse),rtol=0,atol=0)
    assert not torch.equal(first,flow.sample_from_gaussian(z,coarse+.5))
    count=sum(p.numel() for p in flow.parameters())
    assert flow.parameter_counts=={'residual_decoder':count,'total':count}
    with pytest.raises(ValueError):flow.sample_from_gaussian(z.flatten(1)[:,:-1],coarse)
    with pytest.raises(ValueError):flow.sample_from_gaussian(z,coarse[:,:,:1])


def test_identity_initialization_and_conditional_training_gradients():
    torch.manual_seed(1)
    flow=GlobalConditionalSplineDecoder(3,1,2,width=8,bins=4,attention_heads=2).double()
    z=torch.randn(3,3,2,2,dtype=torch.double);coarse=torch.randn(3,1,2,2,dtype=torch.double)
    y,ld=flow.decode(z,coarse)
    torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(ld,torch.zeros_like(ld),rtol=0,atol=1e-12)
    (-flow.log_prob(z,coarse).mean()).backward()
    for layer in flow.layers:
        assert torch.isfinite(layer.conditioner.output.weight.grad).all()
        assert layer.conditioner.output.weight.grad.abs().sum()>0
