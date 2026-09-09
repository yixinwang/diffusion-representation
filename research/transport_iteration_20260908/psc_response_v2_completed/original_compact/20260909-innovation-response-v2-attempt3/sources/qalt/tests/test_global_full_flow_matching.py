import pytest

torch=pytest.importorskip('torch')
from qalt.global_full_flow_matching import GlobalFullFlowMatching


def test_pack_is_exact_complete_permutation_with_unit_jacobian():
    model=GlobalFullFlowMatching(channels=1,size=4,packing_factor=2,width=4,attention_heads=1).double()
    x=torch.arange(16,dtype=torch.float64).reshape(1,1,4,4)
    packed=model.pack(x)
    assert packed.shape == (1,4,2,2)
    assert torch.equal(model.unpack(packed),x)
    assert torch.equal(packed.flatten().sort().values,x.flatten())
    jac=torch.autograd.functional.jacobian(lambda value:model.pack(value.reshape(1,1,4,4)).flatten(),x.flatten())
    assert abs(float(torch.linalg.det(jac))) == 1


@pytest.mark.parametrize('path',['ordinary','gaussian_reference'])
def test_complete_source_zero_field_identity_and_gradients(path):
    model=GlobalFullFlowMatching(channels=1,size=4,packing_factor=2,width=4,attention_heads=1,path=path).double()
    z=torch.randn(2,16,dtype=torch.float64)
    before=torch.random.get_rng_state().clone()
    assert torch.equal(model.sample_from_gaussian(z,steps=3),z.reshape(2,1,4,4))
    assert torch.equal(before,torch.random.get_rng_state())
    loss=model.training_loss(z.reshape(2,1,4,4),generator=torch.Generator().manual_seed(42))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):model.sample_from_gaussian(z[:,:4])
    with pytest.raises(ValueError):model._field(model.pack(z.reshape(2,1,4,4)),torch.zeros(2,dtype=torch.float64),torch.zeros(1))


def test_attention_can_transmit_dependency_between_distant_tokens():
    # Activate output so initialized-zero prediction does not conceal reachability.
    torch.manual_seed(114)
    model=GlobalFullFlowMatching(channels=1,size=32,packing_factor=4,width=8,attention_heads=2).double()
    torch.nn.init.normal_(model.velocity.output.weight,std=.1)
    x=torch.randn(1,16,8,8,dtype=torch.float64,requires_grad=True)
    value=model._field(x,torch.tensor([.4],dtype=torch.float64))[0,0,0,0]
    grad=torch.autograd.grad(value,x)[0]
    assert grad[0,:,7,7].abs().max()>1e-10
    assert model.dimension == x.numel()


def test_path_options_have_identical_count_and_default_full_dimension():
    ordinary=GlobalFullFlowMatching()
    reference=GlobalFullFlowMatching(path='gaussian_reference')
    assert ordinary.dimension==3072
    assert ordinary.packed_channels==48 and ordinary.packed_size==8
    assert ordinary.parameter_counts==reference.parameter_counts
    assert ordinary.parameter_counts['zero_context_input_weights']==124*9
