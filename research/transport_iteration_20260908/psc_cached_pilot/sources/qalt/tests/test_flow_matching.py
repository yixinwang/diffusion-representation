import math
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from qalt.flow_matching import FullTensorFlowMatching, HierarchicalFlowMatching, heun_integrate
from qalt.multiscale_flow import MultiscaleSplineFlow, haar_split, haar_merge


def test_zero_field_preserves_full_source_and_hierarchical_chart():
    g = torch.Generator().manual_seed(61)
    z = torch.randn((3,64), dtype=torch.float64, generator=g)
    full = FullTensorFlowMatching(channels=1,size=8,levels=2,width=4).double()
    hierarchical = HierarchicalFlowMatching(channels=1,size=8,levels=2,width=4).double()
    spline = MultiscaleSplineFlow(channels=1,size=8,levels=2,width=4,coarse_layers=2,detail_layers=2).double()
    assert hierarchical.block_shapes == spline.block_shapes
    assert sum(math.prod(s) for s in hierarchical.block_shapes) == hierarchical.dimension == full.dimension == 64
    assert torch.equal(full.sample_from_gaussian(z,steps=3),z.reshape(3,1,8,8))
    source = hierarchical._source_blocks(z)
    expected = source[0]
    for detail in source[1:]: expected = haar_merge(expected,detail)
    assert torch.equal(hierarchical.sample_from_gaussian(z,steps=3),expected)
    observed=[]; current=expected
    for _ in range(2):
        current,detail=haar_split(current);observed.append(detail)
    reconstructed=torch.cat([current.flatten(1)]+[d.flatten(1) for d in reversed(observed)],dim=1)
    torch.testing.assert_close(reconstructed,z,atol=1e-15,rtol=1e-15)


def test_heun_linear_solution_and_second_order_refinement():
    calls=[]
    def velocity(x,t,context=None):
        assert context is None
        calls.append(t.clone())
        return .7*x
    initial=torch.tensor([[[[1.,-2.]]]],dtype=torch.float64)
    errors=[]
    for steps in (8,16,32):
        calls.clear();out=heun_integrate(velocity,initial,steps)
        exact_discrete=initial*(1+.7/steps+.5*(.7/steps)**2)**steps
        torch.testing.assert_close(out,exact_discrete,atol=1e-14,rtol=1e-14)
        errors.append(float((out-initial*math.exp(.7)).abs().max()))
        assert len(calls)==2*steps and calls[0].item()==0 and calls[-1].item()==1
    assert errors[0]/errors[1]>3.8 and errors[1]/errors[2]>3.8
    assert errors[-1]<.0003


def test_dimension_weighted_losses_and_input_parameter_gradients():
    x=torch.randn((2,1,8,8),dtype=torch.float64,generator=torch.Generator().manual_seed(81))
    for cls in (FullTensorFlowMatching,HierarchicalFlowMatching):
        model=cls(channels=1,size=8,levels=2,width=4).double()
        loss=model.training_loss(x,generator=torch.Generator().manual_seed(92))
        noise=torch.randn((2,64),dtype=torch.float64,generator=torch.Generator().manual_seed(92))
        if cls is FullTensorFlowMatching: target=x.flatten(1)
        else:
            coarse=x;details=[]
            for _ in range(2): coarse,detail=haar_split(coarse);details.append(detail)
            target=torch.cat([coarse.flatten(1)]+[d.flatten(1) for d in reversed(details)],dim=1)
        torch.testing.assert_close(loss,(target-noise).square().mean(),atol=1e-14,rtol=1e-14)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert any(torch.count_nonzero(p.grad)>0 for p in model.parameters())


class ContextRecorder(nn.Module):
    def __init__(self): super().__init__();self.contexts=[]
    def forward(self,x,t,context=None):
        assert context is not None
        self.contexts.append(context.clone())
        return context.repeat(1,3,1,1)


class UnitField(nn.Module):
    def forward(self,x,t,context=None):
        assert context is None
        return torch.ones_like(x)


def test_generation_context_is_only_its_own_previously_generated_parent():
    model=HierarchicalFlowMatching(channels=1,size=8,levels=2,width=4).double()
    # Calling training_loss cannot stash an observed conditional image.
    z=torch.zeros((1,64),dtype=torch.float64)
    before=model.sample_from_gaussian(z,steps=2)
    model.training_loss(torch.full((1,1,8,8),17.,dtype=torch.float64))
    assert torch.equal(model.sample_from_gaussian(z,steps=2),before)
    model.coarse_velocity=UnitField()
    first,second=ContextRecorder(),ContextRecorder()
    model.detail_velocities=nn.ModuleList([first,second])
    out=model.sample_from_gaussian(z,steps=2)
    coarse=torch.ones((1,1,2,2),dtype=torch.float64)
    assert len(first.contexts)==4 and all(torch.equal(c,coarse) for c in first.contexts)
    next_coarse=haar_merge(coarse,coarse.repeat(1,3,1,1))
    assert len(second.contexts)==4 and all(torch.equal(c,next_coarse) for c in second.contexts)
    torch.testing.assert_close(out,haar_merge(next_coarse,next_coarse.repeat(1,3,1,1)))
    with pytest.raises(TypeError): model.sample_from_gaussian(z,context=coarse)


def test_deterministic_random_loss_and_shape_validation():
    model=HierarchicalFlowMatching(channels=1,size=8,levels=2,width=4)
    x=torch.zeros((2,1,8,8))
    a=model.training_loss(x,torch.Generator().manual_seed(12))
    b=model.training_loss(x,torch.Generator().manual_seed(12))
    assert torch.equal(a,b)
    with pytest.raises(ValueError): model.sample_from_gaussian(torch.zeros((2,63)))
    with pytest.raises(ValueError): model.sample_from_gaussian(torch.zeros((2,64)),steps=0)
    with pytest.raises(ValueError): model.training_loss(torch.zeros((2,1,7,7)))
    with pytest.raises(ValueError): HierarchicalFlowMatching(size=7,levels=2)
