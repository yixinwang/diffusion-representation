import copy
import math
import types
import pytest
import torch
from qalt.cached_global_innovation import (
    CachedGlobalInnovationDecoder, CachedInnovationFlow, OrthonormalFrame,
    Transform, integrated_linear, rank_mix, require_valid)


def decoder():
    torch.manual_seed(903)
    d=CachedGlobalInnovationDecoder(1,1,4,width=4,rank=2,channel_embedding=2).double()
    with torch.no_grad():
        for c in d.conditioners:
            c.head.weight.normal_(0,.07);c.head.bias.normal_(0,.07)
            c.eigen_head.weight.normal_(0,.1);c.eigen_head.bias.normal_(0,.1)
    return d


def test_nonidentity_full_residual_jacobian_and_all_parameter_gradients():
    d=decoder();x=torch.randn(1,1,4,4,dtype=torch.double)*.3;c=torch.randn_like(x)
    y,ld=d.decode(x,c);xx,ild=d.encode(y,c)
    assert (y-x).abs().max()>.1
    torch.testing.assert_close(xx,x,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=1e-10,rtol=0)
    j=torch.autograd.functional.jacobian(lambda v:d.decode(v.reshape_as(x),c)[0].flatten(),x.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(j)[1],ld[0],atol=1e-10,rtol=0)
    loss=-d.log_prob(y.detach(),c).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
    assert all(p.grad.abs().sum()>0 for p in d.parameters())


def test_scalar_boundary_tail_and_shape_gradients():
    torch.manual_seed(45)
    x=torch.tensor([-5.,-4.,-3.,0.,3.,4.,5.],dtype=torch.double,requires_grad=True)
    raw=torch.randn(7,7,dtype=torch.double,requires_grad=True)
    out=require_valid(integrated_linear(x,raw));inv=require_valid(integrated_linear(out.value,raw,inverse=True))
    torch.testing.assert_close(inv.value,x,atol=1e-12,rtol=0)
    derivative=torch.autograd.grad(out.value.sum(),x)[0]
    torch.testing.assert_close(derivative.log(),out.logdet,atol=1e-12,rtol=0)
    assert torch.equal(derivative[[0,1,5,6]],torch.ones(4,dtype=torch.double))
    a=torch.tensor([-.231,.612],dtype=torch.double,requires_grad=True)
    b=torch.randn(2,7,dtype=torch.double,requires_grad=True)
    for inverse in (False,True):
        assert torch.autograd.gradcheck(lambda v,r:integrated_linear(v,r,inverse=inverse).value,(a,b))
    with pytest.raises(ValueError):integrated_linear(a,b,bound=float('inf'))


def test_prefix_mask_blocks_current_and_future_information():
    d=decoder();r=torch.randn(1,1,4,4,dtype=torch.double,requires_grad=True);c=torch.randn_like(r)
    head=d.conditioners[2];raw,a=head(r,c)
    grad=torch.autograd.grad(raw.sum()+a.sum(),r)[0]
    assert grad[~head.observed].count_nonzero()==0
    assert grad[head.observed].abs().max()>0


def test_frame_cache_mutation_dtype_load_copy_and_gradient_lifecycle():
    f=OrthonormalFrame(12,2).float().eval();f.prepare_inference()
    with torch.no_grad():
        old=f.matrix().clone();f.bottom.add_(.75);new=f.matrix().clone()
    assert not torch.equal(old,new) and not f.cache_ready
    f.prepare_inference();f.double();assert not f.cache_ready
    with torch.no_grad():
        out=rank_mix(torch.ones(1,12,dtype=torch.double),f.matrix(),torch.full((1,2),.3,dtype=torch.double))
    assert out.valid
    f.prepare_inference();matrix=f.matrix();matrix.square().sum().backward()
    assert f.bottom.grad is not None and f.rotation.grad is not None
    copied=copy.deepcopy(f)
    with torch.no_grad():torch.testing.assert_close(copied.matrix(),f.matrix())
    f.load_state_dict(f.state_dict());assert not f.cache_ready
    f.prepare_inference();f.train();assert not f.cache_ready


def test_aggregate_failure_and_extreme_inputs_are_rejected():
    d=decoder().float()
    def injected(self,v,cache,inverse=False):
        return Transform(v,torch.full((len(v),),torch.finfo(v.dtype).max/2),torch.tensor(True))
    original=d.apply_cached;d.apply_cached=types.MethodType(injected,d)
    x=torch.zeros(1,1,4,4);out=d.transform(x,x)
    assert not out.valid and not torch.isfinite(out.logdet).all()
    with pytest.raises(FloatingPointError):require_valid(out)
    d.apply_cached=original;x.flatten()[0]=float('inf')
    assert not d.transform(x,torch.zeros_like(x)).valid
    huge=torch.full_like(x,torch.finfo(x.dtype).max)
    assert not d.transform(huge,torch.zeros_like(huge)).valid


def test_complete_composition_root_input_gradient_and_source_jacobian():
    torch.manual_seed(19)
    model=CachedInnovationFlow(channels=1,size=8,levels=1,pre_layers=2,
        analysis_coarse_layers=2,analysis_detail_layers=2,coarse_layers=2,
        residual_layers=2,width=4,bins=4,attention_heads=1,
        innovation_rank=2,innovation_channel_embedding=2).double()
    with torch.no_grad():
        for c in model.residual_decoder.conditioners:
            c.head.weight.normal_(0,.03);c.eigen_head.bias.fill_(.15)
    z=torch.randn(1,64,dtype=torch.double)*.15;y,ld=model.decode(z);zz,ild=model.encode(y)
    torch.testing.assert_close(zz,z,atol=1e-10,rtol=0)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=1e-10,rtol=0)
    jac=torch.autograd.functional.jacobian(lambda v:model.decode(v.reshape(1,-1))[0].flatten(),z.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=1e-10,rtol=0)
    for p in model.coarse_decoder.parameters():p.requires_grad_(False)
    model.zero_grad();(-model.log_prob(y.detach()).mean()).backward()
    assert all(p.grad is None for p in model.coarse_decoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model._analysis_parameters())
    # Isolate the frozen root term: it must still propagate through its inputs.
    coarse=torch.randn(1,1,4,4,dtype=torch.double,requires_grad=True)
    root_z,root_ld=model.coarse_decoder.encode(coarse,model._zero(coarse))
    (root_z.square().sum()/2-root_ld.sum()).backward()
    assert coarse.grad is not None and coarse.grad.abs().sum()>0
    assert model.parameter_counts['total']==sum(p.numel() for p in model.parameters())
