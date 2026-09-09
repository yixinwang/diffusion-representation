import copy
from contextlib import ExitStack
from unittest.mock import patch
import pytest
import torch

from qalt.cached_global_innovation import CachedGlobalInnovationDecoder, CachedInnovationFlow
from qalt.innovation_response import InnovationResponse, InnovationResponseDecoder, InnovationResponseFlow


def tiny(mode="innovation", nonzero=True):
    torch.manual_seed(1401)
    model = InnovationResponseDecoder(1, 1, 4, width=4, rank=2,
                                      channel_embedding=2, response_mode=mode).double()
    if nonzero:
        with torch.no_grad():
            for c in model.conditioners:
                c.head.weight.normal_(0, .04)
                c.eigen_head.weight.normal_(0, .05)
            for response in model.responses:
                response.output.weight.normal_(0, .15)
                response.output.bias.normal_(0, .1)
    return model


@pytest.mark.parametrize("mode", ["innovation", "prefix"])
def test_nonidentity_roundtrip_dense_jacobian_and_all_parameter_gradients(mode):
    model = tiny(mode)
    source = torch.randn(2, 1, 4, 4, dtype=torch.double) * .4
    coarse = torch.randn_like(source)
    value, ld = model.decode(source, coarse)
    recovered, ild = model.encode(value, coarse)
    assert (value-source).abs().max() > .01
    torch.testing.assert_close(recovered, source, atol=2e-11, rtol=0)
    torch.testing.assert_close(ld+ild, torch.zeros_like(ld), atol=2e-11, rtol=0)
    jac = torch.autograd.functional.jacobian(
        lambda v: model.decode(v.reshape(1,1,4,4), coarse[:1])[0].flatten(), source[0].flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1], ld[0], atol=2e-11, rtol=0)
    (-model.log_prob(value.detach(), coarse).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in model.parameters())


def test_identity_nesting_and_no_extra_conditioner_or_frame_pass():
    model = tiny(nonzero=False)
    with torch.no_grad():
        for c in model.conditioners:
            c.head.weight.normal_(0,.04)
            c.eigen_head.bias.fill_(.1)
    old = CachedGlobalInnovationDecoder(1,1,4,width=4,rank=2,channel_embedding=2).double()
    old.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('responses.')})
    z = torch.randn(1,1,4,4,dtype=torch.double)
    c = torch.randn_like(z)
    counts = [0]*4
    hooks = [head.register_forward_hook(lambda m,a,o,b=b: counts.__setitem__(b,counts[b]+1))
             for b,head in enumerate(model.conditioners)]
    with ExitStack() as stack:
        calls = [stack.enter_context(patch.object(f,'matrix',wraps=f.matrix)) for f in model.frames]
        actual = model.decode(z,c)
        assert all(call.call_count==1 for call in calls)
    expected = old.decode(z,c)
    assert counts == [1]*4
    for h in hooks: h.remove()
    for a,b in zip(actual,expected): assert torch.equal(a,b)
    raw,alpha = old.conditioners[1](z,c)
    raw2,alpha2,summary = old.conditioners[1](z,c,return_summary=True)
    assert torch.equal(raw,raw2) and torch.equal(alpha,alpha2) and summary.shape==(1,16)
    assert not any(k.startswith('responses.') for k in old.state_dict())


def test_anchor_triangularity_prefix_control_and_masked_summary_gradients():
    model = tiny(); response = model.responses[2]
    z = torch.randn(1,4,dtype=torch.double,requires_grad=True)
    prefix = torch.randn(1,1,4,4,dtype=torch.double,requires_grad=True)
    coarse = torch.randn_like(prefix,requires_grad=True)
    cache = model.cache_context(prefix,coarse,2)
    out = response(z,cache.summary,cache.frame)
    assert torch.equal(out.value[:,response.anchors],z[:,response.anchors])
    gradient = torch.autograd.grad(out.value[:,response.followers].sum(),
                                  (z,prefix,coarse),retain_graph=True)
    assert gradient[0][:,response.anchors].abs().sum()>0
    assert gradient[1][~model.conditioners[2].observed].count_nonzero()==0
    assert gradient[1][model.conditioners[2].observed].abs().sum()>0
    assert gradient[2].abs().sum()>0
    control = copy.deepcopy(response); control.mode='prefix'
    control_out = control(z,cache.summary,cache.frame)
    grad = torch.autograd.grad(control_out.value[:,control.followers].sum(),z)[0]
    assert grad[:,control.anchors].count_nonzero()==0


def test_full_model_counts_source_budget_and_copy():
    baseline = CachedInnovationFlow()
    innovation = InnovationResponseFlow()
    prefix = InnovationResponseFlow(response_mode='prefix')
    assert baseline.parameter_counts['total']==528624
    assert innovation.parameter_counts['total']==prefix.parameter_counts['total']==539120
    assert sum(p.numel() for p in innovation.residual_decoder.responses.parameters())==10496
    for response in innovation.residual_decoder.responses:
        assert len(response.anchors)==16 and len(response.followers)==704
        assert response.anchors.tolist()==[0,48,96,144,192,240,288,336,383,431,479,527,575,623,671,719]
        assert len(torch.unique(torch.cat((response.anchors,response.followers))))==720
    scalar = CachedInnovationFlow(innovation_width=42,use_mixer=False)
    assert scalar.parameter_counts['total']==544080
    torch.manual_seed(1402)
    model = InnovationResponseFlow(channels=1,size=8,levels=1,pre_layers=2,
        analysis_coarse_layers=2,analysis_detail_layers=2,coarse_layers=2,residual_layers=2,
        width=4,bins=4,attention_heads=1,innovation_rank=2,innovation_channel_embedding=2).double()
    with torch.no_grad():
        for r in model.residual_decoder.responses: r.output.weight.normal_(0,.1)
    z=torch.randn(1,64,dtype=torch.double)*.25
    x,ld=model.decode(z);zz,ild=model.encode(x)
    torch.testing.assert_close(z,zz,atol=3e-11,rtol=0)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=3e-11,rtol=0)
    copied=copy.deepcopy(model)
    cx,cld=copied.decode(z)
    assert torch.equal(cx,x) and torch.equal(cld,ld)
    jac=torch.autograd.functional.jacobian(lambda v:model.decode(v.reshape(1,-1))[0].flatten(),z.flatten())
    assert torch.all(jac.abs().sum(0)>0)  # every full-source coordinate affects output
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=3e-11,rtol=0)
    for p in model.coarse_decoder.parameters():p.requires_grad_(False)
    (-model.log_prob(x.detach()).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model._analysis_parameters())
    assert all(p.grad is None for p in model.coarse_decoder.parameters())


def test_public_transform_refreshes_frame_cache_after_mutation_dtype_and_load():
    model=tiny().eval();z=torch.randn(1,1,4,4,dtype=torch.double);c=torch.randn_like(z)
    model.prepare_inference()
    with torch.no_grad():
        old=model.decode(z,c)[0]
        model.frames[1].bottom.add_(.4)
        new=model.decode(z,c)[0]
    assert not torch.equal(old,new)
    fresh=copy.deepcopy(model);fresh.load_state_dict(model.state_dict())
    with torch.no_grad():torch.testing.assert_close(new,fresh.decode(z,c)[0],atol=0,rtol=0)
    model.prepare_inference();model.float()
    assert all(not f.cache_ready for f in model.frames)
    y,ld=model.decode(z.float(),c.float());zz,ild=model.encode(y,c.float())
    torch.testing.assert_close(zz,z.float(),atol=3e-5,rtol=0)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=3e-5,rtol=0)


def test_nonfinite_inputs_and_hidden_scale_overflow_fail_closed():
    response=InnovationResponse(4,2).float()
    z=torch.zeros(1,4);summary=torch.zeros(1,16);frame=torch.ones(4,2)
    with torch.no_grad():
        response.output.bias[2:]=torch.finfo(torch.float32).max
    result=response(z,summary,frame)
    assert torch.isfinite(result.value).all()  # tanh would otherwise hide overflow
    assert not bool(result.valid)
    model=tiny().float();source=torch.zeros(1,1,4,4);source.flatten()[0]=float('nan')
    with pytest.raises(FloatingPointError):model.decode(source,torch.zeros_like(source))
    with pytest.raises(ValueError):InnovationResponseDecoder(1,1,4,width=4,rank=2,use_mixer=False)
    with pytest.raises(ValueError):InnovationResponse(720,17)
