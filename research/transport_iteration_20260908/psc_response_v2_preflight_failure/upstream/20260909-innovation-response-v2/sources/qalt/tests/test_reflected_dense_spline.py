import copy
import torch
from qalt.dense_spline import dense_spline_kernel
from qalt.reflected_dense_spline import reflected_dense_spline_kernel as reflected
from qalt.dense_global_conditional_spline import DenseGlobalConditionalSplineDecoder
from qalt.global_innovation_flow import GlobalInnovationFlow


def test_interior_one_ulp_failure_without_endpoint_snapping():
    # Fabricated raw parameters; Decimal80 forward-bisection reference below.
    raw = ([5.280641078948975,1.09431791305542,2.089104175567627,3.620931625366211,3.2673206329345703,1.609753131866455,-7.457563400268555,-1.148647427558899],
           [-2.8963139057159424,5.16301155090332,-.47221818566322327,3.604236125946045,3.543950080871582,1.9939310550689697,2.1943840980529785,1.897101640701294],
           [-2.4742302894592285,6.432955741882324,-.4769377112388611,-2.051760673522949,-7.446475028991699,3.570924997329712,.9646139144897461])
    p=tuple(torch.tensor([a],dtype=torch.float32) for a in raw)
    x=torch.tensor([.8682846426963806],dtype=torch.float32)
    assert x.item()<.8682847023010254
    assert ((x-(-2.992781162261963))/3.8610658645629883).item()==1.
    assert not bool(dense_spline_kernel(x,*p,inverse=True)[2])
    y,ld,valid=reflected(x,*p,inverse=True)
    assert bool(valid)
    assert 1.235917091369629<=y.item()<=1.306220531463623
    assert abs(y.item()-1.3062205229177157)<1.2e-7
    assert abs(ld.item()+1.9422820720142764)<2e-6


def test_all_input_raw_gradients_of_values_and_logdet():
    g=torch.Generator().manual_seed(184)
    args=(torch.tensor([-.57,.83],dtype=torch.double),)+tuple(torch.randn(2,k,generator=g,dtype=torch.double)*.3 for k in (8,8,7))
    args=tuple(a.requires_grad_() for a in args)
    def function(*a):
        y,ld,valid=reflected(a[0],*a[1:],inverse=True)
        assert bool(valid)
        return torch.cat((y,ld))
    assert torch.autograd.gradcheck(function,args,eps=1e-6,atol=3e-5,rtol=3e-4)
    for inverse in (False,True):
        a=reflected(args[0],*args[1:],inverse=inverse)
        b=dense_spline_kernel(args[0],*args[1:],inverse=inverse)
        assert bool(a[2]&b[2])
        for x,y in zip(a[:2],b[:2]):torch.testing.assert_close(x,y,atol=2e-12,rtol=0)


def test_knot_neighbors_monotonicity_tails_and_extreme_failure():
    g=torch.Generator().manual_seed(185);n=32;k=8
    raw=tuple(torch.randn(n,size,generator=g)*3 for size in (8,8,7))
    h=.001+(1-k*.001)*torch.softmax(raw[1],-1)
    knots=torch.cat((torch.full((n,1),-3.),-3+6*torch.cumsum(h,-1)[:,:-1],torch.full((n,1),3.)),1)
    grid=torch.cat((torch.nextafter(knots,torch.full_like(knots,-float('inf'))),knots,
        torch.nextafter(knots,torch.full_like(knots,float('inf'))),torch.full((n,1),-20.),torch.full((n,1),20.)),1).sort(1).values
    count=grid.shape[1];p=tuple(a[:,None,:].expand(n,count,-1).reshape(-1,a.shape[-1]) for a in raw)
    x=grid.flatten();y,ld,valid=reflected(x,*p,inverse=True)
    assert bool(valid) and torch.isfinite(y).all() and torch.isfinite(ld).all()
    assert (y.reshape_as(grid).diff(dim=1)>=0).all()  # floating plateaus allowed
    tails=(x<=-3)|(x>=3)
    assert torch.equal(y[tails],x[tails]) and ld[tails].count_nonzero()==0
    huge=tuple(torch.zeros(1,k) for k in (8,8,7));huge[2].fill_(1e30)
    assert not bool(reflected(torch.tensor([.1]),*huge,inverse=True)[2])


def test_full_model_state_counts_default_and_gradients():
    torch.manual_seed(186)
    original=GlobalInnovationFlow(channels=1,size=8,levels=1,pre_layers=2,
        analysis_coarse_layers=2,analysis_detail_layers=2,coarse_layers=2,residual_layers=2,
        width=4,bins=4,attention_heads=1).double()
    with torch.no_grad():
        for p in original.parameters():p.add_(.012*torch.randn_like(p))
    model=copy.deepcopy(original)
    replacement=DenseGlobalConditionalSplineDecoder(3,1,4,layers=2,width=4,bins=4,
        attention_heads=1,backend='dense_reflected').double()
    replacement.load_state_dict(original.residual_decoder.state_dict(),strict=True)
    model.residual_decoder=replacement
    default=DenseGlobalConditionalSplineDecoder(3,1,4,layers=2,width=4,bins=4,attention_heads=1)
    assert default.backend=='dense_eager' and default._spline is dense_spline_kernel
    assert model.state_dict().keys()==original.state_dict().keys()
    assert model.parameter_counts==original.parameter_counts
    z=torch.randn(1,64,dtype=torch.double)*.3
    x,ld=model.decode(z);expected,expected_ld=original.decode(z)
    torch.testing.assert_close(x,expected,atol=2e-11,rtol=0)
    torch.testing.assert_close(ld,expected_ld,atol=2e-11,rtol=0)
    zz,ild=model.encode(x)
    torch.testing.assert_close(zz,z,atol=2e-11,rtol=0)
    torch.testing.assert_close(ild+ld,torch.zeros_like(ld),atol=2e-11,rtol=0)
    jac=torch.autograd.functional.jacobian(lambda v:model.decode(v.reshape(1,-1))[0].flatten(),z.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=2e-11,rtol=0)
    gradients=[]
    for network in (original,model):
        observed=x.detach().requires_grad_()
        gradients.append(torch.autograd.grad(-network.log_prob(observed).sum(),(observed,*network.parameters())))
    for a,b in zip(*gradients):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a,b,atol=3e-10,rtol=3e-10)
