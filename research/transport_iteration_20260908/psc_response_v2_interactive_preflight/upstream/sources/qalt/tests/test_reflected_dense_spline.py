import copy
import math
from decimal import Decimal, localcontext
import torch
from qalt.dense_spline import dense_spline_kernel
from qalt.reflected_dense_spline import reflected_dense_spline_kernel as reflected
from qalt.dense_global_conditional_spline import DenseGlobalConditionalSplineDecoder
from qalt.global_innovation_flow import GlobalInnovationFlow


def test_interior_knot_neighbors_against_realized_map():
    # Legacy rejection and softmax rounding differ across Torch platforms.
    # Test the candidate against independent Decimal80 bisection of this
    # platform's realized knots, without requiring the legacy program to fail.
    raw = ([5.280641078948975,1.09431791305542,2.089104175567627,3.620931625366211,3.2673206329345703,1.609753131866455,-7.457563400268555,-1.148647427558899],
           [-2.8963139057159424,5.16301155090332,-.47221818566322327,3.604236125946045,3.543950080871582,1.9939310550689697,2.1943840980529785,1.897101640701294],
           [-2.4742302894592285,6.432955741882324,-.4769377112388611,-2.051760673522949,-7.446475028991699,3.570924997329712,.9646139144897461])
    p=tuple(torch.tensor([a],dtype=torch.float32) for a in raw)
    def knots(raw):
        fractions=.001+(1-8*.001)*torch.softmax(raw,-1)
        return torch.cat((torch.full_like(fractions[:,:1],-3.),
            -3+6*torch.cumsum(fractions,-1)[:,:-1],
            torch.full_like(fractions[:,:1],3.)),dim=-1)
    xk,yk=knots(p[0]),knots(p[1])
    derivatives=torch.cat((torch.ones_like(p[2][:,:1]),
        .001+torch.nn.functional.softplus(p[2]+math.log(math.expm1(1-.001))),
        torch.ones_like(p[2][:,:1])),dim=-1)
    for steps in (1,64):
        x=yk[:,2].clone()
        for _ in range(steps):x=torch.nextafter(x,torch.full_like(x,-float('inf')))
        assert bool((yk[:,1]<x).all() and (x<yk[:,2]).all())
        with localcontext() as ctx:
            ctx.prec=80
            dec=lambda t:Decimal.from_float(t.item())
            xl,xr,yl,yr=map(dec,(xk[0,1],xk[0,2],yk[0,1],yk[0,2]))
            dl,dr=map(dec,(derivatives[0,1],derivatives[0,2]))
            width,height=xr-xl,yr-yl
            delta=height/width;target=(dec(x)-yl)/height
            lo,hi=Decimal(0),Decimal(1)
            for _ in range(300):
                theta=(lo+hi)/2;cross=theta*(1-theta)
                denominator=delta+(dl+dr-2*delta)*cross
                value=(delta*theta*theta+dl*cross)/denominator
                if value<target:lo=theta
                else:hi=theta
            theta=(lo+hi)/2;cross=theta*(1-theta)
            denominator=delta+(dl+dr-2*delta)*cross
            derivative=delta*delta*(dr*theta*theta+2*delta*cross+dl*(1-theta)**2)/denominator**2
            reference=xl+theta*width;reference_ld=-derivative.ln()
            assert xl<reference<xr
            # The wider interior gap makes endpoint snapping detectable at
            # the unchanged value tolerance; one-ULP outputs may round to xr.
            if steps==64:assert xr-reference>Decimal('1.2e-7')
        y,ld,valid=reflected(x,*p,inverse=True)
        assert bool(valid)
        assert float(xl)<=y.item()<=float(xr)
        assert abs(y.item()-float(reference))<1.2e-7
        assert abs(ld.item()-float(reference_ld))<2e-6


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
