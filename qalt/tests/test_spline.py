import pytest

torch = pytest.importorskip("torch")
from qalt.spline import rational_quadratic_spline as spline


def parameters(shape, bins=8, scale=.7, seed=177):
    g = torch.Generator().manual_seed(seed)
    return tuple(scale * torch.randn(shape + (n,), dtype=torch.float64, generator=g) for n in (bins,bins,bins-1))


def test_zero_raw_parameters_are_identity_and_have_unit_derivative():
    x = torch.tensor([-9., -3., -2.7, -1., 0., 1.1, 2.9, 3., 8.], dtype=torch.float64, requires_grad=True)
    p = parameters(x.shape, scale=0.)
    for inverse in (False,True):
        y, ld = spline(x,*p,inverse=inverse)
        torch.testing.assert_close(y,x,atol=2e-15,rtol=2e-15)
        torch.testing.assert_close(ld,torch.zeros_like(x),atol=2e-15,rtol=0.)
        dx, = torch.autograd.grad(y.sum(),x)
        torch.testing.assert_close(dx,torch.ones_like(x),atol=2e-15,rtol=0.)


def test_nontrivial_roundtrip_both_directions_with_boundaries_and_tails():
    g = torch.Generator().manual_seed(114)
    x = (torch.rand((4,5,7),dtype=torch.float64,generator=g)-.5)*9
    x.flatten()[:6] = torch.tensor([-3.,3.,-3.-1e-10,3.+1e-10,-3.+1e-10,3.-1e-10],dtype=torch.float64)
    p = parameters(x.shape,scale=1.6)
    for direction in (False,True):
        y,ld=spline(x,*p,inverse=direction)
        z,ild=spline(y,*p,inverse=not direction)
        torch.testing.assert_close(z,x,atol=2e-10,rtol=2e-10)
        torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=2e-9,rtol=0.)
        tail=x.abs()>=3
        assert torch.equal(y[tail],x[tail])
        assert torch.equal(ld[tail],torch.zeros_like(ld[tail]))


def test_logdet_matches_independent_autograd_jacobian():
    x=torch.tensor([-.8,.2,1.8],dtype=torch.float64,requires_grad=True)
    p=parameters(x.shape,scale=1.2)
    for inverse in (False,True):
        jac=torch.autograd.functional.jacobian(lambda u:spline(u,*p,inverse=inverse)[0],x)
        _,ld=spline(x,*p,inverse=inverse)
        assert torch.all(torch.diag(jac)>0)
        torch.testing.assert_close(jac,torch.diag(torch.diag(jac)),atol=1e-14,rtol=0.)
        torch.testing.assert_close(torch.log(torch.diag(jac)),ld,atol=2e-12,rtol=2e-12)


def test_parameter_and_input_gradients_pass_gradcheck_both_directions():
    x=torch.tensor([-.71,.32],dtype=torch.float64,requires_grad=True)
    p=tuple(v.requires_grad_() for v in parameters(x.shape,bins=4,scale=.4))
    for inverse in (False,True):
        assert torch.autograd.gradcheck(lambda *args:spline(*args,inverse=inverse),(x,*p),eps=1e-6,atol=3e-5,rtol=3e-4)
        y,ld=spline(x,*p,inverse=inverse)
        grads=torch.autograd.grad((y.square()+ld).sum(),(x,*p))
        assert all(torch.isfinite(v).all() for v in grads)


def test_monotonicity_for_shared_nontrivial_parameters_and_float32():
    x=torch.linspace(-4,4,2001,dtype=torch.float64)
    raw=parameters((),scale=1.4)
    p=tuple(v.expand(x.shape+v.shape) for v in raw)
    y,ld=spline(x,*p)
    assert torch.all(y[1:]>y[:-1])
    restored,_=spline(y,*p,inverse=True)
    torch.testing.assert_close(restored,x,atol=2e-11,rtol=2e-11)
    x32=x.float(); p32=tuple(v.float() for v in p)
    y32,ld32=spline(x32,*p32)
    r32,_=spline(y32,*p32,inverse=True)
    assert torch.isfinite(ld32).all()
    torch.testing.assert_close(r32,x32,atol=3e-4,rtol=3e-4)


def test_scalar_empty_all_tail_and_validation():
    x=torch.tensor(.2,dtype=torch.float64)
    y,ld=spline(x,*parameters(()))
    assert y.shape==ld.shape==torch.Size([])
    empty=torch.empty((0,2),dtype=torch.float64)
    assert spline(empty,*parameters(empty.shape))[0].shape==empty.shape
    tail=torch.tensor([-5.,6.],dtype=torch.float64,requires_grad=True)
    p=tuple(v.requires_grad_() for v in parameters(tail.shape))
    y,ld=spline(tail,*p)
    grads=torch.autograd.grad((y+ld).sum(),(tail,*p))
    assert all(torch.isfinite(v).all() for v in grads)
    assert all(torch.count_nonzero(v)==0 for v in grads[1:])
    with pytest.raises(ValueError): spline(x,*parameters((2,)))
    with pytest.raises(ValueError): spline(x,*parameters(()),min_bin_width=.2)
    with pytest.raises(ValueError): spline(x.half(),*(v.half() for v in parameters(())))


def test_internal_knots_interpolate_and_share_prescribed_derivatives():
    import math
    bins=6
    raw=parameters((),bins=bins,scale=.8)
    widths=.001+(1-bins*.001)*torch.softmax(raw[0],dim=-1)
    heights=.001+(1-bins*.001)*torch.softmax(raw[1],dim=-1)
    xknots=-3+6*widths.cumsum(0)[:-1]
    yknots=-3+6*heights.cumsum(0)[:-1]
    derivative=.001+torch.nn.functional.softplus(raw[2]+math.log(math.expm1(.999)))
    p=tuple(v.expand(xknots.shape+v.shape) for v in raw)
    y,ld=spline(xknots,*p)
    torch.testing.assert_close(y,yknots,atol=2e-14,rtol=0.)
    torch.testing.assert_close(ld,derivative.log(),atol=2e-13,rtol=0.)
    for sign in (-1,1):
        _,nearld=spline(xknots+sign*1e-9,*p)
        torch.testing.assert_close(nearld,derivative.log(),atol=2e-6,rtol=0.)
