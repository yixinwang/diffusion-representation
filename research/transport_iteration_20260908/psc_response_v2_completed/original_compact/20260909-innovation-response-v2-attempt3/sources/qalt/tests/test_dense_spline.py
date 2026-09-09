import pytest

torch=pytest.importorskip('torch')
from qalt.spline import rational_quadratic_spline as reference
from qalt.dense_spline import dense_rational_quadratic_spline as dense,dense_spline_kernel


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_reference_forward_inverse_and_gradients(dtype):
    g=torch.Generator().manual_seed(871)
    values=[torch.linspace(-4,4,31,dtype=dtype)]
    values += [torch.randn(31,k,dtype=dtype,generator=g)*.5 for k in (6,6,5)]
    left=[v.clone().requires_grad_() for v in values]
    right=[v.clone().requires_grad_() for v in values]
    y,ld=reference(*left); yy,lld,valid=dense(*right)
    assert bool(valid)
    tol=3e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(yy,y,atol=tol,rtol=tol)
    torch.testing.assert_close(lld,ld,atol=tol,rtol=tol)
    gl=torch.autograd.grad((y+ld*.1).sum(),left)
    gr=torch.autograd.grad((yy+lld*.1).sum(),right)
    for a,b in zip(gl,gr):torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    back,ild,valid=dense(yy,*right[1:],inverse=True)
    original_back,original_ild=reference(yy,*right[1:],inverse=True)
    assert bool(valid)
    torch.testing.assert_close(back,right[0],atol=tol,rtol=tol)
    torch.testing.assert_close(back,original_back,atol=tol,rtol=tol)
    torch.testing.assert_close(ild,original_ild,atol=tol,rtol=tol)
    inverse_left=[v.detach().clone().requires_grad_() for v in (yy,*left[1:])]
    inverse_right=[v.detach().clone().requires_grad_() for v in (yy,*right[1:])]
    il,ill=reference(*inverse_left,inverse=True)
    ir,irl,status=dense(*inverse_right,inverse=True)
    assert bool(status)
    for a,b in zip(torch.autograd.grad((il+.1*ill).sum(),inverse_left),
                   torch.autograd.grad((ir+.1*irl).sum(),inverse_right)):
        torch.testing.assert_close(a,b,atol=tol,rtol=tol)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_all_tails_boundaries_extreme_parameters_and_zero_gradients(dtype):
    x=torch.tensor([-1e20,-3,3,1e20],dtype=dtype,requires_grad=True)
    p=[torch.full((4,k),1e20,dtype=dtype,requires_grad=True) for k in (4,4,3)]
    y,ld,valid=dense(x,*p,inverse=True)
    assert bool(valid) and torch.equal(y,x) and torch.equal(ld,torch.zeros_like(x))
    grads=torch.autograd.grad((y+ld).sum(),(x,*p))
    assert torch.equal(grads[0],torch.ones_like(x))
    assert all(torch.equal(v,torch.zeros_like(v)) for v in grads[1:])


def test_dense_jacobian_matches_logdet():
    g=torch.Generator().manual_seed(431)
    x=torch.tensor([-.7,.2,1.1],dtype=torch.float64)
    params=[torch.randn(3,k,generator=g,dtype=torch.float64)*.2 for k in (4,4,3)]
    y,ld,status=dense(x,*params)
    jac=torch.autograd.functional.jacobian(lambda value:dense(value,*params)[0],x)
    assert bool(status)
    torch.testing.assert_close(torch.log(torch.diagonal(jac)),ld,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_extreme_stress_rejects_invalid_without_silent_clipping(dtype):
    x=torch.linspace(-2.9,2.9,17,dtype=dtype)
    for magnitude in (30.,1000.,1e20):
        g=torch.Generator().manual_seed(322)
        p=[torch.randn(17,k,generator=g,dtype=dtype)*magnitude for k in (4,4,3)]
        for inverse in (False,True):
            y,ld,status=dense(x,*p,inverse=inverse)
            try:ry,rld=reference(x,*p,inverse=inverse)
            except (ValueError,FloatingPointError):
                assert not bool(status)
            else:
                if bool(status):
                    tol=2e-4 if dtype==torch.float32 else 1e-9
                    torch.testing.assert_close(y,ry,atol=tol,rtol=tol)
                    torch.testing.assert_close(ld,rld,atol=tol,rtol=tol)
    p=[torch.zeros(17,k,dtype=dtype) for k in (4,4,3)]
    p[0][0,0]=float('nan')
    assert not bool(dense(x,*p)[2])
    # Even unused invalid tail parameters must invalidate the batch.
    assert not bool(dense(torch.full_like(x,4),*p)[2])


def test_fullgraph_capture_has_no_scalar_read_or_dynamic_indexing():
    # Eager backend tests graph capture without GPU jobs or compilation benchmarking.
    compiled=torch.compile(dense_spline_kernel,backend='eager',fullgraph=True)
    x=torch.tensor([-.3,.4,4.],dtype=torch.float64)
    p=[torch.zeros(3,k,dtype=torch.float64) for k in (4,4,3)]
    y,ld,status=compiled(x,*p)
    assert bool(status)
    torch.testing.assert_close(y,x,atol=1e-14,rtol=1e-14)
