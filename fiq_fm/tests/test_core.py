import torch

from fiqfm.core import OrthogonalChart, fit_flow_moment_chart, random_orthogonal, subspace_sine
from fiqfm.data import ExactQuotientDistribution, digits_splits


def test_orthogonal_chart_roundtrip_and_loss_identity():
    q=random_orthogonal(9,3); chart=OrthogonalChart(q,3); x=torch.randn(64,9); z,r=chart.transform(x)
    assert torch.allclose(chart.inverse(z,r),x,atol=1e-5)
    v=torch.randn_like(x); a=torch.randn_like(x)
    # coordinate residual and induced ambient residual have identical Euclidean norms.
    assert torch.allclose(((v@q-a)**2).sum(-1),((v-a@q.T)**2).sum(-1),atol=1e-5)


def test_flow_moment_recovers_linear_gaussian_active_subspace():
    g=torch.Generator().manual_seed(5); q=random_orthogonal(12,6)
    eig=torch.tensor([5.,4.,3.]+[.4]*9); x=torch.randn(20000,12,generator=g)@torch.diag(eig.sqrt())@q.T
    chart,_=fit_flow_moment_chart(x,3,seed=7,n_pairs=60000)
    assert subspace_sine(chart.q,q,3)<.08


def test_exact_quotient_diagonal_gap_positive():
    d=ExactQuotientDistribution(); s=d.sample(10000,9); gap=d.diagonal_kl_gap(s["z"])
    assert float(gap.mean())>.1
    assert torch.all(gap>=0)


def test_digits_split_has_no_overlap():
    _,_,m=digits_splits(seed=3)
    a,b,c=set(m["train_indices"]),set(m["val_indices"]),set(m["test_indices"])
    assert not (a&b or a&c or b&c)
    assert len(a|b|c)==1797


def test_fiber_gauge_refinement_recovers_pair_blocks():
    from fiqfm.core import refine_fiber_gauge
    d=ExactQuotientDistribution(D=18,d=2,seed=77)
    tr=d.sample(7000,21); va=d.sample(3500,22)
    # Deliberately scramble the true residual axes while preserving the exact active quotient.
    perm=torch.tensor([7,0,13,2,8,3,12,5,1,11,6,15,4,9,14,10])
    q=torch.cat([d.q[:,:2],d.q[:,2:][:,perm]],1)
    chart=OrthogonalChart(q,2)
    refined,diag=refine_fiber_gauge(chart,tr['x'],va['x'],block_size=2,seed=99)
    align=(refined.q[:,2:].T@d.q[:,2:]).abs().argmax(1).tolist()
    groups=[set(align[i:i+2]) for i in range(0,16,2)]
    truth=[{2*j,2*j+1} for j in range(8)]
    assert all(g in truth for g in groups)
    assert diag['fraction_score_within_blocks']>.75
