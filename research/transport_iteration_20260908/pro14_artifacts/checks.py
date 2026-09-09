"""Run once on fabricated inputs only; never import qalt or load native arrays."""
from __future__ import annotations
import hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
import torch
from mechanism import ResponseHead, InnovationResponse, FabricatedFlow, FabricatedBlock

OUT=Path(__file__).resolve().parent
START=time.perf_counter()
report={'scope':'fabricated algebra only; no fits, native data, qalt execution or GPU timings',
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
        'checks':{},'all_checks_pass':False,
        'source_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('checks.py','mechanism.py')}}

def num(x):
    return float(x.detach()) if isinstance(x,torch.Tensor) else float(x)

def record(name,value):
    report['checks'][name]=value
    (OUT/'fabricated_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(name,json.dumps(value),flush=True)

def bound(err,tol):
    assert math.isfinite(err) and err <= tol,(err,tol)

try:
    torch.set_num_threads(1)
    torch.manual_seed(140914)
    torch.set_default_dtype(torch.float64)
    for mode in ('prefix','innovation'):
        f=FabricatedFlow(root=4,n=12,r=3,blocks=2,mode=mode).double()
        z=.7*torch.randn(1,28)
        y,ld=f(z);back,ild=f(y,True)
        jac=torch.autograd.functional.jacobian(lambda v:f(v[None])[0][0],z[0])
        sign,dense=torch.linalg.slogdet(jac)
        rt=num((back-z).abs().max());le=num((ld+ild).abs().max())
        je=num((dense-ld[0]).abs())
        bound(rt,1e-10);bound(le,1e-10);bound(je,1e-10)
        assert float(sign)>0
        loss=-f.log_prob(y.detach()).mean()
        grads=torch.autograd.grad(loss,tuple(f.parameters()))
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        # Directional finite difference checks all active response-head weights.
        params=tuple(f.parameters());dirs=tuple(torch.randn_like(p) for p in params)
        norm=math.sqrt(sum(float(d.square().sum()) for d in dirs));dirs=tuple(d/norm for d in dirs)
        analytic=sum(num((g*d).sum()) for g,d in zip(grads,dirs));eps=1e-5
        saved=[p.detach().clone() for p in params]
        with torch.no_grad():
            for p,s,d in zip(params,saved,dirs):p.copy_(s+eps*d)
        plus=num(-f.log_prob(y.detach()).mean())
        with torch.no_grad():
            for p,s,d in zip(params,saved,dirs):p.copy_(s-eps*d)
        minus=num(-f.log_prob(y.detach()).mean())
        with torch.no_grad():
            for p,s in zip(params,saved):p.copy_(s)
        ge=abs((plus-minus)/(2*eps)-analytic);bound(ge,2e-7)
        record('small_'+mode,{'round_trip_max':rt,'ld_cancellation_max':le,'dense_jacobian_error':je,'directional_gradient_error':ge,'all_parameter_gradients_finite':True})

    for dtype in (torch.float64,torch.float32):
        f=FabricatedFlow().to(dtype=dtype)
        z=torch.randn(4,3072,dtype=dtype);z[0,::157]=20;z[1,::157]=-20
        y,ld=f(z);back,ild=f(y,True)
        rt=num((back-z).abs().max());le=num((ld+ild).abs().max())
        bound(rt,2e-9 if dtype==torch.float64 else 1e-3)
        bound(le,2e-8 if dtype==torch.float64 else 1e-2)
        yy,ll=f(z);assert torch.equal(y,yy) and torch.equal(ld,ll)
        record('full_3072_'+str(dtype),{'batch':4,'round_trip_max':rt,'ld_cancellation_max':le,'same_source_exact_copy':True,'all_3072_coordinates_retained':True})

    b=FabricatedBlock(32,4).double();x=torch.randn(3,32);h=torch.randn(3,4)
    with torch.no_grad():b.response.head.output.weight.zero_();b.response.head.output.bias.zero_()
    y,ld=b(x,h);y0,ld0=b.base(x,h)
    assert torch.equal(y,y0) and torch.equal(ld,ld0)
    record('zero_head_nesting',{'output_equal':True,'logdet_equal':True})

    # Exact model-class witness, instantiated INSIDE the proposed head: n=2,r=1.
    # U-frame rows=(1,1)/sqrt(2), m=0, and
    # s(u)=log(2)*tanh(tanh(tanh(u)^2)). Thus Cov(U,V)=0 but V's
    # conditional energy increases with |U|. All outputs are full-dimensional.
    q=InnovationResponse(2,1,width=32).double();frame=torch.ones(2,1)/math.sqrt(2)
    with torch.no_grad():
        for p in q.parameters():p.zero_()
        q.head.input.weight[0,2]=1
        q.head.output.weight[1,0]=math.sqrt(2)
    def sfun(u):return math.log(2)*math.tanh(math.tanh(math.tanh(u)**2))
    phi=lambda u:math.exp(-u*u/2)/math.sqrt(2*math.pi)
    E=lambda f:quad(lambda u:phi(u)*f(u),-10,10,epsabs=2e-12,epsrel=2e-12,limit=160)[0]
    ev2=E(lambda u:math.exp(2*sfun(u)))
    es=E(sfun);eu2v2=E(lambda u:u*u*math.exp(2*sfun(u)))
    cov_energy=eu2v2-ev2
    gap=.5*math.log(ev2)-es
    assert cov_energy>.1 and gap>0
    grid=torch.linspace(-3,3,81);xx=torch.stack((grid,torch.ones_like(grid)),-1)
    yy,ll=q(xx,torch.zeros(81,1),frame)
    expected=torch.tensor([math.exp(sfun(float(u))) for u in grid])
    bound(num((yy[:,1]-expected).abs().max()),1e-12)
    # Integrate joint normalized density directly over both output coordinates.
    def inner(u):
        s=sfun(u);sig=math.exp(s)
        return quad(lambda v: math.exp(-.5*((v/sig)**2+u*u))/(2*math.pi*sig),
                    -math.inf,math.inf,epsabs=3e-10,epsrel=3e-10,limit=160)[0]
    normal,normal_err=quad(inner,-9,9,epsabs=1e-8,epsrel=1e-8,limit=160)
    bound(abs(normal-1),2e-8)
    record('zero_covariance_nonlinear_energy',{'conditional_scale_formula':'log(2)*tanh(tanh(tanh(u)^2))',
        'cov_U_V_exact':0.0,'E_V_squared':ev2,'cov_U_squared_V_squared':cov_energy,
        'best_independent_Gaussian_KL_gap_nats':gap,
        'density_integral_numeric':normal,'quadrature_error_estimate':normal_err,
        'comparison_scope':'Gaussian witness only; no RQS, FM or image-quality win'})

    # Honest fixed-base failure: independent anchors, but two followers have
    # correlation rho. No diagonal conditional response fits that fixed-base law.
    rho=.6;floor=-.5*math.log(1-rho*rho)
    record('retained_approximation_failure',{'fixed_base':'identity',
        'independent_anchors':True,'follower_correlation':rho,'unremoved_conditional_TC_nats':floor,
        'qualification':'Not a floor for the trainable full model: its base map can change.'})
    # Omitting the new determinant is a WRONG attempt, retained and rejected.
    wrong_cancellation=num(ll.abs().max())
    assert wrong_cancellation>.01
    record('deliberately_wrong_missing_logdet_rejected',{'max_error':wrong_cancellation})
    # Invalid data cannot be silently saturated by tanh.
    failures=[]
    for place in ('input','context','frame','head_parameter','hidden_overflow'):
        layer=InnovationResponse(4,1).double();xf=torch.zeros(2,4);hf=torch.zeros(2,1);uf=torch.ones(4,1)/2
        if place=='input':xf[0,0]=float('nan')
        elif place=='context':hf[0,0]=float('inf')
        elif place=='frame':uf[0,0]=float('inf')
        elif place=='head_parameter':
            with torch.no_grad():layer.head.output.weight[0,0]=float('nan')
        else:
            hf.fill_(2)
            with torch.no_grad():layer.head.input.weight.fill_(1e308)
        try:layer(xf,hf,uf)
        except FloatingPointError:failures.append(place)
    assert len(failures)==5
    record('nonfinite_rejection',{'rejected_cases':failures})
    torch.manual_seed(141001)
    c=InnovationResponse(12,3).double()
    with torch.no_grad(): c.head.output.weight.normal_(0,.15)
    w=(.2*torch.randn(9,3)).requires_grad_();hh=torch.randn(2,3);xx=torch.randn(2,12)
    def frame_objective(w):
        frame=torch.linalg.qr(torch.cat((torch.eye(3),w),0),mode='reduced')[0]
        y,ld=c(xx,hh,frame)
        return .5*y.square().sum()-ld.sum()
    loss=frame_objective(w);g=torch.autograd.grad(loss,w)[0]
    d=torch.randn_like(w);d=d/d.norm();eps=1e-5
    numeric=num((frame_objective(w+eps*d)-frame_objective(w-eps*d))/(2*eps))
    err=abs(numeric-num((g*d).sum()));bound(err,2e-7)
    record('reused_frame_gradient',{'directional_gradient_error':err,'all_finite':bool(torch.isfinite(g).all())})

    # Parameter and asymptotic-operation accounting, not a latency benchmark.
    one=sum(p.numel() for p in ResponseHead().parameters())
    assert one==2624
    target=528624+4*one
    total=lambda w:328188+4*(9*w*w+886*w+885)
    width=next(w for w in range(1,513) if target<=total(w)<=1.05*target)
    record('shape_only_counts',{'extra_parameters_per_block':one,'extra_parameters_total':4*one,
        'candidate_total':target,'candidate_residual':200436+4*one,
        'scalar_width':width,'scalar_total':total(width),'strong_RQS_total':589712,
        'additional_noise_coordinates':0,'new_attention_passes':0,'new_QR_factorizations':0})
    report['all_checks_pass']=True
except BaseException as error:
    report['failure']={'error':repr(error),'traceback':traceback.format_exc()}
    raise
finally:
    report['wall_seconds']=time.perf_counter()-START
    (OUT/'fabricated_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
