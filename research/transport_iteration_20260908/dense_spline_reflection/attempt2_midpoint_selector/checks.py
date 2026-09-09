"""Bounded fabricated algebra audit only. No fit, native inputs or qalt edits."""
from pathlib import Path
from decimal import Decimal,localcontext
import json,math,time,traceback,hashlib
import torch
import prototype as new
import original_dense_spline as old
P=Path(__file__).resolve().parent
R={'scope':'fabricated scalar reflection prototype only; not native attribution or interval certificate','checks':{},'pass':False}
torch.set_num_threads(1);start=time.perf_counter()
def record(name,value):
 R['checks'][name]=value;(P/'results.json').write_text(json.dumps(R,indent=2)+'\n')
def parameters(n,seed=1582364):
 g=torch.Generator().manual_seed(seed)
 return tuple(torch.randn(n,k,generator=g)*3 for k in (8,8,7))
def geometry(rw,rh,rd):
 k=rw.shape[-1];f=lambda raw:.001+(1-k*.001)*torch.softmax(raw,-1)
 def knots(v):return torch.cat((torch.full_like(v[:,:1],-3),-3+6*torch.cumsum(v,-1)[:,:-1],torch.full_like(v[:,:1],3)),1)
 xk,yk=knots(f(rw)),knots(f(rh))
 dd=torch.cat((torch.ones_like(rd[:,:1]),.001+torch.nn.functional.softplus(rd+math.log(math.expm1(.999))),torch.ones_like(rd[:,:1])),1)
 return xk,yk,dd

def decimal_inverse(x,xl,xr,yl,yr,dl,dr):
 with localcontext() as c:
  c.prec=80
  x,xl,xr,yl,yr,dl,dr=map(lambda v:Decimal.from_float(float(v)),(x,xl,xr,yl,yr,dl,dr))
  w=xr-xl;h=yr-yl;delta=h/w;curv=dl+dr-2*delta
  lo=Decimal(0);hi=Decimal(1)
  for _ in range(180):
   t=(lo+hi)/2;q=1-t;den=delta+curv*t*q;y=yl+h*(delta*t*t+dl*t*q)/den
   if y<x:lo=t
   else:hi=t
  t=(lo+hi)/2;q=1-t;den=delta+curv*t*q
  ld=-(delta*delta*(dr*t*t+2*delta*t*q+dl*q*q)/(den*den)).ln()
  return float(xl+t*w),float(ld)
try:
 # Original observed fabricated failure parameters are saved explicitly.
 case=json.loads((P/'boundary_failure_01.json').read_text())['first']
 args=tuple(torch.tensor([case[k]],dtype=torch.float32) for k in ('raw_w','raw_h','raw_d'))
 x=torch.tensor([case['x']],dtype=torch.float32)
 a=old.dense_spline_kernel(x,*args,inverse=True);b=new.dense_spline_kernel(x,*args,inverse=True)
 assert not bool(a[2]) and bool(b[2])
 ref=decimal_inverse(case['x'],case['xl'],case['xr'],case['yl'],case['yr'],case['dl'],case['dr'])
 ulp=abs(float(torch.nextafter(b[0],torch.full_like(b[0],float('inf')))-b[0]))
 assert abs(float(b[0])-ref[0])<=2*ulp
 assert abs(float(b[1])-ref[1])<3e-5
 record('saved_one_ulp_failure',{'old_valid':bool(a[2]),'new_valid':bool(b[2]),'value':float(b[0]),'logdet':float(b[1]),'decimal80_reference':ref,'value_error':abs(float(b[0])-ref[0]),'logdet_error':abs(float(b[1])-ref[1]),'output_ulp':ulp})
 # All internal knots and their neighboring floats, plus exact endpoints/tails.
 n=128;args=parameters(n);xk,yk,dd=geometry(*args)
 grid=torch.cat((torch.nextafter(yk,torch.full_like(yk,-float('inf'))),yk,torch.nextafter(yk,torch.full_like(yk,float('inf'))),torch.full((n,1),-20.),torch.full((n,1),20.)),1).sort(1).values
 count=grid.shape[1];flat=grid.flatten();expanded=tuple(p[:,None,:].expand(n,count,-1).reshape(-1,p.shape[-1]) for p in args)
 val,ld,good=new.dense_spline_kernel(flat,*expanded,inverse=True)
 assert bool(good);assert torch.isfinite(val).all() and torch.isfinite(ld).all()
 assert (val.reshape(n,count)[:,1:]>=val.reshape(n,count)[:,:-1]).all()
 tails=(flat<=-3)|(flat>=3);assert torch.equal(val[tails],flat[tails]);assert ld[tails].count_nonzero()==0
 record('all_knots_neighbors_monotonic_tails',{'templates':n,'inputs':len(flat),'valid':bool(good),'minimum_increment':float(val.reshape(n,count).diff(dim=1).min()),'tail_entries':int(tails.sum()),'floating_plateaus_allowed':True})
 # Direct float64 recomputation of raw softmax is a separate numerical map.
 g=torch.Generator().manual_seed(991);x=(torch.rand(7,generator=g,dtype=torch.double)*4-2)
 pars=tuple(torch.randn(7,k,generator=g,dtype=torch.double)*.3 for k in (8,8,7))
 allargs=(x,*pars)
 def outputs(*aa):
  y,l,v=new.dense_spline_kernel(aa[0],*aa[1:],inverse=True)
  assert bool(v)
  return torch.cat((y,l))
 aa=tuple(v.clone().requires_grad_() for v in allargs)
 assert torch.autograd.gradcheck(outputs,aa,eps=1e-6,atol=3e-5,rtol=3e-4)
 gradients=[];results=[]
 for dtype in (torch.float32,torch.float64):
  local=tuple(v.to(dtype).detach().requires_grad_() for v in allargs)
  out=outputs(*local);weights=torch.linspace(.2,1.3,len(out),dtype=dtype)
  gr=torch.autograd.grad((out*weights).sum(),local);assert all(torch.isfinite(v).all() for v in gr)
  gradients.append(gr);results.append(out)
 maximum=0.
 for a,b in zip(*gradients):
  maximum=max(maximum,float((a.double()-b).abs().max()));torch.testing.assert_close(a.double(),b,atol=5e-5,rtol=5e-4)
 record('all_input_raw_value_logdet_gradients',{'double_gradcheck':True,'float32_float64_max_absolute_gradient_difference':maximum,'max_output_difference':float((results[0].double()-results[1]).abs().max().detach())})
 # Both analytic inverse branches around theta=.5 must agree in derivative.
 xx,hh,der=geometry(*pars)
 forward_x=(xx[:,3]+xx[:,4])/2
 yy=old.dense_spline_kernel(forward_x,*pars)[0]
 val,ld,valid=new.dense_spline_kernel(yy,*pars,inverse=True)
 assert bool(valid);torch.testing.assert_close(val,forward_x,atol=1e-12,rtol=0)
 record('branch_midpoint_roundtrip',{'max_error':float((val-forward_x).abs().max())})
 # Finite extreme parameters can still overflow float32; reject, do not claim repair.
 huge=tuple(torch.zeros(1,k) for k in (8,8,7));huge[2].fill_(1e30)
 value,ld,valid=new.dense_spline_kernel(torch.tensor([.1]),*huge,inverse=True)
 assert not bool(valid)
 record('unresolved_extreme_derivative_overflow',{'raw_derivative':1e30,'correctly_rejected':True,'not_a_universal_float32_inverse_guarantee':True})
 R['pass']=True
except BaseException as e:
 R['failure']={'error':repr(e),'traceback':traceback.format_exc()};raise
finally:
 R['seconds']=time.perf_counter()-start
 R['source_sha256']={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('prototype.py','original_dense_spline.py','checks.py','boundary_failure_01.json')}
 (P/'results.json').write_text(json.dumps(R,indent=2)+'\n')
print(json.dumps(R,indent=2))
