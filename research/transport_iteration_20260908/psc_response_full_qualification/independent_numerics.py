"""Read-only arithmetic audit of authenticated diagnostic capture banks; no fitting."""
import argparse,json,hashlib,math,pathlib,torch,mpmath as mp
mp.mp.dps=100;torch.set_num_threads(1)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def finite(t):return bool(torch.isfinite(t).all())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scalar(root):
 p=root/'layer_2_kernel_locals.pt';x=load(p);bad=x['inside']&~x['good_theta'];coords=bad.nonzero().tolist();invalid_count=len(coords);coords=coords or [[1,22,5,1]];rows=[]
 for cc in coords:
  c=tuple(cc);idx=int(x['index'][c].item()); get=lambda key:float(x[key][c]); xl=get('xl');yl=get('yl');xr=float(x['xk'][c+(idx+1,)]);yr=float(x['yk'][c+(idx+1,)])
  vals={k:get(k) for k in ('x','xl','yl','width','height','dl','dr','delta','curvature','eta','a','b','c','disc','root','den','theta','mapped','ld','denominator','numerator')};vals.update(xr=xr,yr=yr)
  y,xxl,xxr,yyl,yyr,dl,dr=map(mp.mpf,map(float,[vals['x'],xl,xr,yl,yr,vals['dl'],vals['dr']]));w=xxr-xxl;h=yyr-yyl;delta=h/w
  def f(t):
   q=1-t;den=delta*(t*t+q*q)+(dl+dr)*t*q
   return yyl+h*(delta*t*t+dl*t*q)/den
  low,high=mp.mpf(0),mp.mpf(1)
  for _ in range(380):
   mid=(low+high)/2
   if f(mid)<y:low=mid
   else:high=mid
  t=(low+high)/2;q=1-t;den=delta*(t*t+q*q)+(dl+dr)*t*q;num=dr*t*t+2*delta*t*q+dl*q*q;value=xxl+w*t;ld=-(2*mp.log(delta)+mp.log(num)-2*mp.log(den))
  # Independent single-coordinate reflected arithmetic on captured float32 knots/slopes.
  T=lambda z:torch.tensor(z,dtype=torch.float32)
  X,XL,XR,YL,YR,L,R=map(T,[vals['x'],xl,xr,yl,yr,vals['dl'],vals['dr']]);W=XR-XL;H=YR-YL;D=H/W;cur=L+R-2*D;scale=torch.maximum(D,torch.maximum(L,R));mid=(D/scale+L/scale)/(2*D/scale+L/scale+R/scale);el=(X-YL)/H;er=(YR-X)/H;left=bool(el<=mid);e=el if left else er;l=L if left else R;r=R if left else L;B=l-e*cur;A=D-B;C=-D*e;disc=(l*(1-e)-r*e)**2+4*D**2*e*(1-e);rr=torch.sqrt(disc);short=(-B+rr)/(2*A) if bool(B<0) else -2*C/(B+rr);tt=short if left else 1-short;qq=1-short if left else short;v=XL+tt*W if left else XR-qq*W;de=D+cur*tt*qq;nu=R*tt**2+2*D*tt*qq+L*qq**2;logdet=-(2*torch.log(D)+torch.log(nu)-2*torch.log(de))
  rows.append({'coordinate':cc,'captured_good_theta':bool(x['good_theta'][c]),'captured':vals,'oracle':{'theta':mp.nstr(t,70),'one_minus_theta':mp.nstr(q,70),'value':mp.nstr(value,70),'inverse_logdet':mp.nstr(ld,70),'eta':mp.nstr((y-yyl)/h,70)},'independent_cpu_reflected_fixed_captured_knots':{'orientation_left':left,'eta_left':float(el),'eta_right':float(er),'short':float(short),'theta':float(tt),'complement':float(qq),'value':float(v),'ld':float(logdet),'value_abs_error':float(abs(mp.mpf(float(v))-value)),'ld_abs_error':float(abs(mp.mpf(float(logdet))-ld))},'captured_old_value_error':float(abs(mp.mpf(vals['mapped'])-value)),'captured_old_ld_error':float(abs(mp.mpf(vals['ld'])-ld))})
 return {'locals_sha256':sha(p),'invalid_theta_count':invalid_count,'rows':rows}
def full(root):
 f=root/'full_model';s=json.loads((f/'status.json').read_text());b=load(f/'numerical_bank.pt');g=load(f/'joint_parameter_gradients.pt');ig=load(f/'joint_input_gradients.pt');fw=load(f/'joint_forward.pt');expected=set(s['expected_analysis_parameter_names'])|set(s['expected_residual_parameter_names']);z=fw['encoded'];ld=fw['analysis_ld']+fw['residual_ld']+fw['coarse_ld'];loss=(.5*(z.square()+math.log(2*math.pi)).sum(1)-ld).mean()/3072
 metrics={'source_roundtrip_max_abs':float((b['source']-b['recovered_source']).abs().max()),'source_logdet_cancellation_max_abs':float((b['source_decode_ld']+b['source_encode_ld']).abs().max()),'observed_roundtrip_max_abs':float((b['observed_logits']-b['observed_recovered_logits']).abs().max()),'observed_logdet_cancellation_max_abs':float((b['observed_encode_ld']+b['observed_decode_ld']).abs().max())}
 checks={'source_shape':list(b['source'].shape)==[8,3072],'observed_shape':list(b['observed_logits'].shape)==[32,3,32,32],'finite_bank':all(finite(v) for v in b.values()),'finite_joint_forward':all(finite(v) for v in fw.values()) and finite(loss),'exact_reload':torch.equal(b['generated_logits'],b['exact_reload_logits']) and torch.equal(b['source_decode_ld'],b['exact_reload_ld']),'source_seed_exact':torch.equal(b['source'],torch.randn(8,3072,generator=torch.Generator().manual_seed(78300))),'names_exact':set(g)==expected==set(s['actual_trainable_parameter_names']),'all_parameter_gradients_present_finite':bool(g) and all(v is not None and finite(v) for v in g.values()),'all_input_gradients_present_finite':set(ig)=={'logits','coarse','residual','fixed_root_input'} and all(v is not None and finite(v) for v in ig.values()),'reported_metrics_exact':all(metrics[k]==s[k] for k in metrics),'numeric_gates':all(v<=(.01 if 'logdet' in k else .001) for k,v in metrics.items()),'hashes_exact':sha(f/'source_bank.pt')==s['source_bank_sha256'] and sha(f/'numerical_bank.pt')==s['numerical_bank_sha256']}
 local_rng_matches=checks.pop('source_seed_exact')
 assert all(checks.values()),checks
 return {'checks':checks,'metrics':metrics,'local_cross_version_gaussian_regeneration_matches':local_rng_matches,'source_values_sha256':hashlib.sha256(b['source'].numpy().tobytes()).hexdigest(),'parameter_gradient_count':len(g),'parameter_gradient_elements':sum(v.numel() for v in g.values()),'analysis_count':len(s['expected_analysis_parameter_names']),'residual_count':len(s['expected_residual_parameter_names']),'joint_scalar_loss_recomputed':float(loss),'input_gradients_nonzero':{k:bool((v!=0).any()) for k,v in ig.items()},'status_sha256':sha(f/'status.json'),'unrecomputed_state_assertions':['root_requires_grad_false_and_grad_none','model_state_unchanged'],'scope':'Saved arrays independently checked; model state equality and frozen flags are source-reviewed diagnostic assertions, not reconstructable from gradient arrays alone.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.out.exists();d={}
 for name in ('20260909-response-diagnostic-gpu','20260909-response-full-cpu','20260909-response-full-gpu'):
  p=a.root/name;d[name]={'scalar':scalar(p)}
  if (p/'full_model').exists():d[name]['full']=full(p)
 assert d['20260909-response-full-cpu']['full']['source_values_sha256']==d['20260909-response-full-gpu']['full']['source_values_sha256']
 a.out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v.get('full',{}).get('metrics',{}) for k,v in d.items()},indent=2))
