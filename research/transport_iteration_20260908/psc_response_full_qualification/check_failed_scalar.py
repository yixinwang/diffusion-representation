"""Standalone high-precision audit of compact actual captured scalar, no model/data load."""
import json,sys,math
from pathlib import Path
import mpmath as mp
mp.mp.dps=100
f=json.loads(Path(sys.argv[1]).read_text());v=f['captured'];M=lambda k:mp.mpf(float(v[k]));x,xl,xr,yl,yr,dl,dr=map(M,('x','xl','xr','yl','yr','dl','dr'));w=xr-xl;h=yr-yl;d=h/w
assert yl<x<yr and 0<dl and 0<dr
# Monotone bisection independently avoids either quadratic-root formula.
def forward(t):
 c=1-t;q=d*(t*t+c*c)+(dl+dr)*t*c
 return yl+h*(d*t*t+dl*t*c)/q
lo,hi=mp.mpf(0),mp.mpf(1)
for _ in range(380):
 mid=(lo+hi)/2
 if forward(mid)<x:lo=mid
 else:hi=mid
t=(lo+hi)/2;c=1-t;q=d*(t*t+c*c)+(dl+dr)*t*c;a=dr*t*t+2*d*t*c+dl*c*c
value=xl+w*t;ld=-(2*mp.log(d)+mp.log(a)-2*mp.log(q))
assert 0<t<1 and v['theta']>1 and v['mapped']>v['xr']
assert abs(value-mp.mpf(f['oracle']['value']))<mp.mpf('1e-65')
assert abs(ld-mp.mpf(f['oracle']['inverse_logdet']))<mp.mpf('1e-65')
print(json.dumps({'theta':mp.nstr(t,70),'value':mp.nstr(value,70),'inverse_ld':mp.nstr(ld,70),'captured_theta_excess':v['theta']-1,'captured_x_distance_below_yr':float(yr-x),'reflected_saved_value_error':float(abs(mp.mpf(f['independent_cpu_reflected_fixed_captured_knots']['value'])-value)),'checks_pass':True,'scope':'ordinary high-precision bisection, not interval certification or GPU rerun'},indent=2))
