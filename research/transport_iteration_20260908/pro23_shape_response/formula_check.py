import json,math
import mpmath as mp
mp.mp.dps=80
rows=[]
for z in [-1e8,-20.,-1.,0.,1.,20.,1e8]:
 for k,t in [(0.,0.),(1e-9,-1e-9),(-math.log(2),.9),(math.log(2),-.9)]:
  a=math.asinh(z);d=a*math.expm1(-k)+t*math.exp(-k)
  b=2*math.sinh(d/2)**2+(z/math.hypot(1,z))*math.sinh(d)
  q=z+z*2*math.sinh(d/2)**2+math.hypot(1,z)*math.sinh(d)
  ld=-k+math.log1p(b)
  zz,kk,tt=map(mp.mpf,(z,k,t));qq=mp.sinh((mp.asinh(zz)+tt)*mp.exp(-kk))
  ll=-kk+mp.log(mp.cosh((mp.asinh(zz)+tt)*mp.exp(-kk)))-mp.log(1+zz*zz)/2
  rows.append({'z':z,'kappa':k,'tau':t,'relative_value_error':float(abs(mp.mpf(q)-qq)/max(1,abs(qq))),'ld_abs_error':float(abs(mp.mpf(ld)-ll)),'identity_exact':q==z and ld==0 if k==t==0 else None})
result={'scope':'fabricated scalar floating checks against80-digit evaluation, not interval certificate or production qualification','rows':rows,'identity_all':all(r['identity_exact'] for r in rows if r['identity_exact'] is not None),'max_relative_value_error':max(r['relative_value_error'] for r in rows),'max_ld_abs_error':max(r['ld_abs_error'] for r in rows)}
print(json.dumps(result,indent=2,allow_nan=False))
