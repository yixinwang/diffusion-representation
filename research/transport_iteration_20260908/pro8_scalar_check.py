"""Ordinary scalar checks only; NOT an interval certificate; no model fitting."""
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import differential_evolution, minimize_scalar
import math,json

def derivative_bound_value(v):
 t,z,e=v;s2=(1-t)**2+t*t;d2=2*(1-t)**2+t*t
 k=t/math.sqrt(d2);A=2*e*(1-t)/(s2*math.sqrt(d2))
 phi=math.exp(-z*z/2)/math.sqrt(2*math.pi);den=1+e*(2*ndtr(z)-1)
 return A*k*phi*abs(z*den+2*e*phi)/(den*den)
r=differential_evolution(lambda v:-derivative_bound_value(v),[(0,1),(-8,8),(.4,.6)],tol=1e-11,seed=4)
def field(t,y,e):
 s2=(1-t)**2+t*t;d=math.sqrt(2*(1-t)**2+t*t);z=t*y/d
 return 2*e*(1-t)/(s2*d)*math.exp(-z*z/2)/math.sqrt(2*math.pi)/(1+e*(2*ndtr(z)-1))
def median_gap(e,n):
 y=0.;h=1/n
 for i in range(n):
  t=i/n;k1=field(t,y,e);k2=field(t+h,y+h*k1,e);y+=h*(k1+k2)/2
 u=ndtr(y)
 return .5-u-e*(u*u-u)
records=[]
for calls in [4,8,16,32,64]:
 n=calls//2
 opt=minimize_scalar(lambda e:median_gap(e,n),bounds=(.4,.6),method='bounded',options={'xatol':1e-13})
 e=min([(.4,median_gap(.4,n)),(.6,median_gap(.6,n)),(opt.x,opt.fun)],key=lambda a:a[1])
 records.append(dict(calls=calls,n=n,minimizing_e=e[0],minimum_median_gap=e[1],KL_lower_at_numerical_min=2*2880*e[1]**2))
rho=1-9/16*(1-math.exp(-2880*.1**2*.5**2/(24*1.6)))
out=dict(derivative_numerical_max=-r.fun,location=r.x.tolist(),rho=rho,wrong_summary_bound=15*rho**256,wrong_root_bound=math.exp(-32*192/72),expected_KL_bound=96*15*rho**256+224*math.exp(-32*192/72),median_numerical_checks=records,interval_certified=False)
print(json.dumps(out,indent=2))
Path(__file__).with_suffix('.json').open('w').write(json.dumps(out,indent=2)+'\n')
