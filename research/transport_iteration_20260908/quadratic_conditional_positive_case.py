"""Deterministic scalar identities and quadrature; no fitted model or data."""
import json,math
from pathlib import Path
import numpy as np
rho=.65
nodes,weights=np.polynomial.legendre.leggauss(160)
c=(nodes+1)/2;r=c.copy();w=weights/2
x=np.cos(2*np.pi*c)[:,None]*(2*r[None,:]-1)
weight=w[:,None]*w[None,:]
results={}
for theta in [-rho,0.,rho]:
 density=1+theta*x
 mean=float(np.sum(weight*density*6*x))
 variance=float(np.sum(weight*density*(6*x-mean)**2))
 assert abs(mean-theta)<1e-12 and abs(variance-(6-theta**2))<1e-12
 results[str(theta)]={'quadrature_mean':mean,'quadrature_variance':variance}
worst=0.
for a in [-rho,-.1,0.,.1,rho]:
 u=np.linspace(0,1,1001)
 q=2*u/(1-a+np.sqrt((1-u)*(1-a)**2+u*(1+a)**2))
 worst=max(worst,float(np.max(np.abs(q+a*(q*q-q)-u))))
assert worst<1e-14
bounds=[]
for n in [256,4000]:
 v=6-rho*rho;b=6+rho;l=math.log(40)
 radius=b*l/(3*n)+math.sqrt((b*l/(3*n))**2+2*v*l/n)
 bounds.append(dict(n=n,theta=rho,expected_kl_upper=v/(12*(1-rho*rho)*n),
   probability=.95,parameter_error_upper=radius,high_probability_kl_upper=radius*radius/(12*(1-rho*rho)),
   independence_kl_lower=rho*rho/12))
actual=float(np.sum(weight*(1+rho*x)*np.log1p(rho*x)))
assert actual>=rho*rho/12
report=dict(status='scalar_checks_passed',fitting_performed=False,observed_data_used=False,
 moment_quadrature=results,cdf_inverse_max_error=worst,independence_kl_quadrature=actual,bounds=bounds)
output=Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
