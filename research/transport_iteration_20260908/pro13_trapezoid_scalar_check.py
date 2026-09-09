"""Independent fabricated scalar algebra checks; no fit/timing/model experiment."""
from pathlib import Path
import json,math
import numpy as np
from scipy.special import ndtr,ndtri
A=math.sqrt(1.5);T=np.array([0,1/8,3/8,5/8,7/8,1.]);H=A*np.array([1,1,-1,-1,1,1]);P=A*np.array([0,1/8,1/8,-1/8,-1/8,0.])
def psi(u):return np.interp(u,T,H)
def cdf(v,u,theta):
 i=np.searchsorted(T[1:-1],v,side='right');d=v-T[i];s=(H[i+1]-H[i])/(T[i+1]-T[i])
 return v+theta*psi(u)*(P[i]+H[i]*d+s*d*d/2)
def inverse(p,u,theta):
 b=theta*psi(u);knots=T+b[...,None]*P;i=(p[...,None]>=knots[...,1:-1]).sum(-1)
 d=p-knots[np.arange(len(p)),i];s=(H[i+1]-H[i])/(T[i+1]-T[i]);h=1+b*H[i];beta=b*s
 disc=h*h+2*beta*d
 if np.any(disc<=0):raise ArithmeticError('nonpositive discriminant')
 return T[i]+2*d/(h+np.sqrt(disc))
def decode(z,theta):
 u=ndtr(z[:,0]);p=ndtr(z[:,1]);v=inverse(p,u,theta)
 logdet=-.5*(z*z+math.log(2*math.pi)).sum(1)-np.log1p(theta*psi(u)*psi(v))
 return np.column_stack((u,v)),logdet
z=np.array([[-1.25,.27],[.11,-1.31],[.77,1.14],[-.33,-.18]],dtype=float)
y,ld=decode(z,.45);back=np.column_stack((ndtri(y[:,0]),ndtri(cdf(y[:,1],y[:,0],.45))))
errors=[]
for row in range(len(z)):
 step=1e-5;columns=[]
 for j in (0,1):
  shift=np.zeros_like(z[row:row+1]);shift[0,j]=step
  columns.append(((decode(z[row:row+1]+shift,.45)[0]-decode(z[row:row+1]-shift,.45)[0])/(2*step))[0])
 jac=np.stack(columns,axis=1);errors.append(abs(np.linalg.slogdet(jac)[1]-ld[row]))
u=np.linspace(0,1,101);maxcdf=0
for theta in (-.45,0,.45):
 for p in (np.zeros(101),np.ones(101),np.linspace(0,1,101)):
  v=inverse(p,u,theta);maxcdf=max(maxcdf,float(np.max(np.abs(cdf(v,u,theta)-p))))
report={'scope':'ordinary scalar checks only, not interval-certified','gaussian_roundtrip_max':float(np.max(np.abs(back-z))),'dense_finite_difference_logdet_max':max(errors),'boundary_cdf_inverse_max':maxcdf,'nonidentity_pair_displacement':float(np.max(np.abs(y-ndtr(z))))}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
