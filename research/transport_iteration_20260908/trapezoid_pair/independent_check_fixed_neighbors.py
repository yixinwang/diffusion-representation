"""Independent fabricated boundary/parity audit, no fitting or timing study."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('root_trapezoid',P/'reference.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
s=importlib.util.spec_from_file_location('independent_trapezoid',P.parent/'pro13_trapezoid_scalar_check.py');ind=importlib.util.module_from_spec(s);s.loader.exec_module(ind)
u=np.unique(np.r_[np.linspace(0,1,1025),r.KNOTS,np.nextafter(r.KNOTS[1:],0),np.nextafter(r.KNOTS[:-1],1)])
record={'reference_sha256':hashlib.sha256((P/'reference.py').read_bytes()).hexdigest(),'case_count':0,'errors':[],'max_cdf_roundtrip_error':0.,'max_logdet_cancellation':0.,'max_independent_inverse_difference':0.,'scope':'ordinary float64 boundary checks, not interval proof'}
for theta in np.linspace(-.45,.45,129):
 thresholds=r.KNOTS[1:-1]+(theta*r.psi(u))[:,None]*r.INTEGRALS[1:-1]
 pp=np.column_stack((np.zeros(len(u)),np.ones(len(u)),np.full(len(u),np.nextafter(0.,1.)),np.full(len(u),np.nextafter(1.,0.)),thresholds,np.nextafter(thresholds,0),np.nextafter(thresholds,1)))
 uu=np.broadcast_to(u[:,None],pp.shape);record['case_count']+=pp.size
 try:
  v,ld=r.decode(uu,pp,theta);back,ild=r.encode(uu,v,theta)
  independent=ind.inverse(pp.flatten(),uu.flatten(),theta).reshape(pp.shape)
  record['max_cdf_roundtrip_error']=max(record['max_cdf_roundtrip_error'],float(np.max(np.abs(back-pp))))
  record['max_logdet_cancellation']=max(record['max_logdet_cancellation'],float(np.max(np.abs(ld+ild))))
  record['max_independent_inverse_difference']=max(record['max_independent_inverse_difference'],float(np.max(np.abs(v-independent))))
 except Exception as error:
  bad=[]
  for i in range(len(u)):
   try:r.decode(uu[i],pp[i],theta)
   except Exception as local:bad.append({'u':float(u[i]),'theta':float(theta),'error':repr(local)})
   if len(bad)>=3:break
  record['errors'].append({'theta':float(theta),'error':repr(error),'examples':bad})
(P/'independent_check_fixed_neighbors.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
