"""Separate numerical diagnosis; does not modify or overwrite original checks."""
from pathlib import Path
import sys,json,warnings,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri
sys.path.insert(0,str(Path(__file__).resolve().parent/'check_math'))
import pro13 as p
out={}
for name,fn,lo,hi,points in [
 ('ordinary_original',lambda u:(u-.5)*float(p.psi(u)),0,1,p.KNOTS[1:-1]),
 ('gaussianized_original',lambda u:ndtri(u)*float(p.psi(u)),0,1,p.KNOTS[1:-1]),
 ('gaussianized_paired',lambda u:ndtri(u)*(float(p.psi(u))-float(p.psi(1-u))),0,.5,p.KNOTS[(p.KNOTS>0)&(p.KNOTS<.5)])]:
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');value,error=quad(fn,lo,hi,points=points,epsabs=2e-13)
 out[name]={'value':value,'reported_absolute_error':error,'warnings':[str(w.message) for w in caught]}
u=np.r_[np.linspace(.0001,.4999,2000),p.KNOTS[1:-1]]
out['paired_node_checks']={'nodes':len(u),'psi_symmetry_max_error':float(np.max(abs(p.psi(u)-p.psi(1-u)))),'ndtri_antisymmetry_max_error':float(np.max(abs(ndtri(u)+ndtri(1-u))))}
out['source_sha256']=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()
path=Path(__file__).with_suffix('.json')
with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
