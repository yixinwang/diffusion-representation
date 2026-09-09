"""Full-state round trips plus an independent small-map finite-difference Jacobian."""
import json,math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import pro13 as p


def main():
    rows=[]
    for i,path in enumerate(sorted(Path('standalone_results').glob('*_packed_state.json'))):
        st=json.loads(path.read_text());z=np.random.default_rng(1309444+i).normal(size=(32,3072))
        x=p.decode(st,z); back=p.encode(st,x)
        err=float(np.max(abs(back-z))); assert err<5e-10
        rows.append(dict(state=path.name,gaussian_roundtrip_max_error=err))
    # Fixed analytic state is for map algebra only, not a learned-quality claim.
    small=dict(c=2,groups=1,block_size=6,root_bins=8,context_bins=32,
        root_probabilities=[[.1,.1,.2,.1,.1,.1,.15,.15],[.125]*8],pairs=[[(0,5),(1,3),(2,4)]],theta=[[.4]*32])
    z=np.array([[.113,-.227,.34,-.416,.594,-.73,.85,-.961]])
    x=p.decode(small,z); dim=z.shape[1];step=1e-5;J=np.empty((dim,dim))
    for j in range(dim):
        plus=z.copy();minus=z.copy();plus[0,j]+=step;minus[0,j]-=step
        J[:,j]=(p.decode(small,plus)[0]-p.decode(small,minus)[0])/(2*step)
    sign,ld=np.linalg.slogdet(J)
    formula=-.5*(dim*math.log(2*math.pi)+(z*z).sum())-p.log_prob(small,x)[0]
    assert sign>0 and abs(ld-formula)<1e-7
    report=dict(scope='fabricated numerical checks, not interval certification',full_state_roundtrips=rows,
        small_map_jacobian_logdet_finite_difference=float(ld),small_map_logdet_density_formula=float(formula),
        small_map_logdet_abs_difference=float(abs(ld-formula)),finite_difference_step=step)
    Path('bijection_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
