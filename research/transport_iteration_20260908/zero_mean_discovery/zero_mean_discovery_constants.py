"""Concentration constants only: no observations, simulation or model fitting."""
import math,json
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
H=4*720*719//2;M=4*360;t=.075/2

def logfailure(n,tau,J):
    c=math.cos(math.pi/J);x=tau*c;d=t-tau
    null=math.log(J*(H-M))-n*x*x/(1+x)
    true=math.log(M)-n*d*d/(1-2*t*t+2*(1.5+t)*d/3)
    return float(logsumexp([null,true]))

def best(n,J):
    r=minimize_scalar(lambda tau:logfailure(n,tau,J),bounds=(1e-12,t-1e-12),method='bounded',options={'xatol':1e-14})
    return float(r.x),float(r.fun)

rows=[]
for alpha in (.05,.01):
 for J in (4,8,16,32,64):
    lo=1;hi=1000000
    while lo<hi:
        mid=(lo+hi)//2
        if best(mid,J)[1]<=math.log(alpha):hi=mid
        else:lo=mid+1
    tau,logp=best(lo,J)
    rows.append({'graph_error_target':alpha,'net_directions':J,'sufficient_arrays':lo,'threshold':tau,'unclipped_failure_upper':math.exp(logp),'failure_upper_at_n_minus1':math.exp(best(lo-1,J)[1])})
uvar=4*(.5*t*t-t**4)/2000+2*(.5-t*t+t**4)/(2000*1999)
a=.05;eps=t*t/2
ureq=math.ceil(20.25*math.log(H/a)/(eps*eps))+2
r={'scope':'ordinary floating evaluation of rigorously derived sufficient bounds; not interval arithmetic; no fits or synthetic observations','edges':H,'true_edges':M,'amplitude':.075,'mean_vector_norm':t,'graph_n':2000,'null_mean_vector_rms':1/math.sqrt(2000),'true_projection_standard_deviation':math.sqrt((.5-t*t)/2000),'union_bound_at_2000':min(1,math.exp(best(2000,16)[1])),'optimized_constants':rows,'squared_statistic_signal':t*t,'squared_statistic_null_variance':1/(2000*1999),'squared_statistic_true_variance':uvar,'squared_statistic_true_standard_deviation':math.sqrt(uvar),'naive_bounded_U_Hoeffding_sufficient_n_for_05':ureq}
L=4*(math.lgamma(721)-360*math.log(2)-math.lgamma(361))
K=M*.075**2/(4*(1-(1.5*.075)**2))
r['fano']={'log_matching_catalog':L,'per_array_KL_upper':K,'success_upper_at_2000':(2000*K+math.log(2))/L,'necessary_n_for_uniform_95pct':math.ceil((.95*L-math.log(2))/K),'scope':'minimax exact-all-four-matchings recovery only; phases may be revealed; not generative risk'}
p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
