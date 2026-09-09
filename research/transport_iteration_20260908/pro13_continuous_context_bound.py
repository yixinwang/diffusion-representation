"""Ordinary arithmetic for a distinct continuous context-interpolation variant."""
import json,math
from pathlib import Path
n=2000;K=32;pmin=.5/K;G=4;M=1440;L=.5;kappa=.45;a=.35;E=1035360
v=L*L/(12*K*K);B=1/(2*(1-2.25*kappa*kappa));An=K/(n+1)*(1+3/((n+2)*pmin))
H=1/((n+1)*pmin)*(1+3/((n+2)*pmin));bias=9*L*L/(16*K*K)
delta=min(1,2*E*math.exp(-n*a*a/(8*(1+(1.5+kappa)*a/6))))
terms={'root':192*7/4001,'continuous_context_bias':B*M*bias,'sampling':B*(G+M*v)*H,'empty_bins':B*M*kappa*kappa*math.exp(-n*pmin),'structure_failure':delta*M*math.log((1+1.5*kappa)/(1-1.5*kappa))}
report={'ordinary_float_not_interval':True,'variant':'continuous linear interpolation of32 clipped bin estimates with constant endpoint extrapolation','An':An,'H':H,'H_over_An':H/An,'bias_per_pair':bias,'v_within_training_bin':v,'terms':terms,'total':sum(terms.values()),'restricted_product_floor':M*a*a/2}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
