"""Exact polynomial moments and ordinary bound arithmetic, no fitting."""
from fractions import Fraction as Q
import math,json
from pathlib import Path
knots=list(map(Q,['0','1/8','3/8','5/8','7/8','1']));values=[1,1,-1,-1,1,1]
def moment(power,coordpower=0):
 total=Q(0)
 for lo,hi,vl,vh in zip(knots[:-1],knots[1:],values[:-1],values[1:]):
  slope=Q(vh-vl)/(hi-lo);intercept=Q(vl)-slope*lo
  for j in range(power+1):
   n=j+coordpower
   total+=math.comb(power,j)*intercept**(power-j)*slope**j*(hi**(n+1)-lo**(n+1))/Q(n+1)
 return total
k=.45;a=.35;G=4;M=1440;n=ns=2000;K=32;pmin=.5/K;L=.5;E=1035360;amplitude_squared=1.5
B=1/(2*(1-k*k*amplitude_squared**2));v=L*L/(12*K*K);An=K/(n+1)*(1+3/((n+2)*pmin));delta=min(1,2*E*math.exp(-ns*a*a/(8*(1+(amplitude_squared+k)*a/6))))
terms={'root':192*7/4001,'context_approximation':B*M*v,'sampling':B*(G+M*v)*An,'empty_bins':B*M*k*k*math.exp(-n*pmin),'structure_failure':delta*M*math.log((1+amplitude_squared*k)/(1-amplitude_squared*k))}
report={'ordinary_float_bound_not_interval':True,'exact_unscaled_feature_moments':{str(i):str(moment(i)) for i in range(1,5)},'exact_unscaled_coordinate_moments':{'u_times_feature':str(moment(1,1)),'u_squared_times_feature':str(moment(1,2))},'B':B,'delta_structure':delta,'terms':terms,'total':sum(terms.values()),'product_floor':M*a*a/2,'amplitude':math.sqrt(1.5),'density_min':1-k*1.5}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
