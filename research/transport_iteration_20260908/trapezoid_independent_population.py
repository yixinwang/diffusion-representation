"""Direct mixed-measure quadrature, not the experiment's moment-series evaluator."""
import numpy as np
from numpy.polynomial.legendre import leggauss

def direct_population(truth,state,context_order=80,mixture_order=48):
 root=np.asarray(truth['root_probabilities'],dtype=np.float64);qroot=np.asarray(state['root_probabilities'],dtype=np.float64)
 root_kl=float(np.sum(root*np.log(root/qroot)));config=state['config'];bins=config['context_bins']
 continuous=state.get('continuous_context',False);constant=state.get('constant_context',False)
 boundaries=list(np.linspace(0,1,config['root_bins']+1))
 boundaries+=list((np.arange(bins)+.5)/bins if continuous and not constant else np.linspace(0,1,bins+1))
 boundaries=sorted(set(boundaries));cnode,cweight=leggauss(context_order);context=[];cw=[]
 for left,right in zip(boundaries[:-1],boundaries[1:]):
  x=left+(right-left)*(cnode+1)/2;density=root[0,np.minimum((x*config['root_bins']).astype(int),config['root_bins']-1)]*config['root_bins'];context.extend(x);cw.extend(cweight*(right-left)/2*density)
 context=np.asarray(context);cw=np.asarray(cw)
 node,weight=leggauss(mixture_order);amp=np.sqrt(1.5);psi=np.r_[-amp,amp,amp*node];pw=np.r_[.25,.25,weight/4]
 z=(psi[:,None]*psi[None,:]).ravel();zw=(pw[:,None]*pw[None,:]).ravel();assert abs(zw.sum()-1)<1e-14
 blocks=[];overlaps=[]
 for block,pairs in enumerate(state['pairs']):
  theta=truth['signs'][block]*(truth['offset']+.075*np.sin(2*np.pi*context+truth['phases'][block]));coef=np.asarray(state['coefficients'][block])
  if constant:eta=np.repeat(coef[0],len(context))
  elif continuous:eta=np.interp(context,(np.arange(bins)+.5)/bins,coef)
  else:eta=coef[np.minimum((context*bins).astype(int),bins-1)]
  truth_pairs={tuple(sorted(edge)) for edge in truth['pairs'][block]};assert len(truth_pairs)==config['block_size']//2
  fitted_pairs={tuple(sorted(edge)) for edge in pairs};assert len(fitted_pairs)==len(pairs)
  overlap=len(truth_pairs&fitted_pairs);overlaps.append(overlap);values=np.zeros(len(context))
  for first in range(0,len(context),32):
   t=theta[first:first+32,None];e=eta[first:first+32,None];p=1+t*z;logq=np.log1p(e*z)
   entropy=(p*np.log1p(t*z))@zw
   true_edge_cross=(p*logq)@zw;false_edge_cross=logq@zw
   values[first:first+32]=len(truth_pairs)*entropy-overlap*true_edge_cross-(len(fitted_pairs)-overlap)*false_edge_cross
  blocks.append(float(cw@values))
 return {'root_kl':root_kl,'residual_block_kl':blocks,'joint_kl':root_kl+sum(blocks),'correct_pairs_per_block':overlaps,'context_order':context_order,'direct_psi_mixture_order':mixture_order}
