"""Exact-real nonlinear response witness and independent numerical realization."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import torch
repo=Path('work/diffusion-representation').resolve();sys.path.insert(0,str(repo/'qalt/src'))
from qalt.innovation_response import InnovationResponse
from qalt.cached_global_innovation import OrthonormalFrame
out=Path('work/review20-current-response-witness');revision='3b8c0f7bef604a90ff329fd9608639527b67a972'
identity={}
for name in ['qalt/src/qalt/innovation_response.py','qalt/src/qalt/cached_global_innovation.py']:
 raw=(repo/name).read_bytes();assert raw==subprocess.check_output(['git','-C',str(repo),'show',revision+':'+name]);identity[name]=hashlib.sha256(raw).hexdigest()
normal=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
k,kerr=quad(lambda x:math.tanh(x)**2*normal(x),-np.inf,np.inf,epsabs=1e-12,epsrel=1e-12)
fourth,ferr=quad(lambda x:math.tanh(x)**4*normal(x),-np.inf,np.inf,epsabs=1e-12,epsrel=1e-12)
variance=fourth-k*k;mi_upper=.5*math.log1p(variance);assert variance>mi_upper>0
m=InnovationResponse(720,16).double();frame=OrthonormalFrame(720,16).double()
with torch.no_grad():
 for p in m.parameters():p.zero_()
 frame.bottom.zero_();frame.rotation.zero_()
 m.input.weight[0,32]=1;m.input.weight[1,32]=-1
 m.output.weight[1,0]=1;m.output.weight[1,1]=-1;m.output.bias[1]=-k
 g=torch.Generator().manual_seed(2026090921);source=torch.randn(64,3072,dtype=torch.float64,generator=g)
 block=source[:,192:912];summary=torch.zeros(64,16,dtype=torch.float64);w=frame.matrix()
 transformed=m(block,summary,w);assert bool(transformed.valid)
 expected=block.clone();expected[:,1]+=torch.tanh(block[:,0])**2-k
 error=float((expected-transformed.value).abs().max());assert error<1e-12
 inverse=m(transformed.value,summary,w,inverse=True);roundtrip=float((inverse.value-block).abs().max());assert bool(inverse.valid) and roundtrip<1e-12
 assert bool((transformed.logdet==0).all()) and bool((inverse.logdet==0).all())
 result=source.clone();result[:,192:912]=transformed.value
 np.savez(out/'numerical_witness.npz',source=source.numpy(),transformed=result.numpy(),block_recovered=inverse.value.numpy(),frame=w.numpy())
r={'scope':'Restricted mathematical witness, no fitting/data/quality experiment; one actual response block embedded in3072coordinates with othercoordinates unchanged','source_commit':revision,'source_sha256':identity,'normalization_k':k,'quad_reported_abs_error_k':kerr,'fourth_moment':fourth,'quad_reported_abs_error_fourth':ferr,'permutation_score_T_exact_formula':variance,'mutual_information_upper_bound':mi_upper,'mutual_information_exact_value_computed':False,'strict_permutation_vs_MI_gap_at_least':variance-mi_upper,'zero_anchor_follower_covariance':'exact by even/odd symmetry','forward_max_error':error,'inverse_max_error':roundtrip,'all_scales_one':True,'all_logdets_zero':True,'saved_bank_sha256':hashlib.sha256((out/'numerical_witness.npz').read_bytes()).hexdigest(),'torch':torch.__version__,'numpy':np.__version__}
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
