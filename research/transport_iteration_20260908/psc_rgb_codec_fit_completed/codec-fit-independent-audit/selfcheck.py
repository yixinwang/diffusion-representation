"""Fabricated analytic and independently dense-convolution SSIM checks only."""
import json,math
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
from audit import metrics
out=Path(__file__).with_name('selfcheck.json');assert not out.exists()
rng=np.random.default_rng(2026090921);x=rng.random((3,3,32,32)).astype(np.float32);y=rng.random(x.shape).astype(np.float32)
g=np.exp(-np.arange(-5,6,dtype=float)**2/4.5);g/=g.sum();w=np.outer(g,g)
def dense(a,b):
 def c(z):return convolve2d(z,w,mode='valid')
 u,v=c(a),c(b);aa=c(a*a)-u*u;bb=c(b*b)-v*v;ab=c(a*b)-u*v
 return (((2*u*v+.0001)*(2*ab+.0009))/((u*u+v*v+.0001)*(aa+bb+.0009))).mean()
m,s=metrics(x,y);reference=np.array([np.mean([dense(a.astype(float),b.astype(float)) for a,b in zip(xx,yy)]) for xx,yy in zip(x,y)])
error=float(np.max(np.abs(s-reference)));assert error<2e-12
mi,si=metrics(x,x);assert np.array_equal(mi,np.zeros(3)) and np.allclose(si,1,rtol=0,atol=2e-13)
a=np.full((2,3,12,13),.2);b=np.full_like(a,.4);mc,sc=metrics(a,b);want=(2*.2*.4+.0001)/(.2**2+.4**2+.0001);assert np.allclose(mc,.04,rtol=0,atol=1e-15) and np.allclose(sc,want,rtol=0,atol=2e-12)
reject=0
for aa,bb in [(x[:,:,:10,:],y[:,:,:10,:]),(x,np.full_like(x,np.nan)),(x,np.full_like(x,1.1))]:
 try:metrics(aa,bb)
 except ValueError:reject+=1
assert reject==3
report={'scope':'fabricated-only; no model/data loader/experiment outputs','dense2d_vs_separable_max_ssim_error':error,'identity_mse_zero':True,'identity_ssim_one':True,'constant_offset_ssim':float(sc[0]),'constant_offset_analytic':want,'invalid_probes_rejected':reject,'global_psnr_example':-10*math.log10(np.mean([.001,.009])),'mean_image_psnr_distinct_example':float(np.mean(-10*np.log10([.001,.009]))),'pass':True}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
