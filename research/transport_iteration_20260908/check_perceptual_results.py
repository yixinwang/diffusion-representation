"""Independent metrics from preserved features, no model/image loading."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.results
read=lambda name:json.loads((root/name).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read('COMPLETE.json');prov=read('provenance.json');saved=read('metrics.json')
assert c['status']=='completed_evaluation_only_development'
assert set(c['payload_sha256'])=={p.name for p in root.iterdir() if p.is_file()}-{'COMPLETE.json'}
for name,h in c['payload_sha256'].items():assert sha(root/name)==h
assert prov['source_commit']=='22ac114f90f5250fddced652c6a3bccc8942dd9d'
for name,h in prov['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',prov['source_commit']+':'+name],cwd=a.repo)).hexdigest()==h
assert prov['weights_sha256']=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
assert sha(a.native/'status.json')==prov['input_status_sha256']
ns=json.loads((a.native/'status.json').read_text());assert ns['payload_sha256']==prov['input_payload_sha256']
with np.load(a.native/'record_ids.npz',allow_pickle=False) as ids:assert np.array_equal(ids['repair'],np.load(root/'repair_ids.npy',allow_pickle=False))
assert json.loads((a.native/'data_ledger.json').read_text())==prov['data_ledger']
real=np.load(root/'repair_features.npy',allow_pickle=False);assert real.shape==(1000,2048) and real.dtype==np.float32 and np.isfinite(real).all()

def measure(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);n,m=len(x),len(y);d=x.shape[1]
 xx=x@x.T;yy=y@y.T;xy=x@y.T
 kxx=(xx/d+1)**3;kyy=(yy/d+1)**3;kxy=(xy/d+1)**3
 kid=(np.sum(kxx)-np.trace(kxx))/(n*(n-1))+(np.sum(kyy)-np.trace(kyy))/(m*(m-1))-2*np.mean(kxy)
 dx=np.sum(x*x,axis=1);dy=np.sum(y*y,axis=1)
 xx=np.maximum(dx[:,None]+dx[None,:]-2*xx,0);yy=np.maximum(dy[:,None]+dy[None,:]-2*yy,0);xy=np.maximum(dx[:,None]+dy[None,:]-2*xy,0)
 _,ix=np.unique(x,axis=0,return_inverse=True);_,iy=np.unique(y,axis=0,return_inverse=True)
 xx[ix[:,None]==ix[None,:]]=0;yy[iy[:,None]==iy[None,:]]=0
 _,joint=np.unique(np.concatenate((x,y)),axis=0,return_inverse=True)
 xy[joint[:n,None]==joint[None,n:]]=0
 np.fill_diagonal(xx,np.inf);np.fill_diagonal(yy,np.inf)
 rx=np.partition(xx,4,axis=1)[:,4];ry=np.partition(yy,4,axis=1)[:,4]
 inside_x=xy<rx[:,None];inside_y=xy<ry[None,:]
 return {'real_count':n,'fake_count':m,'kid_unbiased_full_bank':float(kid),
  'precision':float(inside_x.any(0).mean()),'recall':float(inside_y.any(1).mean()),
  'density':float(inside_x.sum(0).mean()/5),'coverage':float((xy.min(1)<rx).mean()),
  'nearest_k':5,'real_duplicate_rows':n-len(np.unique(ix)),'fake_duplicate_rows':m-len(np.unique(iy)),
  'cross_radius_ties_real':int((xy==rx[:,None]).sum()),'cross_radius_ties_fake':int((xy==ry[None,:]).sum())}
computed={};discrepancies={}
for name in ('real_real_first500_last500_smaller_pair','analysis_only','coupling',*(f'residual_fm_nfe_{n}' for n in (4,8,16,32,64))):
 if name.startswith('real_real'):x,y=real[:500],real[500:]
 else:
  x=real;y=np.load(root/(name+'_features.npy'),allow_pickle=False)
  assert y.shape==(2000,2048) and y.dtype==np.float32 and np.isfinite(y).all()
 result=measure(x,y);computed[name]=result
 for key,value in result.items():
  discrepancy=abs(value-saved[name][key]);discrepancies[name+'_'+key]=discrepancy
  assert discrepancy<=1e-10,(name,key,value,saved[name][key])
 print(name,'verified',flush=True)
a.output.write_text(json.dumps({'status':'audit_pass','payload_hashes':len(c['payload_sha256']),
 'source_hashes':len(prov['source_sha256']),'reference_ids_and_ledger_match':True,
 'metric_max_abs_discrepancy':max(discrepancies.values()),'recomputed_metrics':computed,
 'no_model_or_image_read':True,'feature_extraction_rerun':False},indent=2)+'\n')
