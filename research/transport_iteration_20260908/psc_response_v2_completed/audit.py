"""Independent saved-bank audit; never fits models or reads canonical data."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
SEEDS=(78201,78202,78203)
ARMS=('P_frozen','P_joint','I_frozen','I_joint','RQS_frozen','RQS_joint','S42')
def read(p): return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def close(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('nonfinite comparison')
    if not np.allclose(a,b,rtol=1e-10,atol=1e-10): raise ValueError(('mismatch',a,b))
def array(path,shape):
    a=np.load(path,mmap_mode='r');assert a.shape==shape and np.isfinite(a).all(),str(path)
    return a
def kid(x,y):
    x,y=x.astype('float64'),y.astype('float64');d=x.shape[1]
    def kernel_sum(a,b,diagonal=False):
        total=0.
        for start in range(0,len(a),128):
            k=(1+a[start:start+128]@b.T/d)**3
            if diagonal:
                rows=np.arange(len(k));k[rows,start+rows]=0
            total+=k.sum()
        return total
    return kernel_sum(x,x,True)/(len(x)*(len(x)-1))+kernel_sum(y,y,True)/(len(y)*(len(y)-1))-2*kernel_sum(x,y)/(len(x)*len(y))
def gates(r):
    j,m,s,q=r['I_joint'],r['I_frozen'],r['S42'],r['RQS_frozen']
    g={'I_frozen_residual_nll_better_S42':m['residual_nll']<s['residual_nll'],
       'I_frozen_covariance_error_half_S42':m['covariance_error']<=s['covariance_error']/2,
       'I_joint_complete_nll_better_I_frozen':j['complete_nll']<m['complete_nll'],
       'I_joint_KID_better_S42':j['kid']<s['kid'],
       'I_joint_KID_better_RQS_frozen':j['kid']<q['kid'],
       'I_joint_gradient_within_10_percent':bool(np.all(np.abs(np.array(j['gradient_means'])-j['repair_gradient_means'])<=.1*np.abs(j['repair_gradient_means'])))}
    for mode in ('frozen','joint'):
        for metric in ('complete_nll','kid'):
            g[f'I_{mode}_{metric}_better_P_{mode}']=r['I_'+mode][metric]<r['P_'+mode][metric]
    for metric in ('complete_nll','kid'):
        g['I_joint_'+metric+'_better_RQS_joint']=j[metric]<r['RQS_joint'][metric]
    for arm in ('S42','I_frozen','RQS_frozen','RQS_joint','P_frozen','P_joint'):
        g['I_joint_energy_not_worse_'+arm]=j['energy_mean']<=r[arm]['energy_mean']+.0005
    refs=[r[a]['kid'] for a in ('RQS_frozen','RQS_joint')]
    g['I_joint_KID_material_5pct_both_RQS']=all(v>0 and j['kid']<=.95*v for v in refs)
    return g

def audit(root):
    status=read(root/'status.json')
    assert status['status']=='completed_development_only','Do not score failed/incomplete study'
    for name,h in status['payload_sha256'].items(): assert digest(root/name)==h,name
    frozen=read(root/'ALL_FITS_FROZEN.json');admit=read(root/'ALL_NUMERICS_ADMITTED.json')
    assert admit['count']==21 and admit['frozen_fits_sha256']==digest(root/'ALL_FITS_FROZEN.json')
    expected={f'{s}/{a}' for s in SEEDS for a in ARMS}
    assert set(frozen['checkpoints'])==set(admit['banks'])==expected
    report={}
    for seed in SEEDS:
        p=root/f'seed_{seed}';evaluation=read(p/'evaluation.json');results={};num={}
        source=array(p/'common_gaussian.npy',(2000,3072))
        assert read(p/'pair_identity.json')['source_sha256']==digest(p/'common_gaussian.npy')
        real=array(p/'repair_features.npy',(1000,2048))
        for arm in ARMS:
            ap=p/arm;key=f'{seed}/{arm}'
            assert digest(p/(arm+'.pt'))==frozen['checkpoints'][key]['sha256']
            for name,h in admit['banks'][key].items(): assert digest(ap/name)==h
            bank=np.load(ap/'numerical.npz');assert all(np.isfinite(bank[k]).all() for k in bank.files)
            for name,shape in {'source':(8,3072),'recovered':(8,3072),'logits':(8,3,32,32),'exact_copy':(8,3,32,32),'ld':(8,),'inverse_ld':(8,)}.items(): assert bank[name].shape==shape
            progress=read(ap/'generation_progress.json');assert progress['status']=='completed' and progress['completed_rows']==progress['logits_written_rows']==progress['planned_rows']==2000
            pixels=array(ap/'generated.npy',(2000,3,32,32));logits=array(ap/'logits.npy',(2000,3,32,32))
            assert pixels.dtype==np.float64 and logits.dtype==np.float32 and (pixels>=0).all() and (pixels<=1).all()
            endpoints={'zero':int((pixels==0).sum()),'one':int((pixels==1).sum())}
            assert endpoints==progress['endpoint_counts']
            n={'finite':True,'source_error':float(np.max(np.abs(bank['recovered']-bank['source']))),'ld_error':float(np.max(np.abs(bank['ld']+bank['inverse_ld']))),'exact_copy':bool(np.array_equal(bank['logits'],bank['exact_copy']))}
            assert n==read(ap/'numerical.json') and n['source_error']<=.001 and n['ld_error']<=.01 and n['exact_copy']
            num[arm]=n
            r=read(ap/'metrics.json');close(r['kid'],kid(real,array(ap/'features.npy',(2000,2048))))
            assert r['sigmoid_endpoint_counts']==endpoints
            parts=array(ap/'repair_nll_components.npy',(1000,5))
            close(parts[:,-1],parts[:,:4].sum(axis=1))
            close(r['complete_nll'],parts[:,-1].mean()/3072);close(r['residual_nll'],parts[:,1].mean()/2880)
            close(r['energy_mean'],array(ap/'energy.npy',(1000,)).mean())
            descriptors=array(ap/'descriptors.npy',(2000,6))
            gray=pixels.mean(axis=1);a=gray[:,:16,:16].mean((1,2));b=gray[:,16:,16:].mean((1,2))
            horizontal=np.diff(gray,axis=2)**2;vertical=np.diff(gray,axis=1)**2
            h=horizontal.mean((1,2));v=vertical.mean((1,2))
            close(descriptors,np.column_stack((a,b,a*b,h,v,h*v)))
            close(r['gradient_means'],descriptors[:,3:5].mean(axis=0))
            assert r==evaluation['arms'][arm]
            results[arm]=r
        g=gates(results);assert g==evaluation['gates']==status['seed_gates'][str(seed)]
        assert all(g.values())==evaluation['all_gates_pass']
        report[str(seed)]={'metrics':results,'numerics':num,'gates':g,'passed':sum(g.values()),'total':len(g)}
    assert all(all(x['gates'].values()) for x in report.values())==status['all_gates_pass']
    return {'status':'PASS saved-bank audit','scope':'Checks saved numeric banks, full manifest, checkpoint identities, feature KID, likelihood components, saved energy means, pixel-recomputed descriptors, and all gates from checked metrics plus reported covariance/repair-gradient summaries; does not rerun model or feature extractor, or reconstruct canonical target pixels.','payload_count':len(status['payload_sha256']),'seeds':report}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.root),indent=2)+'\n')
