"""Read-only post-run audit of original states and arrays; no fitting/RNG calls."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from model import Model,psi

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root):
    package=Path(__file__).resolve().parents[1]
    reg=json.loads((root/'registration.json').read_text())
    assert sha(package/'PROTOCOL.md')==reg['protocol_sha256']
    for path,h in reg['source_sha256'].items():assert sha(package/path)==h
    freeze=json.loads((root/'ALL_FITS_FROZEN.json').read_text());assert freeze['count']==72 and freeze['before_any_population_evaluation']
    summary=json.loads((root/'summary.json').read_text());max_decode=max_ld=max_stat=0.;models=0;copies=0;all_files=[]
    for p,h in freeze['model_sha256'].items():
        p=root/p;assert sha(p)==h;model=Model.load(p);arm=p.stem;cell=p.parent
        saved=np.load(cell/(arm+'_numerical.npz'),allow_pickle=False)
        x,ld=model.sample(saved['source']);max_decode=max(max_decode,float(np.max(abs(x-saved['output']))));max_ld=max(max_ld,float(np.max(abs(ld-saved['forward_logdet']))))
        assert np.array_equal(x,saved['output']) and np.array_equal(ld,saved['forward_logdet'])
        assert np.array_equal(model.log_prob(x),saved['log_prob'])
        assert np.array_equal(saved['source'],np.load(cell/'generation_source.npy',allow_pickle=False))
        if arm=='spectral':
            cp=np.load(cell/'exact_copy.npz',allow_pickle=False)
            assert np.array_equal(cp['output'],x) and np.array_equal(cp['forward_logdet'],ld) and np.array_equal(cp['log_prob'],saved['log_prob']);copies+=1
            # Independent edgewise sums, including every nonedge; no new fit.
            obs=np.load(cell.parent/'observed.npy',mmap_mode='r',allow_pickle=False)[:model.cfg.graph_arrays]
            st=np.load(cell/(arm+'_fit_arrays.npz'),allow_pickle=False);c=obs[:,0]
            hh=[np.sqrt(2)*np.cos(2*np.pi*c),np.sqrt(2)*np.sin(2*np.pi*c)]
            for b in range(model.cfg.blocks):
                f=psi(obs[:,model.cfg.root_dim+b*model.cfg.block_size:model.cfg.root_dim+(b+1)*model.cfg.block_size])
                acc=st[f'block_{b}_correlations']
                for i in range(model.cfg.block_size):
                    for j in range(i+1,model.cfg.block_size):
                        yy=f[:,i]*f[:,j]
                        for k in range(2):max_stat=max(max_stat,abs(float(np.mean(hh[k]*yy)-acc[k,i,j])))
            assert max_stat<1e-12
        models+=1
    for p in sorted(root.rglob('*')):
        if p.is_file():all_files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    return dict(status='saved_payload_audit_pass',no_fits=True,no_new_random_draws=True,
      frozen_sources_verified=len(reg['source_sha256']),frozen_models=models,exact_copies=copies,
      source_to_output_replay_max_error=max_decode,forward_logdet_replay_max_error=max_ld,
      independently_summed_graph_max_error=max_stat,
      direct_quadrature_max_error=max(v['direct_quadrature_error'] for row in summary.values() for v in row.values()),
      context_refinement_max_error=max(v['context_refinement_error'] for row in summary.values() for v in row.values()),
      original_run_payloads=all_files,
      scope='Same-platform saved-state replay and independently ordered graph sums, not an independent full training replication.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.run)
    with a.output.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='original_run_payloads'},indent=2))
