"""Frozen six-cell mechanism validation; no neural or native-data comparison."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback

SEEDS=(13109101,13109102,13109103)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def write_json(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();status={'phase':'source_guards','state':'running','cells':[]}
    def checkpoint():
        status['elapsed_seconds']=time.perf_counter()-started
        tmp=out/'status.tmp';write_json(tmp,status);os.replace(tmp,out/'status.json')
    checkpoint()
    try:
        repo=Path(__file__).resolve().parents[3]
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()!=args.commit:
            raise ValueError('frozen HEAD mismatch')
        source=out/'sources';source.mkdir()
        folders=('trapezoid_mechanism','trapezoid_pair');hashes={}
        for folder in folders:
            for f in sorted((repo/'research/transport_iteration_20260908'/folder).glob('*')):
                if not f.is_file() or f.suffix not in ('.py','.md','.json','.slurm'):continue
                rel=str(f.relative_to(repo));raw=f.read_bytes()
                if raw!=subprocess.check_output(['git','-C',str(repo),'show',args.commit+':'+rel]):raise ValueError('source differs: '+rel)
                dest=source/folder/f.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw);hashes[rel]=sha(raw)
        sys.path.insert(0,str(source/'trapezoid_pair'));sys.path.insert(0,str(source/'trapezoid_mechanism'))
        import numpy as np
        import scipy
        from learner import Learner,Config
        from fixture import make_truth,observe
        from evaluate import population_kl
        write_json(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'source_commit':args.commit,'source_sha256':hashes,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}})
        env=os.environ.copy();env['PYTHONPATH']=str(source/'trapezoid_pair')+os.pathsep+str(source/'trapezoid_mechanism')
        with (out/'preflight.txt').open('x') as log:
            checked=subprocess.run([sys.executable,'-m','pytest','-q',str(source/'trapezoid_pair'/'test_reference.py'),
                str(source/'trapezoid_pair'/'test_learner.py'),str(source/'trapezoid_mechanism'/'test_evaluate.py')],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if checked.returncode:raise RuntimeError('fabricated preflight failed before fitting observations')
        cfg=Config();jobs=[]
        # Complete every fitting call before consulting the evaluator or truth graph.
        for seed in SEEDS:
            for world,offset in (('positive',.35),('zero_mean',0.)):
                cell=out/f'{seed}_{world}';cell.mkdir();status.update(phase='fit',current_cell=cell.name);checkpoint()
                fixture_start=time.perf_counter();truth=make_truth(seed,cfg,offset)
                write_json(cell/'truth_evaluator_only.json',truth)
                z=np.random.default_rng(seed+10000000).standard_normal((4000,cfg.dimension))
                np.save(cell/'fit_gaussian.npy',z,allow_pickle=False)
                x=observe(z,truth,cfg);np.save(cell/'fit_observed.npy',x,allow_pickle=False)
                x.setflags(write=False);del z
                write_json(cell/'data.json',{'fixture_and_save_seconds':time.perf_counter()-fixture_start,
                    'observations':4000,'dimension':cfg.dimension,'common_across_world_source':True,
                    'claim':'floating realization of ideal synthetic law; no continuous-truth versus atomic-output KL claim'})
                costs={}
                for arm,kw in (('continuous',{'continuous_context':True}),('binned',{}),('constant',{'constant_context':True})):
                    before=time.perf_counter();cpu=time.process_time()
                    model=Learner.fit(x,cfg,**kw);model.save(cell/(arm+'.json'))
                    costs[arm]={'wall_seconds_including_validation_fit_serialization':time.perf_counter()-before,
                                'cpu_seconds':time.process_time()-cpu,'graph_failures':list(model.graph_failures)}
                # Optimized product baseline only fits roots, with no discarded graph work.
                before=time.perf_counter();counts=np.ones((cfg.root_dim,cfg.root_bins))
                for j in range(cfg.root_dim):counts[j]+=np.bincount((x[:,j]*cfg.root_bins).astype(int),minlength=cfg.root_bins)
                product=Learner(cfg,counts/(len(x)+cfg.root_bins),[[] for _ in range(cfg.blocks)],
                    np.zeros((cfg.blocks,1)),[True]*cfg.blocks,{'observed_arrays':len(x),'graph_computed':False},constant_context=True)
                product.save(cell/'product.json');costs['product']={'wall_seconds_including_fit_serialization':time.perf_counter()-before}
                write_json(cell/'fit_costs.json',costs);del x,model,product
                jobs.append(cell);status['cells'].append(cell.name);checkpoint()
        fit_end=time.perf_counter()
        write_json(out/'ALL_FITS_FROZEN.json',{'cells':[c.name for c in jobs],'before_any_population_evaluation':True,
            'wall_since_start_including_source_data_and_fits':fit_end-started})
        status['phase']='population_and_numerical_evaluation';checkpoint();summary={}
        for cell in jobs:
            truth=json.loads((cell/'truth_evaluator_only.json').read_text());seed=truth['fixture_seed']
            z=np.random.default_rng(seed+30000000).standard_normal((64,cfg.dimension))
            np.save(cell/'common_gaussian.npy',z,allow_pickle=False);rows={}
            for arm in ('continuous','binned','constant','product'):
                status.update(current_cell=cell.name,current_arm=arm);checkpoint()
                model=Learner.load(cell/(arm+'.json'));s=model.state_dict()
                low=population_kl(truth,s,32);high=population_kl(truth,s,64)
                t=time.perf_counter();x,ld=model.sample_from_gaussian(z);sample_seconds=time.perf_counter()-t
                # Preserve actual arrays before checking numerical tolerances.
                np.savez(cell/(arm+'_numerical.npz'),source=z,output=x,forward_logdet=ld)
                recovered,ild=model.encode_gaussian(x);logp=model.log_prob(x)
                np.savez(cell/(arm+'_inverse.npz'),recovered=recovered,inverse_logdet=ild,log_prob=logp)
                rt=float(np.max(abs(recovered-z)));cancel=float(np.max(abs(ld+ild)))
                agreement=abs(low['joint_kl']-high['joint_kl'])
                write_json(cell/(arm+'_raw_evaluation.json'),{'order32':low,'order64':high,
                    'roundtrip':rt,'logdet_cancellation':cancel,'quadrature_difference':agreement})
                if not all(np.isfinite(v).all() for v in (x,ld,recovered,ild,logp,
                    [rt,cancel,agreement,low['joint_kl'],high['joint_kl'],low['root_kl'],high['root_kl']],
                    low['residual_block_kl'],high['residual_block_kl'])):
                    raise ArithmeticError('nonfinite numerical/population result: '+cell.name+'/'+arm)
                if rt>1e-9 or cancel>1e-8 or agreement>1e-8:raise ArithmeticError('numerical/evaluation gate failed: '+cell.name+'/'+arm)
                row={'population':high,'quadrature_32_64_difference':agreement,'max_source_roundtrip':rt,
                     'max_logdet_cancellation':cancel,'one_batch64_seconds_diagnostic_only':sample_seconds}
                if arm=='continuous':
                    clone=Learner(cfg,s['root_probabilities'],s['pairs'],s['coefficients'],s['graph_failures'],s['diagnostics'],continuous_context=True)
                    cx,cld=clone.sample_from_gaussian(z);cp=clone.log_prob(x)
                    np.savez(cell/'exact_copy.npz',output=cx,forward_logdet=cld,log_prob=cp)
                    row['exact_copy_equal']=bool(np.array_equal(x,cx) and np.array_equal(ld,cld) and np.array_equal(logp,cp))
                    if not row['exact_copy_equal']:raise ArithmeticError('copy mismatch')
                write_json(cell/(arm+'_evaluation.json'),row);rows[arm]=row
            summary[cell.name]=rows;checkpoint()
        write_json(out/'summary.json',summary)
        inventory={str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())}
                   for f in out.rglob('*') if f.is_file() and f.name!='status.json' and '__pycache__' not in f.parts}
        write_json(out/'ARTIFACTS.json',inventory)
        status.update(state='complete',phase='complete',evaluation_and_final_hashing_seconds=time.perf_counter()-fit_end,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as e:
        status.update(state='failed',error=repr(e),traceback=traceback.format_exc())
    finally:checkpoint()
    return 0 if status['state']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
