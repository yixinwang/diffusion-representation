"""Run the registered local fabricated study; no network or native data paths."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,json,hashlib,time,platform,resource,traceback,sys
import numpy as np
import scipy
from model import Config,Model,ExactCopyDecoder
from discover import fit,oracle_graph_diagnostic
from fixture import make_truth,observe
from evaluate import population,direct_population
from bounds import all_bounds

def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(out):
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();package=Path(__file__).resolve().parent.parent
    dump(out/'registration.json',dict(protocol_sha256=sha(package/'PROTOCOL.md'),source_sha256={str(p.relative_to(package)):sha(p) for p in sorted((package/'src').glob('*.py'))},before_any_fits=True))
    dump(out/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),blas_threads_requested=1))
    cfg=Config();cells=[];model_paths=[];data_times={}
    for seed in (150901,150902):
        folder=out/f'seed_{seed}';folder.mkdir();rng=np.random.default_rng(seed+100000)
        z=rng.standard_normal((cfg.graph_arrays+cfg.parameter_arrays,cfg.dimension));np.save(folder/'observed_source.npy',z)
        for case in ('harmonic','null','off_basis'):
            loc=folder/case;loc.mkdir();truth=make_truth(seed,cfg,case);dump(loc/'truth_evaluator_only.json',truth)
            t=time.perf_counter();x=observe(z,truth,cfg);np.save(loc/'observed.npy',x);data_times[f'{seed}/{case}']=time.perf_counter()-t
            for budget in ('low','large'):
                cc=replace(cfg,graph_arrays=2000,parameter_arrays=2000) if budget=='low' else cfg
                xx=x[:cc.graph_arrays+cc.parameter_arrays];cell=loc/budget;cell.mkdir();cells.append((cell,truth,cc,seed))
                for arm in ('spectral','unconditional','histogram_lr','product','oracle_graph_harmonic','oracle_graph_histogram'):
                    t=time.perf_counter()
                    if arm.startswith('oracle_graph_'):m,st=oracle_graph_diagnostic(xx,cc,truth['pairs'],arm.removeprefix('oracle_graph_'))
                    else:m,st=fit(xx,cc,arm)
                    p=cell/(arm+'.json');m.save(p);np.savez_compressed(cell/(arm+'_fit_arrays.npz'),**st)
                    dump(cell/(arm+'_fit_receipt.json'),dict(fit_diagnostics=m.diagnostics,
                      fit_and_serialization_wall_seconds=time.perf_counter()-t,model_sha256=sha(p),
                      observation_sha256=sha(loc/'observed.npy'),row_slice=[0,len(xx)],
                      eligible=not arm.startswith('oracle'),model_path=str(p.relative_to(out))))
                    model_paths.append(p)
                    print('FIT',str(p.relative_to(out)),flush=True)
    dump(out/'ALL_FITS_FROZEN.json',dict(before_any_population_evaluation=True,count=len(model_paths),
        model_sha256={str(p.relative_to(out)):sha(p) for p in model_paths},fixture_generation_and_save_seconds=data_times))
    results={}
    for cell,truth,cc,seed in cells:
        z=np.random.default_rng(seed+200000).standard_normal((128,cc.dimension));np.save(cell/'generation_source.npy',z)
        rows={}
        for arm in ('spectral','unconditional','histogram_lr','product','oracle_graph_harmonic','oracle_graph_histogram'):
            p=cell/(arm+'.json');m=Model.load(p);t=time.perf_counter();x,ld=m.sample(z);decode_time=time.perf_counter()-t
            t=time.perf_counter();logp=m.log_prob(x);density_time=time.perf_counter()-t
            recovered,ild=m.encode(x)
            np.savez_compressed(cell/(arm+'_numerical.npz'),source=z,output=x,forward_logdet=ld,recovered_source=recovered,inverse_logdet=ild,log_prob=logp)
            q=population(truth,m,32);qhi=population(truth,m,64);direct=direct_population(truth,m)
            err=float(np.max(abs(z-recovered)));lderr=float(np.max(abs(ld+ild)))
            density_err=float(np.max(abs(logp-(-.5*recovered**2-.5*np.log(2*np.pi)).sum(1)-ild)))
            row=dict(population=q,context_refinement_error=abs(q['joint_kl']-qhi['joint_kl']),direct_quadrature_error=abs(q['joint_kl']-direct),
              source_roundtrip_error=err,logdet_cancellation_error=lderr,change_of_variables_error=density_err,
              source_to_output_128_seconds=decode_time,full_log_prob_128_seconds=density_time,
              eligible=not arm.startswith('oracle'),fitted_state_sha256=sha(p),diagnostics=m.diagnostics)
            if arm=='spectral':
                cp=ExactCopyDecoder(m);cx,cld=cp.sample(z);clp=cp.log_prob(cx)
                np.savez_compressed(cell/'exact_copy.npz',output=cx,forward_logdet=cld,log_prob=clp)
                row['exact_copy_bitwise']=bool(np.array_equal(x,cx) and np.array_equal(ld,cld) and np.array_equal(logp,clp))
                assert row['exact_copy_bitwise']
            dump(cell/(arm+'_evaluation.json'),row);rows[arm]=row
            assert max(err,lderr,density_err)<1e-8
            assert row['direct_quadrature_error']<1e-10 and row['context_refinement_error']<1e-10
            assert q['joint_kl']>=-1e-12 and q['joint_kl']<=q['deterministic_joint_upper']+1e-10
        key=str(cell.relative_to(out));results[key]=rows;print('EVAL',key,flush=True)
    # Full-D numerical construction: not a learned full-D graph experiment.
    full=out/'full_dimension_smoke';full.mkdir();fc=Config(root_dim=192,blocks=4,block_size=720)
    truth=make_truth(150903,fc,'harmonic');coef=np.zeros((fc.blocks,3))
    for b,(s,p) in enumerate(zip(truth['signs'],truth['phases'])):
        coef[b,1]=s*.075*np.sin(p)/np.sqrt(2);coef[b,2]=s*.075*np.cos(p)/np.sqrt(2)
    m=Model(fc,truth['root'],truth['pairs'],coef,diagnostics={'constructed_from_truth':True,'no_fit':True})
    m.save(full/'constructed_model.json');z=np.random.default_rng(150903).standard_normal((64,fc.dimension));np.save(full/'source.npy',z)
    t=time.perf_counter();x,ld=m.sample(z);dt=time.perf_counter()-t;r,ild=m.encode(x);lp=m.log_prob(x)
    cx,cld=ExactCopyDecoder(m).sample(z)
    np.savez_compressed(full/'numerical.npz',output=x,forward_logdet=ld,recovered=r,inverse_logdet=ild,log_prob=lp,copy_output=cx,copy_logdet=cld)
    smoke=dict(dimension=fc.dimension,constructed_model_not_fit=True,source_to_output_64_seconds=dt,
      source_roundtrip_error=float(np.max(abs(r-z))),logdet_cancellation_error=float(np.max(abs(ld+ild))),
      exact_copy=bool(np.array_equal(x,cx) and np.array_equal(ld,cld)))
    dump(full/'receipt.json',smoke);assert smoke['exact_copy'] and smoke['source_roundtrip_error']<1e-8 and smoke['logdet_cancellation_error']<1e-8
    dump(out/'bounds.json',all_bounds());dump(out/'summary.json',results)
    dump(out/'status.json',dict(status='complete',fits=len(model_paths),population_evaluations=len(model_paths),
      elapsed_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      full_dimension_smoke=smoke,scope='Local fabricated development only. No native, PSC, trained-FM, or latent superiority claim.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    try:run(args.output)
    except BaseException:
        if args.output.exists():
            (args.output/'FAILED_TRACEBACK.txt').write_text(traceback.format_exc())
        raise
