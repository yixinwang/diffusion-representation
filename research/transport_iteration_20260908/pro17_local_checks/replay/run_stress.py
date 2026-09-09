"""One frozen two-condition stress test. Never launches native experiments."""
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from copula import (Model, fit, generate_observed, compile_equal_information_copy,
                    expected_kl, oracle_cell_means)

HERE=Path(__file__).resolve().parent

def digest_array(a):
    return hashlib.sha256(np.asarray(a,dtype='<f8',order='C').tobytes(order='C')).hexdigest()

def main():
    out=HERE/'results.json'
    if out.exists():
        raise RuntimeError('Original results already exist. Refusing to overwrite or regenerate.')
    raw=(HERE/'FROZEN_PROTOCOL.json').read_bytes()
    cfg=json.loads(raw)
    seal={'protocol_sha256':hashlib.sha256(raw).hexdigest(),
          'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                           for name in ['copula.py','test_copula.py','run_stress.py']}}
    (HERE/'PRE_RUN_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
    n,B,A,K=cfg['n_train'],cfg['context_bins'],cfg['anchors'],cfg['groups']
    rho,amp=cfg['rho'],cfg['amplitude']
    rng=np.random.default_rng(cfg['seed'])
    result={'protocol':cfg,'pre_run_seal':seal,'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,
            'rng_initial':rng.bit_generator.state,'scenarios':[],
            'scope':'Synthetic only. No native image/video fits. No latent-FM empirical comparison.'}
    for scenario in cfg['scenarios_in_order']:
        signs=np.ones(K) if scenario=='shared_nonlinear' else np.where(np.arange(K)%2==0,1.,-1.)
        start=time.perf_counter()
        data=generate_observed(rng,n,B,A,K,amp,signs)
        generation_seconds=time.perf_counter()-start
        start=time.perf_counter(); tied=fit(data,B,A,K,True,rho); tied_fit=time.perf_counter()-start
        start=time.perf_counter(); untied=fit(data,B,A,K,False,rho); untied_fit=time.perf_counter()-start
        copy=compile_equal_information_copy(tied)
        product=Model(np.zeros((B,1)),A,K,rho)
        means=oracle_cell_means(B,amp)
        oracle_theta=means[:,None] if scenario=='shared_nonlinear' else np.zeros((B,1))
        oracle_tied=Model(oracle_theta,A,K,rho)
        oracle_untied=Model(means[:,None]*signs[None,:],A,K,rho)
        methods={'population_product_projection':product,'learned_shared_copula':tied,
                 'exact_compiled_triangular_copy':copy,'learned_untied_triangular':untied,
                 'oracle_shared_bin_diagnostic':oracle_tied,
                 'oracle_untied_bin_diagnostic':oracle_untied}
        risks={name:expected_kl(model,amp,signs,64) for name,model in methods.items()}
        check={name:abs(risks[name]-expected_kl(model,amp,signs,128))
               for name,model in methods.items()}
        noise=rng.normal(size=(cfg['timing_batch'],cfg['dimension']))
        decoded=tied.decode(noise)
        copied=copy.decode(noise)
        copy_error=float(np.max(np.abs(decoded-copied)))
        log_copy_error=float(np.max(np.abs(tied.log_prob(data)-copy.log_prob(data))))
        timings={}
        for name in ('learned_shared_copula','exact_compiled_triangular_copy','learned_untied_triangular'):
            model=methods[name]
            decode_time=[]; density_time=[]
            for _ in range(cfg['timing_repetitions']):
                start=time.perf_counter(); sample=model.decode(noise); decode_time.append(time.perf_counter()-start)
                start=time.perf_counter(); model.log_prob(sample); density_time.append(time.perf_counter()-start)
            timings[name]={'complete_toy_decode_seconds_median':float(np.median(decode_time)),
                           'complete_toy_density_seconds_median':float(np.median(density_time)),
                           'output_coordinates_per_batch':int(noise.size),
                           'chart':'identity, cost zero only in this toy; native chart cost NOT inferred'}
        row={'scenario':scenario,'observed_train_shape':list(data.shape),
             'observed_train_sha256_little_endian_float64':digest_array(data),
             'generation_seconds_not_a_model_fit_cost':generation_seconds,
             'fit_seconds_including_observed_gaussian_CDF':{'shared':tied_fit,'untied':untied_fit},
             'population_kl_nats_per_full_observation':risks,
             'quadrature_64_vs_128_absolute_difference':check,
             'shared_gain_over_product_nats':risks['population_product_projection']-risks['learned_shared_copula'],
             'copy_max_sample_difference':copy_error,'copy_max_log_prob_difference':log_copy_error,
             'models':{name:model.theta.tolist() for name,model in methods.items()},
             'complete_toy_cpu_timings':timings,
             'scientific_interpretation':(
                 'Tests shared local nonlinear non-Gaussian law. Copy equivalence prevents a method-specific advantage claim.'
                 if scenario=='shared_nonlinear' else
                 'Predeclared negative: opposing site responses cancel. Population shared optimum is independence, so pooling cannot improve it. No position-aware repair is selected.')}
        if copy_error!=0. or log_copy_error!=0.:
            raise AssertionError('Exact compiled-copy equivalence failed')
        if max(check.values())>1e-10:
            raise ArithmeticError('Population integration qualification failed; no automatic repair')
        result['scenarios'].append(row)
    result['rng_final']=rng.bit_generator.state
    result['series_tail_bound_per_full_observation']=K*3*rho**402/(1-rho*rho)
    result['scope_negative']='No new mechanism-specific gain versus equally cheap tied triangular flow. No real image/video superiority. No universal separation from latent models.'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Pro17 fixed synthetic stress results','',
           'No native image/video model was fitted. Oracle rows are diagnostic, not learned competitors.','',
           '| Condition | Product oracle KL | Shared learned KL | Equal-map copy KL | Untied learned KL |',
           '|---|---:|---:|---:|---:|']
    for row in result['scenarios']:
        r=row['population_kl_nats_per_full_observation']
        lines.append('| '+row['scenario']+' | '+' | '.join(f'{r[k]:.12g}' for k in
            ['population_product_projection','learned_shared_copula','exact_compiled_triangular_copy','learned_untied_triangular'])+' |')
    lines += ['', 'Units: nats per complete 113-dimensional synthetic observation, not bits/dimension.',
              'The failure condition is fixed in advance. Its population shared projection is independence.',
              'A copied conventional triangular flow is pointwise identical and equally costly; this is not a new architecture-level advantage.',
              'Timings are CPU toy timings with identity chart; they establish no native end-to-end speed advantage.',
              'Full parameters, input fingerprints, RNG states, numerical qualification and timings: results.json.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
