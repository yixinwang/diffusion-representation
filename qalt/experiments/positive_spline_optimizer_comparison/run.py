"""Observed-feature optimizer comparison; no generation or population evaluation."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import scipy
import qalt.positive_density_spline as original
import qalt.positive_spline_optimizer as accelerated

ROOT=Path(__file__).resolve().parents[3]
OLD_PATH=ROOT/'qalt/experiments/positive_spline_validation/run.py'
spec=importlib.util.spec_from_file_location('frozen_positive_spline_simulator',OLD_PATH)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
SOURCE_FILES=tuple(dict.fromkeys((
    'qalt/src/qalt/__init__.py','qalt/src/qalt/core.py',
    *frozen.SOURCE_FILES,
    'qalt/src/qalt/positive_spline_optimizer.py',
    'qalt/tests/test_positive_spline_optimizer.py',
    'qalt/experiments/positive_spline_optimizer_comparison/run.py',
    'qalt/experiments/positive_spline_optimizer_comparison/PROTOCOL.md',
    'qalt/experiments/positive_spline_optimizer_comparison/run.slurm',
)))
CELLS=list(itertools.product(('local','distant'),(8,32),(256,1024,4096),(8101,8102,8103)))
SOLVERS={'fw':(original,original.fit_positive_density_spline),
         'accelerated':(accelerated,accelerated.fit_positive_density_spline_accelerated)}


def source_guard(commit):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual!=commit:raise ValueError('full requested source commit must equal HEAD')
    hashes={}
    for relative in SOURCE_FILES:
        payload=(ROOT/relative).read_bytes()
        if payload!=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT):
            raise ValueError(f'dirty registered source: {relative}')
        hashes[relative]=hashlib.sha256(payload).hexdigest()
    for module in (original,accelerated):
        if Path(module.__file__).resolve()!=ROOT/'qalt/src'/Path(*module.__name__.split('.')).with_suffix('.py'):
            raise ValueError('imported optimizer is outside guarded checkout')
    return hashes


class ObservedWork:
    """Count operations without changing their mathematical arguments/results."""
    def __init__(self,shape):
        self.shape=shape;self.latest=None
        self.counts=dict(feature_constructions=0,forward_feature_products=0,
            transpose_feature_products=0,column_sum_passes=0,linear_minimizer_calls=0,
            brent_root_calls=0,brent_derivative_calls=0)
    def capture(self,vector):
        candidate=np.asarray(vector).reshape(self.shape)
        weights=original.response_integral_weights(frozen.BINS)
        if np.isfinite(candidate).all() and candidate.min()>=frozen.LOWER-2e-12 and candidate.max()<=frozen.UPPER+2e-12 and np.max(np.abs(candidate@weights-1))<=2e-12:
            self.latest=candidate.copy()


class FeatureView:
    def __init__(self,matrix,work,transpose=False):self.matrix=matrix;self.work=work;self.transpose=transpose
    @property
    def T(self):return FeatureView(self.matrix.T,self.work,not self.transpose)
    @property
    def nnz(self):return self.matrix.nnz
    def sum(self,*args,**kwargs):
        self.work.counts['column_sum_passes']+=1
        return self.matrix.sum(*args,**kwargs)
    def __matmul__(self,vector):
        self.work.counts['transpose_feature_products' if self.transpose else 'forward_feature_products']+=1
        if not self.transpose:self.work.capture(vector)
        return self.matrix@vector


@contextmanager
def instrument(module,work):
    saved={name:getattr(module,name) for name in ('_joint_features','bounded_row_linear_oracle')}
    if module is original:saved['brentq']=module.brentq
    def features(*args,**kwargs):
        work.counts['feature_constructions']+=1
        return FeatureView(saved['_joint_features'](*args,**kwargs),work)
    def oracle(*args,**kwargs):
        work.counts['linear_minimizer_calls']+=1
        return saved['bounded_row_linear_oracle'](*args,**kwargs)
    def brent(function,*args,**kwargs):
        work.counts['brent_root_calls']+=1
        def counted(step):
            work.counts['brent_derivative_calls']+=1
            return function(step)
        return saved['brentq'](counted,*args,**kwargs)
    module._joint_features=features;module.bounded_row_linear_oracle=oracle
    if module is original:module.brentq=brent
    try:yield
    finally:
        for name,value in saved.items():setattr(module,name,value)


def fresh_check(context,response,model):
    coefficients=model.coefficients
    features=original._joint_features(np.empty((len(response),0)) if context is None else context,response,frozen.BINS)
    density=features@coefficients.ravel()
    gradient=np.asarray(features.T@(-1/(len(response)*density))).reshape(coefficients.shape)
    weights=original.response_integral_weights(frozen.BINS)
    vertex=original.bounded_row_linear_oracle(gradient,weights,frozen.LOWER,frozen.UPPER)
    gap=float(np.sum(gradient*(coefficients-vertex)))
    report=dict(objective=-float(np.log(density).mean()),gap=gap,
        coefficient_min=float(coefficients.min()),coefficient_max=float(coefficients.max()),
        row_integral_max_error=float(np.abs(coefficients@weights-1).max()),
        minimum_observed_density=float(density.min()),
        fresh_work=dict(feature_constructions=1,forward_feature_products=1,transpose_feature_products=1,linear_minimizer_calls=1))
    if not np.isfinite(list(report[k] for k in ('objective','gap','coefficient_min','coefficient_max','row_integral_max_error','minimum_observed_density'))).all() or gap < -1e-11 or report['row_integral_max_error']>2e-12 or coefficients.min()<frozen.LOWER-2e-12 or coefficients.max()>frozen.UPPER+2e-12:
        raise FloatingPointError('returned point fails independent gap/feasibility check')
    report['gap_met']=gap<=frozen.GAP_TOLERANCE
    return report


def compare_cell(cell,out,state,*,max_iterations=500):
    world,dimension,count,seed=cell
    folder=out/f'{world}_d{dimension}_n{count}_seed{seed}';folder.mkdir()
    observations=frozen.simulate_observations(count,dimension,frozen._stream(seed,dimension,count,world,1),world)
    observations.setflags(write=False)
    np.save(folder/'observations.npy',observations)
    parents,groups=frozen.graph(dimension,world)
    results=dict(cell=cell,observations_hash=frozen.array_hash(observations),parents=parents,groups=groups,heads=[])
    frozen.atomic_json(folder/'cell.json',results)
    for group in range(max(groups)+1):
        sites=[j for j,g in enumerate(groups) if g==group]
        response=np.concatenate([observations[:,j] for j in sites])
        context=np.concatenate([observations[:,parents[j]] for j in sites],axis=0) if parents[sites[0]] else None
        response.setflags(write=False)
        if context is not None:context.setflags(write=False)
        context_dimension=0 if context is None else context.shape[1]
        np.savez(folder/f'head_{group}_inputs.npz',response=response,context=np.empty((len(response),0)) if context is None else context)
        head=dict(group=group,sites=sites,independent_array_count=count,pooled_response_count=len(response),
            pooled_responses_independent=False,response_hash=frozen.array_hash(response),
            context_hash=None if context is None else frozen.array_hash(context),solvers={})
        # Fixed order, disclosed below; no outcome determines solver ordering.
        for name,(module,fit) in SOLVERS.items():
            state.update(cell=cell,head=group,solver=name)
            work=ObservedWork(((frozen.BINS+2)**context_dimension,frozen.BINS+1))
            started=time.perf_counter()
            try:
                with instrument(module,work):
                    model,diagnostic=fit(context,response,bins=frozen.BINS,max_iterations=max_iterations,
                        gap_tolerance=frozen.GAP_TOLERANCE,lower=frozen.LOWER,upper=frozen.UPPER)
                fit_seconds=time.perf_counter()-started
                np.save(folder/f'head_{group}_{name}_coefficients.npy',model.coefficients)
                before=time.perf_counter();check=fresh_check(context,response,model)
                check_seconds=time.perf_counter()-before
                if abs(check['objective']-diagnostic.objective)>1e-10 or abs(check['gap']-diagnostic.frank_wolfe_gap)>1e-10:
                    raise AssertionError('reported final objective/gap differs from independent recomputation')
                counts=work.counts.copy()
                counts['objective_evaluations']=(len(diagnostic.objective_trace)+diagnostic.iterations if name=='fw' else diagnostic.objective_evaluations)
                counts['gradient_evaluations']=counts['transpose_feature_products']
                counts['line_derivative_evaluations']=diagnostic.iterations+counts['brent_derivative_calls'] if name=='fw' else 0
                counts['projection_calls']=0 if name=='fw' else diagnostic.projection_calls
                counts['projected_row_count']=0 if name=='fw' else diagnostic.projected_row_count
                if name=='accelerated':
                    for key in ('forward_feature_products','transpose_feature_products','gradient_evaluations','objective_evaluations','projection_calls','projected_row_count','linear_minimizer_calls'):
                        if counts[key]!=getattr(diagnostic,key):raise AssertionError(f'work accounting mismatch: {key}')
                head['solvers'][name]=dict(diagnostics=asdict(diagnostic),work=counts,independent_check=check,
                    instrumented_fit_seconds=fit_seconds,fresh_check_seconds=check_seconds)
                frozen.atomic_json(folder/f'head_{group}_{name}.json',head['solvers'][name])
            except BaseException as error:
                if work.latest is not None:np.save(folder/f'head_{group}_{name}_last_feasible_evaluation.npy',work.latest)
                frozen.atomic_json(folder/f'head_{group}_{name}_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),work=work.counts,
                    elapsed_seconds=time.perf_counter()-started,last_feasible_evaluation_saved=work.latest is not None))
                raise
        results['heads'].append(head);frozen.atomic_json(folder/'cell.json',results)
    return results


def stop(signum,frame):raise TimeoutError('scheduler warning; preserving partial comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    state={'phase':'source_guard'};results=[];started=time.perf_counter()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    try:
        hashes=source_guard(args.source_commit)
        for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            if os.environ.get(variable)!='1':raise ValueError(f'{variable}=1 required')
        frozen.atomic_json(args.output/'manifest.json',dict(source_commit=args.source_commit,source_sha256=hashes,
            cells=CELLS,bins=frozen.BINS,max_iterations=500,gap_tolerance=frozen.GAP_TOLERANCE,
            lower=frozen.LOWER,upper=frozen.UPPER,solver_order=list(SOLVERS),
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,host=platform.node(),
            real_data_accessed=False,teacher_passed_to_fit=False,generation_evaluated=False))
        snapshot=args.output/'sources';snapshot.mkdir()
        for i,path in enumerate(SOURCE_FILES):(snapshot/f'{i:02d}_{Path(path).name}').write_bytes((ROOT/path).read_bytes())
        state['phase']='fitting'
        for cell in CELLS:
            results.append(compare_cell(cell,args.output,state))
            frozen.atomic_json(args.output/'progress.json',dict(completed_cells=len(results),remaining_cells=CELLS[len(results):]))
        frozen.atomic_json(args.output/'summary.json',dict(status='completed_optimizer_comparison',results=results,
            elapsed_seconds=time.perf_counter()-started,quality_advantage_established=False,
            population_advantage_established=False,timing='instrumented single-thread CPU; fixed FW-first order and unequal operation counts'))
    except BaseException as error:
        frozen.atomic_json(args.output/'failure.json',dict(status='failed_partial',state=state,error=repr(error),
            traceback=traceback.format_exc(),completed_cells=len(results),elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        payload={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file() and p.name!='payload_hashes.json'}
        frozen.atomic_json(args.output/'payload_hashes.json',payload)


if __name__=='__main__':main()
