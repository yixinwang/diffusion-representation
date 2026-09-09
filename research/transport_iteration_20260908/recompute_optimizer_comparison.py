"""Independent saved-coefficient audit; no fitting or generation-quality scoring."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.interpolate import BSpline

COMMIT='a397981d13f1bbe2121b58d5dac9f13089c243a7'
CELLS=list(itertools.product(('local','distant'),(8,32),(256,1024,4096),(8101,8102,8103)))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_hash(x):
    x=np.ascontiguousarray(x);d=hashlib.sha256(str((x.shape,x.dtype.str)).encode());d.update(x.view(np.uint8));return d.hexdigest()
def read(path):return json.loads(path.read_text())
def require(ok,message):
    if not ok:raise AssertionError(message)


def reconstruct(cell):
    world,d,n,seed=cell
    rng=np.random.default_rng(np.random.SeedSequence([seed,d,n,{'local':0,'distant':1}[world],1]))
    source=rng.uniform(size=(n,d));values=source.copy()
    children=np.arange(1,d,2) if world=='local' else np.array([d-1])
    parents=children-1 if world=='local' else np.array([0])
    target=source[:,children];amplitude=.65*np.cos(2*np.pi*source[:,parents])
    low,high=np.zeros_like(target),np.ones_like(target)
    for _ in range(54):
        midpoint=low+(high-low)/2
        cdf=midpoint+amplitude*np.sin(2*np.pi*midpoint)/(2*np.pi)
        right=cdf<target;low=np.where(right,midpoint,low);high=np.where(right,high,midpoint)
    values[:,children]=low+(high-low)/2
    return values


def features(context,response):
    # Independent dense basis construction: evaluate each scalar basis and hat.
    knots=np.r_[np.zeros(3),np.arange(1,4)/4,np.ones(3)]
    basis=np.ones((len(response),1)) if context.shape[1]==0 else BSpline(knots,np.eye(6),2)(context[:,0])
    hats=np.maximum(1-np.abs(response[:,None]*4-np.arange(5)[None,:]),0)
    return (basis[:,:,None]*hats[:,None,:]).reshape(len(response),-1)


def measure(matrix,coefficients):
    w=np.array([.125,.25,.25,.25,.125]);q=matrix@coefficients.ravel()
    require(np.isfinite(q).all() and (q>0).all(),'invalid saved density')
    gradient=(matrix.T@(-1/q/len(q))).reshape(coefficients.shape)
    vertex=np.full_like(coefficients,.175)
    for row in range(len(vertex)):
        mass=1-.175*w.sum()
        for column in np.argsort(gradient[row]/w,kind='stable'):
            added=min(mass,(3.3-.175)*w[column]);vertex[row,column]+=added/w[column];mass-=added
        require(abs(mass)<1e-12,'linear oracle leftover mass')
    return dict(objective=float(-np.log(q).mean()),gap=float(np.sum(gradient*(coefficients-vertex))),
        integral_error=float(np.abs(coefficients@w-1).max()),minimum=float(coefficients.min()),maximum=float(coefficients.max()))


def audit(payload,repo):
    manifest=read(payload/'manifest.json');summary=read(payload/'summary.json')
    require(manifest['source_commit']==COMMIT,'source revision mismatch')
    require(manifest['cells']==[list(c) for c in CELLS],'registered cell grid mismatch')
    require((manifest['bins'],manifest['max_iterations'],manifest['gap_tolerance'],manifest['lower'],manifest['upper'])==(4,500,1e-4,.175,3.3),'frozen optimizer parameters mismatch')
    require(summary['status']=='completed_optimizer_comparison' and len(summary['results'])==36,'study incomplete')
    recorded=read(payload/'payload_hashes.json')
    actual={str(p.relative_to(payload)):digest(p) for p in payload.rglob('*') if p.is_file() and p.name!='payload_hashes.json'}
    require(recorded==actual,'payload file inventory or byte hash mismatch')
    for relative,expected in manifest['source_sha256'].items():
        blob=subprocess.check_output(['git','show',f'{COMMIT}:{relative}'],cwd=repo)
        require(hashlib.sha256(blob).hexdigest()==expected,f'Git blob mismatch: {relative}')
    snapshots=list((payload/'sources').iterdir())
    require(Counter(digest(p) for p in snapshots)==Counter(manifest['source_sha256'].values()),'source snapshot multiset mismatch')
    rows=[];stream_checks=[];max_errors=dict(objective=0.,gap=0.,integral=0.)
    totals={name:dict(heads_met=0,cells_met=0,updates=0,fit_seconds=0.,fresh_check_seconds=0.,work=Counter()) for name in ('fw','accelerated')}
    for index,cell in enumerate(CELLS):
        world,d,n,seed=cell;folder=payload/f'{world}_d{d}_n{n}_seed{seed}'
        saved=read(folder/'cell.json');require(saved==summary['results'][index],'summary differs from saved cell')
        observations=np.load(folder/'observations.npy',allow_pickle=False)
        require(array_hash(observations)==saved['observations_hash'],'observation byte hash mismatch')
        rebuilt=reconstruct(cell)
        unchanged=list(range(0,d,2)) if world=='local' else list(range(d-1))
        require(np.array_equal(observations[:,unchanged],rebuilt[:,unchanged]),'untransformed uniform stream differs')
        reconstruction_error=float(np.max(np.abs(observations-rebuilt)))
        require(reconstruction_error<=1e-14,'transformed stream differs beyond floating portability tolerance')
        stream_checks.append(dict(cell=cell,saved_hash=array_hash(observations),reconstructed_hash=array_hash(rebuilt),exact_equal=bool(np.array_equal(observations,rebuilt)),maximum_error=reconstruction_error,differing_coordinates=int(np.count_nonzero(observations!=rebuilt))))
        parents=[[] if j%2==0 else [j-1] for j in range(d)] if world=='local' else [[]]+[[j-1] for j in range(1,d)]
        groups=[j%2 for j in range(d)] if world=='local' else [0]+[1]*(d-1)
        require(saved['parents']==parents and saved['groups']==groups,'old graph/pooling differs')
        require(len(saved['heads'])==2,'missing learned head')
        cell_met={name:True for name in totals}
        for head in saved['heads']:
            group=head['group'];sites=[j for j,g in enumerate(groups) if g==group]
            require(head['sites']==sites and head['independent_array_count']==n,'site or array count mismatch')
            expected_response=np.concatenate([observations[:,j] for j in sites])
            expected_context=np.concatenate([observations[:,parents[j]] for j in sites]) if parents[sites[0]] else np.empty((len(expected_response),0))
            with np.load(folder/f'head_{group}_inputs.npz',allow_pickle=False) as inputs:
                response=inputs['response'];context=inputs['context']
            require(np.array_equal(response,expected_response) and np.array_equal(context,expected_context),'pooled observed inputs differ')
            require(array_hash(response)==head['response_hash'],'response hash differs')
            require((array_hash(context) if context.shape[1] else None)==head['context_hash'],'context hash differs')
            matrix=features(context,response)
            require(np.max(np.abs(matrix.sum(1)-1))<1e-12,'independent features do not partition unity')
            entry=dict(cell=cell,group=group,site_count=len(sites),solvers={})
            for name in totals:
                result=head['solvers'][name];diagnostic=result['diagnostics'];work=result['work']
                require(read(folder/f'head_{group}_{name}.json')==result,'head file differs')
                coefficients=np.load(folder/f'head_{group}_{name}_coefficients.npy',allow_pickle=False)
                measure_result=measure(matrix,coefficients)
                for key,record_key in (('objective','objective'),('gap','frank_wolfe_gap')):
                    error=abs(measure_result[key]-diagnostic[record_key]);max_errors[key]=max(max_errors[key],error)
                    require(error<1e-9,f'independent {key} differs')
                    require(abs(measure_result[key]-result['independent_check'][key])<1e-9,'runner independent check differs')
                max_errors['integral']=max(max_errors['integral'],measure_result['integral_error'])
                require(measure_result['integral_error']<2e-12 and measure_result['minimum']>=.175-2e-12 and measure_result['maximum']<=3.3+2e-12,'coefficient infeasibility')
                met=measure_result['gap']<=1e-4
                require(met==diagnostic['converged']==result['independent_check']['gap_met'],'convergence classification differs')
                u=diagnostic['iterations'];require(0<=u<=500,'update cap exceeded')
                if name=='fw':
                    expected=dict(forward_feature_products=3*u+1,transpose_feature_products=u+1,
                        gradient_evaluations=u+1,objective_evaluations=2*u+1,linear_minimizer_calls=u+1,
                        projection_calls=0,projected_row_count=0,feature_constructions=1,column_sum_passes=0)
                    require(all(work[k]==v for k,v in expected.items()),'FW operation identity differs')
                    require(work['line_derivative_evaluations']==u+work['brent_derivative_calls'],'FW derivative count differs')
                else:
                    require(diagnostic['final_gap_recomputed'],'accelerated final gap not recomputed')
                    for key in ('forward_feature_products','transpose_feature_products','gradient_evaluations','objective_evaluations','projection_calls','projected_row_count','linear_minimizer_calls'):
                        require(work[key]==diagnostic[key],f'accelerated work counter differs: {key}')
                    require(work['projection_calls']==u and work['projected_row_count']==u*len(coefficients),'projection work count differs')
                require(np.isfinite(result['instrumented_fit_seconds']) and result['instrumented_fit_seconds']>0,'invalid timing')
                total=totals[name];total['heads_met']+=int(met);cell_met[name]&=met
                total['updates']+=u;total['fit_seconds']+=result['instrumented_fit_seconds'];total['fresh_check_seconds']+=result['fresh_check_seconds'];total['work'].update(work)
                entry['solvers'][name]={**measure_result,'gap_met':met,'updates':u,'fit_seconds':result['instrumented_fit_seconds'],'work':work}
            entry['accelerated_minus_fw_objective']=entry['solvers']['accelerated']['objective']-entry['solvers']['fw']['objective']
            rows.append(entry)
        for name in totals:totals[name]['cells_met']+=int(cell_met[name])
    for total in totals.values():total['work']=dict(total['work'])
    by_world={world:{name:dict(heads_met=sum(r['solvers'][name]['gap_met'] for r in rows if r['cell'][0]==world),
        fit_seconds=sum(r['solvers'][name]['fit_seconds'] for r in rows if r['cell'][0]==world)) for name in totals} for world in ('local','distant')}
    return dict(status='independent_recomputation_passed',source_commit=COMMIT,
        payload_hash_count=len(actual),snapshot_count=len(snapshots),payload_bytes=sum(p.stat().st_size for p in payload.rglob('*') if p.is_file()),
        completed_cells=36,head_comparisons=len(rows),maximum_absolute_recomputation_errors=max_errors,
        stream_reconstruction=stream_checks,stream_tolerance=1e-14,local_numpy=np.__version__,saved_numpy=manifest['numpy'],
        totals=totals,by_world=by_world,heads=rows,
        accelerated_lower_objective_heads=sum(r['accelerated_minus_fw_objective'] < -1e-10 for r in rows),
        accelerated_higher_objective_heads=sum(r['accelerated_minus_fw_objective'] > 1e-10 for r in rows),
        original_external_seconds=summary['elapsed_seconds'],refitting_performed=False,
        generation_quality_evaluated=False,population_claim=False,
        limitations='Operation counts checked for internal consistency, not replayed optimization; timing is original instrumented single-run FW-first CPU timing.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refusing to replace output')
    report=audit(args.payload,args.repo)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:report[k] for k in ('status','completed_cells','head_comparisons','maximum_absolute_recomputation_errors','totals','accelerated_lower_objective_heads','accelerated_higher_objective_heads')},indent=2))


if __name__=='__main__':main()
