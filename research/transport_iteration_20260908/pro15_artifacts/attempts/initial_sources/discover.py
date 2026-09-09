"""Train-only discovery and fitting. No evaluator import, RNG, phase or graph input."""
import time
import numpy as np
from model import observed,psi,design,Model,project_simplex_floor,KAPPA
from bounds import radius

def isolated_edges(adjacency):
    a=np.array(adjacency,dtype=bool,copy=True);np.fill_diagonal(a,False)
    if not np.array_equal(a,a.T):raise ValueError('undirected adjacency required')
    keep=a & (a.sum(1)==1)[:,None] & (a.sum(0)==1)[None,:]
    return np.argwhere(np.triu(keep,1))

def spectral_graph(x,cfg):
    h=design(x[:,0])[:,1:];n=len(x);r=radius(n,cfg.edges,cfg.delta_graph)
    pairs=[];stats={};fops=0
    for b in range(cfg.blocks):
        start=cfg.root_dim+b*cfg.block_size
        acc=np.zeros((2,cfg.block_size,cfg.block_size))
        # Exact blocked sum: O(batch*m + m^2) working memory per block.
        for lo in range(0,n,2048):
            f=psi(x[lo:lo+2048,start:start+cfg.block_size])
            for k in range(2):acc[k]+=f.T@(h[lo:lo+2048,k,None]*f)
        acc/=n;s=np.linalg.norm(acc,axis=0);a=s>r;np.fill_diagonal(a,False)
        pairs.append(isolated_edges(a));stats[f'block_{b}_correlations']=acc
        stats[f'block_{b}_scores']=s
        fops+=2*n*cfg.block_size**2
    return pairs,stats,dict(threshold=r,method='phase_invariant_bernstein',dense_multiplications=fops,discovery_arrays=n)

def unconditional_graph(x,cfg):
    pairs=[];stats={}
    for b in range(cfg.blocks):
        start=cfg.root_dim+b*cfg.block_size;f=psi(x[:,start:start+cfg.block_size]);s=f.T@f/len(x)
        pairs.append(isolated_edges(abs(s)>.175));stats[f'block_{b}_correlations']=s
    return pairs,stats,dict(threshold=.175,method='old_unconditional',discovery_arrays=len(x),dense_multiplications=len(x)*cfg.blocks*cfg.block_size**2)

def bin_grams(x,cfg,b):
    start=cfg.root_dim+b*cfg.block_size;idx=np.minimum((x[:,0]*cfg.context_bins).astype(int),cfg.context_bins-1)
    out=np.zeros((cfg.context_bins,cfg.block_size,cfg.block_size));counts=np.bincount(idx,minlength=cfg.context_bins)
    for k in range(cfg.context_bins):
        # Bins partition rows: total Gram arithmetic is n*m^2, not B*n*m^2.
        loc=np.flatnonzero(idx==k)
        for lo in range(0,len(loc),2048):
            f=psi(x[loc[lo:lo+2048],start:start+cfg.block_size]);out[k]+=f.T@f
    return out,counts

def histogram_lr_graph(x,cfg):
    """Strong comparator: split conditional-likelihood e-test with 8 context bins.
    log(1+t)<=t gives an exact mathematical rejection screen, saving expensive
    log evaluations. All candidate edges still enter the screening Grams.
    """
    n1=len(x)//2;proposal=x[:n1];test=x[n1:];threshold=np.log(cfg.edges/cfg.delta_graph)
    pairs=[];stats={};survivors=0;logs=0
    for b in range(cfg.blocks):
        g1,counts=bin_grams(proposal,cfg,b);g2,_=bin_grams(test,cfg,b)
        theta=np.divide(g1,counts[:,None,None],out=np.zeros_like(g1),where=counts[:,None,None]>0)
        theta=np.clip(theta,-KAPPA,KAPPA)
        upper=(theta*g2).sum(0);loge=np.full((cfg.block_size,cfg.block_size),-np.inf)
        candidates=np.argwhere(np.triu(upper>=threshold-1e-10*len(test),1))
        start=cfg.root_dim+b*cfg.block_size
        for i,j in candidates:
            val=0.
            for lo in range(0,len(test),2048):
                xx=test[lo:lo+2048];idx=np.minimum((xx[:,0]*cfg.context_bins).astype(int),cfg.context_bins-1)
                y=psi(xx[:,start+i])*psi(xx[:,start+j])
                val+=np.log1p(theta[idx,i,j]*y).sum()
            loge[i,j]=loge[j,i]=val
        a=loge>threshold;pairs.append(isolated_edges(a))
        stats[f'block_{b}_proposal_theta']=theta;stats[f'block_{b}_linear_upper']=upper
        stats[f'block_{b}_loge']=loge
        survivors+=len(candidates);logs+=len(candidates)*len(test)
    return pairs,stats,dict(method='split_histogram_likelihood_ratio',threshold_log_e=threshold,
      proposal_arrays=n1,test_arrays=len(test),screen_survivors=survivors,log1p_evaluations=logs,
      dense_multiplications=len(x)*cfg.blocks*cfg.block_size**2,
      finite_type_I_control='Union bound at delta_graph over all original candidate edges; no null-edge independence assumption.')

def fit_root(x,cfg):
    p=np.full((cfg.root_dim,cfg.root_bins),1/cfg.root_bins)
    for j in range(1,cfg.root_dim):
        idx=np.minimum((x[:,j]*cfg.root_bins).astype(int),cfg.root_bins-1)
        p[j]=project_simplex_floor(np.bincount(idx,minlength=cfg.root_bins)/len(x),cfg.root_floor)
    return p

def fit_parameters(x,cfg,pairs,mode):
    width=3 if mode=='harmonic' else cfg.context_bins;coef=np.zeros((cfg.blocks,width));stats={}
    h=design(x[:,0]);gram=h.T@h/len(x);eig=np.linalg.eigvalsh(gram)
    idx=np.minimum((x[:,0]*cfg.context_bins).astype(int),cfg.context_bins-1)
    counts=np.bincount(idx,minlength=cfg.context_bins)
    for b,p in enumerate(pairs):
        if not len(p):continue
        start=cfg.root_dim+b*cfg.block_size
        y=(psi(x[:,start+p[:,0]])*psi(x[:,start+p[:,1]])).mean(1)
        if mode=='harmonic':
            if eig[0]>=.5:coef[b]=np.linalg.solve(gram,h.T@y/len(x))
        else:
            coef[b]=np.divide(np.bincount(idx,weights=y,minlength=cfg.context_bins),counts,out=np.zeros(cfg.context_bins),where=counts>0)
            coef[b]=np.clip(coef[b],-KAPPA,KAPPA)
        stats[f'block_{b}_pooled_response']=y
    stats['design_gram']=gram;stats['context_counts']=counts
    return coef,stats,dict(design_min_eigenvalue=float(eig[0]),design_gate_pass=bool(eig[0]>=.5),parameter_arrays=len(x),pooling='Conditional independence of true disjoint pairs only; arrays remain the graph iid units.')

def fit(x,cfg,arm):
    """Eligible interface: observed arrays, public config, predeclared arm only."""
    t=time.perf_counter();x=observed(x,cfg)
    if len(x)!=cfg.graph_arrays+cfg.parameter_arrays:raise ValueError('wrong registered sample budget')
    root=fit_root(x,cfg);root_time=time.perf_counter()-t;s=time.perf_counter();g=x[:cfg.graph_arrays]
    mode='histogram' if arm=='histogram_lr' else 'harmonic'
    if arm=='spectral':pairs,stats,gd=spectral_graph(g,cfg)
    elif arm=='unconditional':pairs,stats,gd=unconditional_graph(g,cfg)
    elif arm=='histogram_lr':pairs,stats,gd=histogram_lr_graph(g,cfg)
    elif arm=='product':pairs=[np.empty((0,2),int) for _ in range(cfg.blocks)];stats={};gd={'method':'product'}
    else:raise ValueError('unknown eligible arm')
    gt=time.perf_counter()-s;s=time.perf_counter()
    coef,ps,pd=fit_parameters(x[cfg.graph_arrays:],cfg,pairs,mode);stats.update(ps)
    pt=time.perf_counter()-s
    diagnostics=dict(arm=arm,root_seconds=root_time,graph_seconds=gt,parameter_seconds=pt,
      fit_seconds=time.perf_counter()-t,graph=gd,parameter=pd,root_arrays=len(x),
      total_independent_arrays=len(x),truth_input=False,first_root_known_uniform=True)
    return Model(cfg,root,pairs,coef,mode,diagnostics),stats

def oracle_graph_diagnostic(x,cfg,privileged_pairs,mode='harmonic'):
    """Explicitly ineligible privileged diagnostic; not called by fit()."""
    t=time.perf_counter();x=observed(x,cfg);root=fit_root(x,cfg)
    p=[np.asarray(v,dtype=int) for v in privileged_pairs]
    coef,stats,d=fit_parameters(x[cfg.graph_arrays:],cfg,p,mode)
    return Model(cfg,root,p,coef,mode,dict(arm='oracle_graph_'+mode,privileged_graph=True,
      fit_seconds=time.perf_counter()-t,parameter=d)),stats
