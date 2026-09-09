from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
seeds=(77201,77202,77203);arms=('A_only','root_only','S','M','J','RQS')
labels=('Analysis only','Root only','Scalar S','Mixer M','Joint J','Spline RQS')
fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,seed in zip(axes,seeds):
    rows=[]
    for arm in arms:
        x=np.load(a.results/f'seed_{seed}'/arm/'generated.npy',mmap_mode='r')[:8]
        rows.append(np.concatenate(x.transpose(0,2,3,1),axis=1))
    ax.imshow(np.concatenate(rows,axis=0),vmin=0,vmax=1,interpolation='nearest')
    ax.set_xticks(np.arange(8)*32+15.5,labels=np.arange(1,9));ax.set_yticks(np.arange(6)*32+15.5,labels=labels)
    ax.set_title(f'Seed {seed}');ax.tick_params(length=0);ax.spines[['top','right','bottom','left']].set_visible(False)
fig.suptitle('First eight saved samples, every arm and seed',fontsize=15)
fig.text(.5,.035,'Identical Gaussian inputs within each seed. Fixed sample order; no selection. Development pilot, not a quality win.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(a.output/'cached-pilot-all-seeds.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(12,4.5));compare=('S','M','J','RQS')
for seed,color in zip(seeds,('#2c7a7b','#a84e32','#7566a5')):
    d=json.loads((a.results/f'seed_{seed}'/'evaluation.json').read_text())['arms']
    series=([d[x]['kid'] for x in compare],[d[x]['complete_nll'] for x in compare],
            [max(g/r for g,r in zip(d[x]['gradient_means'],d[x]['repair_gradient_means'])) for x in compare])
    for ax,y in zip(axes,series):ax.plot(compare,y,'o-',color=color,label=str(seed),lw=1.5)
for ax,title in zip(axes,('KID: lower is better','Complete NLL / coordinate: lower is better','Maximum image-gradient / real ratio')):
    ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[2].axhspan(.9,1.1,color='#708f60',alpha=.15,label='Registered ±10% target')
axes[0].legend(title='Independent fit seed',fontsize=8)
axes[2].legend(fontsize=8,loc='lower right')
fig.suptitle('Three-seed pilot: stronger spline baseline remains ahead',fontsize=14)
fig.text(.5,.02,'Paired development comparisons; no confidence intervals or confirmatory superiority claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(a.output/'cached-pilot-quality.png',dpi=160)
