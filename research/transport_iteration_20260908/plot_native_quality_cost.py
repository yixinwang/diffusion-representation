from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description='Plot all verified native pilot energy/latency arms; no dataset reads.')
parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parent/'psc_native_completed')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.records
e=json.loads((root/'evaluations.json').read_text());t=json.loads((root/'latencies.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
for ax,batch in zip(axes,['1','64']):
    nfes=[4,8,16,32,64]
    x=[t[f'residual_fm_nfe_{n}'][batch]['median_seconds']*1000 for n in nfes]
    y=[e[f'residual_fm_nfe_{n}']['energy_mean'] for n in nfes]
    ax.plot(x,y,'o-',color='#2878a1',linewidth=1,markersize=5,label='Residual FM (actual calls)')
    for n,a,b in zip(nfes,x,y):ax.annotate(str(n),(a,b),xytext=(4,5),textcoords='offset points',fontsize=8)
    for name,label,col,mark in [('coupling','Coupling','#bf4930','D'),('analysis_only','Analysis only (lower training budget)','#777777','s')]:
        ax.scatter(t[name][batch]['median_seconds']*1000,e[name]['energy_mean'],c=col,marker=mark,s=45,label=label)
    ax.set_title(f'Batch size {batch}')
    ax.set_xlabel('Whole-generator latency (ms; median of 3)')
    ax.grid(alpha=.2)
    ax.margins(y=.12)
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
axes[0].set_ylabel('Energy score (lower is better)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,fontsize=8)
fig.suptitle('Native pilot: verified scores, mixed quality–cost results',fontsize=13)
fig.text(.5,.022,'One short fit on reused development data; no confirmatory uncertainty claim. Protected data unopened.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.18,1,.94))
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170)
