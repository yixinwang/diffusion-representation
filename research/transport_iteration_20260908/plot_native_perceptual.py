from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description='Plot all frozen native perceptual arms; no dataset reads.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
base=Path(__file__).resolve().parent
m=json.loads((base/'psc_perceptual_appendix/metrics.json').read_text())
names=['analysis_only','coupling']+[f'residual_fm_nfe_{n}' for n in (4,8,16,32,64)]
labels=['Analysis only','Coupling','FM 4','FM 8','FM 16','FM 32','FM 64']
colors=['#858585','#c65b3e']+['#367f9b']*5
fig,axes=plt.subplots(1,2,figsize=(10,5.0))
for ax,key,title in zip(axes,['kid_unbiased_full_bank','coverage'],['KID — lower is better','Coverage — higher is better']):
    values=[m[n][key] for n in names]
    ax.barh(labels,values,color=colors,height=.63)
    ax.invert_yaxis();ax.set_title(title,fontsize=12)
    ax.set_xlim(0,max(values)*1.24)
    for i,v in enumerate(values):ax.text(v+max(values)*.02,i,f'{v:.4f}',va='center',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Native image pilot: coupling does not win across quality measures',fontsize=13)
fig.text(.5,.05,'All 7 arms: 2,000 generated images against 1,000 reused development images. One training seed.',ha='center',fontsize=8)
fig.text(.5,.018,'Analysis-only has a lower training budget. All metrics independently recomputed from preserved features.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.09,1,.94))
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170)
