import sys,json,pathlib,hashlib,torch
W=pathlib.Path(__file__).resolve().parent; src=W/'pro16-local-audit/pro16_artifacts'; repo=W/'diffusion-representation'; out=W/'pro16-independent-replay'
sys.path[:0]=[str(src),str(repo/'qalt/src')]
from stable_spline import stable_spline
from run_supplement import selected
from run_audit import oracle
import mpmath as mp
mp.mp.dps=100
from qalt.reflected_dense_spline import reflected_dense_spline_kernel as ours
from original.dense_spline import dense_spline_kernel as old
torch.set_num_threads(1)
rows=[]
for w in json.loads((src/'results/run_001/original_failure_witnesses.json').read_text()):
 dtype=getattr(torch,w['dtype'].split('.')[-1]); z=torch.tensor([w['input']],dtype=dtype); ps=[torch.tensor([w[k]],dtype=dtype) for k in ('raw_w','raw_h','raw_d')]; row={'witness':w,'outputs':{}}
 for name,fn in [('original',old),('ours_reflected',ours),('pro16_promoted_odds',stable_spline)]:
  v,ld,valid=fn(z,*ps,inverse=True); row['outputs'][name]={'value':float(v[0]),'logdet':float(ld[0]),'valid':bool(valid)}
 rows.append(row)
extreme=[]
for w in json.loads((out/'supplement/extreme_logits.json').read_text()):
 dtype=getattr(torch,w['dtype'].split('.')[-1]); z=torch.tensor([w['input']],dtype=dtype); ps=[torch.tensor([p],dtype=dtype) for p in w['raw']]; v,ld,valid=ours(z,*ps,inverse=True)
 ov,ol=oracle(*[t[0] for t in selected(z,*ps)])
 extreme.append({'dtype':str(dtype),'scale':w['scale'],'input':w['input'],'ours_fixed_own_knots_value_error':float(abs(mp.mpf(float(v[0]))-ov)),'ours_fixed_own_knots_logdet_error':float(abs(mp.mpf(float(ld[0]))-ol)),'ours_value_oracle_error':abs(float(v[0])-float(w['oracle_value'])),'ours_logdet_oracle_error':abs(float(ld[0])-float(w['oracle_ld'])),'ours_rounded_roundtrip':float(abs(ours(v,*ps)[0][0]-z[0])),'ours_valid':bool(valid),'ours_value':float(v[0]),'ours_logdet':float(ld[0]),'pro16_valid':w['candidate_valid'],'pro16_roundtrip':w['rounded_inverse_forward_residual']})
report={'scope':'CPU synthetic diagnostic only; not native replay; ordinary floating arithmetic','ours_source_sha256':hashlib.sha256((repo/'qalt/src/qalt/reflected_dense_spline.py').read_bytes()).hexdigest(),'witnesses':rows,'extreme':extreme,'summary':{'original_witness_rejects':sum(not r['outputs']['original']['valid'] for r in rows),'ours_witness_rejects':sum(not r['outputs']['ours_reflected']['valid'] for r in rows),'pro16_witness_rejects':sum(not r['outputs']['pro16_promoted_odds']['valid'] for r in rows),'ours_extreme_rejects':sum(not r['ours_valid'] for r in extreme)}}
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']))
