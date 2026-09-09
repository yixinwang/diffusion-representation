from pathlib import Path
from fractions import Fraction
import json,hashlib,subprocess,collections
base=Path(__file__).resolve().parent;r=base/'retrieved/20260909-video-timestamps';d=r/'decode'
rows=[json.loads(s) for s in (d/'frames.jsonl').read_text().splitlines()];flags=json.loads((d/'timestamp_flags.json').read_text());status=json.loads((d/'status.json').read_text());launch=json.loads((r/'launcher_status.json').read_text())
errors=[]
check=lambda name,ok:errors.append(name) if not ok else None
files=json.loads((d/'ARTIFACTS.json').read_text())
for name,record in files.items():
 raw=(d/name).read_bytes();check('artifact:'+name,len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'])
check('enumeration',set(files)=={f.name for f in d.iterdir() if f.is_file()}-{'ARTIFACTS.json','status.json'})
manifest=json.loads((d/'manifest.json').read_text());check('manifest',hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='180eabace318325e1b7ee6f2d5465b0e89f86789bcf6e0d4f3a92d3cc88a6f96')
repo=base.parent/'diffusion-representation';prefix='research/transport_iteration_20260908/';rev='effc2c565b050193561978c9bce686b3a3f996eb'
for name,path in {'diagnostic_metadata.py':'video_decode_prerequisite/diagnostic_metadata.py','frozen_helpers.py':'video_decode_prerequisite/helpers.py','frozen_reader.py':'video_archive_prerequisite/archive_reader.py'}.items():
 check('source:'+name,(d/name).read_bytes()==subprocess.check_output(['git','-C',str(repo),'show',rev+':'+prefix+path]))
check('complete',status['status']=='complete' and launch['status']=='complete')
check('source/member',status['member_sha256']=='699175c50544283f3b8537387403ff5b82958e4b18e590611129013131d1601a' and launch['dependency_files_verified']==264)
check('indices',[x['index'] for x in rows]==list(range(len(rows))))
previous=None;derived=[];pairs=[]
for x in rows:
 fs=[];stamp=None
 if x['pts'] is None:fs.append('null_pts')
 if x['time_base'] is None or Fraction(*x['time_base'])<=0:fs.append('null_or_nonpositive_time_base')
 if not fs:stamp=Fraction(x['pts'])*Fraction(*x['time_base'])
 if stamp is not None and previous is not None and stamp<=previous[1]:
  fs.append('nonincreasing_vs_previous_valid_frame');pairs.append([previous[0],x['index'],str(previous[1]),str(stamp)])
 check('row_flags:'+str(x['index']),fs==x['timestamp_flags'])
 if fs:derived.append({'index':x['index'],'flags':fs})
 if stamp is not None:previous=(x['index'],stamp)
check('flagfile',flags==derived)
check('counts',len(rows)==status['diagnostic_frames'] and len(flags)==status['timestamp_flagged_frames'])
pts=[x['pts'] for x in rows];dts=[x['dts'] for x in rows]
selected=[]
for k in range(8):
 target=Fraction(k*(len(rows)-1),7);q,rem=divmod(target.numerator,target.denominator);i=q+(2*rem>target.denominator)
 selected.append({'k':k,'normalized_position':[k,7],'target_index':[target.numerator,target.denominator],'index':i,'pts':rows[i]['pts'],'dts':rows[i]['dts'],'time_base':rows[i]['time_base']})
result={'status':'metadata_audit_pass' if not errors else 'failed','errors':errors,'job':'45579152','frames':len(rows),'timestamp_reversals':len(pairs),'null_pts':pts.count(None),'null_dts':dts.count(None),'pts_exact_permutation_1_to_N':sorted(pts)==list(range(1,len(rows)+1)),'nonterminal_dts_exact_1_to_Nminus1':dts[:-1]==list(range(1,len(rows))),'terminal_dts':dts[-1],'pts_minus_index_plus1_counts':dict(collections.Counter(x['pts']-(x['index']+1) for x in rows)),'metadata_values':{key:sorted({str(x.get(key)) for x in rows}) for key in ['width','height','format','colorspace','color_range','color_primaries','color_trc','interlaced','rotation','time_base']},'reversal_pairs':pairs,'prospective_index_selection_not_executed':selected,'no_arrays':not any(d.glob('*.npy')),'slurm':'COMPLETED07:03 MaxRSS43144K ExitCode0:0','payload_sha256':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
(base/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('reversal_pairs','payload_sha256')},indent=2))
