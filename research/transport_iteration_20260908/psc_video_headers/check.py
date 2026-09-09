import json,hashlib,subprocess,re
from pathlib import Path
p=Path('work/psc-video-headers/results');s=json.loads((p/'status.json').read_text());m=json.loads((p/'manifest.json').read_text());rev='92702635134856f6a0b3a9f82186394b97a7d59e';rel='research/transport_iteration_20260908/video_archive_prerequisite/'
assert s['status']=='complete' and s['payload_accessed'] is False and s['archive_sha256_recomputed'] is False
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==s['manifest_file_sha256']
assert hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()==s['manifest_sha256']
for name in ['archive_reader.py','run_header_audit.py']:
 raw=subprocess.check_output(['git','show',rev+':'+rel+name],cwd='work/diffusion-representation')
 if name=='archive_reader.py':assert raw==(p/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['source']['sha256']
 else:assert hashlib.sha256(raw).hexdigest()==s['runner_sha256'];(p/name).write_bytes(raw)
assert m['archive_identity'][2]==171386880 and len(m['members'])==469
names=set();groups={k:set() for k in ['train','val','test']};counts={k:0 for k in groups};classes=set();end=0
for row in m['members']:
 assert row['path'] not in names;names.add(row['path']);parts=row['path'].split('/');assert not row['path'].startswith('/') and '..' not in parts
 assert row['header_offset']==end and row['offset']==end+512
 assert row['padded_end']==row['offset']+((row['size']+511)//512)*512
 assert row['padded_end']<=m['archive_identity'][2];end=row['padded_end']
 if row['type']=='regular':
  split,action,file=parts[-3:];assert split==row['split'] and action==row['class_name']
  match=re.fullmatch(r'v_(.+)_g(\d{2})_c(\d{2})\.avi',file);assert match and match[1]==action
  assert row['group']==action+'/g'+match[2] and row['clip']=='c'+match[3]
  counts[split]+=1;groups[split].add(row['group']);classes.add(action)
 else:assert row['type']=='directory' and row['size']==0
assert counts=={'train':300,'val':30,'test':75}
assert {k:len(v) for k,v in groups.items()}=={'train':195,'val':25,'test':30}
assert all(not groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]) and len(classes)==10
result={'status':'independent_metadata_audit_pass','entries':469,'regular_files':405,'directories':64,'clip_counts':counts,'group_counts':{k:len(v) for k,v in groups.items()},'classes':len(classes),'split_group_overlap':False,'source_commit':rev,'manifest_logical_sha256':s['manifest_sha256'],'manifest_file_sha256':s['manifest_file_sha256'],'payload_read':False,'whole_archive_hash_recomputed':False,'limitation':'Header block digest is recorded from frozen reader, not independently recomputed from raw headers; only metadata downloaded.'}
Path('work/psc-video-headers/machinecheck.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
