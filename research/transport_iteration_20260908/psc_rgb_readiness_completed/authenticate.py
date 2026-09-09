from pathlib import Path
import json,hashlib,tarfile,subprocess
p=Path(__file__).resolve().parent;manifest=json.loads((p/'transfer.json').read_text());archive=p/'complete-readiness.tar';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert archive.stat().st_size==manifest['tar_bytes'] and sha(archive)==manifest['tar_sha256']
out=p/'full';out.mkdir(exist_ok=False)
with tarfile.open(archive,'r:') as t:
 members=t.getmembers();assert {m.name for m in members}==set(manifest['files']) and len(members)==len(manifest['files'])
 for m in members:
  q=Path(m.name);assert m.isfile() and not q.is_absolute() and '..' not in q.parts
  raw=t.extractfile(m).read();info=manifest['files'][m.name];assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
  dest=out/q;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
result=out/'diffusion-results/20260909-rgb-readiness-84c0cad3';stage=out/'rgb-readiness-launch-20260909-84c0cad3';status=json.loads((result/'status.json').read_text());identity=json.loads((result/'source_identity.json').read_text());config=json.loads((p/'config.json').read_text());repo=p.parent/'diffusion-representation'
assert status['status']=='completed_data_free_readiness' and status['real_data_accessed'] is False
assert {str(f.relative_to(result)) for f in result.rglob('*') if f.is_file() and f.name!='status.json'}==set(status['payload_sha256'])
for rel,value in status['payload_sha256'].items():assert sha(result/rel)==value
assert identity['commit']==config['source_commit'] and identity['sha256']==config['source_sha256'] and len(identity['sha256'])==7
for rel,value in identity['sha256'].items():
 assert sha(result/'source'/rel)==value
 assert (result/'source'/rel).read_bytes()==subprocess.check_output(['git','show',identity['commit']+':'+rel],cwd=repo)
assert (stage/'submit-once.sh').read_bytes()==(p/'submit-once.sh').read_bytes()
startup=json.loads((stage/'startup_verified.json').read_text());assert startup['script_sha256']==sha(p/'submit-once.sh')
assert all(v['project_id']==559736 and v['write_read_remove'] for v in startup['storage_checks'].values())
assert (p/'output-project.txt').read_text().split()[0]=='559736'
prep=json.loads((stage/'preparation.json').read_text());assert prep['status']=='prepared_no_allocation' and prep['original_unchanged'] is True
assert 'Relinquishing job allocation 45633712' in (p/'allocation.stderr').read_text()
account=(p/'terminal-accounting-and-status.txt').read_text();assert '45633712|COMPLETED|0:0|' in account
receipt={'status':'full_payload_authenticated','job_id':'45633712','source_commit':identity['commit'],'source_count':7,'result_payload_count':len(status['payload_sha256']),'transfer_files':len(manifest['files']),'transfer_bytes':manifest['payload_bytes'],'archive_sha256':sha(archive),'result_status_sha256':sha(result/'status.json'),'launcher_sha256':sha(p/'submit-once.sh'),'terminal_completed_exit':'0:0','allocation_relinquished':True,'requested_qos':'gpuinteract','recorded_qos':'gpu','actual_output_project_id':559736,'preallocation_python36_failure_preserved':True,'original_checkout_unchanged_at_preparation':True,'result_path':str(result),'numeric_metric_recomputation':'delegated to root independent audit; not performed here','retrieval_note':'scp connection closed before transfer; successful read-only ssh cat exact archive hash verified, no resubmission'}
(p/'authentication.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
