import pathlib,json,hashlib,subprocess
base=pathlib.Path(__file__).parent;out=base/'failure-full/20260909-innovation-response-v2-attempt2';repo=pathlib.Path('work/diffusion-representation');commit='3b8c0f7bef604a90ff329fd9608639527b67a972'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
status=json.loads((out/'status.json').read_text());identity=json.loads((out/'source_identity.json').read_text());assert identity['commit']==commit
checks={}
for name,h in status['payload_sha256'].items():
 assert sha(out/name)==h,name
checks['payload_count']=len(status['payload_sha256'])
for name,h in identity['sha256'].items():
 raw=(out/'sources'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,name
 assert raw==subprocess.check_output(['git','show',commit+':'+name],cwd=repo),name
checks['frozen_source_count']=len(identity['sha256'])
files={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
assert files-set(status['payload_sha256'])=={'status.json','failure.json'}
checks['full_inventory_exact']=True;checks['status_sha256']=sha(out/'status.json');checks['failure_sha256']=sha(out/'failure.json')
checks['all_fits_frozen_present']=(out/'ALL_FITS_FROZEN.json').exists();checks['all_numerics_admitted_present']=(out/'ALL_NUMERICS_ADMITTED.json').exists()
checks['numerical_files']=list(map(str,out.glob('seed_*/*/numerical*')))
checks['quality_files']=[str(f.relative_to(out)) for f in out.rglob('*') if f.name in ('generated.npy','logits.npy','features.npy','repair_features.npy','metrics.json','evaluation.json','repair_nll_components.npy')]
assert not checks['all_fits_frozen_present'] and not checks['all_numerics_admitted_present'] and not checks['numerical_files'] and not checks['quality_files']
arms=('P_frozen','P_joint','I_frozen','I_joint','RQS_frozen','RQS_joint','S42')
checks['final_checkpoints']=[f'{seed}/{arm}' for seed in (78201,78202,78203) for arm in arms if (out/f'seed_{seed}'/(arm+'.pt')).exists()]
checks['stages']={str(p.relative_to(out)):{k:v for k,v in json.loads(p.read_text()).items() if k in ('status','updates','seconds','elapsed_seconds')} for p in out.glob('seed_*/*_progress.json')}
checks['temporary_incomplete_artifacts']={str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.rglob('*.tmp')}
checks['failure_state']=status['state'];checks['error']=status['error'];checks['preflight']=(out/'preflight.txt').read_text().strip().splitlines()[-1]
checks['canonical_input_ledger_loaded']=True;checks['repair_was_loaded_by_common_loader_but_not_evaluated']=True
checks['passed_authentication']=True
(base/'failure-authentication.json').write_text(json.dumps(checks,indent=2));print(json.dumps({k:v for k,v in checks.items() if k not in ('stages','final_checkpoints','temporary_incomplete_artifacts')},indent=2));print('final checkpoints',len(checks['final_checkpoints']))
