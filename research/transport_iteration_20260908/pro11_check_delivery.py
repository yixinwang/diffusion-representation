"""Read-only Pro11 text-subset inventory; no imports of fitting/evaluation code."""
from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DIRECTORY=HERE/'pro11_artifacts'
ORIGINAL='21f3997aed1447b05f3643032dc133b3e0cf9305'
CHERRY='3a181cc'
manifest=(DIRECTORY/'SHA256SUMS').read_text().splitlines();entries=[]
for line in manifest:
 expected,name=line.split(maxsplit=1);p=DIRECTORY/name
 entries.append({'name':name,'expected_sha256':expected,'present':p.exists(),'actual_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None})
files=[]
for p in sorted(DIRECTORY.iterdir()):
 if not p.is_file():continue
 rel=str(p.relative_to(ROOT));raw=p.read_bytes()
 original=subprocess.check_output(['git','show',f'{ORIGINAL}:{rel}'],cwd=ROOT)
 cherry=subprocess.check_output(['git','show',f'{CHERRY}:{rel}'],cwd=ROOT)
 files.append({'name':p.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'matches_original_git_blob':raw==original,'matches_cherry_git_blob':raw==cherry})
records=[]
for seed in (1109101,1109102,1109103):
 value=json.loads((DIRECTORY/f'learning_seed_{seed}.json').read_text())
 source={name:hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()==sha for name,sha in value['environment']['source_sha256'].items()}
 records.append({'seed':seed,'recorded_source_hashes_match':source})
report={'scope':'source/text manifest checks only; no fitted states, bank regeneration, population-KL recomputation or fitting',
 'original_commit':ORIGINAL,'cherry_commit':subprocess.check_output(['git','rev-parse',CHERRY],cwd=ROOT,text=True).strip(),
 'manifest_entries':entries,'present_manifest_entries':sum(e['present'] for e in entries),'missing_manifest_entries':sum(not e['present'] for e in entries),
 'present_hashes_match':all(e['expected_sha256']==e['actual_sha256'] for e in entries if e['present']),
 'full_original_package_verified':False,'published_files':files,'all_published_files_match_both_git_blobs':all(e['matches_original_git_blob'] and e['matches_cherry_git_blob'] for e in files),
 'original_text_bytes':sum(e['bytes'] for e in files if e['name']!='DELIVERY_SUBSET.md'),'record_source_checks':records}
(HERE/'pro11_delivery_independent_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('manifest_entries','published_files','record_source_checks')},indent=2))
assert report['present_manifest_entries']==14 and report['missing_manifest_entries']==6
assert report['present_hashes_match'] and report['all_published_files_match_both_git_blobs']
assert all(all(x['recorded_source_hashes_match'].values()) for x in records)
