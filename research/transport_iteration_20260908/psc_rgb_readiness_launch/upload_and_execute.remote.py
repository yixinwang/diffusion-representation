import os,json,hashlib,subprocess,base64,time
from pathlib import Path
C={'source_commit': '84c0cad383be64d6632a5fa61e5d1bc4b605ca5f', 'source_sha256': {'qalt/src/qalt/__init__.py': 'a54aca828ea497a35a82724573d32fbba7d029370e9b66f0c170666b076cbdaf', 'qalt/src/qalt/core.py': '4f7893896da3d4330837ca41537c361ca2f65d2e5a17428524620601aec25fd6', 'qalt/src/qalt/rgb_codec_flow_matching.py': '48445678a60ab3ec031e64523c6f642f0ae6b3bc826847c7b458b6185fb6a1c4', 'qalt/tests/test_rgb_codec_flow_matching.py': 'b13e6d73044aed7357187da93a6cc53f56a63b382cb69cf314901dc8e6c49644', 'qalt/experiments/rgb_codec_readiness/run.py': '67063c491c9aa66df8aa72c401dbde362845fa0dceaefe94a3f6cd1e69666b1b', 'qalt/experiments/rgb_codec_readiness/PROTOCOL.md': '7af58f59a218428ddfa172545e2552b23b347f849dbb8ef0b347b0d20035af5f', 'qalt/experiments/rgb_codec_readiness/run.sh': '0749017c4fb60c9daf29e41e10995d8586f6950277d9ba737811d8b3856d3941'}, 'checkout': '/ocean/projects/mth260022p/ywang26/diffusion-rgb-readiness-20260909-84c0cad3', 'staging': '/ocean/projects/mth260022p/ywang26/rgb-readiness-launch-20260909-84c0cad3', 'result_root': '/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-rgb-readiness-84c0cad3', 'new_project': '/ocean/projects/mth260022p/ywang26', 'original_read_only_checkout': '/ocean/projects/mth250006p/ywang26/diffusion-representation', 'remote': 'https://github.com/yixinwang/diffusion-representation.git'}
stage=Path(C['staging']);result=Path(C['result_root']);receipt={}
if (stage/'launch_failure.json').exists():(stage/'launch_failure.json').rename(stage/'preallocation-attempt1-failure.json')
try:
 assert not result.exists()
 for target in (stage,result.parent):
  output=subprocess.check_output(['lfs','project','-d',str(target)],universal_newlines=True)
  assert output.split()[0]=='559736',output
  probe=target/('.rgb-readiness-probe-'+str(os.getpid()));payload=b'bounded readiness probe\n'
  with probe.open('xb') as f:f.write(payload)
  assert probe.read_bytes()==payload;probe.unlink()
  receipt[str(target)]={'lfs_project':output.strip(),'project_id':559736,'write_read_remove':True}
 raw=base64.b64decode('IyEvdXNyL2Jpbi9lbnYgYmFzaApzZXQgLWV1byBwaXBlZmFpbApTVEFHRT0vb2NlYW4vcHJvamVjdHMvbXRoMjYwMDIycC95d2FuZzI2L3JnYi1yZWFkaW5lc3MtbGF1bmNoLTIwMjYwOTA5LTg0YzBjYWQzClJFUE89L29jZWFuL3Byb2plY3RzL210aDI2MDAyMnAveXdhbmcyNi9kaWZmdXNpb24tcmdiLXJlYWRpbmVzcy0yMDI2MDkwOS04NGMwY2FkMwpSRVNVTFQ9L29jZWFuL3Byb2plY3RzL210aDI2MDAyMnAveXdhbmcyNi9kaWZmdXNpb24tcmVzdWx0cy8yMDI2MDkwOS1yZ2ItcmVhZGluZXNzLTg0YzBjYWQzClJFVj04NGMwY2FkMzgzYmU2NGQ2NjMyYTVmYTYxZTVkMWJjNGI2MDVjYTVmCmV4cG9ydCBTT1VSQ0VfQ09NTUlUPSIkUkVWIiBSRVNVTFRfUk9PVD0iJFJFU1VMVCIKZXhwb3J0IFBZVEhPTlBZQ0FDSEVQUkVGSVg9IiRTVEFHRS9weWNhY2hlIiBYREdfQ0FDSEVfSE9NRT0iJFNUQUdFL2NhY2hlIiBUTVBESVI9IiRTVEFHRS90bXAiCmV4cG9ydCBUT1JDSF9IT01FPSIkU1RBR0UvdG9yY2gtY2FjaGUiIENVREFfQ0FDSEVfUEFUSD0iJFNUQUdFL2N1ZGEtY2FjaGUiCmV4cG9ydCBIRl9IT01FPSIkU1RBR0UvaGYtY2FjaGUiIFRSSVRPTl9DQUNIRV9ESVI9IiRTVEFHRS90cml0b24tY2FjaGUiCnRlc3QgISAtZSAiJFJFU1VMVCIKY2QgIiRSRVBPIgojIE9uZSBhbGxvY2F0aW9uLCBvbmUgd29ya2VyLCBpbW1lZGlhdGUgcmVsZWFzZSB3aGVuIHRoZSB3b3JrZXIgZXhpdHMuIE5vIHJldHJpZXMuCmV4ZWMgc2FsbG9jIC0tYWNjb3VudD1jaXMyNjAyNDNwIC0tcGFydGl0aW9uPUdQVS1zaGFyZWQgLS1xb3M9Z3B1aW50ZXJhY3QgLS1ncmVzPWdwdTp2MTAwLTMyOjEgLS1leGNsdWRlPXYwMDUgLU4xIC1uMSAtYzQgLS1tZW09MTYwMDBNIC10MDE6MDA6MDAgLS1qb2ItbmFtZT1yZ2ItY29kZWMtcmVhZGluZXNzIFwKIHNydW4gLS1leHBvcnQ9QUxMIC0tb3V0cHV0PSIkU1RBR0Uvd29ya2VyLSVqLm91dCIgLS1lcnJvcj0iJFNUQUdFL3dvcmtlci0lai5lcnIiIGJhc2ggcWFsdC9leHBlcmltZW50cy9yZ2JfY29kZWNfcmVhZGluZXNzL3J1bi5zaAo=');assert hashlib.sha256(raw).hexdigest()=='d7023072a1172d1fe08828e25bbceaa67373c812051c8d3e732209b037b2a771'
 launch=stage/'submit-once.sh'
 with launch.open('xb') as f:f.write(raw)
 assert hashlib.sha256(launch.read_bytes()).hexdigest()=='d7023072a1172d1fe08828e25bbceaa67373c812051c8d3e732209b037b2a771'
 (stage/'startup_verified.json').write_text(json.dumps({'source_commit':C['source_commit'],'script_sha256':'d7023072a1172d1fe08828e25bbceaa67373c812051c8d3e732209b037b2a771','storage_checks':receipt},indent=2)+'\n')
 print(json.dumps({'status':'verified_before_single_allocation','script_sha256':'d7023072a1172d1fe08828e25bbceaa67373c812051c8d3e732209b037b2a771','storage_checks':receipt}),flush=True)
 os.execvp('bash',['bash',str(launch)])
except BaseException as exc:
 (stage/'launch_failure.json').write_text(json.dumps({'error':repr(exc),'storage_checks':receipt},indent=2)+'\n');raise
