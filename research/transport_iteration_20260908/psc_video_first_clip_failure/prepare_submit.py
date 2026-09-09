import subprocess,json,hashlib,time,traceback
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26'); source=base/'diffusion-validation-20260908'; checkout=base/'diffusion-video-decode-20260909'
rev='7b9703b64573748cbd2b3e6a832834c06baf417f';rel='research/transport_iteration_20260908/video_decode_prerequisite'
stage=base/'video-decode-launch';out=base/'diffusion-results/20260909-video-first-clip'
record={'commit':rev,'checkout':str(checkout),'output':str(out),'phase':'fetch'}
def run(args,timeout=300):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
try:
 print(json.dumps(record),flush=True)
 result=run(['git','-C',str(source),'fetch','origin','agent/observation-transport-audit-20260908']);print(result.stdout,result.stderr,flush=True)
 if checkout.exists():raise FileExistsError('new checkout already exists; inspect before mutation')
 result=run(['git','-C',str(source),'worktree','add','--detach',str(checkout),rev]);print(result.stdout,result.stderr,flush=True)
 if run(['git','-C',str(checkout),'rev-parse','HEAD']).stdout.strip()!=rev:raise ValueError('HEAD mismatch')
 if out.exists():raise FileExistsError('output already exists')
 stage.mkdir(exist_ok=False)
 hashes={}
 for name in ['launch_video_decode_guarded.py','video_decode_first_clip.slurm']:
  raw=(checkout/rel/name).read_bytes();frozen=subprocess.check_output(['git','-C',str(checkout),'show',rev+':'+rel+'/'+name],timeout=60)
  if raw!=frozen:raise ValueError('external file differs from frozen Git')
  (stage/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
 record.update(phase='ready_to_submit',external_sha256=hashes)
 (stage/'preparation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
 submission=run(['sbatch','--parsable','--job-name=video-first-clip','--output',str(stage/'slurm-%j.out'),'--error',str(stage/'slurm-%j.err'),str(stage/'video_decode_first_clip.slurm'),rev,str(checkout),str(base/'diffusion-results/20260909-video-headers/manifest.json'),str(out)],timeout=180)
 record.update(phase='submitted',submission_stdout=submission.stdout,submission_stderr=submission.stderr)
 (stage/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
except BaseException as exc:
 record.update(phase='failed_or_uncertain',error=repr(exc),traceback=traceback.format_exc());print(json.dumps(record),flush=True);raise
