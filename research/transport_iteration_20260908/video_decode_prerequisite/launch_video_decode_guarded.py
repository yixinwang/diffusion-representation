"""External PSC source/dependency guard. No remote connection/submission logic."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback

PYTHON='/ocean/projects/mth250006p/ywang26/pytorch/bin/python'
DEPS=Path('/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909/dependencies-attempt3')
PREFIX='research/transport_iteration_20260908/'
SOURCES=[PREFIX+'video_decode_prerequisite/run_decode.py',PREFIX+'video_decode_prerequisite/helpers.py',PREFIX+'video_archive_prerequisite/archive_reader.py']
DEP_MANIFEST=PREFIX+'video_dependency/extraction-attempt3.json'


def sha(x):return hashlib.sha256(x).hexdigest()
def atomic(path,data):
    temp=path.with_suffix('.tmp')
    with temp.open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True);p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    status={'status':'guarding','expected_commit':a.commit,'checkout':str(a.checkout),'started':time.time()}
    atomic(a.output/'launcher_status.json',status)
    try:
        if not re.fullmatch('[0-9a-f]{40}',a.commit):raise ValueError('full frozen commit required')
        def git(*v):return subprocess.check_output(['git','-C',str(a.checkout),*v],stderr=subprocess.PIPE,timeout=30)
        if git('rev-parse','HEAD').decode().strip()!=a.commit:raise ValueError('HEAD mismatch')
        if git('status','--porcelain').strip():raise ValueError('checkout must be clean and frozen')
        source={}
        snapshots=a.output/'sources';snapshots.mkdir()
        for path in SOURCES+[DEP_MANIFEST]:
            frozen=git('show',f'{a.commit}:{path}');local=(a.checkout/path).read_bytes()
            if frozen!=local:raise ValueError('Git source byte mismatch: '+path)
            source[path]=sha(local);(snapshots/Path(path).name).write_bytes(local)
        dep=json.loads((a.checkout/DEP_MANIFEST).read_text())
        if len(dep['files'])!=264:raise ValueError('expected exactly264 frozen dependency files')
        verified=[]
        for name,record in dep['files'].items():
            rel=Path(name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe dependency manifest')
            path=DEPS/rel
            if path.is_symlink() or not path.is_file():raise ValueError('dependency must be regular file')
            raw=path.read_bytes()
            if len(raw)!=record['bytes'] or sha(raw)!=record['sha256']:raise ValueError('dependency byte mismatch: '+name)
            verified.append(name)
        # Historical __pycache__ files are bypassed by a new isolated pycache
        # prefix. No unmanifested modules/shared libraries are permitted.
        extras=[str(f.relative_to(DEPS)) for f in DEPS.rglob('*') if f.is_file() and str(f.relative_to(DEPS)) not in dep['files'] and '__pycache__' not in f.parts]
        if extras:raise ValueError('unmanifested dependency files: '+repr(extras))
        status.update(status='launching_once',source_sha256=source,dependency_files_verified=len(verified),dependency_root=str(DEPS))
        atomic(a.output/'launcher_status.json',status)
        env=os.environ.copy();env.update(PYTHONPATH=str(DEPS),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(a.output/'fresh_pycache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[PYTHON,str(a.checkout/SOURCES[0]),'--output',str(a.output/'decode'),'--manifest',str(a.manifest),
                 '--reader-source',str(a.checkout/SOURCES[2]),'--reader-sha256',source[SOURCES[2]],
                 '--helpers-sha256',source[SOURCES[1]],'--runner-sha256',source[SOURCES[0]],'--numpy-version','2.2.6']
        status['command']=command;atomic(a.output/'launcher_status.json',status)
        with (a.output/'stdout.txt').open('xb') as out,(a.output/'stderr.txt').open('xb') as err:
            run=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=900)
        status.update(status='complete' if run.returncode==0 else 'decode_failed',returncode=run.returncode)
    except BaseException as exc:status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        status['finished']=time.time();atomic(a.output/'launcher_status.json',status)
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
