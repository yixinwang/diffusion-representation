"""Guarded metadata-only audit. The only archive path is the pinned PSC path."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

ARCHIVE = Path('/ocean/projects/mth250006p/ywang26/datasets/ucf101-subset/UCF101_subset.tar.gz')
MODULE = Path(__file__).resolve().with_name('archive_reader.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary, path)


def guarded_source(module, *, expected_sha256=None, expected_commit=None, repo=None):
    """Return frozen bytes after explicit digest and/or full Git blob verification."""
    module = Path(module)
    if expected_sha256 is None and expected_commit is None:
        raise ValueError('expected module SHA256 or full Git commit required')
    raw = module.read_bytes()
    report = {'path': str(module.resolve()), 'sha256': sha(raw), 'bytes': len(raw)}
    if expected_sha256 is not None:
        if not re.fullmatch('[0-9a-f]{64}', expected_sha256) or sha(raw) != expected_sha256:
            raise ValueError('module SHA256 mismatch')
    if expected_commit is not None:
        if repo is None or not re.fullmatch('[0-9a-f]{40}', expected_commit):
            raise ValueError('Git verification requires repo and full 40-character commit')
        repo = Path(repo).resolve()
        relative = module.resolve().relative_to(repo).as_posix()
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE,timeout=20)
        if git('rev-parse','HEAD').decode().strip()!=expected_commit:
            raise ValueError('Git HEAD mismatch')
        frozen = git('show',f'{expected_commit}:{relative}')
        if frozen != raw:
            raise ValueError('module differs from frozen Git blob')
        report.update(git_commit=expected_commit, git_relative_path=relative,
                      git_blob=git('rev-parse',f'{expected_commit}:{relative}').decode().strip())
    return raw, report


def load_snapshot(path):
    """Import exactly the newly preserved source, never a mutable search-path copy."""
    spec = importlib.util.spec_from_file_location('video_header_frozen_reader',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=Path(path).resolve():
        raise ValueError('imported module origin mismatch')
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--expected-module-sha256')
    parser.add_argument('--expected-commit')
    parser.add_argument('--repo',type=Path)
    args=parser.parse_args(argv)
    # Exclusive directory acquisition precedes any archive access. Existing output
    # is never reopened even to record failure.
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status={'status':'running','phase':'source_guard','archive':str(ARCHIVE),
      'environment':{'python':sys.version,'platform':platform.platform()},
      'runner_sha256':sha(Path(__file__).read_bytes()),'payload_accessed':False,
      'scope':'header/footer metadata only; no member reader, extraction, or decoder'}
    atomic_json(args.output/'status.json',status)
    try:
        raw, source=guarded_source(MODULE,expected_sha256=args.expected_module_sha256,
                                   expected_commit=args.expected_commit,repo=args.repo)
        snapshot=args.output/'archive_reader.py'
        with snapshot.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        status.update(source=source,phase='header_index')
        atomic_json(args.output/'status.json',status)
        module=load_snapshot(snapshot)
        manifest=module.index_archive(ARCHIVE)
        # Retain actual metadata even when canonical counts reject the archive.
        module.save_manifest(manifest,args.output/'manifest.json')
        manifest_bytes=(args.output/'manifest.json').read_bytes()
        status.update(phase='canonical_metadata_check',manifest_sha256=manifest.sha256,
                      manifest_file_sha256=sha(manifest_bytes),members=len(manifest.members))
        atomic_json(args.output/'status.json',status)
        module.check_pinned_metadata(manifest)
        status.update(status='complete',phase='complete',
                      archive_sha256_recorded=module.PINNED_ARCHIVE_SHA256,
                      archive_sha256_recomputed=False,
                      upstream_revision=module.UPSTREAM_REVISION)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        status['elapsed_seconds']=time.monotonic()-started
        atomic_json(args.output/'status.json',status)
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
