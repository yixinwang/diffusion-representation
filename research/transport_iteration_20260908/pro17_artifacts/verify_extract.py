#!/usr/bin/env python3
"""Verify all original archive parts and files before extracting into a NEW path.

Usage: python verify_extract.py --parts . --destination ./verified_pro17
No network requests, fit regeneration, symlinks, path traversal, or overwrites.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

MAX_TOTAL=64*1024*1024

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def safe_name(value: str) -> bool:
    p=PurePosixPath(value)
    return (not p.is_absolute() and '..' not in p.parts and '\\' not in value
            and len(p.parts)>1 and p.parts[0]=='pro17_artifacts')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parts',type=Path,default=Path('.'))
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args()
    if args.destination.exists():
        raise SystemExit('Destination already exists; refusing to overwrite anything.')
    m=json.loads((args.parts/'TRANSPORT_MANIFEST.json').read_text())
    if not 0<m['archive_size']<=MAX_TOTAL:
        raise ValueError('Archive size exceeds the declared bound')
    chunks=[]
    for part in m['parts']:
        if Path(part['path']).name!=part['path']:
            raise ValueError('Unsafe part name')
        raw=(args.parts/part['path']).read_bytes()
        if len(raw)!=part['size'] or sha(raw)!=part['sha256']:
            raise ValueError('Part integrity failure: '+part['path'])
        chunks.append(raw)
    archive=b''.join(chunks)
    if len(archive)!=m['archive_size'] or sha(archive)!=m['archive_sha256']:
        raise ValueError('Reassembled original archive hash mismatch')
    expected={x['path']:x for x in m['payload_files']}
    if len(expected)!=len(m['payload_files']) or any(not safe_name(x) for x in expected):
        raise ValueError('Duplicate or unsafe manifest paths')
    if sum(x['size'] for x in expected.values())>MAX_TOTAL:
        raise ValueError('Expanded payload exceeds bound')
    verified={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as tar:
        for member in tar:
            if not member.isfile() or not safe_name(member.name) or member.name not in expected:
                raise ValueError('Unexpected archive member: '+member.name)
            if member.name in verified:
                raise ValueError('Duplicate member: '+member.name)
            item=expected[member.name]
            if member.size!=item['size']:
                raise ValueError('Member size mismatch')
            source=tar.extractfile(member)
            if source is None:raise ValueError('Missing member data')
            raw=source.read(member.size+1)
            if len(raw)!=item['size'] or sha(raw)!=item['sha256']:
                raise ValueError('Original member hash mismatch: '+member.name)
            verified[member.name]=raw
    if set(verified)!=set(expected):
        raise ValueError('Incomplete original payload')
    args.destination.mkdir(parents=True,exist_ok=False)
    for name,raw in verified.items():
        target=args.destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
    print('Verified and extracted',len(verified),'unchanged original files.')
    print('Original archive SHA256:',m['archive_sha256'])

if __name__=='__main__':main()
