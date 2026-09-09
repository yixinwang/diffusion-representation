"""Verify and extract the complete Pro16 package; no network or code execution."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='new output directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    delivery = json.loads((root/'DELIVERY.json').read_text())
    parts = []
    for item in delivery['parts']:
        data = (root/'archive_parts'/item['path']).read_bytes()
        if len(data) != item['bytes'] or sha256(data) != item['sha256']:
            raise ValueError('archive part mismatch: '+item['path'])
        parts.append(data)
    archive = b''.join(parts)
    if len(archive) != delivery['archive_bytes'] or sha256(archive) != delivery['archive_sha256']:
        raise ValueError('complete archive mismatch')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as tar:
        for member in tar.getmembers():
            p = PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or p.parts[0] != 'pro16_artifacts':
                raise ValueError('unsafe archive member: '+member.name)
            if member.name in files:
                raise ValueError('duplicate archive member')
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('missing archive member data')
            files[member.name] = stream.read()
    manifest_name = 'pro16_artifacts/MANIFEST.json'
    if sha256(files[manifest_name]) != delivery['manifest_sha256']:
        raise ValueError('manifest mismatch')
    manifest = json.loads(files[manifest_name])
    expected = {manifest_name}
    for item in manifest['files']:
        name = 'pro16_artifacts/'+item['path']
        data = files[name]
        if len(data) != item['bytes'] or sha256(data) != item['sha256']:
            raise ValueError('payload mismatch: '+name)
        expected.add(name)
    if set(files) != expected or len(files) != delivery['payload_files']:
        raise ValueError('payload inventory mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = args.out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f'Verified {len(files)} files; extracted to {args.out.resolve()}')


if __name__ == '__main__':
    main()
