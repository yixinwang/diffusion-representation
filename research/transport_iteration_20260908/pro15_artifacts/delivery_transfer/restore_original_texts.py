#!/usr/bin/env python3
"""Recover original text bytes; stdlib only. Never imports research source.

Run from any directory. Locates the package via this script's parent directory.
Checks both lossless bundles before writing, refuses symlinks/path traversal and
refuses replacement of differing files. Rerunning skips already-identical files.
"""
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
TRANSFER = ROOT / 'delivery_transfer'
MANIFEST = ('a9833ccfc16d68be73bac882a158413e2fecc195796ce985d01a7ccdcd20f76c', 90816)
SUMS = ('a03bc35e7d1c6d18b1e66fd0d1db1d686accc60bf830443567ba0657516ad8f7', 59933)
BUNDLES = {
    'priority_manifests': ('8fceccbd44bd35cb7f273e8f52540f79271d100a1da424a1057c82493738c29b', 19332, 2),
    'remaining_texts': ('5b8dcf46efe2a5fc57a8598fe8c727ce0218c761b59f3471b0cd27c58481c118', 56376, 101),
}


def check(data, sha, size, label):
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('Size or SHA256 mismatch: ' + label)


def safe_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name:
        raise ValueError('Unsafe or noncanonical path: ' + repr(name))
    target = ROOT.joinpath(*p.parts)
    for q in [target, *target.parents]:
        if q == ROOT.parent:
            break
        if q.is_symlink():
            raise ValueError('Symlink refused: ' + str(q))
    return target


def unpack(name, spec):
    chunks = []
    for part in spec['parts']:
        if '/' in part['file'] or '\\' in part['file']:
            raise ValueError('Unsafe part filename')
        p = safe_path('delivery_transfer/' + part['file'])
        data = p.read_bytes()
        check(data, part['sha256'], part['bytes'], str(p))
        gitsha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if gitsha != part['git_blob_sha']:
            raise ValueError('Git blob identity mismatch: ' + str(p))
        chunks.append(data)
    archive = b''.join(chunks)
    sha, size, count = BUNDLES[name]
    check(archive, sha, size, name)
    found = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:xz') as tf:
        for member in tf.getmembers():
            safe_path(member.name)
            if not member.isfile() or member.name in found:
                raise ValueError('Nonregular or duplicate member: ' + member.name)
            data = tf.extractfile(member).read()
            data.decode('utf-8', errors='strict')
            if b'\0' in data:
                raise ValueError('NUL-containing member: ' + member.name)
            found[member.name] = data
    if len(found) != count:
        raise ValueError('Unexpected member count: ' + name)
    return found


def main():
    index = json.loads((TRANSFER / 'RECOVERY_INDEX.json').read_bytes())
    recovered = unpack('priority_manifests', index['bundles']['priority_manifests'])
    if set(recovered) != {'MANIFEST.json', 'SHA256SUMS'}:
        raise ValueError('Unexpected priority members')
    check(recovered['MANIFEST.json'], *MANIFEST, 'MANIFEST.json')
    check(recovered['SHA256SUMS'], *SUMS, 'SHA256SUMS')
    manifest = json.loads(recovered['MANIFEST.json'])
    expected = {}
    for row in manifest['files']:
        safe_path(row['path'])
        if row['path'] in expected:
            raise ValueError('Duplicate original manifest entry')
        expected[row['path']] = (row['sha256'], row['bytes'])
    expected.update({'MANIFEST.json': MANIFEST, 'SHA256SUMS': SUMS})
    if len(expected) != 451:
        raise ValueError('Wrong original inventory count')
    sums = {}
    for line in recovered['SHA256SUMS'].decode('utf-8').splitlines():
        sha, name = line.split('  ', 1)
        if name in sums or name not in expected or expected[name][0] != sha:
            raise ValueError('Inconsistent original SHA256SUMS: ' + name)
        sums[name] = sha
    if len(sums) != 450 or set(sums) != set(expected) - {'SHA256SUMS'}:
        raise ValueError('Wrong original checksum inventory')
    rest = unpack('remaining_texts', index['bundles']['remaining_texts'])
    if set(rest) & set(recovered):
        raise ValueError('Duplicate bundle member')
    recovered.update(rest)
    binaries = {n: v for n, v in expected.items() if Path(n).suffix in {'.npy', '.npz'}}
    text = {n: v for n, v in expected.items() if n not in binaries}
    if len(text) != 266 or len(binaries) != 185 or len(recovered) != 103:
        raise ValueError('Wrong text/binary split')
    # Validate all existing/recovered originals before the first write.
    for name, (sha, size) in text.items():
        target = safe_path(name)
        if name in recovered:
            data = recovered[name]
            check(data, sha, size, name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Refusing to overwrite different original: ' + name)
        else:
            data = target.read_bytes()
            check(data, sha, size, name)
        data.decode('utf-8', errors='strict')
    if set(recovered) - set(text):
        raise ValueError('Bundle contains unknown or binary originals')
    written = 0
    for name, data in recovered.items():
        target = safe_path(name)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(data)
        written += 1
    for name, (sha, size) in text.items():
        check(safe_path(name).read_bytes(), sha, size, name)
    missing = [n for n in binaries if not safe_path(n).exists()]
    print(json.dumps({
        'status': 'all_original_texts_verified', 'original_text_files': len(text),
        'original_text_bytes': sum(v[1] for v in text.values()),
        'newly_extracted_files': written, 'packed_original_files': 103,
        'original_binaries_not_in_delivery': 185,
        'original_binary_bytes_not_in_delivery': sum(v[1] for v in binaries.values()),
        'binary_files_currently_missing': len(missing),
        'research_source_executed': False,
    }, indent=2))


if __name__ == '__main__':
    main()
