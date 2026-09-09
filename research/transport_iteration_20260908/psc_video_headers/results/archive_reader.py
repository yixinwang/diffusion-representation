"""Header-only plain-tar indexing and explicit train-member bounded I/O.

This is an application boundary, not a sandbox against hostile Python callers.
PAX/GNU extension records are deliberately unsupported pending archive review.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from dataclasses import asdict, dataclass

PINNED_ARCHIVE_SHA256 = 'e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb'
PINNED_ARCHIVE_SIZE = 171386880
UPSTREAM_REVISION = 'b9984b8d2a95e4a1879e1b071e9433858d0bc24a'
SPLITS = ('train', 'val', 'test')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identity(fd):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('archive must be a regular file')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def safe_parts(name):
    # Reject normalization ambiguities as well as traversal.
    if not name or name.startswith('/') or '\\' in name:
        raise ValueError('unsafe member path')
    parts = name.rstrip('/').split('/')
    if any(x in ('', '.', '..') for x in parts):
        raise ValueError('unsafe member path')
    return parts


def classify(name):
    parts = safe_parts(name)
    positions = [i for i, p in enumerate(parts) if p in SPLITS]
    if len(positions) != 1:
        raise ValueError('missing or ambiguous split')
    i = positions[0]
    tail = parts[i+1:]
    if tail and tail[0] == 'UCF101':
        tail = tail[1:]
    if len(tail) != 2:
        raise ValueError('unexpected UCF layout')
    action, filename = tail
    m = re.fullmatch(r'v_(.+)_g(\d{2})_c(\d{2})\.avi', filename)
    if m is None or m[1] != action:
        raise ValueError('class/filename mismatch')
    return parts[i], action, f'{action}/g{m[2]}', f'c{m[3]}'


@dataclass(frozen=True)
class Member:
    path: str
    type: str
    header_offset: int
    offset: int
    size: int
    padded_end: int
    split: str | None
    class_name: str | None
    group: str | None
    clip: str | None


@dataclass(frozen=True)
class Manifest:
    archive_identity: tuple
    header_sha256: str
    members: tuple[Member, ...]

    def payload(self):
        return {'schema': 1, 'archive_identity': list(self.archive_identity),
                'header_sha256': self.header_sha256,
                'members': [asdict(m) for m in self.members]}

    @property
    def sha256(self):
        return digest(self.payload())


def _index_fd(fd):
    before = identity(fd)
    total = before[2]
    if total < 1024 or total % 512:
        raise ValueError('invalid plain-tar size')
    pos = 0
    zeros = 0
    headers = hashlib.sha256()
    rows, names, clips, groups = [], set(), set(), {}
    while pos < total:
        raw = os.pread(fd, 512, pos)  # exactly header/footer blocks, never member bodies
        if len(raw) != 512:
            raise ValueError('truncated header')
        headers.update(pos.to_bytes(8, 'big') + raw)
        if raw == bytes(512):
            zeros += 1
            pos += 512
            continue
        if zeros:
            raise ValueError('nonzero record after tar terminator')
        try:
            t = tarfile.TarInfo.frombuf(raw, 'utf-8', 'strict')
        except (tarfile.HeaderError, UnicodeError) as e:
            raise ValueError('invalid tar header') from e
        safe_parts(t.name)
        if t.name.rstrip('/') in names:
            raise ValueError('duplicate member path')
        names.add(t.name.rstrip('/'))
        if t.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError('links/extensions/special members forbidden')
        if t.linkname or t.size < 0 or (t.isdir() and t.size != 0):
            raise ValueError('invalid member size/link')
        end = pos + 512 + ((t.size+511)//512)*512
        if end > total - 1024:
            raise ValueError('member outside archive or missing terminator')
        fields = (None,)*4 if t.isdir() else classify(t.name)
        if not t.isdir():
            split, _, group, clip = fields
            if (group, clip) in clips:
                raise ValueError('duplicate clip identity')
            clips.add((group, clip))
            if group in groups and groups[group] != split:
                raise ValueError('cross-split group overlap')
            groups[group] = split
        rows.append(Member(t.name, 'directory' if t.isdir() else 'regular', pos,
                           pos+512, t.size, end, *fields))
        pos = end
    if zeros < 2 or before != identity(fd):
        raise ValueError('missing terminator or changing archive')
    return Manifest(before, headers.hexdigest(), tuple(rows))


def _open(path):
    return os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))


def index_archive(path):
    fd = _open(path)
    try:
        return _index_fd(fd)
    finally:
        os.close(fd)


def save_manifest(manifest, path):
    with Path(path).open('x') as f:
        json.dump(manifest.payload(), f, sort_keys=True, indent=2)
        f.write('\n')


def load_manifest(path, archive, *, expected_manifest_sha256):
    data = json.loads(Path(path).read_text())
    if digest(data) != expected_manifest_sha256:
        raise ValueError('manifest digest mismatch')
    fresh = index_archive(archive)
    if fresh.payload() != data:
        raise ValueError('stale or mismatched manifest')
    return fresh


def check_pinned_metadata(manifest):
    """Check recorded size/counts only; does NOT establish whole-file SHA256."""
    if manifest.archive_identity[2] != PINNED_ARCHIVE_SIZE:
        raise ValueError('canonical archive size mismatch')
    regular = [m for m in manifest.members if m.type == 'regular']
    for s, n, g in zip(SPLITS, (300, 30, 75), (195, 25, 30)):
        rows = [m for m in regular if m.split == s]
        if len(rows) != n or len({m.group for m in rows}) != g:
            raise ValueError('canonical split/group counts mismatch')


class TrainMemberReader(io.RawIOBase):
    def __init__(self, archive, manifest, member_path, *, train_allowlist):
        super().__init__()
        self._fd = None
        allowed = frozenset(train_allowlist)
        known = {m.path: m for m in manifest.members if m.type == 'regular'}
        if not allowed or any(p not in known or known[p].split != 'train' for p in allowed):
            raise ValueError('allowlist must contain only explicit train members')
        if member_path not in allowed:
            raise PermissionError('member not explicitly train-allowlisted')
        fd = _open(archive)
        try:
            if _index_fd(fd) != manifest:
                raise ValueError('stale archive manifest')
            self._member = known[member_path]
            self._identity = manifest.archive_identity
            self._pos = 0
            self._fd = fd
        except BaseException:
            os.close(fd)
            raise

    def _check(self):
        self._checkClosed()
        if identity(self._fd) != self._identity:
            raise ValueError('archive modified since indexing')

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def tell(self):
        self._check()
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        self._check()
        if not isinstance(offset, int) or whence not in (0, 1, 2):
            raise ValueError('invalid seek')
        pos = offset + (0 if whence == 0 else self._pos if whence == 1 else self._member.size)
        if not 0 <= pos <= self._member.size:
            raise ValueError('seek outside selected member')
        self._pos = pos
        return pos

    def read(self, size=-1):
        self._check()
        if size is None:
            size = -1
        if not isinstance(size, int) or size < -1:
            raise ValueError('read size must be -1 or nonnegative integer')
        remaining = self._member.size-self._pos
        n = remaining if size == -1 else min(size, remaining)
        data = os.pread(self._fd, n, self._member.offset+self._pos)
        self._check()
        if len(data) != n:
            raise ValueError('truncated member')
        self._pos += n
        return data

    def readinto(self, buffer):
        view = memoryview(buffer).cast('B')
        data = self.read(len(view))
        view[:len(data)] = data
        return len(data)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        super().close()
