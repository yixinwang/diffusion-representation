import io
import json
import os
import tarfile
import pytest
import archive_reader as ar

TRAIN='UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c03.avi'
VAL='UCF101_subset/val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
TEST='UCF101_subset/test/ApplyEyeMakeup/v_ApplyEyeMakeup_g99_c01.avi'


def make(path, entries=None):
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as t:
        for name, body, kind in entries or [(TRAIN,b'abcdef',tarfile.REGTYPE),(VAL,b'VAL-SEALED',tarfile.REGTYPE),(TEST,b'TEST-SEALED',tarfile.REGTYPE)]:
            info=tarfile.TarInfo(name);info.type=kind;info.size=len(body)
            if kind == tarfile.SYMTYPE:info.linkname=TRAIN
            t.addfile(info,io.BytesIO(body))
    return path


def test_header_only_reads_and_offsets(tmp_path,monkeypatch):
    p=make(tmp_path/'a.tar'); calls=[]; real=ar.os.pread
    def observed(fd,n,off):calls.append((n,off));return real(fd,n,off)
    monkeypatch.setattr(ar.os,'pread',observed)
    m=ar.index_archive(p)
    assert [x.split for x in m.members]==['train','val','test']
    assert all(n==512 for n,_ in calls)
    for member in m.members:
        assert all(not (member.offset<=off<member.padded_end) for _,off in calls)
    assert m.members[0].size==6


def test_confinement_seek_negative_and_buffered(tmp_path):
    p=make(tmp_path/'a.tar');m=ar.index_archive(p)
    with ar.TrainMemberReader(p,m,TRAIN,train_allowlist=[TRAIN]) as r:
        assert r.read(2)==b'ab'
        assert r.seek(-1,1)==1
        assert r.read(10**9)==b'bcdef'
        assert r.read()==b''
        assert r.seek(-2,2)==4 and r.read(-1)==b'ef'
        for off,whence in [(-1,0),(7,0),(1,2),(-7,2),(0,3)]:
            with pytest.raises(ValueError):r.seek(off,whence)
        with pytest.raises(ValueError):r.read(-2)
        r.seek(0); b=bytearray(20);assert r.readinto(b)==6 and b[:6]==b'abcdef'
    with pytest.raises(ValueError):r.read()
    with io.BufferedReader(ar.TrainMemberReader(p,m,TRAIN,train_allowlist=[TRAIN])) as r:
        assert r.read()==b'abcdef'


@pytest.mark.parametrize('name',[VAL,TEST,'unlisted'])
def test_no_protected_open(tmp_path,name):
    p=make(tmp_path/'a.tar');m=ar.index_archive(p)
    with pytest.raises(PermissionError):ar.TrainMemberReader(p,m,name,train_allowlist=[TRAIN])
    with pytest.raises(ValueError):ar.TrainMemberReader(p,m,TRAIN,train_allowlist=[TRAIN,VAL])


@pytest.mark.parametrize('name',['../'+TRAIN,'/'+TRAIN,TRAIN.replace('/train/','/../train/'),TRAIN.replace('/train/','//train/'),TRAIN.replace('/train/','/./train/')])
def test_traversal(tmp_path,name):
    with pytest.raises(ValueError):ar.index_archive(make(tmp_path/'a.tar',[(name,b'x',tarfile.REGTYPE)]))


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.FIFOTYPE,tarfile.XHDTYPE])
def test_nonregular(tmp_path,kind):
    with pytest.raises(ValueError):ar.index_archive(make(tmp_path/'a.tar',[(TRAIN,b'',kind)]))


def test_duplicates_overlap_and_truncation(tmp_path):
    for entries in [[(TRAIN,b'x',tarfile.REGTYPE)]*2,
                    [(TRAIN,b'x',tarfile.REGTYPE),(TRAIN.replace('train','val'),b'x',tarfile.REGTYPE)],
                    [(TRAIN,b'x',tarfile.REGTYPE),(TRAIN.replace('train','val').replace('c03','c04'),b'x',tarfile.REGTYPE)]]:
        with pytest.raises(ValueError):ar.index_archive(make(tmp_path/'a.tar',entries))
    p=make(tmp_path/'a.tar');raw=p.read_bytes();p.write_bytes(raw[:1024])
    with pytest.raises(ValueError):ar.index_archive(p)
    t=tarfile.TarInfo(TRAIN);t.size=10**6;p.write_bytes(t.tobuf()+bytes(1536))
    with pytest.raises(ValueError):ar.index_archive(p)


def test_cache_tamper_and_staleness(tmp_path):
    p=make(tmp_path/'a.tar');m=ar.index_archive(p);cache=tmp_path/'m.json';ar.save_manifest(m,cache)
    assert ar.load_manifest(cache,p,expected_manifest_sha256=m.sha256)==m
    with pytest.raises(FileExistsError):ar.save_manifest(m,cache)
    data=json.loads(cache.read_text());data['members'][0]['offset']+=512;cache.write_text(json.dumps(data))
    with pytest.raises(ValueError):ar.load_manifest(cache,p,expected_manifest_sha256=m.sha256)
    with ar.TrainMemberReader(p,m,TRAIN,train_allowlist=[TRAIN]) as r:
        with p.open('r+b') as f:f.seek(m.members[0].offset);f.write(b'Z')
        with pytest.raises(ValueError):r.read()
    with pytest.raises(ValueError):ar.TrainMemberReader(p,m,TRAIN,train_allowlist=[TRAIN])


def test_symlink_archive_and_header_checksum(tmp_path):
    p=make(tmp_path/'a.tar');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(OSError):ar.index_archive(link)
    raw=bytearray(p.read_bytes());raw[0]^=1;p.write_bytes(raw)
    with pytest.raises(ValueError):ar.index_archive(p)
