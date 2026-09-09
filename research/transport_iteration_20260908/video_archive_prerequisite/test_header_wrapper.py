from pathlib import Path
import hashlib
import json
import subprocess
import pytest
import run_header_audit as run


def test_source_guards_git_blob_and_sha(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    module=tmp_path/'reader.py';module.write_text('x=1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','reader.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD']).decode().strip()
    raw=module.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert run.guarded_source(module,expected_sha256=digest,expected_commit=commit,repo=tmp_path)[0]==raw
    for kw in [{},{'expected_sha256':'0'*64},{'expected_commit':commit[:7],'repo':tmp_path}]:
        with pytest.raises(ValueError):run.guarded_source(module,**kw)
    module.write_text('x=2\n')
    with pytest.raises(ValueError):run.guarded_source(module,expected_commit=commit,repo=tmp_path)


def test_guard_failure_saved_no_archive_and_no_overwrite(tmp_path):
    out=tmp_path/'out'
    assert run.main(['--output',str(out),'--expected-module-sha256','0'*64])==1
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='failed' and status['phase']=='source_guard'
    assert status['payload_accessed'] is False and 'traceback' in status
    assert not (out/'manifest.json').exists()
    before=(out/'status.json').read_bytes()
    with pytest.raises(FileExistsError):run.main(['--output',str(out)])
    assert (out/'status.json').read_bytes()==before


def test_no_archive_override_and_snapshot_origin(tmp_path):
    with pytest.raises(SystemExit):run.main(['--output',str(tmp_path/'out'),'--archive','fixture.tar'])
    snapshot=tmp_path/'reader.py';snapshot.write_text('VALUE=7\n')
    mod=run.load_snapshot(snapshot)
    assert mod.VALUE==7 and Path(mod.__file__)==snapshot
    path=tmp_path/'status.json';run.atomic_json(path,{'status':'running'});run.atomic_json(path,{'status':'complete'})
    assert json.loads(path.read_text())=={'status':'complete'}
    assert not path.with_name('status.json.tmp').exists()
