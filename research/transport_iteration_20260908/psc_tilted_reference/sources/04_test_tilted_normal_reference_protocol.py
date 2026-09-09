"""Fabricated report and failure checks; no reference experiment is executed."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'experiments/tilted_normal_reference/run.py'
spec=importlib.util.spec_from_file_location('tilted_reference_protocol_test',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_declared_grid_and_report_gates():
    assert runner.TIMING_SEEDS==(2026090901,2026090902,2026090903)
    assert len(runner.RECOVERY_CELLS)==32
    assert len({c[2] for c in runner.RECOVERY_CELLS})==32
    assert {(c[0],c[1]) for c in runner.RECOVERY_CELLS}=={(g,j) for g in (-.5,.5) for j in range(16)}
    report={'numerical':dict(full_source_roundtrip_max=1e-12,head_tail_roundtrip_max_on_abs50=1e-12,orthogonal_dictionary_error=1e-15,exact_copy_bitwise=True),
        'timing':{str(b):{'raw_seconds':{label:[.1]*9 for label in ('exact','fm_4','fm_8','fm_16','fm_32','fm_64')}} for b in (1,64)}}
    runner.validate_reference_report(report)
    report['timing']['64']['raw_seconds']['fm_64'].pop()
    with pytest.raises(AssertionError):runner.validate_reference_report(report)
    report['numerical']['full_source_roundtrip_max']=float('nan')
    with pytest.raises(FloatingPointError):runner.validate_reference_report(report)


def test_main_failure_preserves_completed_report_hashes_and_rejects_reuse(tmp_path,monkeypatch):
    out=tmp_path/'fresh'
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    fake=types.SimpleNamespace(NFES=(4,8,16,32,64),D=3072,COARSE=192,DETAIL=2880,
        orthogonal_dictionary=lambda:np.eye(2))
    monkeypatch.setattr(runner,'verified_reference',lambda commit,path:fake)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(name,'1')
    def fail(reference,path,state):
        runner.write_json(path/'completed_raw.json',{'fabricated':True})
        state['phase']='injected_failure';raise RuntimeError('deliberate protocol test')
    monkeypatch.setattr(runner,'execute',fail)
    monkeypatch.setattr(sys,'argv',['runner','--source-commit','fixture','--output',str(out)])
    with pytest.raises(RuntimeError):runner.main()
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['status']=='failed_partial'
    assert 'completed_raw.json' in complete['payload_sha256'] and 'failure.json' in complete['payload_sha256']
    assert 'deliberate protocol test' in json.loads((out/'failure.json').read_text())['traceback']
    before={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):runner.main()
    assert before=={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}


def test_wrong_commit_cannot_import_reference_or_simulate(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'full-real-commit\n')
    with pytest.raises(ValueError,match='commit'):
        runner.verified_reference('wrong',tmp_path)
    assert not list(tmp_path.iterdir())
