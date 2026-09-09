"""CPU protocol checks only; these do not execute or simulate GPU compilation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'experiments/dense_model_benchmark/run.py'
spec=importlib.util.spec_from_file_location('dense_model_protocol_test',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_gradient_tolerance_and_nonfinite_whole_case_rejection():
    ref=np.array([0.,2.,-3.],dtype=np.float32)
    assert runner.error_stats(ref+np.array([.00009,.001,.002]),ref,atol=1e-4,rtol=1e-3)['passed']
    assert not runner.error_stats(ref+np.array([.00011,0.,0.]),ref,atol=1e-4,rtol=1e-3)['passed']
    with pytest.raises(FloatingPointError):runner.error_stats(np.array([np.nan]),np.array([0.]),atol=1e-4)
    with pytest.raises(FloatingPointError):runner.error_stats(np.zeros(2),np.zeros(3),atol=1e-4)


def test_pinned_native_status_and_accessed_checkpoint_hashes(tmp_path,monkeypatch):
    names=('source_identity.json','shared_latest.pt','coupling_latest.pt','coupling/chunk_0000.npz')
    identity={'commit':runner.NATIVE_COMMIT,'sha256':{}}
    for name in runner.MODULES:
        if name not in ('dense_spline','dense_global_conditional_spline'):
            relative='qalt/src/qalt/'+name+'.py';identity['sha256'][relative]=runner.digest(runner.ROOT/relative)
    relative='qalt/experiments/observed_flow_pilot/run_shared.py';identity['sha256'][relative]=runner.digest(runner.ROOT/relative)
    (tmp_path/'source_identity.json').write_text(json.dumps(identity))
    for name in names[1:]:
        file=tmp_path/name;file.parent.mkdir(exist_ok=True);file.write_bytes(b'fabricated checkpoint or bank bytes')
    status={'status':'completed_development_only','all_fits_frozen':True,'payload_sha256':{name:runner.digest(tmp_path/name) for name in names}}
    (tmp_path/'status.json').write_text(json.dumps(status));monkeypatch.setattr(runner,'STATUS_SHA',runner.digest(tmp_path/'status.json'))
    assert runner.verify_native(tmp_path)['accessed_payload_sha256']==status['payload_sha256']
    (tmp_path/'coupling_latest.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='payload hash'):runner.verify_native(tmp_path)


def test_failure_preserves_outputs_and_declares_cache_exclusions(tmp_path,monkeypatch):
    out=tmp_path/'new'
    monkeypatch.setattr(runner.signal,'signal',lambda *a:None);monkeypatch.setattr(runner.signal,'alarm',lambda *a:None)
    def fail(args,state):
        np.save(args.output/'completed_outputs.npy',np.array([1.]))
        folder=args.output/'inductor-cache';folder.mkdir();(folder/'compiler.tmp').write_text('fixture')
        state.update(phase='pipeline_first_use',arm='dense_compiled');raise RuntimeError('deliberate first-use failure')
    monkeypatch.setattr(runner,'run',fail)
    monkeypatch.setattr(sys,'argv',['run','--expected-commit','fixture','--output',str(out)])
    with pytest.raises(RuntimeError):runner.main()
    report=json.loads((out/'COMPLETE.json').read_text())
    assert report['status']=='failed' and 'completed_outputs.npy' in report['payload_sha256']
    assert 'inductor-cache/compiler.tmp' not in report['payload_sha256']
    assert (out/'inductor-cache/compiler.tmp').exists() and (out/'FAILED.json').exists()
    before=(out/'COMPLETE.json').read_bytes()
    with pytest.raises(FileExistsError):runner.main()
    assert (out/'COMPLETE.json').read_bytes()==before
