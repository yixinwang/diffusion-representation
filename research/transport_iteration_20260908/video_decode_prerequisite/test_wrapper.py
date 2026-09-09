from fractions import Fraction
from types import SimpleNamespace as NS
import json
import pytest
import run_decode as run


def test_frozen_color_policy():
    assert run.color_policy('yuv420p',2,0)['assumption_used']
    assert not run.color_policy('yuv420p',6,1)['assumption_used']
    for args in [('rgb24',2,0),('yuv420p',1,1),('yuv420p',6,2),('yuv420p10le',2,0)]:
        with pytest.raises(ValueError):run.color_policy(*args)


def test_runner_guard_failure_preserves_status_without_dependencies(tmp_path):
    out=tmp_path/'out'
    args=['--output',str(out),'--manifest','unused','--reader-source','unused',
          '--reader-sha256','0'*64,'--helpers-sha256','0'*64,'--runner-sha256','0'*64,'--numpy-version','unused']
    assert run.main(args)==1
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='failed' and status['phase']=='guards' and status['first_pass_frames']==0
    with pytest.raises(FileExistsError):run.main(args)


def test_first_pass_preserves_prior_metadata_when_decode_fails(tmp_path):
    class Handle:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    stream=NS(index=0,codec_context=NS(name='fabricated',width=16,height=16),sample_aspect_ratio=None)
    class Container(Handle):
        streams=NS(video=[stream])
        def decode(self,s):
            yield NS(pts=0,time_base=Fraction(1,16),width=16,height=16,
                     colorspace=2,color_range=0,format=NS(name='yuv420p'))
            raise ValueError('fabricated decoder failure')
    reader=NS(TrainMemberReader=lambda *a,**k:Handle())
    av=NS(open=lambda *a,**k:Container())
    status={}
    with pytest.raises(ValueError,match='fabricated decoder failure'):
        run.first_pass(av,reader,None,tmp_path,status,lambda:None)
    row=json.loads((tmp_path/'frames.jsonl').read_text())
    assert row['index']==0 and row['policy']['assumption_used']
    assert row['color_primaries'] is None and status['first_pass_frames']==1
