from fractions import Fraction
from types import SimpleNamespace as NS
import json
import diagnostic_metadata as diagnostic


def test_all_rows_preserved_including_offending_and_null(tmp_path):
    class Handle:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    stream=NS(index=0,codec_context=NS(name='fabricated',width=16,height=16),sample_aspect_ratio=None)
    class Container(Handle):
        streams=NS(video=[stream])
        def decode(self,stream):
            for p in [1,2,4,3,None,5]:
                yield NS(pts=p,dts=None,time_base=Fraction(1,25),width=16,height=16,format=NS(name='yuv420p'),colorspace=2,color_range=0)
    status={};reader=NS(TrainMemberReader=lambda *a,**k:Handle());av=NS(open=lambda *a,**k:Container())
    count,flags=diagnostic.diagnostic_rows(av,reader,None,tmp_path,status,lambda:None)
    rows=[json.loads(r) for r in (tmp_path/'frames.jsonl').read_text().splitlines()]
    assert count==6 and [r['pts'] for r in rows]==[1,2,4,3,None,5]
    assert rows[3]['previous_valid_time']==[4,25]
    assert [r['index'] for r in flags]==[3,4]
    assert rows[4]['timestamp_flags']==['null_pts']
    assert not list(tmp_path.glob('*.npy'))
