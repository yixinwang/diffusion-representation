from fractions import Fraction
from types import SimpleNamespace as NS
import pytest
import run_index_decode as run


def test_nearest_indices_endpoints_ties_earlier():
    assert [x['index'] for x in run.select_indices(216)]==[0,31,61,92,123,154,184,215]
    assert [x['index'] for x in run.select_indices(8)]==list(range(8))
    # Denominator7 means no half ties for this exact8-target rule; independently
    # verify nearest index with earlier tie key for many valid sequence sizes.
    for n in range(8,30):
        for row in run.select_indices(n):
            t=Fraction(*row['requested_index'])
            assert row['index']==min(range(n),key=lambda i:(abs(i-t),i))
    with pytest.raises(ValueError):run.select_indices(7)


def test_metadata_keeps_swapped_pts_and_null_dts_without_repair():
    frame=NS(pts=3,dts=None,time_base=Fraction(1,25),width=320,height=240,format=NS(name='yuv420p'),colorspace=2,color_range=0,rotation=0)
    expected=run.exposed_metadata(frame,3);expected['timestamp_flags']=['nonincreasing_vs_previous_valid_frame']
    assert run.compare_frame(frame,3,expected)['pts']==3
    assert run.compare_frame(frame,3,expected)['dts'] is None
    frame.pts=4
    with pytest.raises(ValueError):run.compare_frame(frame,3,expected)


def test_one_pass_full_coverage_and_late_mismatch_preservation(tmp_path,monkeypatch):
    class Handle:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    frames=[NS(pts=i+1,dts=i+1,time_base=Fraction(1,25),width=320,height=240,format=NS(name='yuv420p'),colorspace=2,color_range=0,rotation=0) for i in range(9)]
    timeline=[run.exposed_metadata(f,i) for i,f in enumerate(frames)]
    codec=NS(name='mpeg4',width=320,height=240,color_primaries=2,color_trc=2)
    stream=NS(index=0,codec_context=codec,sample_aspect_ratio=1)
    class Container(Handle):
        streams=NS(video=[stream])
        def decode(self,s):yield from frames
    av=NS(open=lambda *a,**k:Container());reader=NS(TrainMemberReader=lambda *a,**k:Handle())
    calls=[]
    monkeypatch.setattr(run,'convert_selected',lambda av,np,h,f,i,*rest:calls.append(i))
    selected=run.select_indices(9);status={}
    assert run.decode_indices(av,None,None,reader,None,timeline,selected,tmp_path,status,lambda:None)==9
    assert calls==[x['index'] for x in selected]
    bad=tmp_path/'bad';bad.mkdir();frames[-1].pts=99
    with pytest.raises(ValueError):run.decode_indices(av,None,None,reader,None,timeline,selected,bad,{},lambda:None)
    assert (bad/'metadata_mismatch.json').exists()
