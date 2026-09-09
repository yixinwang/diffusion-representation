from fractions import Fraction
import numpy as np
import pytest
from helpers import select_frames,frame_timestamp,resize_geometry,resize_rgb,dequantize


def records(points):
    return [{'index':i,'pts':p,'time_base':Fraction(1,16)} for i,p in enumerate(points)]


def test_exact_timestamp_selection_and_earlier_ties():
    selected=select_frames(records(range(17)))
    assert [r['index'] for r in selected]==[1,3,5,7,9,11,13,15]
    # Target half-tick ties select earlier frame without float comparison.
    selected=select_frames(records(range(18)))
    assert [r['index'] for r in selected]==[1,3,5,7,9,11,13,15]
    assert selected[0]['requested_time']==[3,32]
    assert selected[0]['actual_time']==[1,16]


def test_bad_timestamp_short_window_and_duplicate_selection():
    for pts in [range(8),[0,1,2,3,3,8,10,16],[0,1,2,3,2,8,10,16],[0,1,2,3,4,5,6,100]]:
        with pytest.raises(ValueError):select_frames(records(pts))
    for pts,tb in [(None,Fraction(1,10)),(1,.1),(1,Fraction(-1,10)),(True,Fraction(1,10))]:
        with pytest.raises(ValueError):frame_timestamp(pts,tb)


def test_geometry_and_guarded_fabricated_resize():
    assert resize_geometry(128,256)=={'resized_height':64,'resized_width':128,'crop_top':0,'crop_left':32,'target':64}
    assert resize_geometry(128,129)['resized_width']==65 # exact positive half-up
    import PIL
    x=np.full((16,32,3),137,np.uint8)
    with pytest.raises(ValueError):resize_rgb(x,expected_pillow_version='wrong')
    # Local implementation smoke explicitly records its actual version; it is
    # not a claim that the required PSC Pillow12.1 runtime was tested here.
    y,meta=resize_rgb(x,expected_pillow_version=PIL.__version__)
    assert y.shape==(64,64,3) and np.all(y==137) and meta['pillow_version']==PIL.__version__


def test_dequantization_extreme_pixels_determinism_and_jacobian():
    x=np.array([[[0,255,128],[255,0,128]]],dtype=np.uint8)
    a=dequantize(x,frame_index=7);b=dequantize(x,frame_index=7);c=dequantize(x,frame_index=8)
    np.testing.assert_array_equal(a['uint32_noise'],b['uint32_noise'])
    assert not np.array_equal(a['uint32_noise'],c['uint32_noise'])
    assert np.all((a['unit_float64']>0)&(a['unit_float64']<1))
    np.testing.assert_array_equal(a['logit_float64'].astype(np.float32),a['logit_float32'])
    p=a['unit_float64'];expected=np.sum(np.log(1/(256*p*(1-p))))
    np.testing.assert_allclose(a['outer_logdet_float64'],expected,atol=2e-14)
    with pytest.raises(ValueError):dequantize(x.astype(float),frame_index=7)
