"""Fabricated-test helpers for one prospective training clip; no video I/O."""
from fractions import Fraction
import hashlib
import json
import numbers
import numpy as np

MEMBER='UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c03.avi'
MANIFEST_SHA256='180eabace318325e1b7ee6f2d5465b0e89f86789bcf6e0d4f3a92d3cc88a6f96'
ROLE='first-train-clip-feasibility-v1'


def rational(value):
    value=Fraction(value)
    return [value.numerator,value.denominator]


def frame_timestamp(pts,time_base):
    if not isinstance(pts,numbers.Integral) or isinstance(pts,bool):
        raise ValueError('integer PTS required')
    if not isinstance(time_base,Fraction) or time_base<=0:
        raise ValueError('positive exact Fraction time base required')
    return int(pts)*time_base


def select_frames(records):
    """Presentation-order records with index, integer pts and Fraction time_base."""
    if len(records)<8:raise ValueError('at least eight observed frames required')
    times=[]
    for i,r in enumerate(records):
        if r['index']!=i:raise ValueError('consecutive presentation indices required')
        t=frame_timestamp(r['pts'],r['time_base'])
        if times and t<=times[-1]:raise ValueError('strictly increasing timestamps required')
        times.append(t)
    center=(times[0]+times[-1])/2
    targets=[center+Fraction(2*k-7,16) for k in range(8)]
    if targets[0]<times[0] or targets[-1]>times[-1]:raise ValueError('requested window outside source')
    # Exact rational comparisons; earlier index resolves equal-distance ties.
    chosen=[min(range(len(times)),key=lambda i:(abs(times[i]-t),times[i])) for t in targets]
    if len(set(chosen))!=8:raise ValueError('selection repeats a frame; no replacement')
    return [{'index':i,'pts':int(records[i]['pts']),'time_base':rational(records[i]['time_base']),
             'requested_time':rational(t),'actual_time':rational(times[i])} for i,t in zip(chosen,targets)]


def resize_geometry(height,width,target=64):
    if any(type(v) is not int or v<=0 for v in (height,width,target)):
        raise ValueError('positive integer dimensions required')
    short=min(height,width)
    # floor(a/b+1/2), implemented without binary float rounding.
    nh=(2*height*target+short)//(2*short)
    nw=(2*width*target+short)//(2*short)
    top=(nh-target)//2;left=(nw-target)//2
    return {'resized_height':nh,'resized_width':nw,'crop_top':top,'crop_left':left,'target':target}


def validate_rgb(rgb):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('nonempty H by W by 3 RGB uint8 required')
    return rgb


def resize_rgb(rgb, *, expected_pillow_version='12.1.0'):
    """Pillow RGB uint8 bilinear short-edge resize and centered crop, no augmentation."""
    import PIL
    from PIL import Image
    if PIL.__version__!=expected_pillow_version:raise ValueError('Pillow version mismatch')
    rgb=validate_rgb(rgb);g=resize_geometry(int(rgb.shape[0]),int(rgb.shape[1]))
    image=Image.fromarray(rgb).resize((g['resized_width'],g['resized_height']),resample=Image.Resampling.BILINEAR,reducing_gap=None)
    image=image.crop((g['crop_left'],g['crop_top'],g['crop_left']+64,g['crop_top']+64))
    result=np.asarray(image,dtype=np.uint8).copy()
    return result,dict(g,pillow_version=PIL.__version__,mode='RGB',resample='BILINEAR',reducing_gap=None)


def dequantize(rgb, *, record=MEMBER, frame_index, role=ROLE):
    rgb=validate_rgb(rgb)
    if type(frame_index) is not int or frame_index<0:raise ValueError('nonnegative frame index required')
    key={'schema':1,'record':record,'frame_index':frame_index,'role':role,
         'algorithm':'SHA256-big-endian-seed-PCG64-uint32-midpoint-v1'}
    encoded=json.dumps(key,sort_keys=True,separators=(',',':')).encode()
    seed_bytes=hashlib.sha256(encoded).digest()
    rng=np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes,'big')))
    bits=rng.integers(0,2**32,size=rgb.shape,dtype=np.uint32)
    noise=(bits.astype(np.float64)+.5)/2**32
    cube=(rgb.astype(np.float64)+noise)/256
    if not np.all((cube>0)&(cube<1)):raise ArithmeticError('noninterior dequantized endpoint')
    logit=np.log(cube)-np.log1p(-cube)
    cube_to_logit_ld=np.sum(-np.log(cube)-np.log1p(-cube),dtype=np.float64)
    outer_ld=cube_to_logit_ld-rgb.size*np.log(256.)
    logit32=logit.astype(np.float32)
    if not (np.isfinite(logit).all() and np.isfinite(logit32).all() and np.isfinite(outer_ld)):
        raise ArithmeticError('nonfinite transformed endpoint')
    return {'uint32_noise':bits,'unit_float64':cube,'logit_float64':logit,'logit_float32':logit32,
            'outer_logdet_float64':float(outer_ld),'cube_to_logit_logdet_float64':float(cube_to_logit_ld),
            'key':key,'seed_sha256':seed_bytes.hex(),'numpy_version':np.__version__}
