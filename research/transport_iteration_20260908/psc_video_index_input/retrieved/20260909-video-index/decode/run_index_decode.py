"""Distinct normalized decoder-index input endpoint, one authenticated decode pass."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from fractions import Fraction

ARCHIVE=Path('/ocean/projects/mth250006p/ywang26/datasets/ucf101-subset/UCF101_subset.tar.gz')
MEMBER='UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c03.avi'
MANIFEST_SHA='180eabace318325e1b7ee6f2d5465b0e89f86789bcf6e0d4f3a92d3cc88a6f96'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()

def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def snapshot_import(path, expected, output, name):
    raw=path.read_bytes()
    if digest(raw)!=expected:raise ValueError('source digest mismatch: '+str(path))
    target=output/(name+'.py')
    with target.open('xb') as f:f.write(raw)
    spec=importlib.util.spec_from_file_location(name,target)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def stream_metadata(container):
    if len(container.streams.video)!=1:raise ValueError('exactly one video stream required')
    s=container.streams.video[0];s.thread_type='NONE';s.codec_context.thread_count=1
    return s,{'index':s.index,'codec':s.codec_context.name,'width':s.codec_context.width,'height':s.codec_context.height,
        'color_primaries':getattr(s.codec_context,'color_primaries',None),
        'color_trc':getattr(s.codec_context,'color_trc',None),
        'sample_aspect_ratio':str(s.sample_aspect_ratio),
        'aspect_policy':'coded raster dimensions; no display-aspect resampling'}


def color_policy(pixel_format, colorspace, color_range):
    # FFmpeg AVColorSpace values: RGB0, BT7091, unspecified2, FCC4,
    # BT470BG5, SMPTE170M6. AVColorRange: unspecified0, MPEG1, JPEG2.
    if pixel_format not in ('yuv420p','yuv422p','yuv444p'):
        raise ValueError('unsupported pixel format; no implicit conversion')
    if colorspace not in (2,5,6) or color_range not in (0,1):
        raise ValueError('unsupported explicit color matrix/range')
    return {'matrix':'BT.601','range':'limited','assumption_used':colorspace==2 or color_range==0,
            'source_colorspace':colorspace,'source_range':color_range}


def select_indices(count):
    if type(count) is not int or count<8:raise ValueError('at least eight decoded frames required')
    result=[]
    for k in range(8):
        target=Fraction(k*(count-1),7);floor,remainder=divmod(target.numerator,target.denominator)
        index=floor+int(2*remainder>target.denominator)
        result.append({'k':k,'index':index,'normalized_position':[k,7],
                       'requested_index':[target.numerator,target.denominator]})
    if len({x['index'] for x in result})!=8:raise ValueError('selected indices must be distinct')
    return result


def exposed_metadata(frame,index):
    tb=frame.time_base
    return {'index':index,'pts':frame.pts,'dts':frame.dts,
            'time_base':None if tb is None else [int(tb.numerator),int(tb.denominator)],
            'width':frame.width,'height':frame.height,'format':frame.format.name,
            'colorspace':int(frame.colorspace),'color_range':int(frame.color_range),
            'color_primaries':getattr(frame,'color_primaries',None),
            'color_trc':getattr(frame,'color_trc',None),
            'interlaced':bool(getattr(frame,'interlaced_frame',False)),
            'rotation':getattr(frame,'rotation',None)}


def compare_frame(frame,index,expected):
    actual=exposed_metadata(frame,index)
    # Flag annotations are diagnostic calculations, not decoder attributes.
    wanted={k:expected[k] for k in actual}
    if actual!=wanted:raise ValueError('decoded metadata differs from authenticated ledger at index '+str(index))
    return actual


def convert_selected(av,np,helper,frame,index,selection,metadata,out):
    if metadata['interlaced'] or metadata['rotation']!=0:raise ValueError('unsupported interlacing/rotation')
    policy=color_policy(metadata['format'],metadata['colorspace'],metadata['color_range'])
    from av.video.reformatter import VideoReformatter,Colorspace,ColorRange
    rgb=VideoReformatter().reformat(frame,format='rgb24',src_colorspace=Colorspace.ITU601,
        dst_colorspace=Colorspace.ITU601,src_color_range=ColorRange.MPEG,dst_color_range=ColorRange.JPEG).to_ndarray(format='rgb24')
    np.save(out/f'frame_{index:06d}_original_rgb.npy',rgb,allow_pickle=False)
    processed,geometry=helper.resize_rgb(rgb)
    np.save(out/f'frame_{index:06d}_processed_rgb.npy',processed,allow_pickle=False)
    values=helper.dequantize(processed,frame_index=index)
    for key in ('uint32_noise','unit_float64','logit_float64','logit_float32'):
        np.save(out/f'frame_{index:06d}_{key}.npy',values.pop(key),allow_pickle=False)
    atomic(out/f'frame_{index:06d}.json',{'selection':selection,'exposed_metadata':metadata,
          'color_policy':policy,'geometry':geometry,'dequantization':values})


def decode_indices(av,np,helper,reader,manifest,timeline,selection,out,status,checkpoint):
    wanted={x['index']:x for x in selection};seen=0;saved=[]
    with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
        with av.open(handle,mode='r',format='avi') as container:
            stream,info=stream_metadata(container);atomic(out/'stream.json',info)
            expected_stream={'index':0,'codec':'mpeg4','width':320,'height':240,
                             'color_primaries':2,'color_trc':2,'sample_aspect_ratio':'1',
                             'aspect_policy':'coded raster dimensions; no display-aspect resampling'}
            if info!=expected_stream:raise ValueError('stream differs from original metadata diagnostic')
            with (out/'matched_frames.jsonl').open('x') as ledger:
                for index,frame in enumerate(container.decode(stream)):
                    if index>=len(timeline):raise ValueError('extra decoded frame')
                    try:actual=compare_frame(frame,index,timeline[index])
                    except BaseException:
                        atomic(out/'metadata_mismatch.json',{'index':index,'actual':exposed_metadata(frame,index),'expected':timeline[index]})
                        raise
                    ledger.write(json.dumps(actual,sort_keys=True)+'\n');ledger.flush()
                    seen=index+1;status['matched_frames']=seen
                    if index in wanted:
                        status['current_frame']=index;checkpoint()
                        convert_selected(av,np,helper,frame,index,wanted[index],actual,out)
                        saved.append(index);status['saved_frames']=saved.copy()
                    checkpoint()
    if seen!=len(timeline) or saved!=sorted(wanted):raise ValueError('incomplete full-ledger coverage')
    return seen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reader-source',type=Path,required=True);p.add_argument('--reader-sha256',required=True)
    p.add_argument('--helpers-sha256',required=True);p.add_argument('--runner-sha256',required=True)
    p.add_argument('--numpy-version',required=True)
    p.add_argument('--ledger',type=Path,required=True)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status={'status':'running','phase':'guards','member':MEMBER,'archive':str(ARCHIVE),
      'allowlist_sha256':digest(canonical([MEMBER])),'manifest_logical_sha256':MANIFEST_SHA,
      'python':sys.version,'platform':platform.platform(),'matched_frames':0,'saved_frames':[], 'scope':'normalized decoder-output index endpoint; not physical time or presentation order'}
    last_phase=status['phase'];phase_started=started
    status['completed_phase_seconds']={}
    def checkpoint():
        nonlocal last_phase,phase_started
        now=time.monotonic()
        if status['phase']!=last_phase:
            status['completed_phase_seconds'][last_phase]=now-phase_started
            last_phase=status['phase'];phase_started=now
        status['current_phase_seconds']=now-phase_started
        status['elapsed_seconds']=now-started
        atomic(args.output/'status.json',status)
    checkpoint()
    try:
        raw=Path(__file__).read_bytes()
        if digest(raw)!=args.runner_sha256:raise ValueError('runner source digest mismatch')
        (args.output/'run_index_decode.py').write_bytes(raw)
        helper=snapshot_import(Path(__file__).with_name('helpers.py'),args.helpers_sha256,args.output,'frozen_helpers')
        reader=snapshot_import(args.reader_source,args.reader_sha256,args.output,'frozen_reader')
        import av
        import PIL
        import numpy as np
        if av.__version__!='16.1.0' or PIL.__version__!='12.1.0' or np.__version__!=args.numpy_version:
            raise ValueError('frozen dependency version mismatch')
        from av.video.reformatter import VideoReformatter,Colorspace,ColorRange
        # Validate the explicit enum API before any archive access.
        _=(VideoReformatter,Colorspace.ITU601,ColorRange.MPEG,ColorRange.JPEG)
        status['dependencies']={'av':av.__version__,'pillow':PIL.__version__,'numpy':np.__version__,
                                'av_libraries':av.library_versions}
        source_status={'runner':digest(raw),'helpers':args.helpers_sha256,'reader':args.reader_sha256}
        status['source_sha256']=source_status
        if digest(canonical(json.loads(args.manifest.read_text())))!=MANIFEST_SHA:raise ValueError('manifest digest mismatch')
        ledger_raw=args.ledger.read_bytes()
        expected_ledger_sha='9344c98746b1b76c45357731936e0a2c111e575a6f8da874d0789228fef9a5d0'
        if digest(ledger_raw)!=expected_ledger_sha:raise ValueError('diagnostic full-ledger hash mismatch')
        timeline=[json.loads(line) for line in ledger_raw.decode().splitlines()]
        if len(timeline)!=216 or [r['index'] for r in timeline]!=list(range(216)):
            raise ValueError('authenticated ledger dimension mismatch')
        (args.output/'authenticated_frames.jsonl').write_bytes(ledger_raw)
        selection=select_indices(len(timeline));atomic(args.output/'selection.json',selection)
        status.update(authenticated_ledger_sha256=expected_ledger_sha,
          reused_diagnostic_job='45579152',reused_metadata_frames=216,
          cost_reuse='prior diagnostic7:03 scheduler time recorded separately; current cost includes one new decode, member hash, all guards and output',
          original_physical_timestamp_policy='failed; unchanged')
        status['phase']='manifest_header_validation';checkpoint()
        manifest=reader.load_manifest(args.manifest,ARCHIVE,expected_manifest_sha256=MANIFEST_SHA)
        reader.check_pinned_metadata(manifest)
        atomic(args.output/'manifest.json',manifest.payload())
        status['phase']='bounded_member_hash';checkpoint()
        h=hashlib.sha256()
        with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
            for chunk in iter(lambda:handle.read(65536),b''):h.update(chunk)
        status['member_sha256']=h.hexdigest()
        if h.hexdigest()!='699175c50544283f3b8537387403ff5b82958e4b18e590611129013131d1601a':
            raise ValueError('member hash differs from preserved first failure')
        status['phase']='index_decode';checkpoint()
        seen=decode_indices(av,np,helper,reader,manifest,timeline,selection,args.output,status,checkpoint)
        status.update(matched_frames=seen,scientific_status='input feasibility only; no physical-timing or model-quality claim')
        files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f.read_bytes())} for f in args.output.iterdir() if f.is_file() and f.name!='status.json'}
        atomic(args.output/'ARTIFACTS.json',files)
        status.update(status='complete',phase='complete')
    except BaseException as exc:status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:checkpoint()
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
