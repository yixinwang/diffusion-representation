"""One frozen training clip, two-pass bounded decode. No path/member override."""
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


def color_policy(pixel_format, colorspace, color_range):
    # FFmpeg AVColorSpace values: RGB0, BT7091, unspecified2, FCC4,
    # BT470BG5, SMPTE170M6. AVColorRange: unspecified0, MPEG1, JPEG2.
    if pixel_format not in ('yuv420p','yuv422p','yuv444p'):
        raise ValueError('unsupported pixel format; no implicit conversion')
    if colorspace not in (2,5,6) or color_range not in (0,1):
        raise ValueError('unsupported explicit color matrix/range')
    return {'matrix':'BT.601','range':'limited','assumption_used':colorspace==2 or color_range==0,
            'source_colorspace':colorspace,'source_range':color_range}


def frame_metadata(frame,index):
    if frame.pts is None or frame.time_base is None:raise ValueError('missing frame timestamp')
    if frame.width<=0 or frame.height<=0:raise ValueError('invalid dimensions')
    if bool(getattr(frame,'interlaced_frame',False)):raise ValueError('interlaced frame unsupported')
    if getattr(frame,'rotation',0)!=0:raise ValueError('rotated frame unsupported')
    cs=int(frame.colorspace);cr=int(frame.color_range)
    policy=color_policy(frame.format.name,cs,cr)
    return {'index':index,'pts':int(frame.pts),'time_base':[int(frame.time_base.numerator),int(frame.time_base.denominator)],
            'width':frame.width,'height':frame.height,'format':frame.format.name,
            'colorspace':cs,'color_range':cr,'color_primaries':getattr(frame,'color_primaries',None),
            'color_trc':getattr(frame,'color_trc',None),'policy':policy}


def stream_metadata(container):
    if len(container.streams.video)!=1:raise ValueError('exactly one video stream required')
    s=container.streams.video[0];s.thread_type='NONE';s.codec_context.thread_count=1
    return s,{'index':s.index,'codec':s.codec_context.name,'width':s.codec_context.width,'height':s.codec_context.height,
        'color_primaries':getattr(s.codec_context,'color_primaries',None),
        'color_trc':getattr(s.codec_context,'color_trc',None),
        'sample_aspect_ratio':str(s.sample_aspect_ratio),
        'aspect_policy':'coded raster dimensions; no display-aspect resampling'}


def first_pass(av,reader,manifest,out,status,checkpoint):
    timeline=[]
    with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
        with av.open(handle,mode='r',format='avi') as container:
            stream,info=stream_metadata(container)
            atomic(out/'stream.json',info)
            with (out/'frames.jsonl').open('x') as ledger:
                previous=None
                for index,frame in enumerate(container.decode(stream)):
                    row=frame_metadata(frame,index)
                    stamp=Fraction(row['pts'])*Fraction(*row['time_base'])
                    if previous is not None and stamp<=previous:raise ValueError('nonmonotone timestamps')
                    previous=stamp;timeline.append(row)
                    ledger.write(json.dumps(row,sort_keys=True)+'\n');ledger.flush()
                    status['first_pass_frames']=len(timeline);checkpoint()
    return timeline,info


def second_pass(av,np,helper,reader,manifest,timeline,selection,info,out,status,checkpoint):
    wanted={row['index']:row for row in selection};saved=[];seen=0
    with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
        with av.open(handle,mode='r',format='avi') as container:
            stream,actual=stream_metadata(container)
            if actual!=info:raise ValueError('stream changed between passes')
            for index,frame in enumerate(container.decode(stream)):
                if index>=len(timeline) or frame_metadata(frame,index)!=timeline[index]:raise ValueError('frame metadata changed between passes')
                seen+=1
                if index not in wanted:continue
                status['current_frame']=index;checkpoint()
                # Explicit source colorspace/range prevents adaptive interpretation
                # of unspecified metadata. BT601 limited is frozen prospectively.
                from av.video.reformatter import VideoReformatter,Colorspace,ColorRange
                rgbframe=VideoReformatter().reformat(frame,format='rgb24',src_colorspace=Colorspace.ITU601,
                    dst_colorspace=Colorspace.ITU601,src_color_range=ColorRange.MPEG,dst_color_range=ColorRange.JPEG)
                rgb=rgbframe.to_ndarray(format='rgb24')
                np.save(out/f'frame_{index:06d}_original_rgb.npy',rgb,allow_pickle=False)
                processed,geometry=helper.resize_rgb(rgb)
                np.save(out/f'frame_{index:06d}_processed_rgb.npy',processed,allow_pickle=False)
                values=helper.dequantize(processed,frame_index=index)
                for key in ('uint32_noise','unit_float64','logit_float64','logit_float32'):
                    np.save(out/f'frame_{index:06d}_{key}.npy',values.pop(key),allow_pickle=False)
                atomic(out/f'frame_{index:06d}.json',{'selection':wanted[index],'geometry':geometry,'dequantization':values})
                saved.append(index);status['saved_frames']=saved.copy();checkpoint()
    if seen!=len(timeline) or saved!=sorted(wanted):raise ValueError('second-pass frame coverage mismatch')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reader-source',type=Path,required=True);p.add_argument('--reader-sha256',required=True)
    p.add_argument('--helpers-sha256',required=True);p.add_argument('--runner-sha256',required=True)
    p.add_argument('--numpy-version',required=True)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status={'status':'running','phase':'guards','member':MEMBER,'archive':str(ARCHIVE),
      'allowlist_sha256':digest(canonical([MEMBER])),'manifest_logical_sha256':MANIFEST_SHA,
      'python':sys.version,'platform':platform.platform(),'first_pass_frames':0,'saved_frames':[]}
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
        (args.output/'run_decode.py').write_bytes(raw)
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
        probe=av.VideoFrame(2,2,'yuv420p')
        for i,plane in enumerate(probe.planes):plane.update(bytes([16 if i==0 else 128])*plane.buffer_size)
        probe_rgb=VideoReformatter().reformat(probe,format='rgb24',src_colorspace=Colorspace.ITU601,
            dst_colorspace=Colorspace.ITU601,src_color_range=ColorRange.MPEG,dst_color_range=ColorRange.JPEG).to_ndarray(format='rgb24')
        if probe_rgb.shape!=(2,2,3) or probe_rgb.dtype!=np.uint8:raise ValueError('fabricated conversion API probe failed')
        status['dependencies']={'av':av.__version__,'pillow':PIL.__version__,'numpy':np.__version__,
                                'av_libraries':av.library_versions}
        source_status={'runner':digest(raw),'helpers':args.helpers_sha256,'reader':args.reader_sha256}
        status['source_sha256']=source_status
        if digest(canonical(json.loads(args.manifest.read_text())))!=MANIFEST_SHA:raise ValueError('manifest digest mismatch')
        status['phase']='manifest_header_validation';checkpoint()
        manifest=reader.load_manifest(args.manifest,ARCHIVE,expected_manifest_sha256=MANIFEST_SHA)
        reader.check_pinned_metadata(manifest)
        atomic(args.output/'manifest.json',manifest.payload())
        status['phase']='bounded_member_hash';checkpoint()
        h=hashlib.sha256()
        with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
            for chunk in iter(lambda:handle.read(65536),b''):h.update(chunk)
        status['member_sha256']=h.hexdigest();status['phase']='first_pass';checkpoint()
        timeline,info=first_pass(av,reader,manifest,args.output,status,checkpoint)
        records=[dict(row,time_base=Fraction(*row['time_base'])) for row in timeline]
        selection=helper.select_frames(records);atomic(args.output/'selection.json',selection)
        status['phase']='second_pass';checkpoint()
        second_pass(av,np,helper,reader,manifest,timeline,selection,info,args.output,status,checkpoint)
        files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f.read_bytes())} for f in args.output.iterdir() if f.is_file() and f.name!='status.json'}
        atomic(args.output/'ARTIFACTS.json',files)
        status.update(status='complete',phase='complete')
    except BaseException as exc:status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:checkpoint()
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
