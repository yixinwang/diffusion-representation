"""Metadata-only diagnostic of the same failed training clip. No RGB/selection."""
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


def diagnostic_rows(av,reader,manifest,out,status,checkpoint):
    flags=[];previous=None;count=0
    with reader.TrainMemberReader(ARCHIVE,manifest,MEMBER,train_allowlist=[MEMBER]) as handle:
        with av.open(handle,mode='r',format='avi') as container:
            stream,info=stream_metadata(container)
            atomic(out/'stream.json',info)
            with (out/'frames.jsonl').open('x') as ledger:
                for index,frame in enumerate(container.decode(stream)):
                    if index>=10000:raise ValueError('diagnostic fixed10000-frame cap reached')
                    tb=frame.time_base
                    row={'index':index,'pts':frame.pts,'dts':frame.dts,
                         'time_base':None if tb is None else [int(tb.numerator),int(tb.denominator)],
                         'width':frame.width,'height':frame.height,'format':frame.format.name,
                         'colorspace':int(frame.colorspace),'color_range':int(frame.color_range),
                         'color_primaries':getattr(frame,'color_primaries',None),
                         'color_trc':getattr(frame,'color_trc',None),
                         'interlaced':bool(getattr(frame,'interlaced_frame',False)),
                         'rotation':getattr(frame,'rotation',None),'timestamp_flags':[]}
                    stamp=None
                    if frame.pts is None:row['timestamp_flags'].append('null_pts')
                    if tb is None or tb<=0:row['timestamp_flags'].append('null_or_nonpositive_time_base')
                    if not row['timestamp_flags']:stamp=Fraction(frame.pts)*Fraction(tb)
                    if stamp is not None and previous is not None and stamp<=previous[1]:
                        row['timestamp_flags'].append('nonincreasing_vs_previous_valid_frame')
                        row['previous_valid_index']=previous[0]
                        row['previous_valid_time']=[previous[1].numerator,previous[1].denominator]
                    # Write the offending row BEFORE updating flags/status or continuing.
                    ledger.write(json.dumps(row,sort_keys=True)+'\n');ledger.flush();os.fsync(ledger.fileno())
                    if row['timestamp_flags']:flags.append({'index':index,'flags':row['timestamp_flags']})
                    if stamp is not None:previous=(index,stamp)
                    count=index+1;status.update(diagnostic_frames=count,timestamp_flagged_frames=len(flags));checkpoint()
    atomic(out/'timestamp_flags.json',flags)
    return count,flags


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reader-source',type=Path,required=True);p.add_argument('--reader-sha256',required=True)
    p.add_argument('--helpers-sha256',required=True);p.add_argument('--runner-sha256',required=True)
    p.add_argument('--numpy-version',required=True)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status={'status':'running','phase':'guards','member':MEMBER,'archive':str(ARCHIVE),
      'allowlist_sha256':digest(canonical([MEMBER])),'manifest_logical_sha256':MANIFEST_SHA,
      'python':sys.version,'platform':platform.platform(),'diagnostic_frames':0,'saved_frames':[], 'scope':'same-clip timestamp diagnosis; no RGB, selection, replacement or timestamp repair'}
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
        (args.output/'diagnostic_metadata.py').write_bytes(raw)
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
        status['phase']='metadata_diagnostic';checkpoint()
        count,flags=diagnostic_rows(av,reader,manifest,args.output,status,checkpoint)
        status.update(diagnostic_frames=count,timestamp_flagged_frames=len(flags),
                      selection_policy_changed=False,rgb_arrays_saved=False,
                      scientific_status='diagnostic only; original prerequisite remains failed')
        files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f.read_bytes())} for f in args.output.iterdir() if f.is_file() and f.name!='status.json'}
        atomic(args.output/'ARTIFACTS.json',files)
        status.update(status='complete',phase='complete')
    except BaseException as exc:status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:checkpoint()
    return 0 if status['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
