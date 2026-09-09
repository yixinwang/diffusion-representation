"""Fabricated codec smoke only. Never opens any dataset or archive."""
import sys,json,io,hashlib,platform
from pathlib import Path
import numpy as np
import av
root=Path(sys.argv[1]).resolve();assert Path(av.__file__).resolve().is_relative_to(root/'packages')
buffer=io.BytesIO();original=[]
with av.open(buffer,mode='w',format='matroska') as output:
 stream=output.add_stream('ffv1',rate=8);stream.width=16;stream.height=16;stream.pix_fmt='bgr0'
 for i in range(8):
  frame=np.zeros((16,16,3),dtype=np.uint8);frame[:,:,0]=i*20;frame[:,:,1]=np.arange(16,dtype=np.uint8)[None,:]*10;frame[:,:,2]=np.arange(16,dtype=np.uint8)[:,None]*10;original.append(frame)
  for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):output.mux(packet)
 for packet in stream.encode():output.mux(packet)
payload=buffer.getvalue();buffer.seek(0)
with av.open(buffer,mode='r') as container:
 decoded=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
assert len(decoded)==8 and np.array_equal(np.stack(original),np.stack(decoded))
record={'status':'fabricated_lossless_smoke_pass','av_version':av.__version__,'av_origin':av.__file__,'python':platform.python_version(),'library_versions':av.library_versions,'codec':'ffv1','container':'matroska','frames':8,'shape':[16,16,3],'bytes':len(payload),'fabricated_container_sha256':hashlib.sha256(payload).hexdigest(),'exact_rgb_roundtrip':True,'native_archive_opened':False}
with (root/'smoke.json').open('x') as out:json.dump(record,out,indent=2)
print(json.dumps(record))
