"""Build/load the exact packed-sign C kernel; first build is timed and recorded.
The architecture-specific shared object is a rebuildable runtime output, not
one of the source artifact payload files. No network or native-data access.
"""
from __future__ import annotations
import ctypes,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
import numpy as np

_LIB=None
BUILD_RECORD=None


def load_kernel():
    global _LIB,BUILD_RECORD
    if _LIB is not None:return _LIB
    src=Path(__file__).with_name('packed_graph.c')
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    build=Path(tempfile.mkdtemp(prefix='pro13-graph-'))
    lib=build/'packed_graph.so'
    cc=os.environ.get('CC','cc')
    cmd=[cc,'-O3','-march=native','-std=c11','-fPIC','-shared',str(src),'-o',str(lib)]
    t=time.perf_counter(); result=subprocess.run(cmd,capture_output=True,text=True,check=False)
    seconds=time.perf_counter()-t
    BUILD_RECORD=dict(command=cmd,seconds=seconds,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        source_sha256=digest,compiler_version=subprocess.check_output([cc,'--version'],text=True),
        shared_object_sha256=hashlib.sha256(lib.read_bytes()).hexdigest() if lib.exists() else None,
        shared_object_bytes=lib.stat().st_size if lib.exists() else None,
        delivery='source/compiler record supplied; architecture-specific runtime shared object is not a payload')
    if result.returncode:raise RuntimeError('kernel compilation failed: '+result.stderr)
    dll=ctypes.CDLL(str(lib)); fun=dll.pro13_graph
    fun.argtypes=[np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS'),ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,
        ctypes.c_int64,ctypes.c_int64,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ctypes.POINTER(ctypes.c_int64)]
    fun.restype=ctypes.c_int; _LIB=fun
    return fun


def compiled_graph(block:np.ndarray):
    fun=load_kernel()
    x=np.asarray(block,float)
    if x.ndim!=2 or not np.all(np.isfinite(x)) or np.any((x<0)|(x>1)) or x.shape[1]%2:
        raise ValueError('finite [N, even D] unit block required')
    n,d=x.shape
    # np.packbits returns bytes, then explicitly pad each column to whole uint64s.
    raw=np.packbits((x<=.25)|(x>=.75),axis=0,bitorder='little').T
    padded=np.zeros((d,((n+63)//64)*8),dtype=np.uint8)
    padded[:,:raw.shape[1]]=raw
    words=padded.view(np.uint64)
    degrees=np.zeros(d,np.int32); partners=np.full(d,-1,np.int32); count=ctypes.c_int64()
    ret=fun(words,d,words.shape[1],n,159,1000,degrees,partners,ctypes.byref(count))
    if ret:raise RuntimeError('C kernel rejected input')
    accepted=bool(np.all(degrees==1))
    edges=[(i,int(j)) for i,j in enumerate(partners) if i<j] if accepted else []
    return edges,dict(accepted=accepted,threshold_edges=count.value,zero_degree=int(np.sum(degrees==0)),
        multiple_degree=int(np.sum(degrees>1)),packed_payload_bytes=int(padded.nbytes))
