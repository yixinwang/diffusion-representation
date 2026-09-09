#!/usr/bin/env python3
"""Rebuild and verify Pro12 locally. No network, repository writes, fits, or jobs.

Results go to a NEW directory. Failed/timeout attempts retain stdout, stderr,
return status, source hashes, and completed outputs. COMPLETE is written last.
"""
from __future__ import annotations
import argparse
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from decimal import Decimal
from frontier import summarize

HERE = Path(__file__).resolve().parent
NS = (4,8,16,32,64)
SOURCES = ('certify.cpp','mpfr_minimal.h','test_certificate.cpp','global_bounds.cpp',
           'risk_bounds.cpp','frontier.py','run.py','sources/psc_summary.json')

def write(path: Path, value: dict):
    with path.open('x') as f:
        json.dump(value,f,indent=2); f.write('\n')

def invoke(command: list[str], stem: Path, timeout: float, env: dict[str,str]):
    start=time.monotonic(); timed_out=False; rc=None
    try:
        with stem.with_suffix('.json').open('x') as stdout, stem.with_suffix('.log').open('x') as stderr:
            proc=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,timeout=timeout,check=False)
            rc=proc.returncode
    except subprocess.TimeoutExpired:
        timed_out=True
    finally:
        write(stem.with_suffix('.process.json'),{'command':command,'returncode':rc,
              'timeout':timed_out,'timeout_limit_seconds':timeout,'elapsed_seconds':time.monotonic()-start})
    if timed_out or rc!=0:
        raise RuntimeError(f'bounded command failed: {stem.name}; inspect retained logs')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='new result directory, must not exist')
    p.add_argument('--timeout',type=float,default=240,help='hard per-process timeout, not a runtime prediction')
    p.add_argument('--mpfr-link',default=None,help='optional linker flags; default uses installed MPFR')
    p.add_argument('--n',type=int,nargs='+',choices=NS,default=NS)
    p.add_argument('--repeat-order8',action='store_true',help='additional N4 check at 192 bits, changed grid/order')
    a=p.parse_args()
    if a.timeout<=0: p.error('timeout must be positive')
    if len(set(a.n))!=len(a.n): p.error('duplicate grid')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    write(out/'SOURCE_HASHES.json',manifest)
    compiler=shlex.split(os.environ.get('CXX','g++'))
    lib=ctypes.util.find_library('mpfr')
    flags=shlex.split(a.mpfr_link) if a.mpfr_link else ([f'-l:{lib}'] if lib and '.so' in lib else ['-lmpfr'])
    write(out/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                               'compiler':compiler,'mpfr_library':lib,'link_flags':flags,'fit_performed':False})
    try:
        bins={}
        for name in ('certify','test_certificate','global_bounds','risk_bounds'):
            binary=out/name;bins[name]=binary
            cmd=compiler+['-O2','-std=c++17','-fno-fast-math','-ffp-contract=off',str(HERE/f'{name}.cpp')]+flags+['-lgmp','-o',str(binary)]
            invoke(cmd,out/f'build_{name}',min(a.timeout,60),env)
        invoke([str(bins['test_certificate'])],out/'unit_tests',min(a.timeout,60),env)
        invoke([str(bins['global_bounds'])],out/'global_bounds',min(a.timeout,30),env)
        for name in ('unit_tests','global_bounds'):
            if json.loads((out/f'{name}.json').read_text())['status']!='pass':
                raise RuntimeError(f'{name} failed')
        for n in a.n:
            invoke([str(bins['certify']),str(n),'6','48','4','5e-9','128'],out/f'n{n}_first_attempt',a.timeout,env)
            raw=json.loads((out/f'n{n}_first_attempt.json').read_text(),parse_float=Decimal)
            lo,hi=raw['conditional_KL']
            if raw['N']!=n or not 0<=lo<=hi or hi-lo>Decimal('2e-8'):
                raise RuntimeError(f'N{n}: requested conditional enclosure failed')
            invoke([str(bins['risk_bounds']),str(n),str(lo),str(hi)],out/f'n{n}_risk',min(a.timeout,30),env)
            risk=json.loads((out/f'n{n}_risk.json').read_text(),parse_float=Decimal)
            l,u=risk['unconditional_KL']
            if not 0<=l<=u or u-l>Decimal('2e-8'):
                raise RuntimeError(f'N{n}: requested expected-risk enclosure failed')
        if a.repeat_order8:
            invoke([str(bins['certify']),'4','8','32','3','5e-9','192'],out/'n4_order8_precision192',a.timeout,env)
        if set(a.n)==set(NS):
            write(out/'frontier.json',summarize(out,HERE/'sources/psc_summary.json'))
        payload={f.relative_to(out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
        write(out/'COMPLETE.json',{'status':'completed_all_requested_checks','grid':a.n,
                                'full_fixed_grid':set(a.n)==set(NS),'payload_sha256':payload})
    except Exception as exc:
        write(out/'FAILED.json',{'status':'failed','error':repr(exc),'outputs_preserved':True})
        raise

if __name__=='__main__':main()
