"""Source-only numerical validation. No observed-data loader is imported."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from qalt.rgb_block import FixedShapeGSM
from qalt.radial_transport import decode_radial, encode_radial
from qalt.output_screen import displacement_screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260908)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior results; supply a fresh path')
    rng = np.random.default_rng(args.seed)
    shape = np.array([[1., .4, -.1], [.4, 1.3, .2], [-.1, .2, .8]])
    model = FixedShapeGSM(np.array([.2,.3,.5]), np.array([.08,.5,1.8]), shape/np.linalg.det(shape)**(1/3))
    base = rng.normal(size=(4096,3))
    # Warm both maps, then alternate order; three paired timing repetitions.
    decode_radial(model, base[:8]); model.sample(base[:8])
    timing = {'radial': [], 'legacy': []}
    for trial in range(3):
        names = ['radial','legacy'] if trial%2 == 0 else ['legacy','radial']
        for name in names:
            start = time.perf_counter()
            result = decode_radial(model, base) if name == 'radial' else model.sample(base)
            timing[name].append(time.perf_counter()-start)
    decoded = decode_radial(model, base)
    encoded = encode_radial(model, decoded.values)
    # Nonlinear triangular six-dimensional composition retains every input.
    u = rng.normal(size=(512,6))
    first = decode_radial(model,u[:,:3])
    second = decode_radial(model,u[:,3:])
    x = np.concatenate([first.values, second.values + .2*np.sin(first.values)],axis=1)
    inverse = np.concatenate([encode_radial(model,x[:,:3]).values,
                              encode_radial(model,x[:,3:]-.2*np.sin(x[:,:3])).values],axis=1)
    # The tanh fixture has known displacement <= epsilon*sqrt(3), independently
    # of samples. Its tiny correction is designed to be rejected at margin .005.
    sources = rng.normal(size=(2048,64))
    old = np.tanh(sources+.1*sources**3)
    new = old.copy()
    new[:,:3] = np.tanh(sources[:,:3]+.1*sources[:,:3]**3+.001*np.sin(sources[:,3:6]))
    screen = displacement_screen(old,new,displacement_bound=.001*np.sqrt(3)/8,
                                 distance_scale=8,margin=.005)
    repository = Path(__file__).resolve().parents[3]
    git = lambda *cmd: subprocess.check_output(['git','-C',str(repository),*cmd],text=True).strip()
    paths = ['qalt/src/qalt/radial_transport.py','qalt/src/qalt/rgb_block.py',
             'qalt/src/qalt/output_screen.py','qalt/experiments/transport_validation/run.py']
    output = {'status':'numerical_fixture_only','real_data_accessed':False,'test_data_accessed':False,
              'seed':args.seed,'source_commit':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),
              'source_sha256':{p:hashlib.sha256((repository/p).read_bytes()).hexdigest() for p in paths},
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'platform':platform.platform(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
              'base_vectors':len(base),'base_dimension':3,'timing_seconds':timing,
              'radial_iterations':decoded.iterations,'relative_root_bracket':decoded.max_relative_bracket,
              'roundtrip_max':float(np.max(np.abs(base-encoded.values))),
              'logdet_cancellation_max':float(np.max(np.abs(decoded.log_abs_det+encoded.log_abs_det))),
              'nonlinear_six_dimensional_roundtrip_max':float(np.max(np.abs(u-inverse))),
              'screen':screen,'quality_advantage_established':False,'cost_advantage_established':False}
    assert output['roundtrip_max']<1e-9
    assert output['nonlinear_six_dimensional_roundtrip_max']<1e-9
    assert screen['reject_margin']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
