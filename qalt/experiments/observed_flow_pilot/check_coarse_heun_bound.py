"""Checkpoint-only conservative Heun bijection bound; never loads observations.

Computed scalars use ordinary float64 arithmetic, not directed rounding or
interval enclosures. Passing the numerical test is a diagnostic indication of
a mathematical sufficient condition, not a validated interval certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import numpy as np
from qalt.flow_matching import ConvolutionalVelocity

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ('qalt/src/qalt/__init__.py', 'qalt/src/qalt/core.py',
           'qalt/src/qalt/spline.py', 'qalt/src/qalt/multiscale_flow.py',
           'qalt/src/qalt/flow_matching.py')


def source_identity(commit):
    resolved = subprocess.check_output(['git','rev-parse',f'{commit}^{{commit}}'],cwd=ROOT,text=True).strip()
    if resolved != commit:
        raise ValueError('source commit must be a full commit identifier')
    hashes = {}
    for relative in SOURCES:
        current = (ROOT/relative).read_bytes()
        frozen = subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        if current != frozen:
            raise ValueError(f'local architecture differs from specified checkpoint source: {relative}')
        hashes[relative] = hashlib.sha256(current).hexdigest()
    if Path(sys.modules['qalt.flow_matching'].__file__).resolve() != (ROOT/'qalt/src/qalt/flow_matching.py').resolve():
        raise ValueError('imported architecture is outside checked source tree')
    return hashes


def offset_frobenius_sum(weight):
    """Sum channel-matrix Frobenius norms over spatial offsets, in float64."""
    array = weight.detach().cpu().double()
    if array.ndim != 4 or not bool(torch.isfinite(array).all()):
        raise ValueError('finite OIHW weights required')
    values = [[math.hypot(*array[:,:,i,j].reshape(-1).tolist())
               for j in range(array.shape[3])] for i in range(array.shape[2])]
    return math.fsum(v for row in values for v in row), values


def padded_fourier_bound(weight, spatial_size):
    """Bound a same-size zero-padded convolution by a larger periodic one.

    Embed the input isometrically into a (size+kernel-1)-square torus and
    project its convolution output onto the original sites. Fourier transform
    diagonalizes the periodic channel operator; the largest channel-matrix
    singular value over all frequencies is its norm. Compression cannot
    increase that norm. FFT/SVD arithmetic is not interval validated.
    """
    value = weight.detach().cpu().double().numpy()
    if value.ndim != 4 or value.shape[2] != value.shape[3] or value.shape[2] % 2 != 1 or not np.isfinite(value).all():
        raise ValueError('finite square odd-kernel OIHW weights required')
    kernel_size = value.shape[2]
    period = spatial_size + kernel_size - 1
    kernel = np.zeros((*value.shape[:2], period, period), dtype=np.float64)
    radius = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[:, :, (i-radius) % period, (j-radius) % period] += value[:, :, i, j]
    spectrum = np.fft.fft2(kernel, axes=(-2, -1)).transpose(2, 3, 0, 1)
    singular = np.linalg.svd(spectrum, compute_uv=False)
    return float(singular[..., 0].max()), period


def calculate(state, *, prefix='coarse_prior.velocity', steps=16, spatial_size=8):
    if not isinstance(steps,int) or steps<1 or not isinstance(spatial_size,int) or spatial_size<1:
        raise ValueError('positive integer steps and spatial size required')
    # Explicitly select this field; never silently substitute another decoder.
    selected = {key[len(prefix)+1:]:value for key,value in state.items() if key.startswith(prefix+'.')}
    if not selected or any(not isinstance(v,torch.Tensor) or not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in selected.values()):
        raise ValueError('selected field must contain finite tensor state')
    final = selected.get('net.4.weight')
    first = selected.get('net.0.weight')
    if final is None or first is None or final.ndim!=4 or first.ndim!=4:
        raise ValueError('checkpoint lacks the expected three-convolution field')
    channels, width = final.shape[0], first.shape[0]
    expected = ConvolutionalVelocity(channels=channels,width=width)
    template = expected.state_dict()
    if selected.keys()!=template.keys() or any(selected[k].shape!=template[k].shape for k in template):
        raise ValueError('field differs from unconditional conv-SiLU-conv-SiLU-conv architecture')
    if any(not isinstance(expected.net[i],torch.nn.SiLU) for i in (1,3)):
        raise ValueError('activation changed')
    for index,padding,kernel in ((0,(1,1),(3,3)),(2,(1,1),(3,3)),(4,(0,0),(1,1))):
        layer=expected.net[index]
        if not isinstance(layer,torch.nn.Conv2d) or layer.stride!=(1,1) or layer.padding!=padding or layer.kernel_size!=kernel or layer.dilation!=(1,1) or layer.groups!=1 or layer.padding_mode!='zeros':
            raise ValueError('convolution geometry changed')
    norms=[];offsets=[];fourier=[];periods=[]
    for index in (0,2,4):
        weight=selected[f'net.{index}.weight']
        if index==0:
            weight=weight[:,:channels]  # Fixed time channel contributes no state derivative.
        norm,entries=offset_frobenius_sum(weight)
        norms.append(norm);offsets.append(entries)
        spectral,period=padded_fourier_bound(weight,spatial_size)
        fourier.append(spectral);periods.append(period)
    beta=1+1/math.e
    selected_norms=[min(a,b) for a,b in zip(norms,fourier)]
    bound=beta**2*math.prod(selected_norms)
    h=1/steps
    a=h*bound+.5*(h*bound)**2
    threshold=steps*(math.sqrt(3)-1)
    passes=math.isfinite(a) and a<1
    dimension=channels*spatial_size**2
    result=dict(prefix=prefix,channels=channels,width=width,spatial_size=spatial_size,
        dimension=dimension,steps=steps,actual_velocity_calls=2*steps,
        convolution_offset_frobenius_sums=norms,per_offset_frobenius=offsets,
        padded_fourier_operator_upper_estimates=fourier,periodic_embedding_sizes=periods,
        selected_convolution_upper_estimates=selected_norms,
        silu_derivative_bound=beta,uniform_state_lipschitz_upper_estimate=bound,
        heun_residual_lipschitz_upper_estimate=a,strict_sufficient_L_threshold=threshold,
        ordinary_float_test_passes=passes,validated_interval_certificate=False,
        checkpoint_state_validated='unconditional three-convolution SiLU field; finite biases and weights',
        interpretation='Failure is inconclusive about invertibility; passing requires rounding-controlled verification for a numerical certificate.')
    if passes:
        result['composition_bounds']={
            'log_lower_lipschitz':steps*math.log1p(-a),
            'log_upper_lipschitz':steps*math.log1p(a),
            'log_lower_determinant':dimension*steps*math.log1p(-a),
            'log_upper_determinant':dimension*steps*math.log1p(a)}
    if bound>0 and math.isfinite(bound):
        result['last_layer_scale_strict_upper_for_bound']=threshold/bound
        result['sufficient_steps_strictly_greater_than']=bound/(math.sqrt(3)-1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',choices=['coarse_prior.velocity','velocity'],default='coarse_prior.velocity')
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--spatial-size',type=int,default=8)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refusing to replace output')
    hashes=source_identity(args.source_commit)
    # weights_only avoids general checkpoint pickle execution. No model fitting,
    # observation loading, or sampling is performed.
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(state,dict):raise ValueError('plain state_dict checkpoint required')
    report=calculate(state,prefix=args.prefix,steps=args.steps,spatial_size=args.spatial_size)
    report.update(source_commit=args.source_commit,source_sha256=hashes,
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hash,
        calculator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_accessed=False,trained_weights_modified=False,
        checkpoint_source_link='source commit is caller-declared; source files checked, checkpoint provenance must also be checked against the run manifest')
    # Unrepresentably large bounds must fail JSON serialization instead of being
    # reported as finite guarantees.
    payload=json.dumps(report,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:handle.write(payload)
    print(payload,end='')


if __name__=='__main__':main()
