"""Fabricated weights only; no checkpoints or observations from a study."""
import importlib.util
from pathlib import Path
import math
import pytest
import torch
from qalt.flow_matching import ConvolutionalVelocity

path=Path(__file__).resolve().parents[1]/'experiments/observed_flow_pilot/check_coarse_heun_bound.py'
spec=importlib.util.spec_from_file_location('heun_bound_test',path)
calculator=importlib.util.module_from_spec(spec);spec.loader.exec_module(calculator)


def state():
    return {f'coarse_prior.velocity.{k}':v for k,v in ConvolutionalVelocity(3,width=4).state_dict().items()}


def test_zero_final_layer_and_fixed_time_bias_do_not_change_bound():
    weights=state();report=calculator.calculate(weights)
    assert report['uniform_state_lipschitz_upper_estimate']==0
    assert report['ordinary_float_test_passes'] and not report['validated_interval_certificate']
    weights['coarse_prior.velocity.net.0.weight'][:,3]=1e20
    weights['coarse_prior.velocity.net.0.bias'].fill_(1e20)
    modified=calculator.calculate(weights)
    assert report['convolution_offset_frobenius_sums']==modified['convolution_offset_frobenius_sums']
    assert report['composition_bounds']==modified['composition_bounds']


def test_one_entry_per_layer_gives_explicit_product_and_step_threshold():
    weights=state()
    for value in weights.values():value.zero_()
    for index,scale,center in ((0,2.,1),(2,3.,1),(4,5.,0)):
        weights[f'coarse_prior.velocity.net.{index}.weight'][0,0,center,center]=scale
    report=calculator.calculate(weights,steps=16)
    assert report['convolution_offset_frobenius_sums']==[2,3,5]
    expected=(1+1/math.e)**2*30
    assert report['uniform_state_lipschitz_upper_estimate']==pytest.approx(expected)
    assert not report['ordinary_float_test_passes']
    factor=.5*report['last_layer_scale_strict_upper_for_bound']
    weights['coarse_prior.velocity.net.4.weight']*=factor
    assert calculator.calculate(weights)['ordinary_float_test_passes']


def test_sum_offset_frobenius_bounds_dense_zero_padded_convolution():
    torch.manual_seed(48)
    layer=torch.nn.Conv2d(2,3,3,padding=1,bias=False).double()
    eye=torch.eye(18,dtype=torch.double).reshape(18,2,3,3)
    matrix=layer(eye).flatten(1).T
    operator=torch.linalg.svdvals(matrix)[0].item()
    bound,_=calculator.offset_frobenius_sum(layer.weight)
    assert operator<=bound


def test_shape_nonfinite_and_conditional_fields_rejected():
    weights=state();weights['coarse_prior.velocity.net.2.bias'][0]=float('nan')
    with pytest.raises(ValueError):calculator.calculate(weights)
    conditional={f'coarse_prior.velocity.{k}':v for k,v in ConvolutionalVelocity(3,context_channels=3,width=4).state_dict().items()}
    with pytest.raises(ValueError):calculator.calculate(conditional)
    with pytest.raises(ValueError):calculator.calculate(state(),prefix='velocity')


def test_padded_fourier_bound_dominates_dense_zero_padding_operator():
    torch.manual_seed(583)
    for size,kernel in ((2,3),(3,3),(3,1)):
        layer=torch.nn.Conv2d(2,3,kernel,padding=kernel//2,bias=False).double()
        eye=torch.eye(2*size*size,dtype=torch.double).reshape(-1,2,size,size)
        dense=layer(eye).flatten(1).T
        actual=torch.linalg.svdvals(dense)[0].item()
        upper,period=calculator.padded_fourier_bound(layer.weight,size)
        rough,_=calculator.offset_frobenius_sum(layer.weight)
        assert period==size+kernel-1
        assert actual <= upper+1e-12
        assert upper <= rough+1e-12
        if kernel==1: assert actual==pytest.approx(upper)
