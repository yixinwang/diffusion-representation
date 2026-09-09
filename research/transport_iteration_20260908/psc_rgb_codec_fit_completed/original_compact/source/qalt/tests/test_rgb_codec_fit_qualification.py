"""Small fabricated metric/order checks; never access canonical data or run fitting."""
import ast,importlib.util,math
from pathlib import Path
import numpy as np
import pytest
import torch
PATH=Path(__file__).parents[1]/'experiments/rgb_codec_fit_qualification/run.py'
spec=importlib.util.spec_from_file_location('rgb_codec_fit_test',PATH);run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def test_global_psnr_is_global_mse_not_average_psnr():
 mse=np.array([.001,.009]);actual=run.global_psnr(mse)
 assert actual==pytest.approx(-10*math.log10(.005))
 assert actual!=pytest.approx(float((-10*np.log10(mse)).mean()))
 assert math.isinf(run.global_psnr([0.,0.]))
 for bad in ([float('nan')],[-1.],[]):
  with pytest.raises(ValueError):run.global_psnr(bad)

def test_ssim_identity_and_known_constant_offset():
 old=torch.get_num_threads();torch.set_num_threads(1)
 try:
  x=torch.rand(2,3,12,12,generator=torch.Generator().manual_seed(994),dtype=torch.float64)
  mse,ssim=run.reconstruction_metrics(x,x)
  assert torch.equal(mse,torch.zeros(2,dtype=torch.float64))
  torch.testing.assert_close(ssim,torch.ones_like(ssim),atol=2e-13,rtol=0)
  a=torch.full_like(x,.2);b=torch.full_like(x,.4);mse,ssim=run.reconstruction_metrics(a,b)
  torch.testing.assert_close(mse,torch.full_like(mse,.04),atol=1e-15,rtol=0)
  expected=(2*.2*.4+.01**2)/(.2**2+.4**2+.01**2)
  torch.testing.assert_close(ssim,torch.full_like(ssim,expected),atol=1e-12,rtol=0)
 finally:torch.set_num_threads(old)

def test_evaluation_stops_before_any_model_read_without_freeze(tmp_path):
 class Forbidden:
  @property
  def codec(self):raise AssertionError('model read before freeze receipt')
 with pytest.raises(ValueError,match='freeze'):run.normalize_and_evaluate(Forbidden(),None,tmp_path,{})

def test_frozen_order_no_repair_accessor_and_declared_schedule():
 tree=ast.parse(PATH.read_text());attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
 assert 'repair' not in attrs and 'repair_ids' not in attrs
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');text=ast.unparse(main)
 assert text.index('fit_codec(')<text.index('normalize_and_evaluate(')
 assert run.SEED==79201 and run.CHECKPOINT_SECONDS==(60.,150.,300.,450.,600.)
 assert run.learning_rate(0)==pytest.approx(2e-5)
 assert run.learning_rate(30)==pytest.approx(2e-4)
 assert run.learning_rate(600)==pytest.approx(2e-5)
