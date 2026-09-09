"""Fabricated-only source/clock/checkpoint checks; no canonical loading."""
import ast,importlib.util,json,types
from pathlib import Path
import numpy as np
import pytest
import torch
PATH=Path(__file__).parents[1]/'experiments/rgb_full_generation_fit/run.py'
spec=importlib.util.spec_from_file_location('rgb_full_fit_tests',PATH);run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def test_exact_native_pixel_logit_policy_and_registered_seeds():
 x=np.linspace(.001,.999,2*3*12*12).reshape(2,3,12,12)
 actual=run.native.logit_inputs(x)[0]
 assert torch.equal(actual,torch.from_numpy((np.log(x)-np.log1p(-x)).astype(np.float32)))
 assert run.SEEDS==(78201,78202,78203)
 with pytest.raises(ValueError):run.model_spec('latent',79201)
 assert run.model_spec('pixel',78201)['source_dimension']==3072
 assert run.model_spec('latent',78201)['source_dimension']==1024
 assert run.model_spec('pixel',78201)['streams']['FIT_index_PCG64_reset_each_stage']==78232

def test_no_evaluation_or_repair_access_and_preloader_guard():
 tree=ast.parse(PATH.read_text());attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
 assert not {'repair','repair_ids','sample_from_gaussian','reconstruction_metrics','make_extractor'}&attrs
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');text=ast.unparse(main)
 assert text.index('source_guard(')<text.index('load_observed_flow_data(')
 assert 'RGB_PROCESS_START_NS' in PATH.read_text()
 assert run.PIXEL_POINTS==(360.,780.,1200.,1800.) and run.LATENT_POINTS==(180.,600.,1200.,1800.)
 assert run.learning_rate(90,1800)==pytest.approx(2e-4) and run.learning_rate(1800,1800)==pytest.approx(2e-5)

def test_small_stage_deadline_checkpoint_rng_and_prefix_cost(tmp_path,monkeypatch):
 old=torch.get_num_threads();torch.set_num_threads(1)
 try:
  clock=[0.];monkeypatch.setattr(run,'time',types.SimpleNamespace(perf_counter=lambda:clock[0]))
  model=run.PixelFlowMatching(size=12,width=4,blocks=1,groups=2)
  targets=torch.randn(3,3,12,12,generator=torch.Generator().manual_seed(71));ids=np.array([9,12,99])
  def loss(x,g):clock[0]+=.6;return model.training_loss(x,g)
  report=run.fit_stage(model,targets,ids,tmp_path,'field',1.,(.5,1.),78201,loss,{'synthetic':True},-5.,{},True)
  assert report['updates']==2 and report['overrun_seconds']==pytest.approx(.2)
  checkpoints=report['checkpoints'];assert len(checkpoints)==3
  assert all(r['standalone_prefix_seconds_through_hash']>=5 for r in checkpoints)
  saved=torch.load(tmp_path/checkpoints[-1]['path'],weights_only=True)
  assert saved['updates']==2 and saved['path_rng'] is not None and saved['optimizer']['state']
  draws=[json.loads(s) for s in (tmp_path/'field_steps.jsonl').read_text().splitlines() if json.loads(s)['event']=='draw']
  rng=np.random.default_rng(78201+31)
  for row in draws:assert row['indices']==rng.integers(0,3,size=32).tolist()
 finally:torch.set_num_threads(old)
