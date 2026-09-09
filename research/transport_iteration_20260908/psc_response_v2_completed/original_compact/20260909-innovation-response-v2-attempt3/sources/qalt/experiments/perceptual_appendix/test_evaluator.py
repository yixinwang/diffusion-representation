"""Fabricated tensors only; no pretrained weights or data accesses."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('appendix_evaluator',Path(__file__).with_name('evaluator.py'))
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)


class FakeExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.batch_sizes=[]
    def forward(self,x):
        assert x.dtype==torch.float32 and x.shape[1:]==(3,32,32)
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        self.batch_sizes.append(len(x))
        return [x.mean((1,2,3)).reshape(-1,1,1,1).expand(-1,2048,1,1)]


class EvaluatorTest(unittest.TestCase):
    def test_chart_batching_and_endpoints(self):
        images=np.full((33,3,32,32),.375,dtype=np.float64)
        images[0]=0;images[-1]=1
        model=FakeExtractor();features=evaluator.extract(model,images,'cpu')
        self.assertEqual(model.batch_sizes,[32,1]);self.assertEqual(features.shape,(33,2048))
        self.assertEqual(features.dtype,np.float32)
        np.testing.assert_array_equal(features[:,0],images.mean((1,2,3)))

    def test_reject_no_clipping(self):
        model=FakeExtractor()
        with self.assertRaises(ValueError):evaluator.extract(model,np.zeros((1,3,32,32),np.float32),'cpu')
        for invalid in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError):
                evaluator.extract(model,np.full((1,3,32,32),invalid,np.float64),'cpu')

    def test_source_hash_failure_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'bad.py';source.write_text('raise RuntimeError("must not import")')
            with self.assertRaisesRegex(ValueError,'source hash'):
                evaluator.make_extractor(source,Path(directory)/'missing.pth','cpu')


if __name__=='__main__':unittest.main()
