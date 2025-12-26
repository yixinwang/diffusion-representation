"""
Checkpoint loading utilities for the MNISTDiffusion codebase (model.py/train_mnist.py).

Matches the checkpoint format saved by save_checkpoint() in train_mnist.py:
    {
      "model": ...,
      "model_ema": ...,
      "optimizer": ...,
      "scheduler": ...,
      "epoch": int,
      "global_steps": int,
      "args": dict,
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os
import torch

from model import MNISTDiffusion
from utils import ExponentialMovingAverage


@dataclass
class LoadedDiffusion:
    model: MNISTDiffusion
    model_ema: ExponentialMovingAverage
    ckpt: Dict[str, Any]
    device: torch.device

    @property
    def score_model(self) -> MNISTDiffusion:
        """Return the recommended score model (EMA module)."""
        return self.model_ema.module

    @property
    def train_model(self) -> MNISTDiffusion:
        """Return the raw training model."""
        return self.model


def load_mnist_diffusion_checkpoint(
    ckpt_path: str,
    *,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
    override_args: Optional[Dict[str, Any]] = None,
) -> Tuple[LoadedDiffusion, MNISTDiffusion]:
    """
    Load MNISTDiffusion + EMA model from a checkpoint and return (bundle, model_for_scoring).

    Parameters
    ----------
    ckpt_path:
        Path to .pt checkpoint produced by train_mnist.py
    device:
        'cpu', 'cuda', or torch.device
    use_ema:
        If True, return EMA module as `model_for_scoring`; else return training model.
    override_args:
        Optional overrides for model hyperparams if you need to force them.

    Returns
    -------
    bundle:
        LoadedDiffusion with model, model_ema, ckpt dict
    model_for_scoring:
        MNISTDiffusion instance (either EMA module or raw training model)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    args = dict(ckpt.get("args", {}) or {})
    if override_args:
        args.update(override_args)

    # Defaults match train_mnist.py
    timesteps = int(args.get("timesteps", 1000))
    base_dim = int(args.get("model_base_dim", 64))
    image_size = int(args.get("image_size", 28))
    in_channels = int(args.get("in_channels", 1))
    dim_mults = args.get("dim_mults", [2, 4])

    model = MNISTDiffusion(
        timesteps=timesteps,
        image_size=image_size,
        in_channels=in_channels,
        base_dim=base_dim,
        dim_mults=dim_mults,
    ).to(device)

    # EMA wrapper: decay value is irrelevant for loading; only structure matters.
    model_ema = ExponentialMovingAverage(model, device=str(device), decay=0.0)

    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    model.eval()
    model_ema.eval()

    bundle = LoadedDiffusion(model=model, model_ema=model_ema, ckpt=ckpt, device=device)
    model_for_scoring = bundle.score_model if use_ema else bundle.train_model
    return bundle, model_for_scoring
