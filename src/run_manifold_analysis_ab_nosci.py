#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion_checkpoint import load_mnist_diffusion_checkpoint
from knn_metrics import knn_graph, score_diagonal_metric
from intrinsic_dim import intrinsic_dim_report

from ltsa_riemannian import ltsa, ltsa_riemannian
from graph_embedding_torch import (
    laplacian_eigenmap_from_knn_torch,
    diffusion_map_knn_torch,
    participation_ratio_from_evals,
)


def make_mnist(n: int, root: str = "./mnist_data", image_size: int = 28) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ds = MNIST(root=root, train=True, download=True, transform=tfm)
    n = min(int(n), len(ds))
    x = torch.stack([ds[i][0] for i in range(n)], dim=0)  # (N,1,28,28)
    return x


def _ltsa_residual_energy_from_eigs(eigvals_full: np.ndarray) -> np.ndarray:
    """Residual energy fraction after keeping first k nontrivial eigenvalues (drops the smallest)."""
    ev = np.asarray(eigvals_full, dtype=np.float64)
    ev = np.sort(ev)
    if ev.size <= 1:
        return np.array([])
    ev = ev[1:]  # drop constant mode
    ev = np.clip(ev, 0.0, None)
    tot = ev.sum()
    if tot <= 0:
        return np.ones_like(ev)
    cum = np.cumsum(ev)
    return 1.0 - cum / tot


def _dims_at_thresholds(residual_curve: np.ndarray, thresholds=(0.20, 0.10, 0.05)) -> dict:
    residual_curve = np.asarray(residual_curve, dtype=np.float64)
    out = {}
    if residual_curve.size == 0:
        for thr in thresholds:
            out[thr] = None
        return out
    for thr in thresholds:
        idx = np.where(residual_curve <= float(thr))[0]
        out[thr] = int(idx[0] + 1) if idx.size else int(residual_curve.size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "score_riemannian_diag", "initial_noise"],
    )
    p.add_argument("--t", type=int, default=20)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--candidate_k", type=int, default=200)

    p.add_argument("--out_dim", type=int, default=10)
    p.add_argument("--do_ltsa_riemannian", action="store_true")
    p.add_argument("--do_graph_embed", action="store_true")
    p.add_argument("--graph_dim", type=int, default=10)

    args = p.parse_args()

    device = torch.device(args.device)

    bundle, score_model = load_mnist_diffusion_checkpoint(
        args.ckpt, device=args.device, use_ema=args.use_ema
    )
    score_model = score_model.to(device)

    # ---- data ----
    x0 = make_mnist(args.n).to(device)  # (N,1,28,28)

    # x_work depends on metric:
    # - euclidean / score metric: analyze x_t (noisy) because score metric is defined at x_t
    # - initial_noise: analyze x0 but neighbor metric uses recovered z0
    if args.metric in ("euclidean", "score_riemannian_diag"):
        # same helper pattern used in your original script: construct x_t via bundle (if present)
        # If bundle has precomputed schedules, use them; otherwise fall back to simple DDPM forward.
        # Most of your codebase used bundle diffusion schedules; try them first.
        # Use diffusion schedules from the loaded diffusion model (MNISTDiffusion)
        # score_model is an MNISTDiffusion instance (either EMA or train model).
        if hasattr(score_model, "sqrt_alphas_cumprod") and hasattr(score_model, "sqrt_one_minus_alphas_cumprod"):
            t_batch = torch.full((x0.shape[0],), int(args.t), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            a = score_model.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
            s = score_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
            x_work = a * x0 + s * noise
        else:
            x_work = x0
            print("[warn] score_model missing diffusion schedules; using x0 as x_work")
    else:
        x_work = x0

    cand = None if args.candidate_k <= 0 else int(args.candidate_k)

    # ---- kNN graph (this is where metric is applied) ----
    nn_idx, nn_d2 = knn_graph(
        x_work,
        k=int(args.k),
        metric=args.metric,
        diffusion_model=score_model,
        t=int(args.t),
        candidate_k=cand,
        device=str(device),
    )

    X_flat = x_work.view(x_work.shape[0], -1).detach().cpu().numpy()

    # ---- baseline LTSA (Euclidean in ambient coords; useful as reference) ----
    ltsa_res = ltsa(X_flat, nn_idx, out_dim=int(args.out_dim))
    ltsa_decay = None
    if ltsa_res.diagnostics and "eigvals_full" in ltsa_res.diagnostics:
        ltsa_decay = _ltsa_residual_energy_from_eigs(ltsa_res.diagnostics["eigvals_full"])
    else:
        ltsa_decay = np.array([])

    # ---- Option A: Riemannian (metric-whitened) LTSA ----
    ltsa_r_decay = None
    if args.do_ltsa_riemannian:
        if args.metric != "score_riemannian_diag":
            print("[warn] --do_ltsa_riemannian is most meaningful with --metric score_riemannian_diag")

        gdiag_t = score_diagonal_metric(score_model, x_work, int(args.t))
        gdiag = gdiag_t.detach().cpu().numpy()
        ltsa_r = ltsa_riemannian(X_flat, nn_idx, gdiag, out_dim=int(args.out_dim))
        ltsa_r_decay = _ltsa_residual_energy_from_eigs(ltsa_r.diagnostics["eigvals_full"])

    # ---- kNN intrinsic dimension report ----
    id_res = intrinsic_dim_report(X_flat, nn_idx, nn_d2)

    # ---- Option B: graph-only embeddings (no SciPy) ----
    graph_le = None
    graph_dm = None
    if args.do_graph_embed:
        graph_le = laplacian_eigenmap_from_knn_torch(nn_idx, nn_d2, out_dim=int(args.graph_dim))
        graph_dm = diffusion_map_knn_torch(nn_idx, nn_d2, out_dim=int(args.graph_dim))

    # ---- print ----
    print("\n=== Intrinsic dimension summary ===")
    print(f"Metric = {args.metric}, t = {args.t}, n = {args.n}, kNN k = {args.k}")

    print("\nLTSA residual-energy cutoffs:")
    for thr, kdim in _dims_at_thresholds(ltsa_decay).items():
        print(f"  residual <= {thr:0.2f} at k = {kdim}")

    if ltsa_r_decay is not None:
        print("\nRiemannian LTSA (diag metric) residual-energy cutoffs:")
        for thr, kdim in _dims_at_thresholds(ltsa_r_decay).items():
            print(f"  residual <= {thr:0.2f} at k = {kdim}")

    if graph_dm is not None:
        pr = participation_ratio_from_evals(graph_dm.evals, drop_first=True)
        print(f"\nGraph diffusion (torch, symmetric) participation-ratio dim ≈ {pr:.2f}")

    if graph_le is not None:
        ev = np.asarray(graph_le.evals, dtype=np.float64)
        print("\nGraph Laplacian smallest eigenvalues:")
        print([float(x) for x in ev[:6]])

    print("\nLocal kNN-based estimates:")
    print(f"  Levina–Bickel MLE dim ≈ {float(id_res.lb_mle):.2f}")
    print(f"  Local PCA participation ratio ≈ {float(id_res.local_pca_pr):.2f}")
    print("=================================\n")


if __name__ == "__main__":
    main()
