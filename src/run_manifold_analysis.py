#!/usr/bin/env python3
"""
Run LTSA + Diffusion Maps + intrinsic-dimension diagnostics on MNIST diffusion states.

Guarantees:
- ALWAYS prints numeric intrinsic-dimension summaries
- NO matplotlib dependency (safe with NumPy 2.x on HPC)
- CSV curves always saved if --save_plots is given
- PNG plots only if Pillow is available
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion_checkpoint import load_mnist_diffusion_checkpoint
from knn_metrics import knn_graph, pairwise_sq_dists
from ltsa import ltsa
from diffusion_map import diffusion_map, estimate_intrinsic_dim_from_spectrum
from intrinsic_dim import intrinsic_dim_report


# ---------------------------------------------------------------------
# Helpers for explained-variance / residual-energy curves
# ---------------------------------------------------------------------

def _ltsa_explained_variance(eigvals_full: np.ndarray) -> np.ndarray:
    ev = np.asarray(eigvals_full, dtype=np.float64)
    ev = np.sort(ev)
    if ev.size <= 1:
        return np.array([])
    ev = ev[1:]  # drop constant mode
    total = ev.sum()
    if total <= 0:
        return np.zeros_like(ev)
    cum = np.cumsum(ev)
    return 1.0 - cum / total


def _diffmap_energy_decay(evals: np.ndarray) -> np.ndarray:
    mu = np.asarray(evals, dtype=np.float64)
    mu = np.sort(mu)[::-1]
    if mu.size <= 1:
        return np.array([])
    mu = mu[1:]  # drop trivial eigenvalue
    e = mu * mu
    total = e.sum()
    if total <= 0:
        return np.zeros_like(e)
    cum = np.cumsum(e)
    return 1.0 - cum / total


def _dims_at_thresholds(decay: np.ndarray | None, thresholds=(0.20, 0.10, 0.05)) -> dict:
    if decay is None:
        return {thr: None for thr in thresholds}
    decay = np.asarray(decay, dtype=np.float64)
    if decay.size == 0:
        return {thr: None for thr in thresholds}
    out = {}
    for thr in thresholds:
        idx = np.where(decay <= float(thr))[0]
        out[float(thr)] = int(idx[0] + 1) if idx.size > 0 else None
    return out


def _save_decay_plot(curves: dict, max_dim: int, outpath: str, title: str):
    """
    Save residual-energy curves.
    - CSV always
    - PNG only if Pillow is available
    """
    series = []
    for name, c in curves.items():
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        if c.size == 0:
            continue
        m = min(int(max_dim), c.size)
        xs = np.arange(1, m + 1)
        ys = c[:m]
        series.append((name, xs, ys))

    if not series:
        return

    # Always save CSV
    csv_path = os.path.splitext(outpath)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,k,residual_energy\n")
        for name, xs, ys in series:
            for k, y in zip(xs, ys):
                f.write(f"{name},{k},{y}\n")

    # Try Pillow for PNG
    try:
        from PIL import Image, ImageDraw
        W, H = 900, 520
        pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 60
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        x0, y0 = pad_l, H - pad_b
        x1, y1 = W - pad_r, pad_t
        dr.line((x0, y0, x1, y0), fill=(0, 0, 0), width=2)
        dr.line((x0, y0, x0, y1), fill=(0, 0, 0), width=2)

        max_k = max(int(xs[-1]) for _, xs, _ in series)

        def to_xy(k, y):
            px = x0 + (x1 - x0) * (k - 1) / max(1, (max_k - 1))
            yy = min(1.0, max(0.0, float(y)))
            py = y0 - (y0 - y1) * yy
            return px, py

        shades = [(30, 30, 30), (100, 100, 100), (170, 170, 170)]
        for i, (_, xs, ys) in enumerate(series):
            pts = [to_xy(int(k), float(y)) for k, y in zip(xs, ys)]
            dr.line(pts, fill=shades[i % len(shades)], width=3)

        dr.text((pad_l, 10), title[:80], fill=(0, 0, 0))
        dr.text((W // 2 - 90, H - 40), "Embedding dimension k", fill=(0, 0, 0))
        dr.text((10, H // 2 - 20), "Residual energy", fill=(0, 0, 0))

        img.save(outpath)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def make_mnist(n: int, root: str = "./mnist_data", image_size: int = 28) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = MNIST(root=root, train=True, download=True, transform=tfm)
    n = min(n, len(ds))
    return torch.stack([ds[i][0] for i in range(n)], dim=0)


@torch.no_grad()
def make_xt(diffusion, x0: torch.Tensor, t: int) -> torch.Tensor:
    device = x0.device
    t_batch = torch.full((x0.shape[0],), int(t), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    a = diffusion.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    s = diffusion.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    return a * x0 + s * noise


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--metric", type=str, default="euclidean",
                   choices=["euclidean", "score_riemannian_diag", "initial_noise"])
    p.add_argument("--t", type=int, default=200)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--candidate_k", type=int, default=200)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out_dim", type=int, default=2)
    p.add_argument("--dm_dim", type=int, default=8)
    p.add_argument("--save_npz", type=str, default="")
    p.add_argument("--save_plots", type=str, default="")
    p.add_argument("--max_curve_dim", type=int, default=100)
    args = p.parse_args()

    bundle, score_model = load_mnist_diffusion_checkpoint(
        args.ckpt, device=args.device, use_ema=args.use_ema
    )

    device = torch.device(args.device)
    x0 = make_mnist(args.n).to(device)

    if args.metric in ("euclidean", "score_riemannian_diag"):
        x_work = make_xt(score_model, x0, args.t)
    else:
        x_work = x0

    cand = None if args.candidate_k <= 0 else args.candidate_k
    nn_idx, nn_d2 = knn_graph(
        x_work, k=args.k, metric=args.metric,
        diffusion_model=score_model, t=args.t,
        candidate_k=cand, device=str(device)
    )

    X_flat = x_work.view(x_work.shape[0], -1).cpu().numpy()

    ltsa_res = ltsa(X_flat, nn_idx, out_dim=args.out_dim)
    id_res = intrinsic_dim_report(X_flat, nn_idx, nn_d2)

    D2 = pairwise_sq_dists(X_flat)
    dm_res = diffusion_map(D2, out_dim=args.dm_dim, alpha=0.5, t=1)

    # -----------------------------------------------------------------
    # ALWAYS print numeric summaries
    # -----------------------------------------------------------------

    ltsa_decay = None
    if ltsa_res.diagnostics and "eigvals_full" in ltsa_res.diagnostics:
        ltsa_decay = _ltsa_explained_variance(ltsa_res.diagnostics["eigvals_full"])

    dm_decay = _diffmap_energy_decay(dm_res.evals)

    print("\n=== Intrinsic dimension summary ===")
    print(f"Metric = {args.metric}, t = {args.t}, n = {args.n}")

    print("\nLTSA residual-energy cutoffs:")
    for thr, k in _dims_at_thresholds(ltsa_decay).items():
        print(f"  residual <= {thr:0.2f} at k = {k}")

    print("\nDiffusion map residual-energy cutoffs:")
    for thr, k in _dims_at_thresholds(dm_decay).items():
        print(f"  residual <= {thr:0.2f} at k = {k}")

    mu = np.asarray(dm_res.evals, dtype=np.float64)
    if mu.size > 1:
        mu = np.sort(mu)[::-1][1:]
        pr = (mu @ mu) ** 2 / ((mu ** 2) @ (mu ** 2))
        print(f"\nDiffusion map participation-ratio dim ≈ {pr:.2f}")

    print("\nLocal kNN-based estimates:")
    print(f"  Levina–Bickel MLE dim ≈ {id_res.lb_mle:.2f}")
    print(f"  Local PCA participation ratio ≈ {id_res.local_pca_pr:.2f}")
    print("=================================\n")

    # -----------------------------------------------------------------
    # Optional saving
    # -----------------------------------------------------------------

    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
        _save_decay_plot(
            {"LTSA": ltsa_decay},
            args.max_curve_dim,
            os.path.join(args.save_plots, "ltsa_residual_energy.png"),
            f"LTSA residual energy (metric={args.metric}, t={args.t})",
        )
        _save_decay_plot(
            {"Diffusion map": dm_decay},
            args.max_curve_dim,
            os.path.join(args.save_plots, "diffmap_residual_energy.png"),
            f"Diffusion map residual energy (metric={args.metric}, t={args.t})",
        )

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            ltsa_decay=ltsa_decay,
            dm_decay=dm_decay,
            nn_idx=nn_idx,
            nn_d2=nn_d2,
            intrinsic_dim=asdict(id_res),
            run_args=vars(args),
        )


if __name__ == "__main__":
    main()
