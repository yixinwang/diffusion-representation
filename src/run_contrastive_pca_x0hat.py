#!/usr/bin/env python3
"""run_contrastive_pca_x0hat.py

Case 3: cPCA on denoised outputs.

Instead of raw x_t, use the denoised estimator

  x0_hat(t) \approx E[x0 | x_t]

For DDPM parameterized by predicted noise eps_theta(x_t,t), the common x0 estimator is:

  x0_hat = (x_t - sqrt(1-\bar\alpha_t) * eps_theta(x_t,t)) / sqrt(\bar\alpha_t)

We take:
- target = covariance of x0_hat(t_target) (moderate noise)
- background = covariance of x0_hat(t_background) (higher noise)

This script prints PCA explained-variance diagnostics for *each* dataset and also
runs contrastive PCA via:

  M(\alpha) = C_target - \alpha C_background

No SciPy / scikit-learn.

Example
-------
python src/run_contrastive_pca_x0hat.py \
  --ckpt results/steps_00023919.pt --use_ema \
  --n 1024 --t_target 50 --t_background 400 \
  --alpha 1.0 --components 20

Alpha sweep:
  --alpha_list 0,0.1,0.3,1,3,10

Multiple timestep pairs:
  --t_pairs 50,400;100,400;200,400
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion_checkpoint import load_mnist_diffusion_checkpoint


def _maybe_save_eig_decay_png(
    *,
    evals_target: np.ndarray,
    evals_background: np.ndarray,
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    """Save a simple eigenvalue-decay plot as a PNG.

    Uses matplotlib if available; otherwise prints a note and skips.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not import matplotlib for plotting ({e}); skipping PNG save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    k = int(min(len(evals_target), len(evals_background)))
    x = np.arange(1, k + 1, dtype=np.int64)

    plt.figure(figsize=(6.0, 4.0))
    plt.semilogy(x, np.maximum(evals_target[:k], 1e-30), label="target")
    plt.semilogy(x, np.maximum(evals_background[:k], 1e-30), label="background")
    plt.xlabel("principal component index")
    plt.ylabel("eigenvalue (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue decay plot: {path}")


def _make_mnist(n: int, root: str = "./mnist_data", image_size: int = 28) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ds = MNIST(root=root, train=True, download=True, transform=tfm)
    n = min(int(n), len(ds))
    return torch.stack([ds[i][0] for i in range(n)], dim=0)  # (N,1,H,W)


def _parse_pairs(spec: str) -> List[Tuple[int, int]]:
    """Parse "tA,tB; tA,tB; ..."."""
    spec = (spec or "").strip()
    if not spec:
        return []
    pairs: List[Tuple[int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad --t_pairs entry: '{chunk}' (expected 'tA,tB')")
        a, b = int(parts[0]), int(parts[1])
        if a == b:
            raise ValueError("tA and tB must differ")
        if a < 0 or b < 0:
            raise ValueError("Timesteps must be non-negative")
        pairs.append((a, b))
    return pairs


def _parse_floats_list(spec: str) -> List[float]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: List[float] = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


@torch.no_grad()
def _forward_xt(diffusion_model, x0: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
    t_batch = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    a = diffusion_model.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    s = diffusion_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    return a * x0 + s * eps


@torch.no_grad()
def _pred_eps(diffusion_model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    t_batch = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
    if hasattr(diffusion_model, "model") and callable(getattr(diffusion_model, "model")):
        return diffusion_model.model(x_t, t_batch)
    if callable(diffusion_model):
        return diffusion_model(x_t, t_batch)
    raise TypeError("diffusion_model must be MNISTDiffusion-like or a callable")


@torch.no_grad()
def x0_hat_vectors(
    *,
    score_model,
    x0: torch.Tensor,
    eps0: torch.Tensor,
    t: int,
    clip: bool,
) -> torch.Tensor:
    """Return flattened x0_hat(t) vectors (N,d) for timestep t."""
    x_t = _forward_xt(score_model, x0, int(t), eps0)
    eps_pred = _pred_eps(score_model, x_t, int(t))

    N = x_t.shape[0]
    t_batch = torch.full((N,), int(t), device=x_t.device, dtype=torch.long)
    a = score_model.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(N, 1, 1, 1).clamp_min(1e-12)
    s = score_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(N, 1, 1, 1)

    x0_hat = (x_t - s * eps_pred) / a
    if bool(clip):
        x0_hat = x0_hat.clamp(-1.0, 1.0)
    return x0_hat.view(N, -1)


def cov_centered(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (X_centered, covariance) with covariance shape (d,d)."""
    if X.ndim != 2:
        raise ValueError("X must be (N,d)")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples")
    Xc = X - X.mean(dim=0, keepdim=True)
    C = (Xc.T @ Xc) / float(Xc.shape[0] - 1)
    return Xc, C


def pca_evr_report(X: torch.Tensor, *, components: int, name: str) -> dict:
    """Return PCA spectrum diagnostics from the covariance eigenvalues."""
    _, C = cov_centered(X)
    evals = torch.linalg.eigvalsh(C)  # ascending
    evals = torch.clamp(evals, min=0.0).flip(0)  # descending

    total = torch.sum(evals)
    if float(total) <= 0.0:
        evr = torch.zeros((int(min(components, evals.numel())),), device=evals.device)
    else:
        evr_full = evals / (total + 1e-12)
        evr = evr_full[: int(min(components, evals.numel()))]

    cev = torch.cumsum(evr, dim=0)
    v = evr
    pr = float((v.sum() ** 2) / (torch.sum(v * v) + 1e-12))

    def k_at(frac: float) -> int | None:
        idx = torch.where(cev >= float(frac))[0]
        return int(idx[0].item() + 1) if idx.numel() else None

    return {
        "name": str(name),
        "feature_dim": int(C.shape[0]),
        "total_var": float(torch.trace(C).detach().cpu().item()),
        "evals": evals[: int(min(components, evals.numel()))].detach().cpu().numpy().astype(np.float64),
        "evr": evr.detach().cpu().numpy().astype(np.float64),
        "cev": cev.detach().cpu().numpy().astype(np.float64),
        "k@0.90": k_at(0.90),
        "k@0.95": k_at(0.95),
        "k@0.99": k_at(0.99),
        "participation_ratio": pr,
    }


def contrastive_pca(X_target: torch.Tensor, X_background: torch.Tensor, *, alpha: float, components: int) -> dict:
    Xc_t, C_t = cov_centered(X_target)
    Xc_b, C_b = cov_centered(X_background)
    if C_t.shape != C_b.shape:
        raise ValueError("Target/background feature dims differ")

    M = C_t - float(alpha) * C_b
    evals, evecs = torch.linalg.eigh(M)  # ascending
    evals = evals.flip(0)
    evecs = evecs.flip(1)

    k = int(min(int(components), evecs.shape[1]))
    W = evecs[:, :k]
    lam = evals[:k]

    Yt = Xc_t @ W
    Yb = Xc_b @ W

    var_t = torch.var(Yt, dim=0, unbiased=True)
    var_b = torch.var(Yb, dim=0, unbiased=True)
    contrast = var_t - float(alpha) * var_b

    total_var_t = torch.trace(C_t).clamp_min(0.0)
    total_var_b = torch.trace(C_b).clamp_min(0.0)
    evr_t = var_t / (total_var_t + 1e-12)
    evr_b = var_b / (total_var_b + 1e-12)
    cev_t = torch.cumsum(evr_t, dim=0)
    cev_b = torch.cumsum(evr_b, dim=0)

    def k_at(cev: torch.Tensor, frac: float) -> int | None:
        idx = torch.where(cev >= float(frac))[0]
        return int(idx[0].item() + 1) if idx.numel() else None

    return {
        "alpha": float(alpha),
        "components": int(k),
        "eigenvalues": lam.detach().cpu().numpy().astype(np.float64),
        "var_target": var_t.detach().cpu().numpy().astype(np.float64),
        "var_background": var_b.detach().cpu().numpy().astype(np.float64),
        "total_var_target": float(total_var_t.detach().cpu().item()),
        "total_var_background": float(total_var_b.detach().cpu().item()),
        "evr_target": evr_t.detach().cpu().numpy().astype(np.float64),
        "evr_background": evr_b.detach().cpu().numpy().astype(np.float64),
        "cev_target": cev_t.detach().cpu().numpy().astype(np.float64),
        "cev_background": cev_b.detach().cpu().numpy().astype(np.float64),
        "k@0.90_target": k_at(cev_t, 0.90),
        "k@0.95_target": k_at(cev_t, 0.95),
        "k@0.99_target": k_at(cev_t, 0.99),
        "k@0.90_background": k_at(cev_b, 0.90),
        "k@0.95_background": k_at(cev_b, 0.95),
        "k@0.99_background": k_at(cev_b, 0.99),
        "contrast_variance": contrast.detach().cpu().numpy().astype(np.float64),
        "W": W.detach().cpu().numpy().astype(np.float32),
    }


def _print_pca(rep: dict) -> None:
    print(f"\n=== {rep['name']} ===")
    print(f"feature_dim = {rep['feature_dim']} | total_var = {rep['total_var']:.6e}")
    print(f"k@0.90 = {rep['k@0.90']}, k@0.95 = {rep['k@0.95']}, k@0.99 = {rep['k@0.99']}")
    print(f"participation_ratio ≈ {rep['participation_ratio']:.2f}")
    evr10 = [float(x) for x in rep["evr"][:10]]
    cev10 = [float(x) for x in rep["cev"][:10]]
    evals10 = [float(x) for x in rep["evals"][:10]]
    print(f"first 10 evals: {evals10}")
    print(f"first 10 EVR:   {evr10}")
    print(f"first 10 CEV:   {cev10}")


def _print_cpca(rep: dict, *, header: str) -> None:
    print(f"\n=== {header} ===")
    print(f"alpha = {rep['alpha']}")
    print(f"total_var_target = {rep['total_var_target']:.6e} | total_var_background = {rep['total_var_background']:.6e}")
    print(
        "k@0.90 = {k90}, k@0.95 = {k95}, k@0.99 = {k99}  (target, along cPCs)".format(
            k90=rep.get("k@0.90_target"),
            k95=rep.get("k@0.95_target"),
            k99=rep.get("k@0.99_target"),
        )
    )
    print(
        "k@0.90 = {k90}, k@0.95 = {k95}, k@0.99 = {k99}  (background, along cPCs)".format(
            k90=rep.get("k@0.90_background"),
            k95=rep.get("k@0.95_background"),
            k99=rep.get("k@0.99_background"),
        )
    )

    ev = rep["eigenvalues"]
    vt = rep["var_target"]
    vb = rep["var_background"]
    cv = rep["contrast_variance"]
    et = rep["evr_target"]
    eb = rep["evr_background"]
    ct = rep["cev_target"]
    cb = rep["cev_background"]

    for i in range(int(rep["components"])):
        print(
            f"  cPC{i+1:02d}: eig={ev[i]: .4e}"
            f" | var_target={vt[i]: .4e} (EVR={et[i]:.4e}, CEV={ct[i]:.4e})"
            f" | var_bg={vb[i]: .4e} (EVR={eb[i]:.4e}, CEV={cb[i]:.4e})"
            f" | var_target-alpha*var_bg={cv[i]: .4e}"
        )


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mnist_root", type=str, default="./mnist_data")

    p.add_argument("--t_target", type=int, default=50)
    p.add_argument("--t_background", type=int, default=400)
    p.add_argument("--t_pairs", type=str, default="", help="Optional: 'tA,tB;tA,tB;...' overrides --t_target/--t_background")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--alpha_list", type=str, default="", help="Optional sweep: '0,0.1,0.3,1,3,10' overrides --alpha")

    p.add_argument("--components", type=int, default=10, help="Number of cPCs to print")
    p.add_argument("--pca_components", type=int, default=50, help="How many PCA eigenvalues/EVR entries to compute/print")
    p.add_argument("--clip", action="store_true", help="Optionally clamp x0_hat to [-1,1]")
    p.add_argument("--save_png", action="store_true", help="Save eigenvalue decay plots as PNG")
    p.add_argument("--out", type=str, default="", help="Optional output directory for npz")

    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    _, score_model = load_mnist_diffusion_checkpoint(args.ckpt, device=args.device, use_ema=args.use_ema)
    score_model = score_model.to(args.device)
    score_model.eval()

    timesteps_max = int(getattr(score_model, "timesteps", 1000))

    pairs = _parse_pairs(args.t_pairs) if str(args.t_pairs).strip() else [(int(args.t_target), int(args.t_background))]
    for (ta, tb) in pairs:
        if ta >= timesteps_max or tb >= timesteps_max:
            raise SystemExit(f"Timesteps must be < {timesteps_max}; got ({ta},{tb})")

    alphas = _parse_floats_list(args.alpha_list) if str(args.alpha_list).strip() else [float(args.alpha)]
    if any(a < 0 for a in alphas):
        raise SystemExit("alpha must be non-negative")

    x0 = _make_mnist(args.n, root=args.mnist_root, image_size=int(getattr(score_model, "image_size", 28))).to(args.device)
    eps0 = torch.randn_like(x0)

    needed_ts = sorted(set([t for pair in pairs for t in pair]))
    x0hats = {int(t): x0_hat_vectors(score_model=score_model, x0=x0, eps0=eps0, t=int(t), clip=bool(args.clip)) for t in needed_ts}

    print("\n==============================")
    print("cPCA on denoised x0_hat(t)")
    print("------------------------------")
    print(f"ckpt = {args.ckpt}")
    print(f"n = {int(args.n)} | seed = {int(args.seed)}")
    print(f"clip = {bool(args.clip)}")
    print(f"pairs = {pairs}")
    print(f"alphas = {alphas}")
    print(f"pca_components = {int(args.pca_components)} | cpca_components = {int(args.components)}")
    print("==============================")

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for (ta, tb) in pairs:
        Xa = x0hats[int(ta)]
        Xb = x0hats[int(tb)]

        rep_pca_t = pca_evr_report(Xa, components=int(args.pca_components), name=f"PCA(x0_hat(t={ta}))")
        rep_pca_b = pca_evr_report(Xb, components=int(args.pca_components), name=f"PCA(x0_hat(t={tb}))")

        _print_pca(rep_pca_t)
        _print_pca(rep_pca_b)

        if bool(args.save_png):
            out_dir = str(args.out).strip() or "."
            _maybe_save_eig_decay_png(
                evals_target=rep_pca_t["evals"],
                evals_background=rep_pca_b["evals"],
                out_dir=out_dir,
                filename=f"eig_decay_x0hat_t{ta}_vs_t{tb}.png",
                title=f"x0_hat PCA eigenvalue decay: t={ta} (target) vs t={tb} (background)",
            )

        for alpha in alphas:
            rep = contrastive_pca(Xa, Xb, alpha=float(alpha), components=int(args.components))
            _print_cpca(rep, header=f"cPCA: t_target={ta} vs t_background={tb}")

            if args.out:
                out_path = os.path.join(args.out, f"cpca_x0hat_t{ta}_vs_t{tb}_alpha{alpha}.npz")
                np.savez_compressed(
                    out_path,
                    t_target=np.array([ta], dtype=np.int64),
                    t_background=np.array([tb], dtype=np.int64),
                    alpha=np.array([float(alpha)], dtype=np.float64),
                    clip=np.array([int(bool(args.clip))], dtype=np.int64),
                    pca_target_evals=rep_pca_t["evals"],
                    pca_target_evr=rep_pca_t["evr"],
                    pca_target_cev=rep_pca_t["cev"],
                    pca_bg_evals=rep_pca_b["evals"],
                    pca_bg_evr=rep_pca_b["evr"],
                    pca_bg_cev=rep_pca_b["cev"],
                    eigenvalues=rep["eigenvalues"],
                    var_target=rep["var_target"],
                    var_background=rep["var_background"],
                    total_var_target=np.array([rep["total_var_target"]], dtype=np.float64),
                    total_var_background=np.array([rep["total_var_background"]], dtype=np.float64),
                    evr_target=rep["evr_target"],
                    evr_background=rep["evr_background"],
                    cev_target=rep["cev_target"],
                    cev_background=rep["cev_background"],
                    contrast_variance=rep["contrast_variance"],
                    W=rep["W"],
                )
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
