#!/usr/bin/env python3
"""run_pca_trajectory_pairs.py

Compute PCA explained-variance on diffusion *trajectories* and on *pairwise trajectory differences*.

You provide a trajectory representation z(x_i, t) for each sample i and timestep t.
This script supports two input modes:

1) Load precomputed trajectories from .npz via --z_npz.
   Expected shapes (any of these are accepted):
     - (T, N, D)
     - (N, T, D)
   Key can be 'z', 'Z', or (if only one array exists) that array.

2) Generate trajectories from an MNIST diffusion checkpoint via --ckpt.
   We sample MNIST x0, then construct x_t using the forward diffusion schedule.
   You can choose z_kind:
     - x_t: flattened x_t
     - eps_pred: flattened model-predicted noise eps_theta(x_t,t)
     - score_pred: flattened score ≈ -eps_theta(x_t,t) / sqrt(1-alpha_bar_t)

For each sample i, we form a single trajectory vector by concatenating z over timesteps:
  z_traj(x_i) = concat_t z(x_i,t)  -> shape (N, T*D)

Then we compare:
  A) PCA on z_traj(x_i)
  B) PCA on pairwise differences Δ_ij = z_traj(x_i) - z_traj(x_j)

Notes
-----
- Pairwise differences are O(N^2). Use --max_pairs to subsample.
- PCA is computed using torch/numpy linear algebra (no SciPy / scikit-learn).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from diffusion_checkpoint import load_mnist_diffusion_checkpoint


def _parse_t_list(spec: str, timesteps_max: int | None = None) -> np.ndarray:
    """Parse timestep list.

    Supported:
      - Comma list: "0,50,100"
      - Range: "start:stop:step" (python-like, stop exclusive)
        e.g. "0:1000:50"
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("--t_list cannot be empty")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("Range format must be start:stop:step")
        start, stop, step = (int(p) for p in parts)
        if step <= 0:
            raise ValueError("step must be positive")
        ts = np.arange(start, stop, step, dtype=np.int64)
    else:
        ts = np.array([int(x) for x in spec.split(",") if x.strip() != ""], dtype=np.int64)

    if ts.size == 0:
        raise ValueError("Parsed empty t_list")
    if (ts < 0).any():
        raise ValueError("t_list must be non-negative")
    if timesteps_max is not None and (ts >= int(timesteps_max)).any():
        raise ValueError(f"t_list contains t >= timesteps ({timesteps_max})")

    # unique + preserve order
    seen = set()
    out = []
    for t in ts.tolist():
        if t not in seen:
            out.append(t)
            seen.add(t)
    return np.array(out, dtype=np.int64)


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


@torch.no_grad()
def _forward_xt_from_fixed_eps(diffusion_model, x0: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
    device = x0.device
    t_batch = torch.full((x0.shape[0],), int(t), device=device, dtype=torch.long)
    a = diffusion_model.sqrt_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    s = diffusion_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
    return a * x0 + s * eps


@torch.no_grad()
def _pred_eps(diffusion_model, x_t: torch.Tensor, t: int) -> torch.Tensor:
    t_batch = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
    # MNISTDiffusion has .model, but allow callable too
    if hasattr(diffusion_model, "model") and callable(getattr(diffusion_model, "model")):
        return diffusion_model.model(x_t, t_batch)
    if callable(diffusion_model):
        return diffusion_model(x_t, t_batch)
    raise TypeError("diffusion_model must be MNISTDiffusion-like or a callable")


@torch.no_grad()
def generate_z_trajectory(
    *,
    ckpt: str,
    device: str,
    use_ema: bool,
    n: int,
    t_list: np.ndarray,
    z_kind: str,
    seed: int,
    mnist_root: str = "./mnist_data",
) -> np.ndarray:
    """Return z[t,i,d] as a numpy array with shape (T,N,D)."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    _, score_model = load_mnist_diffusion_checkpoint(ckpt, device=device, use_ema=use_ema)
    score_model = score_model.to(device)
    score_model.eval()

    x0 = _make_mnist(n, root=mnist_root, image_size=int(getattr(score_model, "image_size", 28))).to(device)
    eps0 = torch.randn_like(x0)

    T = int(t_list.shape[0])
    z_chunks = []
    for t in t_list.tolist():
        x_t = _forward_xt_from_fixed_eps(score_model, x0, int(t), eps0)
        if z_kind == "x_t":
            z_t = x_t.view(x_t.shape[0], -1)
        elif z_kind == "eps_pred":
            z_t = _pred_eps(score_model, x_t, int(t)).view(x_t.shape[0], -1)
        elif z_kind == "score_pred":
            eps_pred = _pred_eps(score_model, x_t, int(t))
            t_batch = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
            sigma = score_model.sqrt_one_minus_alphas_cumprod.gather(-1, t_batch).reshape(-1, 1, 1, 1)
            sigma = sigma.clamp_min(1e-12)
            z_t = (-eps_pred / sigma).view(x_t.shape[0], -1)
        else:
            raise ValueError(f"Unknown z_kind: {z_kind}")

        z_chunks.append(z_t.detach().cpu().numpy())

    Z = np.stack(z_chunks, axis=0)  # (T,N,D)
    assert Z.shape[0] == T
    return Z


def load_z_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = list(data.keys())
        if "z" in keys:
            Z = data["z"]
        elif "Z" in keys:
            Z = data["Z"]
        elif len(keys) == 1:
            Z = data[keys[0]]
        else:
            raise ValueError(f"Ambiguous npz keys: {keys}. Provide key 'z' or 'Z'.")
    else:
        Z = data

    Z = np.asarray(Z)
    if Z.ndim != 3:
        raise ValueError(f"Expected trajectory tensor with 3 dims, got shape {Z.shape}")
    return Z


def ensure_TND(Z: np.ndarray, *, assume: str = "TND") -> np.ndarray:
    """Normalize to (T,N,D).

    assume:
      - "TND": treat Z as (T,N,D)
      - "NTD": treat Z as (N,T,D)
      - "auto": choose based on first two dims (prefers smaller as T)
    """
    Z = np.asarray(Z)
    if Z.ndim != 3:
        raise ValueError("Z must be 3D")

    if assume == "TND":
        return Z
    if assume == "NTD":
        return np.transpose(Z, (1, 0, 2))
    if assume != "auto":
        raise ValueError("assume must be TND|NTD|auto")

    T0, N0, _ = Z.shape
    # Heuristic: T usually <= 1000, N might be 1e3-1e5.
    # If first dim looks like N and second like T, transpose.
    if T0 > N0:
        return np.transpose(Z, (1, 0, 2))
    return Z


def trajectory_matrix(Z_TND: np.ndarray, *, mode: str = "concat") -> np.ndarray:
    """Convert Z (T,N,D) to per-sample trajectory vectors (N, Dtraj)."""
    Z = np.asarray(Z_TND)
    if mode != "concat":
        raise ValueError("Only mode='concat' is implemented")
    T, N, D = Z.shape
    return np.transpose(Z, (1, 0, 2)).reshape(N, T * D)


def _sample_pairs(N: int, max_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random pairs (i<j) without constructing all pairs."""
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    # Note: This samples (i,j) with replacement; for large N that's fine.
    # Avoid i==j.
    i = rng.integers(0, N, size=max_pairs, dtype=np.int64)
    j = rng.integers(0, N, size=max_pairs, dtype=np.int64)
    mask = i != j
    if not np.all(mask):
        # resample collisions
        need = int((~mask).sum())
        while need > 0:
            ii = rng.integers(0, N, size=need, dtype=np.int64)
            jj = rng.integers(0, N, size=need, dtype=np.int64)
            ok = ii != jj
            i[~mask][: ok.sum()] = ii[ok]
            j[~mask][: ok.sum()] = jj[ok]
            mask = i != j
            need = int((~mask).sum())
    # order pairs so i<j (purely cosmetic / consistency)
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    return a, b


def _pca_report(X: np.ndarray, n_components: int, *, name: str) -> dict:
    """PCA explained-variance report without sklearn.

    Computes eigenvalues of the sample covariance and converts them to explained
    variance ratios (EVR). Uses a sample-space eigendecomposition of
    (X_centered X_centered^T)/(n-1) to avoid building a (D x D) covariance when
    the feature dimension D is large.
    """
    X = np.asarray(X, dtype=np.float32)
    n_components = int(n_components)
    n_components = min(n_components, min(X.shape[0], X.shape[1]))

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for PCA")

    Xt = torch.from_numpy(X)
    Xt = Xt - Xt.mean(dim=0, keepdim=True)

    G = (Xt @ Xt.T) / float(Xt.shape[0] - 1)
    evals = torch.linalg.eigvalsh(G)  # ascending
    evals = torch.clamp(evals, min=0.0).flip(0)  # descending
    total = torch.sum(evals)

    if float(total) <= 0.0:
        evr = np.zeros((n_components,), dtype=np.float64)
    else:
        evr_full = (evals / total).detach().cpu().numpy().astype(np.float64)
        evr = evr_full[:n_components]

    cev = np.cumsum(evr)

    def k_at(frac: float) -> int | None:
        idx = np.where(cev >= float(frac))[0]
        return int(idx[0] + 1) if idx.size else None

    # participation ratio on explained-variance spectrum
    # PR = (sum v)^2 / sum v^2 for v = explained variance ratios
    v = evr
    pr = float((v.sum() ** 2) / (np.sum(v * v) + 1e-12))

    out = {
        "name": name,
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "n_components": int(n_components),
        "cum_explained_variance": cev,
        "explained_variance_ratio": evr,
        "k@0.90": k_at(0.90),
        "k@0.95": k_at(0.95),
        "k@0.99": k_at(0.99),
        "pca_participation_ratio": pr,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()

    # Input
    p.add_argument("--z_npz", type=str, default="", help="Load trajectory tensor Z from npz")
    p.add_argument("--z_assume", type=str, default="auto", choices=["auto", "TND", "NTD"], help="Interpretation of loaded Z")

    p.add_argument("--ckpt", type=str, default="", help="Generate trajectories from checkpoint")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--n", type=int, default=256, help="Number of MNIST samples if generating")
    p.add_argument("--t_list", type=str, default="0:1000:50", help="Timesteps, e.g. 0,10,20 or 0:1000:50")
    p.add_argument("--z_kind", type=str, default="x_t", choices=["x_t", "eps_pred", "score_pred"], help="Representation z(x,t) when generating")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mnist_root", type=str, default="./mnist_data")

    # PCA + pairing
    p.add_argument("--traj_mode", type=str, default="concat", choices=["concat"])
    p.add_argument("--pca_components", type=int, default=200)

    p.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help=(
            "(Deprecated) Previously sampled explicit pairwise differences for PCA. "
            "Now difference-PCA is computed analytically (same EVR as PCA(z_traj))."
        ),
    )

    p.add_argument("--out", type=str, default="", help="Optional output directory (writes npz summaries)")

    args = p.parse_args()

    if bool(args.z_npz) == bool(args.ckpt):
        raise SystemExit("Provide exactly one of --z_npz or --ckpt")

    # Load / generate Z
    if args.z_npz:
        Z = load_z_npz(args.z_npz)
        Z = ensure_TND(Z, assume=args.z_assume)
        t_list = np.arange(Z.shape[0], dtype=np.int64)
        z_kind = "loaded"
    else:
        # need timesteps for validation
        bundle, score_model = load_mnist_diffusion_checkpoint(args.ckpt, device=args.device, use_ema=args.use_ema)
        timesteps_max = int(getattr(score_model, "timesteps", 1000))
        t_list = _parse_t_list(args.t_list, timesteps_max=timesteps_max)
        Z = generate_z_trajectory(
            ckpt=args.ckpt,
            device=args.device,
            use_ema=args.use_ema,
            n=int(args.n),
            t_list=t_list,
            z_kind=str(args.z_kind),
            seed=int(args.seed),
            mnist_root=str(args.mnist_root),
        )
        z_kind = str(args.z_kind)

    # Convert to per-sample trajectory vectors
    X_traj = trajectory_matrix(Z, mode=args.traj_mode)  # (N, Dtraj)
    N, Dtraj = X_traj.shape

    # PCA on raw trajectories
    rep_raw = _pca_report(X_traj, args.pca_components, name="PCA(z_traj)")

    # PCA on pairwise differences without enumerating pairs.
    # For ordered pairs i != j sampled uniformly, the sample covariance satisfies:
    #   Cov(x_i - x_j) = 2 * Cov_sample(x)
    # Therefore eigenvalues scale by 2 and explained-variance ratios are identical.
    rep_diff = dict(rep_raw)
    rep_diff["name"] = "PCA(z_traj_i - z_traj_j)"
    rep_diff["note"] = "Computed analytically: difference covariance = 2 * sample covariance; EVR identical."

    # Print summary
    def _print(rep: dict) -> None:
        print(f"\n=== {rep['name']} ===")
        print(f"X shape = {rep['shape'][0]} x {rep['shape'][1]}")
        print(f"n_components = {rep['n_components']}")
        print(f"k@0.90 = {rep['k@0.90']}, k@0.95 = {rep['k@0.95']}, k@0.99 = {rep['k@0.99']}")
        print(f"PCA participation ratio ≈ {rep['pca_participation_ratio']:.2f}")
        print(f"First 10 EVR: {[float(x) for x in rep['explained_variance_ratio'][:10]]}")
        if "note" in rep:
            print(f"Note: {rep['note']}")

    print("\n==============================")
    print("Trajectory PCA diagnostics")
    print("------------------------------")
    print(f"Z source = {'npz' if args.z_npz else 'ckpt'}")
    print(f"z_kind = {z_kind}")
    print(f"Z shape (T,N,D) = {tuple(int(x) for x in Z.shape)}")
    print(f"t_list size = {int(len(t_list))}  (min={int(t_list.min())}, max={int(t_list.max())})")
    print(f"traj_mode = {args.traj_mode} -> X_traj shape = {N} x {Dtraj}")
    if int(args.max_pairs) > 0:
        print("[note] --max_pairs is ignored (analytic difference-PCA).")
    print("==============================")

    _print(rep_raw)
    _print(rep_diff)

    # Optional save
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "pca_trajectory_pairs_summary.npz")
        np.savez_compressed(
            out_path,
            Z_shape=np.array(Z.shape, dtype=np.int64),
            t_list=t_list.astype(np.int64),
            rep_raw_shape=np.array(rep_raw["shape"], dtype=np.int64),
            rep_diff_shape=np.array(rep_diff["shape"], dtype=np.int64),
            raw_evr=rep_raw["explained_variance_ratio"],
            raw_cev=rep_raw["cum_explained_variance"],
            diff_evr=rep_diff["explained_variance_ratio"],
            diff_cev=rep_diff["cum_explained_variance"],
            raw_k90=np.array([rep_raw["k@0.90"]], dtype=np.int64),
            raw_k95=np.array([rep_raw["k@0.95"]], dtype=np.int64),
            raw_k99=np.array([rep_raw["k@0.99"]], dtype=np.int64),
            diff_k90=np.array([rep_diff["k@0.90"]], dtype=np.int64),
            diff_k95=np.array([rep_diff["k@0.95"]], dtype=np.int64),
            diff_k99=np.array([rep_diff["k@0.99"]], dtype=np.int64),
            raw_pr=np.array([rep_raw["pca_participation_ratio"]], dtype=np.float64),
            diff_pr=np.array([rep_diff["pca_participation_ratio"]], dtype=np.float64),
        )
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
