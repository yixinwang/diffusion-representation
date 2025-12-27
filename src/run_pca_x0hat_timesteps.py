#!/usr/bin/env python3
"""run_pca_x0hat_timesteps.py

PCA on denoised x0-prediction (x0_hat) across diffusion noise levels.

For a DDPM parameterized by predicted noise eps_theta(x_t,t), the standard
x0 estimator is:

  x0_hat(t) = (x_t - sqrt(1-\bar\alpha_t) * eps_theta(x_t,t)) / sqrt(\bar\alpha_t)

This script:
- loads a pretrained MNIST diffusion checkpoint
- samples a fixed batch of MNIST x0 and fixed Gaussian eps (so x_t is coupled across t)
- for each timestep t in --t_list, computes x0_hat(t) vectors and runs PCA
- prints eigenvalue / EVR / CEV diagnostics + k@0.90/0.95/0.99
- optionally saves eigenvalue-decay PNG plots

Example
-------
python src/run_pca_x0hat_timesteps.py \
  --ckpt results/steps_00023919.pt --use_ema \
  --n 1024 --t_list 0,50,100,200,400,800 \
  --pca_components 100 --save_png --out results/plots
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
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
    seen: set[int] = set()
    out: list[int] = []
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


def _make_mnist_with_labels(
    n: int,
    *,
    root: str = "./mnist_data",
    image_size: int = 28,
) -> tuple[torch.Tensor, np.ndarray]:
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ds = MNIST(root=root, train=True, download=True, transform=tfm)
    n = min(int(n), len(ds))
    xs = []
    ys = []
    for i in range(n):
        x, y = ds[i]
        xs.append(x)
        ys.append(int(y))
    X = torch.stack(xs, dim=0)  # (N,1,H,W)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


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
def x0_hat_vectors(*, score_model, x0: torch.Tensor, eps0: torch.Tensor, t: int, clip: bool) -> torch.Tensor:
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


def pca_report(X: torch.Tensor, *, components: int, name: str) -> dict:
    if X.ndim != 2:
        raise ValueError("X must be (N,d)")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples")

    Xc = X - X.mean(dim=0, keepdim=True)
    C = (Xc.T @ Xc) / float(Xc.shape[0] - 1)

    evals = torch.linalg.eigvalsh(C)  # ascending
    evals = torch.clamp(evals, min=0.0).flip(0)  # descending

    k = int(min(int(components), int(evals.numel())))
    evals_k = evals[:k]

    total = torch.sum(evals)
    if float(total) <= 0.0:
        evr = torch.zeros((k,), device=evals.device)
    else:
        evr = (evals_k / (total + 1e-12)).clamp_min(0.0)

    cev = torch.cumsum(evr, dim=0)

    def k_at(frac: float) -> int | None:
        idx = torch.where(cev >= float(frac))[0]
        return int(idx[0].item() + 1) if idx.numel() else None

    v = evr
    pr = float((v.sum() ** 2) / (torch.sum(v * v) + 1e-12))

    return {
        "name": str(name),
        "feature_dim": int(C.shape[0]),
        "total_var": float(torch.trace(C).detach().cpu().item()),
        "evals": evals_k.detach().cpu().numpy().astype(np.float64),
        "evr": evr.detach().cpu().numpy().astype(np.float64),
        "cev": cev.detach().cpu().numpy().astype(np.float64),
        "k@0.90": k_at(0.90),
        "k@0.95": k_at(0.95),
        "k@0.99": k_at(0.99),
        "participation_ratio": pr,
    }


def pca_project(
    X: torch.Tensor,
    *,
    components: int = 2,
) -> torch.Tensor:
    """Project X onto the top principal components.

    Returns Y with shape (N, components).
    """
    if X.ndim != 2:
        raise ValueError("X must be (N,d)")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples")
    k = int(components)
    k = max(1, k)
    k = min(k, int(min(X.shape[0], X.shape[1])))

    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD: Xc = U S Vh; principal directions are rows of Vh
    # Project: Y = Xc @ V_k
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh[:k].T  # (d,k)
    return Xc @ V


def pca_fit_project_train_test(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA on X_train and project both train and test.

    Uses SVD on centered X_train (no leakage).
    """
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be (N,d)")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dims differ")
    if X_train.shape[0] < 2:
        raise ValueError("Need at least 2 train samples")

    k = int(components)
    k = max(1, k)
    k = min(k, int(min(X_train.shape[0], X_train.shape[1])))

    mean = X_train.mean(dim=0, keepdim=True)
    Xc_train = X_train - mean
    Xc_test = X_test - mean

    _, _, Vh = torch.linalg.svd(Xc_train, full_matrices=False)
    V = Vh[:k].T  # (d,k)
    return Xc_train @ V, Xc_test @ V


def standardize_train_test(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (X_train - mean) / std, (X_test - mean) / std


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, batch_size: int = 512) -> float:
    model.eval()
    correct = 0
    total = 0
    for i in range(0, X.shape[0], int(batch_size)):
        xb = X[i : i + int(batch_size)]
        yb = y[i : i + int(batch_size)]
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return float(correct) / float(max(total, 1))


def train_mlp_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[float, float]:
    device = X_train.device
    model = MLPClassifier(int(X_train.shape[1]), int(hidden_dim), out_dim=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.CrossEntropyLoss()

    n = int(X_train.shape[0])
    for _ in range(int(epochs)):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, int(batch_size)):
            idx = perm[i : i + int(batch_size)]
            xb = X_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    train_acc = accuracy(model, X_train, y_train, batch_size=batch_size)
    test_acc = accuracy(model, X_test, y_test, batch_size=batch_size)
    return train_acc, test_acc


def _print_pca(rep: dict) -> None:
    print(f"\n=== {rep['name']} ===")
    print(f"feature_dim = {rep['feature_dim']} | total_var = {rep['total_var']:.6e}")
    print(f"k@0.90 = {rep['k@0.90']}, k@0.95 = {rep['k@0.95']}, k@0.99 = {rep['k@0.99']}")
    print(f"participation_ratio ≈ {rep['participation_ratio']:.2f}")
    evals10 = [float(x) for x in rep["evals"][:10]]
    evr10 = [float(x) for x in rep["evr"][:10]]
    cev10 = [float(x) for x in rep["cev"][:10]]
    print(f"first 10 evals: {evals10}")
    print(f"first 10 EVR:   {evr10}")
    print(f"first 10 CEV:   {cev10}")


def _maybe_save_eig_decay_png(*, evals: np.ndarray, out_dir: str, filename: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not import matplotlib for plotting ({e}); skipping PNG save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    x = np.arange(1, len(evals) + 1, dtype=np.int64)
    plt.figure(figsize=(6.0, 4.0))
    plt.semilogy(x, np.maximum(evals, 1e-30))
    plt.xlabel("principal component index")
    plt.ylabel("eigenvalue (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue decay plot: {path}")


def _maybe_save_eig_decay_overlay_png(
    *,
    evals_a: np.ndarray,
    evals_b: np.ndarray,
    label_a: str,
    label_b: str,
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    """Save an overlay eigenvalue-decay plot as a PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not import matplotlib for plotting ({e}); skipping PNG save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    k = int(min(len(evals_a), len(evals_b)))
    x = np.arange(1, k + 1, dtype=np.int64)
    plt.figure(figsize=(6.0, 4.0))
    plt.semilogy(x, np.maximum(evals_a[:k], 1e-30), label=label_a)
    plt.semilogy(x, np.maximum(evals_b[:k], 1e-30), label=label_b)
    plt.xlabel("principal component index")
    plt.ylabel("eigenvalue (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue decay plot: {path}")


def _maybe_save_pca_scatter_by_label_png(
    *,
    Y2: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    """Save a PC1-vs-PC2 scatter colored by digit label."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not import matplotlib for plotting ({e}); skipping PNG save.")
        return

    if Y2.ndim != 2 or Y2.shape[1] != 2:
        raise ValueError("Y2 must be (N,2)")
    if labels.ndim != 1 or labels.shape[0] != Y2.shape[0]:
        raise ValueError("labels must be (N,)")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    plt.figure(figsize=(6.0, 5.0))
    sc = plt.scatter(Y2[:, 0], Y2[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8, linewidths=0)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    cb = plt.colorbar(sc, ticks=list(range(10)))
    cb.set_label("digit")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved PCA class scatter: {path}")


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")

    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mnist_root", type=str, default="./mnist_data")

    p.add_argument("--t_list", type=str, default="0,50,100,200,400", help="Timesteps, e.g. 0,10,20 or 0:1000:50")
    p.add_argument("--clip", action="store_true", help="Optionally clamp x0_hat to [-1,1]")

    p.add_argument("--no_raw", action="store_true", help="Disable PCA on raw x0 baseline")

    p.add_argument("--pca_components", type=int, default=100)
    p.add_argument("--save_png", action="store_true", help="Save eigenvalue decay plots as PNG")
    p.add_argument("--save_class_png", action="store_true", help="Save PC1-vs-PC2 scatter colored by digit class")
    p.add_argument("--out", type=str, default="", help="Optional output directory for PNG + NPZ")

    p.add_argument("--eval_clf", action="store_true", help="Train/test a small MLP to predict digit label from PCA features")
    p.add_argument("--train_frac", type=float, default=0.8, help="Train fraction for classifier evaluation")
    p.add_argument("--clf_pca_dim", type=int, default=50, help="PCA dimension used as classifier input")
    p.add_argument("--clf_hidden", type=int, default=128, help="Hidden width of the MLP classifier")
    p.add_argument("--clf_epochs", type=int, default=30, help="Training epochs for the MLP classifier")
    p.add_argument("--clf_lr", type=float, default=1e-3, help="Learning rate for the MLP classifier")
    p.add_argument("--clf_batch", type=int, default=256, help="Batch size for the MLP classifier")

    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    _, score_model = load_mnist_diffusion_checkpoint(args.ckpt, device=args.device, use_ema=args.use_ema)
    score_model = score_model.to(args.device)
    score_model.eval()

    timesteps_max = int(getattr(score_model, "timesteps", 1000))
    t_list = _parse_t_list(str(args.t_list), timesteps_max=timesteps_max)

    x0, y = _make_mnist_with_labels(args.n, root=args.mnist_root, image_size=int(getattr(score_model, "image_size", 28)))
    x0 = x0.to(args.device)
    eps0 = torch.randn_like(x0)

    out_dir = str(args.out).strip() or "."
    if args.out:
        os.makedirs(out_dir, exist_ok=True)

    # Precompute train/test split indices once (shared across representations)
    if bool(args.eval_clf):
        train_frac = float(args.train_frac)
        if not (0.05 < train_frac < 0.95):
            raise SystemExit("--train_frac must be in (0.05, 0.95)")
        rng = np.random.default_rng(int(args.seed))
        perm = rng.permutation(int(x0.shape[0]))
        n_train = int(round(train_frac * float(x0.shape[0])))
        n_train = max(2, min(n_train, int(x0.shape[0]) - 2))
        idx_train = perm[:n_train]
        idx_test = perm[n_train:]
        y_torch = torch.from_numpy(y).to(args.device, dtype=torch.long)

    print("\n==============================")
    print("PCA on denoised x0_hat(t)")
    print("------------------------------")
    print(f"ckpt = {args.ckpt}")
    print(f"n = {int(args.n)} | seed = {int(args.seed)}")
    print(f"clip = {bool(args.clip)}")
    print(f"raw_x0_pca = {not bool(args.no_raw)}")
    print(f"t_list (len={len(t_list)}) = {t_list.tolist()}")
    print(f"pca_components = {int(args.pca_components)}")
    print("==============================")

    reports: list[dict] = []

    rep_raw = None
    if not bool(args.no_raw):
        Xraw = x0.view(x0.shape[0], -1)
        rep_raw = pca_report(Xraw, components=int(args.pca_components), name="PCA(raw x0)")
        _print_pca(rep_raw)
        reports.append({"t": -1, **rep_raw})
        if bool(args.save_png):
            _maybe_save_eig_decay_png(
                evals=rep_raw["evals"],
                out_dir=out_dir,
                filename="eig_decay_raw_x0.png",
                title="raw x0 PCA eigenvalue decay",
            )
        if bool(args.save_class_png):
            Y2 = pca_project(Xraw, components=2).detach().cpu().numpy().astype(np.float64)
            _maybe_save_pca_scatter_by_label_png(
                Y2=Y2,
                labels=y,
                out_dir=out_dir,
                filename="pca_scatter_raw_x0_by_digit.png",
                title="raw x0 PCA scatter (colored by digit)",
            )

        if bool(args.eval_clf):
            X_train = Xraw[idx_train]
            X_test = Xraw[idx_test]
            y_train = y_torch[idx_train]
            y_test = y_torch[idx_test]

            Z_train, Z_test = pca_fit_project_train_test(X_train, X_test, components=int(args.clf_pca_dim))
            Z_train, Z_test = standardize_train_test(Z_train, Z_test)
            train_acc, test_acc = train_mlp_classifier(
                Z_train,
                y_train,
                Z_test,
                y_test,
                hidden_dim=int(args.clf_hidden),
                epochs=int(args.clf_epochs),
                lr=float(args.clf_lr),
                batch_size=int(args.clf_batch),
            )
            print(
                "[clf] raw x0 -> PCA({k}) -> MLP: train_acc={tr:.4f} test_acc={te:.4f}".format(
                    k=int(args.clf_pca_dim),
                    tr=float(train_acc),
                    te=float(test_acc),
                )
            )
    for t in t_list.tolist():
        X = x0_hat_vectors(score_model=score_model, x0=x0, eps0=eps0, t=int(t), clip=bool(args.clip))
        rep = pca_report(X, components=int(args.pca_components), name=f"PCA(x0_hat(t={int(t)}))")
        _print_pca(rep)
        reports.append({"t": int(t), **rep})

        if bool(args.save_png):
            _maybe_save_eig_decay_png(
                evals=rep["evals"],
                out_dir=out_dir,
                filename=f"eig_decay_x0hat_t{int(t)}.png",
                title=f"x0_hat PCA eigenvalue decay (t={int(t)})",
            )

            if rep_raw is not None:
                _maybe_save_eig_decay_overlay_png(
                    evals_a=rep_raw["evals"],
                    evals_b=rep["evals"],
                    label_a="raw x0",
                    label_b=f"x0_hat(t={int(t)})",
                    out_dir=out_dir,
                    filename=f"eig_decay_compare_raw_vs_x0hat_t{int(t)}.png",
                    title=f"PCA eigenvalue decay: raw x0 vs x0_hat(t={int(t)})",
                )

        if bool(args.save_class_png):
            Y2 = pca_project(X, components=2).detach().cpu().numpy().astype(np.float64)
            _maybe_save_pca_scatter_by_label_png(
                Y2=Y2,
                labels=y,
                out_dir=out_dir,
                filename=f"pca_scatter_x0hat_t{int(t)}_by_digit.png",
                title=f"x0_hat(t={int(t)}) PCA scatter (colored by digit)",
            )

        if bool(args.eval_clf):
            X_train = X[idx_train]
            X_test = X[idx_test]
            y_train = y_torch[idx_train]
            y_test = y_torch[idx_test]

            Z_train, Z_test = pca_fit_project_train_test(X_train, X_test, components=int(args.clf_pca_dim))
            Z_train, Z_test = standardize_train_test(Z_train, Z_test)
            train_acc, test_acc = train_mlp_classifier(
                Z_train,
                y_train,
                Z_test,
                y_test,
                hidden_dim=int(args.clf_hidden),
                epochs=int(args.clf_epochs),
                lr=float(args.clf_lr),
                batch_size=int(args.clf_batch),
            )
            print(
                "[clf] x0_hat(t={t}) -> PCA({k}) -> MLP: train_acc={tr:.4f} test_acc={te:.4f}".format(
                    t=int(t),
                    k=int(args.clf_pca_dim),
                    tr=float(train_acc),
                    te=float(test_acc),
                )
            )

    if args.out:
        out_path = os.path.join(out_dir, "pca_x0hat_timesteps_summary.npz")
        np.savez_compressed(
            out_path,
            t_list=t_list.astype(np.int64),
            pca_components=np.array([int(args.pca_components)], dtype=np.int64),
            reports=np.array(reports, dtype=object),
        )
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
