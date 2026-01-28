
"""
Cocycle Pairwise Flow Matching (PFM) vs Latent Flow Matching (latent-FM) on MNIST.

Implements the draft idea in /mnt/data/unsupervised_cocycle.pdf:
- encoder h_phi(y) -> x in R^d
- conditional bridge vector field w_psi(y_t, t | x, x')
- pairwise flow matching loss || w - (y' - y) ||^2 with y_t = (1-t)y + t y'

Also implements a baseline latent-FM:
- encoder e(y)->z, decoder d(z)->y
- flow matching in latent space: v(z_t,t|z,z') ≈ (z' - z)
- generation/translation via: encode y -> z, encode y' -> z', integrate latent ODE from z to z'
  then decode.

Evaluation:
- Translation quality on paired test images: MSE, PSNR, SSIM, classifier-FID (small MNIST CNN features),
  classifier accuracy on generated images vs target labels.
- Embedding quality: linear probe accuracy, kNN accuracy, t-SNE visualization.

Runs on GPU if available.

Notes:
- No torchvision dependency (some environments have a torchvision/torch mismatch). MNIST is downloaded directly
  from Yann LeCun's website and parsed locally.
"""

import argparse
from pathlib import Path
import math
import random
import time
import gzip
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

# ----------------------------
# MNIST download + parsing
# ----------------------------

MNIST_URLS = {
    "train_images": [
        "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}

def _download(urls, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    last_err = None
    for url in urls:
        try:
            print(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, str(path))
            if path.exists() and path.stat().st_size > 0:
                return
        except Exception as e:
            last_err = e
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
            print(f"  failed: {e}")

    raise RuntimeError(f"All MNIST download URLs failed for {path.name}. Last error: {last_err}")


def _read_idx_images(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic} in {gz_path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, rows, cols)
        return data

def _read_idx_labels(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic} in {gz_path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def prepare_mnist(root: Path):
    root = Path(root)
    raw = root / "mnist_raw"

    paths = {
        k: raw / Path(urls[0]).name   # <-- urls is a list now
        for k, urls in MNIST_URLS.items()
    }

    for k, urls in MNIST_URLS.items():
        _download(urls, paths[k])

    train_x = _read_idx_images(paths["train_images"])
    train_y = _read_idx_labels(paths["train_labels"])
    test_x  = _read_idx_images(paths["test_images"])
    test_y  = _read_idx_labels(paths["test_labels"])
    return (train_x, train_y), (test_x, test_y)


class MNISTNumpy(Dataset):
    def __init__(self, images_u8: np.ndarray, labels_u8: np.ndarray):
        self.x = images_u8  # (N,28,28) uint8
        self.y = labels_u8  # (N,) uint8

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        img = torch.from_numpy(self.x[idx].astype(np.float32) / 255.0).unsqueeze(0)  # (1,28,28)
        lab = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return img, lab


# ----------------------------
# Utils
# ----------------------------

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ssim_batch(x, y, data_range=1.0, K1=0.01, K2=0.03, win=7):
    pad = win // 2
    mu_x = F.avg_pool2d(x, win, 1, pad)
    mu_y = F.avg_pool2d(y, win, 1, pad)
    sigma_x = F.avg_pool2d(x * x, win, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, win, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, win, 1, pad) - mu_x * mu_y

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean().item()


def psnr_from_mse(mse, data_range=1.0):
    return 10.0 * math.log10((data_range ** 2) / (mse + 1e-12))


def pair_shuffle(y, labels=None):
    idx = torch.randperm(y.size(0), device=y.device)
    y2 = y[idx]
    if labels is None:
        return y2, None
    return y2, labels[idx]


# ----------------------------
# Model blocks
# ----------------------------

class FiLMMod(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, h, cond):
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        emb = F.silu(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class MLPEncoder(nn.Module):
    def __init__(self, in_dim=784, latent_dim=32, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, y_flat):
        return self.net(y_flat)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=32, out_dim=784, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class CocycleVectorField(nn.Module):
    def __init__(self, data_dim=784, cond_latent_dim=32, hidden=1024, time_dim=128):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.fc1 = nn.Linear(data_dim + time_dim, hidden)
        self.film1 = FiLMMod(cond_dim=2 * cond_latent_dim, hidden_dim=hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film2 = FiLMMod(cond_dim=2 * cond_latent_dim, hidden_dim=hidden)
        self.fc3 = nn.Linear(hidden, data_dim)

    def forward(self, y_flat, t, x, x2):
        te = self.time(t)
        inp = torch.cat([y_flat, te], dim=-1)
        h = F.silu(self.fc1(inp))
        cond = torch.cat([x, x2], dim=-1)
        h = self.film1(h, cond)
        h = F.silu(self.fc2(h))
        h = self.film2(h, cond)
        out = self.fc3(F.silu(h))
        return out


class LatentVectorField(nn.Module):
    """Traditional latent FM MLP without FiLM modulation."""
    def __init__(self, latent_dim=32, hidden=512, time_dim=64, depth=3):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        in_dim = latent_dim * 3 + time_dim  # [zt, z, z2, t_emb]
        layers = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(dim, hidden))
            dim = hidden
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(dim, latent_dim)

    def forward(self, zt, t, z, z2):
        te = self.time(t)
        h = torch.cat([zt, z, z2, te], dim=-1)
        for layer in self.hidden_layers:
            h = F.silu(layer(h))
        return self.out(h)


# ----------------------------
# ODE Integrator (Euler)
# ----------------------------

@torch.no_grad()
def euler_integrate(vf, y0, x, x2, steps=25, reverse=False):
    y = y0
    if reverse:
        ts = torch.linspace(1, 0, steps + 1, device=y0.device)
    else:
        ts = torch.linspace(0, 1, steps + 1, device=y0.device)

    for i in range(steps):
        t = ts[i].expand(y.size(0))
        dt = (ts[i + 1] - ts[i]).item()
        dy = vf(y, t, x, x2)
        y = y + dt * dy
    return y


# ----------------------------
# MNIST classifier for features/metrics
# ----------------------------

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_feat=False):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = h.view(h.size(0), -1)
        feat = F.relu(self.fc1(h))
        logits = self.fc2(feat)
        if return_feat:
            return logits, feat
        return logits


@torch.no_grad()
def compute_fid(feat_real, feat_fake, eps=1e-6):
    import numpy as np
    from scipy import linalg
    mu1 = feat_real.mean(axis=0)
    mu2 = feat_fake.mean(axis=0)
    sigma1 = np.cov(feat_real, rowvar=False)
    sigma2 = np.cov(feat_fake, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# ----------------------------
# Training & Eval
# ----------------------------

def train_classifier(device, train_ds, test_ds, epochs=2, batch=256, lr=1e-3):
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    te = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    clf = SmallCNN().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for ep in range(epochs):
        clf.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(clf(x), y)
            loss.backward()
            opt.step()
        clf.eval()
        correct = 0
        total = 0
        for x, y in te:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = clf(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc = correct / total
        print(f"[clf] epoch {ep+1}/{epochs} test acc={acc:.4f}")
    return clf


def train_cocycle_fm(device, train_ds, latent_dim=32, epochs=5, batch=256, lr=2e-4):
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    enc = MLPEncoder(784, latent_dim).to(device)
    vf = CocycleVectorField(784, latent_dim).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(vf.parameters()), lr=lr, weight_decay=1e-4)

    enc.train(); vf.train()
    for ep in range(epochs):
        t0 = time.time()
        losses = []
        for x, _ in tr:
            x = x.to(device, non_blocking=True)
            y = x.view(x.size(0), -1)
            y2, _ = pair_shuffle(y)
            t = torch.rand(y.size(0), device=device)
            yt = (1 - t)[:, None] * y + t[:, None] * y2
            u = y2 - y
            x_lat = enc(y)
            x2_lat = enc(y2)
            pred = vf(yt, t, x_lat, x2_lat)
            loss = F.mse_loss(pred, u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[cocycleFM] epoch {ep+1}/{epochs} loss={np.mean(losses):.6f} time={time.time()-t0:.1f}s")
    return enc.eval(), vf.eval()


def train_latent_fm(device, train_ds, latent_dim=32, epochs=5, batch=256, lr=2e-4, ae_warmup_epochs=1):
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    enc = MLPEncoder(784, latent_dim).to(device)
    dec = MLPDecoder(latent_dim, 784).to(device)
    vf = LatentVectorField(latent_dim).to(device)

    opt_ae = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    opt_vf = torch.optim.AdamW(vf.parameters(), lr=lr, weight_decay=1e-4)

    # Phase 1: fit the autoencoder only
    for ep in range(ae_warmup_epochs):
        enc.train(); dec.train()
        losses = []
        for x, _ in tr:
            x = x.to(device, non_blocking=True)
            y = x.view(x.size(0), -1)
            z = enc(y)
            yhat = dec(z).sigmoid()
            loss = F.mse_loss(yhat, y)
            opt_ae.zero_grad(set_to_none=True)
            loss.backward()
            opt_ae.step()
            losses.append(loss.item())
        print(f"[latentFM][AE warmup] epoch {ep+1}/{ae_warmup_epochs} recon={np.mean(losses):.6f}")

    # Freeze AE for classical latent FM
    for param in list(enc.parameters()) + list(dec.parameters()):
        param.requires_grad_(False)
    enc.eval(); dec.eval()

    # Phase 2: train the latent flow using fixed AE latents
    for ep in range(epochs):
        vf.train()
        losses = []
        for x, _ in tr:
            x = x.to(device, non_blocking=True)
            y = x.view(x.size(0), -1)
            y2, _ = pair_shuffle(y)
            with torch.no_grad():
                z = enc(y)
                z2 = enc(y2)
            t = torch.rand(z.size(0), device=device)
            zt = (1 - t)[:, None] * z + t[:, None] * z2
            u = z2 - z
            pred = vf(zt, t, z, z2)
            loss_fm = F.mse_loss(pred, u)
            opt_vf.zero_grad(set_to_none=True)
            loss_fm.backward()
            opt_vf.step()
            losses.append(loss_fm.item())
        print(f"[latentFM] epoch {ep+1}/{epochs} fm={np.mean(losses):.6f}")

    return enc.eval(), dec.eval(), vf.eval()

def eval_models(device, test_ds, clf, cocycle_enc, cocycle_vf, latent_enc, latent_dec, latent_vf,
                latent_dim=32, batch=256, steps=25, num_pairs_eval=4096, viz_n=16, outdir="./runs"):
    te = DataLoader(test_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    def img(t): return t.view(-1, 1, 28, 28)

    # -------- paired translation eval set (no grads needed) --------
    with torch.no_grad():
        Ys, Ys2, Ls2 = [], [], []
        for x, y in te:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_flat = x.view(x.size(0), -1)
            y2_flat, y2_lab = pair_shuffle(y_flat, y)
            Ys.append(y_flat); Ys2.append(y2_flat); Ls2.append(y2_lab)
            if sum(t.size(0) for t in Ys) >= num_pairs_eval:
                break
        y = torch.cat(Ys, 0)[:num_pairs_eval]
        y2 = torch.cat(Ys2, 0)[:num_pairs_eval]
        lab2 = torch.cat(Ls2, 0)[:num_pairs_eval]

    # -------- translation + image-quality metrics (no grads needed) --------
    with torch.no_grad():
        # CocycleFM translation
        x_lat = cocycle_enc(y)
        x2_lat = cocycle_enc(y2)
        yhat_c = euler_integrate(cocycle_vf, y, x_lat, x2_lat, steps=steps).clamp(0, 1)

        # LatentFM translation
        z = latent_enc(y)
        z2 = latent_enc(y2)
        zt = euler_integrate(latent_vf, z, z, z2, steps=steps)
        yhat_l = latent_dec(zt).sigmoid().clamp(0, 1)

        mse_c = F.mse_loss(yhat_c, y2).item()
        mse_l = F.mse_loss(yhat_l, y2).item()
        psnr_c = psnr_from_mse(mse_c)
        psnr_l = psnr_from_mse(mse_l)
        ssim_c = ssim_batch(img(yhat_c), img(y2))
        ssim_l = ssim_batch(img(yhat_l), img(y2))

        # Classifier-based metrics
        clf.eval()
        pred_c = clf(img(yhat_c)).argmax(1)
        pred_l = clf(img(yhat_l)).argmax(1)
        acc_c = (pred_c == lab2).float().mean().item()
        acc_l = (pred_l == lab2).float().mean().item()

        _, feat_real = clf(img(y2), return_feat=True)
        _, feat_c = clf(img(yhat_c), return_feat=True)
        _, feat_l = clf(img(yhat_l), return_feat=True)
        fid_c = compute_fid(feat_real.detach().cpu().numpy(), feat_c.detach().cpu().numpy())
        fid_l = compute_fid(feat_real.detach().cpu().numpy(), feat_l.detach().cpu().numpy())

    # -------- embedding quality (linear probe NEEDS grads) --------
    def linear_probe(encoder, epochs=5, lr=1e-2):
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        clf_lin = nn.Linear(latent_dim, 10).to(device)
        opt = torch.optim.Adam(clf_lin.parameters(), lr=lr)

        for ep in range(epochs):
            clf_lin.train()
            for xb, yb in te:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                zb = encoder(xb.view(xb.size(0), -1))  # encoder frozen; OK
                loss = F.cross_entropy(clf_lin(zb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                break  # quick

        clf_lin.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in te:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                zb = encoder(xb.view(xb.size(0), -1))
                pred = clf_lin(zb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
                if total >= 5000:
                    break
        return correct / total

    lin_c = linear_probe(cocycle_enc)
    lin_l = linear_probe(latent_enc)

    # -------- kNN (no grads) --------
    @torch.no_grad()
    def knn_acc(encoder, k=5, n_train=5000, n_test=2000):
        Xtr, Ytr = [], []
        for xb, yb in te:
            xb, yb = xb.to(device), yb.to(device)
            z = encoder(xb.view(xb.size(0), -1))
            Xtr.append(z); Ytr.append(yb)
            if sum(t.size(0) for t in Xtr) >= n_train:
                break
        Xtr = torch.cat(Xtr, 0)[:n_train]
        Ytr = torch.cat(Ytr, 0)[:n_train]

        Xte, Yte = [], []
        for xb, yb in te:
            xb, yb = xb.to(device), yb.to(device)
            z = encoder(xb.view(xb.size(0), -1))
            Xte.append(z); Yte.append(yb)
            if sum(t.size(0) for t in Xte) >= n_test:
                break
        Xte = torch.cat(Xte, 0)[:n_test]
        Yte = torch.cat(Yte, 0)[:n_test]

        dists = torch.cdist(Xte, Xtr)
        nn_idx = dists.topk(k, largest=False).indices
        nn_lab = Ytr[nn_idx]
        pred = torch.mode(nn_lab, dim=1).values
        return (pred == Yte).float().mean().item()

    knn_c = knn_acc(cocycle_enc)
    knn_l = knn_acc(latent_enc)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def save_grid(src, tgt, gen, name):
        n = min(viz_n, src.size(0))
        fig, axes = plt.subplots(3, n, figsize=(n * 1.2, 3.6))
        for i in range(n):
            axes[0, i].imshow(src[i, 0].cpu(), cmap="gray"); axes[0, i].axis("off")
            axes[1, i].imshow(tgt[i, 0].cpu(), cmap="gray"); axes[1, i].axis("off")
            axes[2, i].imshow(gen[i, 0].cpu(), cmap="gray"); axes[2, i].axis("off")
        axes[0, 0].set_ylabel("src", rotation=0, labelpad=20)
        axes[1, 0].set_ylabel("tgt", rotation=0, labelpad=20)
        axes[2, 0].set_ylabel("gen", rotation=0, labelpad=20)
        plt.tight_layout()
        p = outdir / name
        plt.savefig(p, dpi=150)
        plt.close(fig)
        return str(p)

    grid_c = save_grid(img(y), img(y2), img(yhat_c), "cocycle_translation_grid.png")
    grid_l = save_grid(img(y), img(y2), img(yhat_l), "latentfm_translation_grid.png")

    tsne_path_c = None
    tsne_path_l = None
    if TSNE is not None:
        with torch.no_grad():
            Xc, Yc = [], []
            for xb, yb in te:
                xb, yb = xb.to(device), yb.to(device)
                Xc.append(cocycle_enc(xb.view(xb.size(0), -1)).cpu()); Yc.append(yb.cpu())
                if sum(t.size(0) for t in Xc) >= 2000: break
            Xc = torch.cat(Xc, 0)[:2000].numpy()
            Yc = torch.cat(Yc, 0)[:2000].numpy()

            Xl, Yl = [], []
            for xb, yb in te:
                xb, yb = xb.to(device), yb.to(device)
                Xl.append(latent_enc(xb.view(xb.size(0), -1)).cpu()); Yl.append(yb.cpu())
                if sum(t.size(0) for t in Xl) >= 2000: break
            Xl = torch.cat(Xl, 0)[:2000].numpy()
            Yl = torch.cat(Yl, 0)[:2000].numpy()

        def plot_tsne(X, y, name):
            Z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(X)
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(Z[:, 0], Z[:, 1], s=6, c=y, cmap="tab10")
            plt.colorbar(ticks=range(10))
            plt.tight_layout()
            p = outdir / name
            plt.savefig(p, dpi=150)
            plt.close(fig)
            return str(p)

        tsne_path_c = plot_tsne(Xc, Yc, "tsne_cocycle_enc.png")
        tsne_path_l = plot_tsne(Xl, Yl, "tsne_latent_enc.png")

    metrics = {
        "translation": {
            "cocycleFM": {"mse": mse_c, "psnr": psnr_c, "ssim": ssim_c, "clf_acc_vs_target": acc_c, "fid": fid_c},
            "latentFM":  {"mse": mse_l, "psnr": psnr_l, "ssim": ssim_l, "clf_acc_vs_target": acc_l, "fid": fid_l},
        },
        "embedding": {
            "cocycleFM": {"linear_probe_acc": lin_c, "knn5_acc": knn_c},
            "latentFM":  {"linear_probe_acc": lin_l, "knn5_acc": knn_l},
        },
        "artifacts": {
            "grid_cocycle": grid_c,
            "grid_latentfm": grid_l,
            "tsne_cocycle": tsne_path_c,
            "tsne_latent": tsne_path_l,
        },
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./runs_cocycle_mnist")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clf-epochs", type=int, default=2)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (train_x, train_y), (test_x, test_y) = prepare_mnist(outdir)
    train_ds = MNISTNumpy(train_x, train_y)
    test_ds = MNISTNumpy(test_x, test_y)

    clf = train_classifier(device, train_ds, test_ds, epochs=args.clf_epochs, batch=args.batch)

    cocycle_enc, cocycle_vf = train_cocycle_fm(device, train_ds, latent_dim=args.latent_dim,
                                               epochs=args.epochs, batch=args.batch)

    latent_enc, latent_dec, latent_vf = train_latent_fm(device, train_ds, latent_dim=args.latent_dim,
                                                        epochs=args.epochs, batch=args.batch)

    metrics = eval_models(device, test_ds, clf, cocycle_enc, cocycle_vf,
                          latent_enc, latent_dec, latent_vf,
                          latent_dim=args.latent_dim, steps=args.steps, outdir=outdir)

    import json
    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Saved:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
