"""Cocycle vs Latent Flow Matching on a 2-D pinwheel mixture."""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from cocycle_fm_mnist import (
    seed_all,
    pair_shuffle,
    euler_integrate,
    MLPEncoder,
    MLPDecoder,
    CocycleVectorField,
    LatentVectorField,
)


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0.0, 2 * np.pi, num_classes, endpoint=False)
    feats = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    feats[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)
    angles = rads[labels] + rate * np.exp(feats[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = rotations.T.reshape(-1, 2, 2)
    data = 10.0 * np.einsum("ti,tij->tj", feats, rotations)
    return data.astype(np.float32), labels.astype(np.int64)


class PinwheelDataset(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(points)
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_cocycle(device, dataset, data_dim=2, latent_dim=16, hidden=128,
                  epochs=400, batch=256, lr=1e-4):
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False)
    enc = MLPEncoder(in_dim=data_dim, latent_dim=latent_dim, hidden=hidden).to(device)
    vf = CocycleVectorField(data_dim=data_dim, cond_latent_dim=latent_dim,
                            hidden=hidden * 2, time_dim=64).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(vf.parameters()), lr=lr, weight_decay=1e-4)

    enc.train(); vf.train()
    for ep in range(epochs):
        losses = []
        for pts, _ in loader:
            y = pts.to(device)
            y2, _ = pair_shuffle(y)
            t = torch.rand(y.size(0), device=device)
            yt = (1 - t)[:, None] * y + t[:, None] * y2
            target = y2 - y
            x_lat = enc(y)
            x2_lat = enc(y2)
            pred = vf(yt, t, x_lat, x2_lat)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 50 == 0:
            print(f"[cocycle] epoch {ep+1}/{epochs} loss={np.mean(losses):.6f}")
    return enc.eval(), vf.eval()


def train_latent(device, dataset, data_dim=2, latent_dim=16, hidden=128,
                 epochs=400, batch=256, lr=1e-4, ae_warmup=400):
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False)
    enc = MLPEncoder(in_dim=data_dim, latent_dim=latent_dim, hidden=hidden).to(device)
    dec = MLPDecoder(latent_dim=latent_dim, out_dim=data_dim, hidden=hidden).to(device)
    vf = LatentVectorField(latent_dim=latent_dim, hidden=hidden, time_dim=64).to(device)

    opt_ae = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    opt_vf = torch.optim.AdamW(vf.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(ae_warmup):
        enc.train(); dec.train()
        losses = []
        for pts, _ in loader:
            y = pts.to(device)
            z = enc(y)
            recon = dec(z)
            loss = F.mse_loss(recon, y)
            opt_ae.zero_grad(set_to_none=True)
            loss.backward()
            opt_ae.step()
            losses.append(loss.item())
        if (ep + 1) % 50 == 0:
            print(f"[latent][AE] epoch {ep+1}/{ae_warmup} recon={np.mean(losses):.6f}")

    for param in list(enc.parameters()) + list(dec.parameters()):
        param.requires_grad_(False)
    enc.eval(); dec.eval()

    for ep in range(epochs):
        vf.train()
        losses = []
        for pts, _ in loader:
            y = pts.to(device)
            y2, _ = pair_shuffle(y)
            with torch.no_grad():
                z = enc(y)
                z2 = enc(y2)
            t = torch.rand(z.size(0), device=device)
            zt = (1 - t)[:, None] * z + t[:, None] * z2
            target = z2 - z
            pred = vf(zt, t, z, z2)
            loss = F.mse_loss(pred, target)
            opt_vf.zero_grad(set_to_none=True)
            loss.backward()
            opt_vf.step()
            losses.append(loss.item())
        if (ep + 1) % 50 == 0:
            print(f"[latent][VF] epoch {ep+1}/{epochs} fm={np.mean(losses):.6f}")
    return enc.eval(), dec.eval(), vf.eval()


def linear_probe(encoder, dataset, latent_dim, num_classes, device, epochs=200, lr=1e-2):
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    for p in encoder.parameters():
        p.requires_grad_(False)
    clf = nn.Linear(latent_dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for ep in range(epochs):
        clf.train()
        for pts, lbs in loader:
            pts = pts.to(device)
            lbs = lbs.to(device)
            logits = clf(encoder(pts))
            loss = F.cross_entropy(logits, lbs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for pts, lbs in loader:
            pts = pts.to(device)
            lbs = lbs.to(device)
            pred = clf(encoder(pts)).argmax(1)
            correct += (pred == lbs).sum().item()
            total += lbs.numel()
    return correct / max(total, 1)


def knn_acc(encoder, dataset, device, num_samples=2000, k=5):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for pts, lbs in loader:
            pts = pts.to(device)
            feats.append(encoder(pts).cpu())
            labels.append(lbs.cpu())
    X = torch.cat(feats, dim=0)
    Y = torch.cat(labels, dim=0)
    if X.size(0) > num_samples:
        idx = torch.randperm(X.size(0))[:num_samples]
        X = X[idx]
        Y = Y[idx]
    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    nn_idx = dists.topk(k, largest=False).indices
    nn_labels = Y[nn_idx]
    pred = torch.mode(nn_labels, dim=1).values
    return (pred == Y).float().mean().item()


def plot_translations(all_points, all_labels, y, y2, yhat_c, yhat_l,
                      outdir: Path, num_classes, max_pairs=200):
    palette = plt.cm.get_cmap("tab10", num_classes)
    idx = torch.randperm(y.size(0))[:min(max_pairs, y.size(0))]
    src = y[idx].cpu().numpy()
    tgt = y2[idx].cpu().numpy()
    pred_c = yhat_c[idx].cpu().numpy()
    pred_l = yhat_l[idx].cpu().numpy()
    labs = all_labels.cpu().numpy()
    pts = all_points.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    titles = ["CocycleFM translations", "LatentFM translations"]
    preds = [pred_c, pred_l]
    for ax, title, pred in zip(axes, titles, preds):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=labs, cmap=palette, s=10, alpha=0.25)
        for s, t, p in zip(src, tgt, pred):
            ax.plot([s[0], t[0]], [s[1], t[1]], color="gray", alpha=0.3, linestyle="--")
            ax.arrow(s[0], s[1], p[0] - s[0], p[1] - s[1],
                     color="crimson" if title.startswith("Cocycle") else "navy",
                     alpha=0.7, length_includes_head=True, head_width=0.2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    path = outdir / "pinwheel_translation_vectors.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_generations(all_points, all_labels, yhat_c, yhat_l, target_labels,
                     outdir: Path, num_classes):
    palette = plt.cm.get_cmap("tab10", num_classes)
    pts = all_points.cpu().numpy()
    labs = all_labels.cpu().numpy()
    gen_c = yhat_c.cpu().numpy()
    gen_l = yhat_l.cpu().numpy()
    tgt = target_labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    titles = ["CocycleFM generations", "LatentFM generations"]
    gens = [gen_c, gen_l]
    for ax, title, gen in zip(axes, titles, gens):
        ax.scatter(pts[:, 0], pts[:, 1], c=labs, cmap=palette, s=10, alpha=0.15)
        sc = ax.scatter(gen[:, 0], gen[:, 1], c=tgt, cmap=palette, s=25, edgecolors="k", linewidths=0.2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    path = outdir / "pinwheel_generations.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_latents(cocycle_enc, latent_enc, dataset, device, latent_dim, num_classes, outdir: Path):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    feats_c, feats_l, labels = [], [], []
    with torch.no_grad():
        for pts, lbs in loader:
            pts = pts.to(device)
            feats_c.append(cocycle_enc(pts).cpu())
            feats_l.append(latent_enc(pts).cpu())
            labels.append(lbs)
    Zc = torch.cat(feats_c).numpy()
    Zl = torch.cat(feats_l).numpy()
    Y = torch.cat(labels).numpy()
    def to_two(feats):
        if feats.shape[1] >= 2:
            return feats[:, :2]
        pad = np.zeros((feats.shape[0], 2 - feats.shape[1]), dtype=feats.dtype)
        return np.concatenate([feats, pad], axis=1)
    Zc2 = to_two(Zc)
    Zl2 = to_two(Zl)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feats, title in zip(axes, [Zc2, Zl2], ["Cocycle encoder", "Latent encoder"]):
        ax.scatter(feats[:, 0], feats[:, 1], c=Y, cmap=plt.cm.get_cmap("tab10", num_classes), s=12, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    path = outdir / "latent_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def evaluate(device, dataset, cocycle_enc, cocycle_vf,
             latent_enc, latent_dec, latent_vf,
             latent_dim, num_classes, steps, outdir: Path,
             num_pairs_eval=None, batch=512):
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")
    if num_pairs_eval is None:
        num_pairs_eval = len(dataset)
    eff_batch = min(batch, max(1, len(dataset)))
    loader = DataLoader(dataset, batch_size=eff_batch, shuffle=True, drop_last=False)
    Ys, Ys2, Labs = [], [], []
    with torch.no_grad():
        for pts, lbs in loader:
            y = pts.to(device)
            y2, lab2 = pair_shuffle(y, lbs.to(device))
            Ys.append(y)
            Ys2.append(y2)
            Labs.append(lab2)
            if sum(t.size(0) for t in Ys) >= num_pairs_eval:
                break
    if not Ys:
        raise RuntimeError("Evaluation dataset produced no batches; reduce batch size or ensure dataset is non-empty.")
    y = torch.cat(Ys, dim=0)[:num_pairs_eval]
    y2 = torch.cat(Ys2, dim=0)[:num_pairs_eval]
    lab2 = torch.cat(Labs, dim=0)[:num_pairs_eval]

    with torch.no_grad():
        x_lat = cocycle_enc(y)
        x2_lat = cocycle_enc(y2)
        yhat_c = euler_integrate(cocycle_vf, y, x_lat, x2_lat, steps=steps).cpu()

        z = latent_enc(y)
        z2 = latent_enc(y2)
        zt = euler_integrate(latent_vf, z, z, z2, steps=steps)
        yhat_l = latent_dec(zt).cpu()

    mse_c = F.mse_loss(yhat_c, y.cpu()).item()
    mse_l = F.mse_loss(yhat_l, y.cpu()).item()
    mse_c_tgt = F.mse_loss(yhat_c, y2.cpu()).item()
    mse_l_tgt = F.mse_loss(yhat_l, y2.cpu()).item()

    lin_c = linear_probe(cocycle_enc, dataset, latent_dim, num_classes, device)
    lin_l = linear_probe(latent_enc, dataset, latent_dim, num_classes, device)
    knn_c = knn_acc(cocycle_enc, dataset, device)
    knn_l = knn_acc(latent_enc, dataset, device)

    all_points = torch.stack([dataset[i][0] for i in range(len(dataset))])
    all_labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    grid_path = plot_translations(all_points, all_labels, y.cpu(), y2.cpu(), yhat_c, yhat_l,
                                  outdir, num_classes)
    gen_path = plot_generations(all_points, all_labels, yhat_c, yhat_l, lab2.cpu(),
                                outdir, num_classes)
    latent_scatter_path = plot_latents(cocycle_enc, latent_enc, dataset, device,
                                       latent_dim, num_classes, outdir)

    metrics = {
        "translation": {
            "mse_vs_source": {"cocycle": mse_c, "latent": mse_l},
            "mse_vs_target": {"cocycle": mse_c_tgt, "latent": mse_l_tgt},
        },
        "embedding": {
            "linear_probe_acc": {"cocycle": lin_c, "latent": lin_l},
            "knn5_acc": {"cocycle": knn_c, "latent": knn_l},
        },
        "artifacts": {
            "translation_vectors": grid_path,
            "generation_scatter": gen_path,
            "latent_scatter": latent_scatter_path,
        },
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./runs_pinwheel")
    ap.add_argument("--num-clusters", type=int, default=5)
    ap.add_argument("--samples-per-cluster", type=int, default=1000)
    ap.add_argument("--radial-std", type=float, default=0.3)
    ap.add_argument("--tangential-std", type=float, default=0.05)
    ap.add_argument("--rate", type=float, default=0.25)
    ap.add_argument("--latent-dim", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--ae-warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data, labels = make_pinwheel_data(args.radial_std, args.tangential_std,
                                      args.num_clusters, args.samples_per_cluster,
                                      args.rate)
    dataset = PinwheelDataset(data, labels)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Pinwheel dataset is empty; increase samples per cluster or num clusters.")
    if total < 2:
        train_ds = test_ds = dataset
    else:
        train_frac = 0.8
        train_size = max(1, int(total * train_frac))
        if train_size >= total:
            train_size = total - 1
        test_size = total - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cocycle_enc, cocycle_vf = train_cocycle(device, train_ds, data_dim=2,
                                            latent_dim=args.latent_dim, epochs=args.epochs,
                                            batch=args.batch)
    latent_enc, latent_dec, latent_vf = train_latent(device, train_ds, data_dim=2,
                                                     latent_dim=args.latent_dim, epochs=args.epochs,
                                                     batch=args.batch, ae_warmup=args.ae_warmup)

    metrics = evaluate(device, test_ds, cocycle_enc, cocycle_vf,
                       latent_enc, latent_dec, latent_vf,
                       latent_dim=args.latent_dim, num_classes=args.num_clusters,
                       steps=args.steps, outdir=outdir)

    metrics_path = outdir / "pinwheel_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Saved:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
