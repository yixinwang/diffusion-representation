from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

OUT = Path("build/out")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260829
CIFAR_REV = "0b2714987fa478483af9968de7c934580d0bb9a2"


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def ih(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


ds = load_dataset("uoft-cs/cifar10", revision=CIFAR_REV)

def arrays(split: str):
    part = ds[split]
    x = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in part["img"]])
    y = np.asarray(part["label"], dtype=np.int64)
    return x, y

x, y = arrays("train")
xt, yt = arrays("test")

# Reconstruct every v1 source ID.
v1_rng = np.random.default_rng(20260827)
v1_train: set[int] = set()
for cls in range(10):
    order = v1_rng.permutation(np.flatnonzero(y == cls))
    v1_train.update(order[:1800].tolist())
v1_test: set[int] = set()
for cls in range(10):
    order = np.random.default_rng(20260827 + 991 + cls).permutation(np.flatnonzero(yt == cls))
    v1_test.update(order[:200].tolist())

# Reconstruct every v2 source ID from the remaining records.
v2_train: set[int] = set()
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(y == cls).tolist()) - v1_train), dtype=np.int64)
    order = np.random.default_rng(20260828 + 17 * cls).permutation(available)
    v2_train.update(order[:2500].tolist())
v2_test: set[int] = set()
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(yt == cls).tolist()) - v1_test), dtype=np.int64)
    order = np.random.default_rng(20260828 + 991 + cls).permutation(available)
    v2_test.update(order[:200].tolist())

used_train = v1_train | v2_train
used_test = v1_test | v2_test
assert len(used_train) == 43000
assert len(used_test) == 4000

counts = {"train": 400, "validation": 100, "development": 100}
roles = {name: [] for name in counts}
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(y == cls).tolist()) - used_train), dtype=np.int64)
    order = np.random.default_rng(SEED + 31 * cls).permutation(available)
    cursor = 0
    for role, count in counts.items():
        roles[role].extend(order[cursor:cursor + count].tolist())
        cursor += count

confirmation = []
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(yt == cls).tolist()) - used_test), dtype=np.int64)
    confirmation.extend(np.random.default_rng(SEED + 991 + cls).permutation(available)[:200].tolist())
confirmation = sorted(confirmation)

assert not set().union(*(set(v) for v in roles.values())) & used_train
assert not set(confirmation) & used_test

payload = {}
for role, values in roles.items():
    idx = np.asarray(sorted(values), dtype=np.int64)
    payload[f"{role}_x"] = x[idx]
    payload[f"{role}_y"] = y[idx]
    payload[f"{role}_source_id"] = idx
np.savez_compressed(OUT / "cifar10_32_rmpf_train_dev.npz", **payload)
idx = np.asarray(confirmation, dtype=np.int64)
np.savez_compressed(
    OUT / "cifar10_32_rmpf_confirmation.npz",
    confirmation_x=xt[idx],
    confirmation_y=yt[idx],
    confirmation_source_id=50000 + idx,
)

manifest = {
    "repo": "uoft-cs/cifar10",
    "revision": CIFAR_REV,
    "shape": [32, 32, 3],
    "seed": SEED,
    "split_sizes": {**{k: len(v) for k, v in roles.items()}, "confirmation": len(confirmation)},
    "split_hashes": {**{k: ih(np.asarray(sorted(v))) for k, v in roles.items()}, "confirmation": ih(50000 + np.asarray(confirmation))},
    "excluded_v1_v2_train": len(used_train),
    "excluded_v1_v2_test": len(used_test),
    "artifacts": {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in OUT.glob("cifar10_32_rmpf_*.npz")
    },
}
(OUT / "cifar_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
