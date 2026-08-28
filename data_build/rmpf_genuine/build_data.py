from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from datasets import Video, load_dataset

OUT = Path("build/out")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260829
CIFAR_REV = "0b2714987fa478483af9968de7c934580d0bb9a2"
HMDB_REV = "0b7ec9fef2ce1809ebc7a669227b99f731075683"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ih(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# CIFAR-10 32x32. Reconstruct and exclude every v1/v2 role before v3.
# ---------------------------------------------------------------------------
cifar = load_dataset("uoft-cs/cifar10", revision=CIFAR_REV)

def cifar_arrays(split: str):
    part = cifar[split]
    x = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in part["img"]])
    y = np.asarray(part["label"], dtype=np.int64)
    return x, y

x, y = cifar_arrays("train")
xt, yt = cifar_arrays("test")

# v1 IDs
v1_rng = np.random.default_rng(20260827)
v1_counts = [1200, 200, 200, 200]
v1_train: set[int] = set()
for cls in range(10):
    idx = v1_rng.permutation(np.flatnonzero(y == cls))
    v1_train.update(idx[: sum(v1_counts)].tolist())
v1_test: set[int] = set()
for cls in range(10):
    idx = np.random.default_rng(20260827 + 991 + cls).permutation(np.flatnonzero(yt == cls))
    v1_test.update(idx[:200].tolist())

# v2 IDs, generated from records not in v1.
v2_train: set[int] = set()
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(y == cls).tolist()) - v1_train), dtype=np.int64)
    idx = np.random.default_rng(20260828 + 17 * cls).permutation(available)
    v2_train.update(idx[:2500].tolist())
v2_test: set[int] = set()
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(yt == cls).tolist()) - v1_test), dtype=np.int64)
    idx = np.random.default_rng(20260828 + 991 + cls).permutation(available)
    v2_test.update(idx[:200].tolist())

used_train = v1_train | v2_train
used_test = v1_test | v2_test
assert len(used_train) == 43000 and len(used_test) == 4000

counts = {"train": 400, "validation": 100, "development": 100}
roles: dict[str, list[int]] = {key: [] for key in counts}
for cls in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(y == cls).tolist()) - used_train), dtype=np.int64)
    idx = np.random.default_rng(SEED + 31 * cls).permutation(available)
    cursor = 0
    for role, count in counts.items():
        roles[role].extend(idx[cursor : cursor + count].tolist())
        cursor += count

confirmation: list[int] = []
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

# ---------------------------------------------------------------------------
# HMDB51 6x32x32 RGB. Official test remains a separate confirmation file.
# ---------------------------------------------------------------------------
hmdb = load_dataset("divm/hmdb51", revision=HMDB_REV)
hmdb = {key: value.cast_column("video", Video(decode=False)) for key, value in hmdb.items()}
labels = sorted(set(hmdb["train"]["label"]) & set(hmdb["validation"]["label"]) & set(hmdb["test"]["label"]))[:10]
label_id = {label: index for index, label in enumerate(labels)}


def video_path(value: dict, suffix: str) -> tuple[Path, tempfile.NamedTemporaryFile | None]:
    if value.get("path"):
        return Path(value["path"]), None
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.write(value["bytes"])
    handle.flush()
    return Path(handle.name), handle


def decode(value: dict) -> np.ndarray:
    path, handle = video_path(value, ".mp4")
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if handle is not None:
        handle.close()
        path.unlink(missing_ok=True)
    if len(frames) < 6:
        raise RuntimeError(f"video has only {len(frames)} frames")
    array = np.stack(frames)
    h, w = array.shape[1:3]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    array = array[:, top : top + side, left : left + side]
    ids = np.linspace(0, len(array) - 1, 6).round().astype(int)
    return np.stack([cv2.resize(array[i], (32, 32), interpolation=cv2.INTER_AREA) for i in ids]).astype(np.uint8)


def select_records(split: str, count_per_label: int, offset: int):
    ds = hmdb[split]
    rows = []
    for label in labels:
        candidates = [i for i, value in enumerate(ds["label"]) if value == label]
        order = np.random.default_rng(SEED + offset + 101 * label_id[label]).permutation(candidates)
        for index in order[:count_per_label]:
            row = ds[int(index)]
            rows.append((decode(row["video"]), label_id[label], int(index), str(row.get("video_id", index))))
    return rows

# Train is partitioned without overlap; official test is untouched confirmation.
train_all = {}
for label in labels:
    candidates = [i for i, value in enumerate(hmdb["train"]["label"]) if value == label]
    train_all[label] = np.random.default_rng(SEED + 211 * label_id[label]).permutation(candidates)

train_rows = []
validation_rows = []
development_rows = []
for label in labels:
    order = train_all[label]
    for role, slc, target in [
        ("train", slice(0, 50), train_rows),
        ("validation", slice(50, 60), validation_rows),
        ("development", slice(60, 70), development_rows),
    ]:
        for index in order[slc]:
            row = hmdb["train"][int(index)]
            target.append((decode(row["video"]), label_id[label], int(index), str(row.get("video_id", index))))
confirmation_rows = select_records("test", 20, 9000)


def pack(rows):
    return {
        "x": np.stack([row[0] for row in rows]),
        "y": np.asarray([row[1] for row in rows], dtype=np.int64),
        "source_index": np.asarray([row[2] for row in rows], dtype=np.int64),
        "source_key": np.asarray([row[3] for row in rows]),
    }

video_payload = {}
for role, rows in [("train", train_rows), ("validation", validation_rows), ("development", development_rows)]:
    for key, value in pack(rows).items():
        video_payload[f"{role}_{key}"] = value
np.savez_compressed(OUT / "hmdb51_6x32_rmpf_train_dev.npz", **video_payload)
np.savez_compressed(
    OUT / "hmdb51_6x32_rmpf_confirmation.npz",
    **{f"confirmation_{key}": value for key, value in pack(confirmation_rows).items()},
)

artifacts = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in OUT.glob("*.npz")}
manifest = {
    "dataset_version": "rmpf-genuine-v1",
    "seed": SEED,
    "cifar10": {
        "repo": "uoft-cs/cifar10",
        "revision": CIFAR_REV,
        "shape": [32, 32, 3],
        "split_sizes": {**{k: len(v) for k, v in roles.items()}, "confirmation": len(confirmation)},
        "split_hashes": {**{k: ih(np.asarray(sorted(v))) for k, v in roles.items()}, "confirmation": ih(50000 + np.asarray(confirmation))},
        "excluded_v1_v2_train": len(used_train),
        "excluded_v1_v2_test": len(used_test),
    },
    "hmdb51": {
        "repo": "divm/hmdb51",
        "revision": HMDB_REV,
        "labels": labels,
        "shape": [6, 32, 32, 3],
        "split_sizes": {"train": len(train_rows), "validation": len(validation_rows), "development": len(development_rows), "confirmation": len(confirmation_rows)},
        "official_test_is_confirmation": True,
    },
    "artifacts": artifacts,
}
(OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
(OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(OUT.glob("*")) if p.name != "SHA256SUMS"))
print(json.dumps(manifest, indent=2, sort_keys=True))
