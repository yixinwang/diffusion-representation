from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("build/out")
RAW = Path("build/raw")
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
V1 = json.loads((ROOT / "data_build/mcqf_v2/cifar_v1_exclusions.json").read_text())
V2_SEED = 20260828


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def downsample_2x(images: np.ndarray) -> np.ndarray:
    sums = images.astype(np.uint16).reshape(len(images), 16, 2, 16, 2, 3).sum((2, 4))
    return ((sums + 2) // 4).astype(np.uint8)


api = HfApi()
cifar_revision = api.dataset_info("uoft-cs/cifar10").sha
cifar = load_dataset("uoft-cs/cifar10", revision=cifar_revision)


def cifar_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    part = cifar[split]
    images = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in part["img"]])
    labels = np.asarray(part["label"], dtype=np.int64)
    return images, labels


x32, y = cifar_arrays("train")
xt32, yt = cifar_arrays("test")
source_ids = np.arange(50000, dtype=np.int64)
test_ids = np.arange(50000, 60000, dtype=np.int64)

# Reconstruct all MCQF-v1 IDs and assert their frozen hashes before excluding them.
old_rng = np.random.default_rng(int(V1["prior_seed"]))
old_counts = {"train": 1200, "validation": 200, "fiber_fit": 200, "development": 200}
old: dict[str, list[int]] = {key: [] for key in old_counts}
for class_id in range(10):
    indices = old_rng.permutation(np.flatnonzero(y == class_id))
    cursor = 0
    for name, count in old_counts.items():
        old[name].extend(indices[cursor : cursor + count].tolist())
        cursor += count
old_test: list[int] = []
for class_id in range(10):
    indices = np.random.default_rng(int(V1["prior_seed"]) + 991 + class_id).permutation(
        np.flatnonzero(yt == class_id)
    )
    old_test.extend(indices[:200].tolist())
old_hashes = {key: index_hash(np.asarray(sorted(values))) for key, values in old.items()}
old_hashes["confirmation"] = index_hash(test_ids[np.asarray(sorted(old_test))])
if old_hashes != V1["prior_split_source_id_hashes"]:
    raise RuntimeError({"expected": V1["prior_split_source_id_hashes"], "actual": old_hashes})
used_train = set().union(*(set(values) for values in old.values()))
used_test = set(old_test)
assert len(used_train) == 18000 and len(used_test) == 2000

# Fresh official CIFAR-10 data, disjoint from every v1 role.
new_counts = {"train": 1600, "validation": 300, "fiber_fit": 300, "development": 300}
new: dict[str, list[int]] = {key: [] for key in new_counts}
for class_id in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(y == class_id).tolist()) - used_train), dtype=np.int64)
    permutation = np.random.default_rng(V2_SEED + 17 * class_id).permutation(available)
    cursor = 0
    for name, count in new_counts.items():
        new[name].extend(permutation[cursor : cursor + count].tolist())
        cursor += count
new_test: list[int] = []
for class_id in range(10):
    available = np.asarray(sorted(set(np.flatnonzero(yt == class_id).tolist()) - used_test), dtype=np.int64)
    new_test.extend(np.random.default_rng(V2_SEED + 991 + class_id).permutation(available)[:200].tolist())
new_test_array = np.asarray(sorted(new_test), dtype=np.int64)
all_new = set().union(*(set(values) for values in new.values()))
if all_new & used_train or set(new_test_array.tolist()) & used_test:
    raise AssertionError("MCQF-v1/v2 overlap")

x = downsample_2x(x32)
xt = downsample_2x(xt32)
payload: dict[str, np.ndarray] = {}
for name, values in new.items():
    indices = np.asarray(sorted(values), dtype=np.int64)
    payload[f"{name}_x"] = x[indices]
    payload[f"{name}_y"] = y[indices]
    payload[f"{name}_source_id"] = source_ids[indices]
np.savez_compressed(OUT / "cifar10_16_v2_train_dev.npz", **payload)
np.savez_compressed(
    OUT / "cifar10_16_v2_confirmation.npz",
    confirmation_x=xt[new_test_array],
    confirmation_y=yt[new_test_array],
    confirmation_source_id=test_ids[new_test_array],
)

# Pin and download a released real UCF101 subset.
ucf_repo = "sayakpaul/ucf101-subset"
ucf_revision = api.dataset_info(ucf_repo).sha
archive = Path(
    hf_hub_download(
        ucf_repo,
        "UCF101_subset.tar.gz",
        repo_type="dataset",
        revision=ucf_revision,
        local_dir=RAW,
    )
)
expected_archive_sha = "e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb"
if sha(archive) != expected_archive_sha:
    raise RuntimeError(("UCF archive hash", sha(archive)))
extract_root = RAW / "ucf"
extract_root.mkdir(exist_ok=True)
# The pinned file is an uncompressed POSIX tar despite its historical .tar.gz suffix.
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".avi"):
            continue
        target = (extract_root / member.name).resolve()
        if extract_root.resolve() not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = handle.extractfile(member)
        if source is None:
            continue
        with target.open("wb") as destination:
            destination.write(source.read())

videos = sorted(extract_root.rglob("*.avi"))
if len(videos) < 300:
    raise RuntimeError(f"expected at least 300 UCF videos, found {len(videos)}")
pattern = re.compile(r"^v_(?P<action>.+)_g(?P<group>\d+)_c(?P<clip>\d+)\.avi$", re.I)
metadata: list[tuple[Path, str, int, int]] = []
for path in videos:
    match = pattern.match(path.name)
    if match:
        metadata.append((path, match.group("action"), int(match.group("group")), int(match.group("clip"))))
if len(metadata) < 300:
    raise RuntimeError(f"parsed only {len(metadata)} UCF videos")
actions = sorted({action for _, action, _, _ in metadata})
action_id = {action: index for index, action in enumerate(actions)}
groups = sorted({group for _, _, group, _ in metadata})

# Outcome-blind source-group roles are fixed from filenames before pixel decoding.
role_by_group: dict[int, str] = {}
for index, group in enumerate(groups):
    residue = index % 10
    role_by_group[group] = (
        "confirmation"
        if residue in (8, 9)
        else "validation"
        if residue == 7
        else "development"
        if residue == 6
        else "fiber_fit"
        if residue == 5
        else "train"
    )
records: dict[str, list[tuple[np.ndarray, int, int, int, int, int]]] = {
    name: [] for name in ["train", "validation", "fiber_fit", "development", "confirmation"]
}
source_videos: list[dict[str, object]] = []
for video_id, (path, action, group, clip_id) in enumerate(metadata):
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < 12:
        continue
    array = np.stack(frames)
    height, width = array.shape[1:3]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    array = array[:, top : top + side, left : left + side]
    array = np.stack([cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA) for frame in array]).astype(
        np.uint8
    )
    starts = [0, max(0, len(array) - 12)] if len(array) >= 24 else [0]
    for view_id, start in enumerate(sorted(set(starts))):
        frame_ids = np.linspace(start, min(len(array) - 1, start + 11), 6).round().astype(int)
        records[role_by_group[group]].append(
            (array[frame_ids], action_id[action], video_id, group, clip_id, view_id)
        )
    source_videos.append(
        {
            "video_id": video_id,
            "file": path.name,
            "action": action,
            "group": group,
            "clip_id": clip_id,
            "role": role_by_group[group],
            "frames": len(array),
        }
    )


def pack(rows: list[tuple[np.ndarray, int, int, int, int, int]]) -> dict[str, np.ndarray]:
    return {
        "x": np.stack([row[0] for row in rows]),
        "y": np.asarray([row[1] for row in rows], dtype=np.int64),
        "video_id": np.asarray([row[2] for row in rows], dtype=np.int64),
        "group_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "clip_id": np.asarray([row[4] for row in rows], dtype=np.int64),
        "view_id": np.asarray([row[5] for row in rows], dtype=np.int64),
    }


for role, rows in records.items():
    if len(rows) < 20:
        raise RuntimeError((role, len(rows)))
video_payload: dict[str, np.ndarray] = {}
for role in ["train", "validation", "fiber_fit", "development"]:
    for name, value in pack(records[role]).items():
        video_payload[f"{role}_{name}"] = value
np.savez_compressed(OUT / "ucf101_6x16_v2_train_dev.npz", **video_payload)
np.savez_compressed(
    OUT / "ucf101_6x16_v2_confirmation.npz",
    **{f"confirmation_{name}": value for name, value in pack(records["confirmation"]).items()},
)

artifacts = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in OUT.glob("*.npz")}
manifest = {
    "dataset_version": "mcqf-v2-fresh-cifar-ucf-v1",
    "builder_seed": V2_SEED,
    "builder_environment": {"numpy": np.__version__, "opencv": cv2.__version__},
    "cifar10": {
        "source_repo": "uoft-cs/cifar10",
        "source_revision": cifar_revision,
        "dataset_fingerprints": {key: value._fingerprint for key, value in cifar.items()},
        "resolution": [16, 16, 3],
        "prior_exclusion_hashes": old_hashes,
        "split_sizes": {**{key: len(value) for key, value in new.items()}, "confirmation": len(new_test_array)},
        "split_source_id_hashes": {
            **{key: index_hash(source_ids[np.asarray(sorted(value))]) for key, value in new.items()},
            "confirmation": index_hash(test_ids[new_test_array]),
        },
        "v1_v2_train_overlap": 0,
        "v1_v2_confirmation_overlap": 0,
    },
    "ucf101_subset": {
        "source_repo": ucf_repo,
        "source_revision": ucf_revision,
        "archive_sha256": expected_archive_sha,
        "actions": actions,
        "clip_shape": [6, 16, 16, 3],
        "split_sizes": {key: len(value) for key, value in records.items()},
        "group_roles": {str(key): value for key, value in role_by_group.items()},
        "source_videos": source_videos,
    },
    "artifacts": artifacts,
}
(OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
(OUT / "SHA256SUMS").write_text(
    "".join(f"{sha(path)}  {path.name}\n" for path in sorted(OUT.glob("*")) if path.name != "SHA256SUMS")
)
print(json.dumps({"dataset_version": manifest["dataset_version"], "cifar10": manifest["cifar10"], "ucf101_subset": {key: value for key, value in manifest["ucf101_subset"].items() if key != "source_videos"}}, indent=2))
