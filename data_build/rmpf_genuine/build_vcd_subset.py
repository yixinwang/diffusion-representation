from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

OUT = Path("build/out")
RAW = Path("build/raw/vcd")
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://vcdpublic.blob.core.windows.net/vcd1/"
VCD_REPO_COMMIT = "e4c6598cb349bcf2f2cd5df55ef76cf58e947e77"
SEED = 20260829

# Six fixed source videos per scenario. Positions 0:3 train, 3 validation,
# 4 development, and 5 untouched confirmation.
SCENARIOS = {
    "th": [
        "mp4/th/54e9f7b6a15b07e90b4836eb6ffb58ae_1920x1080_30.mp4",
        "mp4/th/2326ac1d3e069fbd86e2ca79082e19f9_1920x1080_30.mp4",
        "mp4/th/fbdfa436d72b83ff284396579bcd6da5_1920x1080_30.mp4",
        "mp4/th/8bff85c461e9f6ae39f3900d581b1d05_1920x1080_30.mp4",
        "mp4/th/0380a333cb0fef69001fb4260e2a705f_1920x1080_30.mp4",
        "mp4/th/2242e0be13a37cf78b00aaeafa54d4a7_1920x1080_30.mp4",
    ],
    "th-ob": [
        "mp4/th-ob/1c63cf94eb2bcdfbb57dbf3727c7d695_1920x1080_30.mp4",
        "mp4/th-ob/bff43254e9a55ecf50b9589a4317c189_1920x1080_30.mp4",
        "mp4/th-ob/1eec2811670a1dfa0f697d099b0dfcdb_1920x1080_30.mp4",
        "mp4/th-ob/be19ce722a9e52e2613a6d8f775af82d_1920x1080_30.mp4",
        "mp4/th-ob/58ec81ddb62061e49a986beb8cc9c212_1920x1080_30.mp4",
        "mp4/th-ob/544c8d35bfcd3575f0bfdba2b79dbec6_1920x1080_30.mp4",
    ],
    "th-bb": [
        "mp4/th-bb/6f05900f2375ca5a01202460975afd79_1920x1080_30.mp4",
        "mp4/th-bb/91154c046bba79ee8b24550a8cb2870b_1920x1080_30.mp4",
        "mp4/th-bb/c3089ec3080756f925b5b83fb66af24c_1920x1080_30.mp4",
        "mp4/th-bb/218522b5c782bfd3a4ed1ed57b0686a0_1920x1080_30.mp4",
        "mp4/th-bb/4d4d189e384d477b27a3a3dbbf73b444_1920x1080_30.mp4",
        "mp4/th-bb/b350cada8743a8f22953de74e2b20f8c_1920x1080_30.mp4",
    ],
    "th-m": [
        "mp4/th-m/0296bcc0fdd2a47380e798289dcc099f_1080x1920_30.mp4",
        "mp4/th-m/3f7df79cd3338701dfce79b3ec82531c_1080x1920_30.mp4",
        "mp4/th-m/6b9b28f4a7b953660e611ae4f7140dd4_1080x1920_30.mp4",
        "mp4/th-m/f7b584ac0829ebf36464b22c7860e3f9_1080x1920_30.mp4",
        "mp4/th-m/70a45f44a21a1d3a18d046e41acb497d_1080x1920_30.mp4",
        "mp4/th-m/633188f7e2aee4ac1f4ec0fa25b3ecdd_1080x1920_30.mp4",
    ],
}
ROLE_BY_POSITION = {0: "train", 1: "train", 2: "train", 3: "validation", 4: "development", 5: "confirmation"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(relative: str) -> Path:
    target = RAW / Path(relative).name
    if not target.exists():
        with urlopen(BASE + relative, timeout=120) as source, target.open("wb") as destination:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                destination.write(chunk)
    return target


def decode(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 24:
        raise RuntimeError(f"{path.name}: only {len(frames)} decoded frames")
    return np.stack(frames)


def make_clips(frames: np.ndarray, count: int) -> tuple[list[np.ndarray], list[int]]:
    h, w = frames.shape[1:3]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = frames[:, top:top + side, left:left + side]
    span = 21
    starts = np.unique(np.linspace(0, len(frames) - span, count, dtype=int))
    clips = []
    for start in starts:
        ids = start + np.arange(6) * 4
        clip = np.stack([cv2.resize(frames[i], (32, 32), interpolation=cv2.INTER_AREA) for i in ids]).astype(np.uint8)
        clips.append(clip)
    return clips, starts.tolist()

records = {role: [] for role in ["train", "validation", "development", "confirmation"]}
sources = []
source_id = 0
for scenario_id, (scenario, paths) in enumerate(SCENARIOS.items()):
    for position, relative in enumerate(paths):
        role = ROLE_BY_POSITION[position]
        local = download(relative)
        frames = decode(local)
        count = 20 if role == "train" else 8
        clips, starts = make_clips(frames, count)
        for view_id, (clip, start) in enumerate(zip(clips, starts)):
            records[role].append((clip, scenario_id, source_id, view_id, start))
        sources.append({
            "source_id": source_id,
            "relative_path": relative,
            "url": BASE + relative,
            "sha256": sha(local),
            "bytes": local.stat().st_size,
            "frames": len(frames),
            "scenario": scenario,
            "role": role,
            "clip_count": len(clips),
        })
        source_id += 1


def pack(rows):
    return {
        "x": np.stack([row[0] for row in rows]),
        "y": np.asarray([row[1] for row in rows], dtype=np.int64),
        "source_id": np.asarray([row[2] for row in rows], dtype=np.int64),
        "view_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "start": np.asarray([row[4] for row in rows], dtype=np.int64),
    }

payload = {}
for role in ["train", "validation", "development"]:
    for key, value in pack(records[role]).items():
        payload[f"{role}_{key}"] = value
np.savez_compressed(OUT / "vcd_6x32_rmpf_train_dev.npz", **payload)
np.savez_compressed(
    OUT / "vcd_6x32_rmpf_confirmation.npz",
    **{f"confirmation_{key}": value for key, value in pack(records["confirmation"]).items()},
)

# Source videos are disjoint across every role by construction.
role_sources = {role: sorted({row[2] for row in rows}) for role, rows in records.items()}
for left, left_ids in role_sources.items():
    for right, right_ids in role_sources.items():
        if left < right:
            assert not set(left_ids) & set(right_ids)

manifest = {
    "dataset": "Microsoft VCD v1 public MP4 subset",
    "source_repo": "microsoft/VCD",
    "source_repo_commit": VCD_REPO_COMMIT,
    "base_url": BASE,
    "seed": SEED,
    "shape": [6, 32, 32, 3],
    "split_sizes": {role: len(rows) for role, rows in records.items()},
    "source_ids_by_role": role_sources,
    "confirmation_array_not_opened_by_scientific_runner": True,
    "sources": sources,
    "artifacts": {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in OUT.glob("vcd_6x32_rmpf_*.npz")
    },
}
(OUT / "vcd_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2, sort_keys=True))
