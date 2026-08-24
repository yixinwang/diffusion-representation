"""Manifest and leakage guards for observed-data QALT studies."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


UCF_BASENAME = re.compile(
    r"^v_(?P<action>.+)_g(?P<group>\d{2})_c(?P<clip>\d{2})\.avi$"
)
UCF_SPLITS = {"train", "val", "test"}
CIFAR_TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))


@dataclass(frozen=True)
class UCFRecord:
    split: str
    class_name: str
    group_id: str
    clip_id: str
    source_relpath: str
    source_size: int


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_ucf_member(name: str, size: int) -> UCFRecord:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    split_positions = [index for index, part in enumerate(path.parts) if part in UCF_SPLITS]
    if len(split_positions) != 1:
        raise ValueError(f"missing or ambiguous split: {name}")
    split_index = split_positions[0]
    tail = path.parts[split_index + 1 :]
    if tail and tail[0] == "UCF101":
        tail = tail[1:]
    if len(tail) != 2:
        raise ValueError(f"unexpected UCF path layout: {name}")
    split = path.parts[split_index]
    class_name = tail[0]
    match = UCF_BASENAME.fullmatch(path.name)
    if match is None or match.group("action") != class_name:
        raise ValueError(f"filename/class mismatch: {name}")
    group = int(match.group("group"))
    clip = int(match.group("clip"))
    return UCFRecord(
        split=split,
        class_name=class_name,
        group_id=f"{class_name}/g{group:02d}",
        clip_id=f"c{clip:02d}",
        source_relpath=name,
        source_size=size,
    )


def inspect_ucf_archive(path: Path) -> list[UCFRecord]:
    records: list[UCFRecord] = []
    identities: set[tuple[str, str]] = set()
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsupported archive member: {member.name}")
            record = parse_ucf_member(member.name, member.size)
            identity = (record.group_id, record.clip_id)
            if identity in identities:
                raise ValueError(f"duplicate UCF clip: {identity}")
            identities.add(identity)
            records.append(record)
    assert_group_disjoint(records)
    return sorted(records, key=lambda row: (row.split, row.source_relpath))


def assert_group_disjoint(records: Iterable[UCFRecord]) -> None:
    split_groups: dict[str, set[str]] = {split: set() for split in UCF_SPLITS}
    for record in records:
        split_groups[record.split].add(record.group_id)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise ValueError(f"UCF group leakage between {left} and {right}: {sorted(overlap)}")


def load_cifar_training_batches(root: Path) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_name in CIFAR_TRAIN_BATCHES:
        path = root / batch_name
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        data = np.asarray(batch[b"data"], dtype=np.uint8)
        target = np.asarray(batch[b"labels"], dtype=np.int64)
        if data.shape != (10_000, 3_072) or target.shape != (10_000,):
            raise ValueError(f"unexpected CIFAR batch shape in {batch_name}")
        images.append(data.reshape(10_000, 3, 32, 32))
        labels.append(target)
    return np.concatenate(images), np.concatenate(labels)


def stratified_cifar_split(labels: np.ndarray, seed: int = 20260823) -> dict[str, list[int]]:
    labels = np.asarray(labels)
    if labels.shape != (50_000,) or set(np.unique(labels)) != set(range(10)):
        raise ValueError("expected all 50,000 CIFAR training labels and ten classes")
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    validation: list[int] = []
    for class_id in range(10):
        indices = np.flatnonzero(labels == class_id)
        if indices.size != 5_000:
            raise ValueError(f"expected 5,000 examples for class {class_id}")
        shuffled = rng.permutation(indices)
        fit.extend(shuffled[:4_500].tolist())
        validation.extend(shuffled[4_500:].tolist())
    return {"fit": sorted(fit), "validation": sorted(validation)}


def require_test_confirmation(
    phase: str,
    config_hash: str | None,
    frozen_confirmation_hash: str | None,
) -> None:
    if phase != "confirmation":
        raise PermissionError("test data require phase='confirmation'")
    if not config_hash or config_hash != frozen_confirmation_hash:
        raise PermissionError("test data require the frozen confirmation config hash")
