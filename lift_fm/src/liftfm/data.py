from __future__ import annotations

import gzip
import hashlib
from importlib import resources
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


_RECORDS = 1_797
_PIXELS = 64


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digits_path() -> Path:
    return Path(resources.files("sklearn.datasets").joinpath("data/digits.csv.gz"))


def _load_labels_only() -> np.ndarray:
    """Read only the final label field; do not parse any image pixels."""
    labels = np.empty(_RECORDS, dtype=np.int64)
    with gzip.open(_digits_path(), "rt", encoding="ascii", newline="") as handle:
        for record_id, line in enumerate(handle):
            if record_id >= _RECORDS:
                raise ValueError("digits source contains too many records")
            labels[record_id] = int(line.rsplit(",", 1)[1])
    if record_id + 1 != _RECORDS:
        raise ValueError("digits source has an unexpected record count")
    return labels


def fixed_split(labels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    target = _load_labels_only() if labels is None else np.asarray(labels, dtype=np.int64)
    if target.shape != (_RECORDS,):
        raise ValueError("expected 1,797 labels")
    indices = np.arange(len(target))
    train, remainder = train_test_split(
        indices,
        test_size=0.4,
        random_state=20260826,
        stratify=target,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=20260827,
        stratify=target[remainder],
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def split_manifest() -> dict[str, object]:
    split = fixed_split()
    return {
        "sizes": {name: int(len(values)) for name, values in split.items()},
        "hashes": {
            name: hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()
            for name, values in split.items()
        },
        "test_sealed_hash": hashlib.sha256(
            np.asarray(split["test"], dtype="<i8").tobytes()
        ).hexdigest(),
        "pixel_access_contract": (
            "labels are read to freeze the stratified split; pixel fields are parsed only for "
            "the requested partition, and test pixel parsing requires allow_test=True"
        ),
    }


def _record_uniform(record_ids: np.ndarray, seed: int, dimension: int = 64) -> np.ndarray:
    ids = np.asarray(record_ids, dtype=np.uint64)
    pixel = np.arange(dimension, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        counter = ids[:, None] * np.uint64(dimension) + pixel + np.uint64(seed) * np.uint64(0x9E3779B1)
        counter ^= counter >> np.uint64(30)
        counter *= np.uint64(0xBF58476D1CE4E5B9)
        counter ^= counter >> np.uint64(27)
        counter *= np.uint64(0x94D049BB133111EB)
        counter ^= counter >> np.uint64(31)
    return ((counter >> np.uint64(40)).astype(np.float64) + 0.5) / float(1 << 24)


def load_partition(name: str, seed: int, allow_test: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name not in {"train", "validation", "test"}:
        raise ValueError(f"unknown partition: {name}")
    if name == "test" and not allow_test:
        raise PermissionError("test partition pixels are sealed")

    labels = _load_labels_only()
    split = fixed_split(labels)
    indices = split[name]
    selected = set(int(value) for value in indices)
    values = np.empty((len(indices), _PIXELS), dtype=np.float64)
    targets = np.empty(len(indices), dtype=np.int64)
    locations = {int(record_id): position for position, record_id in enumerate(indices)}

    parsed = 0
    with gzip.open(_digits_path(), "rt", encoding="ascii", newline="") as handle:
        for record_id, line in enumerate(handle):
            if record_id not in selected:
                continue
            fields = np.fromstring(line, sep=",", dtype=np.float64)
            if fields.shape != (_PIXELS + 1,):
                raise ValueError(f"malformed digits record {record_id}")
            position = locations[record_id]
            values[position] = fields[:-1]
            targets[position] = int(fields[-1])
            parsed += 1
    if parsed != len(indices):
        raise ValueError(f"parsed {parsed} records for {name}, expected {len(indices)}")
    if not np.array_equal(targets, labels[indices]):
        raise AssertionError("streamed pixel records and label manifest disagree")

    values = (values + _record_uniform(indices, seed)) / 17.0
    return values.reshape(-1, 8, 8), targets, indices
