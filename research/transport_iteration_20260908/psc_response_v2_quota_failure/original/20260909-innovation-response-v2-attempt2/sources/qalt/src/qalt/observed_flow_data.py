"""Native CIFAR development inputs with preserved adaptive split history.

Only the five official training files are eligible for loading. Their complete
contents are deserialized by the existing allowlisted loader; discovery pixels
never enter returned model arrays, fitting statistics or evaluation arrays.
The repair records were previously used in development and are not untouched
confirmation. Class labels are used only for the frozen stratified selection.
There is deliberately no test-loading argument, phase or API in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from qalt.data_integrity import (
    adaptive_cifar_repair_split,
    cifar_training_batch_paths,
    load_cifar_training_batches,
    sha256_file,
    stable_json_hash,
)

SELECTION_SALT = "observed-flow-cifar-selection-v1"
DEQUANTIZATION_SALT = "observed-flow-cifar-dequantization-v1"
FIT_PER_CLASS = 400
REPAIR_PER_CLASS = 100
EXPECTED_SPLIT_HASH = "814c2280cea33403d9370b02ea12c83df3a5f89bbbf593567fed6e960ef321e7"
EXPECTED_ID_HASHES = {
    "fit": "4f002c7cfe2e3d1ca54a7c8847982941d655060e6989a66f87094e4d632be457",
    "repair_holdout": "ced1fb315bba0eeb81c03b6f2fb296afd3208cb02a7cff6dc513327708ea8939",
}
MANIFEST = Path(__file__).resolve().parents[2] / "data" / "observed_manifest_v1.json"


@dataclass(frozen=True)
class ObservedFlowData:
    """Selected float64 NCHW observations, canonical record IDs and provenance.

    Arrays are read-only. Labels and excluded discovery pixels are not returned.
    ``ledger['canonical_dataset_verified']`` must be true for a real-data run.
    """

    fit: np.ndarray
    repair: np.ndarray
    fit_ids: np.ndarray
    repair_ids: np.ndarray
    ledger: dict


def _array_hash(array: np.ndarray) -> str:
    """Raw C-contiguous byte hash, matching the existing split-ID convention."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _record_key(salt: str, role: str, record_id: int) -> bytes:
    if role not in {"fit", "repair"}:
        raise ValueError("role must be fit or repair")
    if not 0 <= record_id < 50_000:
        raise ValueError("canonical record ID must be in [0, 50000)")
    return hashlib.sha256(f"{salt}|{role}|{record_id:05d}".encode("ascii")).digest()


def _select_ids(eligible: list[int], labels: np.ndarray, role: str) -> np.ndarray:
    count = FIT_PER_CLASS if role == "fit" else REPAIR_PER_CLASS
    eligible_array = np.asarray(eligible, dtype=np.int64)
    selected = []
    for class_id in range(10):
        candidates = eligible_array[labels[eligible_array] == class_id].tolist()
        if len(candidates) < count:
            raise ValueError("too few eligible records for the frozen class quota")
        ranked = sorted(candidates, key=lambda i: (_record_key(SELECTION_SALT, role, i), i))
        selected.extend(ranked[:count])
    return np.asarray(sorted(selected), dtype="<i8")


def _dequantize_record(pixels: np.ndarray, record_id: int, role: str) -> np.ndarray:
    """Deterministic record-keyed finite-precision uniform dequantization.

    The uniform is a midpoint on the 2**32-cell grid. Its positive distance
    from both endpoints keeps (255+U)/256 strictly below one in float64,
    without clipping. PCG64 is specified explicitly rather than implicitly.
    """
    if pixels.shape != (3, 32, 32) or pixels.dtype != np.uint8:
        raise ValueError("expected native 3x32x32 uint8 pixels")
    seed = int.from_bytes(_record_key(DEQUANTIZATION_SALT, role, record_id), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = rng.integers(0, 1 << 32, size=pixels.shape, dtype=np.uint64).astype(np.float64)
    result += 0.5
    result *= 2.0 ** -32
    result += pixels
    result *= 1.0 / 256.0
    if not np.all((result > 0.0) & (result < 1.0)):
        raise FloatingPointError("dequantization left the open unit cube")
    return result


def _build_selected(images: np.ndarray, ids: np.ndarray, role: str) -> tuple[np.ndarray, str]:
    result = np.empty((len(ids), 3, 32, 32), dtype=np.float64)
    pixel_digest = hashlib.sha256()
    for position, record_id in enumerate(ids):
        pixels = images[int(record_id)]
        pixel_digest.update(memoryview(np.ascontiguousarray(pixels)).cast("B"))
        result[position] = _dequantize_record(pixels, int(record_id), role)
    return result, pixel_digest.hexdigest()


def load_observed_flow_data(
    root: Path, *, allow_noncanonical_fixture: bool = False
) -> ObservedFlowData:
    """Load only frozen training/repair subsets; never load official test data.

    The default checks all five canonical file hashes before deserialization,
    then verifies the full 40k/5k/5k split and existing fit/repair ID hashes
    before selecting any pixels. ``allow_noncanonical_fixture=True`` exists
    solely for fabricated-file unit tests: it bypasses canonical hash equality,
    retains all loader/shape/split checks, and marks returned metadata as a
    noncanonical fixture. It must not be used for empirical results.
    """
    if not isinstance(allow_noncanonical_fixture, bool):
        raise TypeError("allow_noncanonical_fixture must be an explicit bool")
    root = Path(root)
    paths = cifar_training_batch_paths(root)
    expected_files = json.loads(MANIFEST.read_text())["cifar10"]["allowed_development_batches"]
    file_ledger = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"training batch must be a regular nonsymlink file: {path.name}")
        actual = sha256_file(path)
        if not allow_noncanonical_fixture and actual != expected_files[path.name]:
            raise ValueError(f"canonical training batch hash mismatch: {path.name}")
        file_ledger.append({"name": path.name, "path": str(path.resolve()),
                            "size_bytes": path.stat().st_size, "sha256": actual})
    opened_files: list[str] = []
    images, labels = load_cifar_training_batches(root, opened_files=opened_files)
    if opened_files != [str(p.resolve()) for p in paths]:
        raise AssertionError("training loader opened a file outside its exclusive allowlist")
    split = adaptive_cifar_repair_split(labels)
    expected_sizes = {"fit": 40_000, "repair_holdout": 5_000, "excluded_discovery": 5_000}
    if {k: len(v) for k, v in split.items()} != expected_sizes:
        raise AssertionError("full adaptive split counts changed")
    split_hash = stable_json_hash(split)
    id_hashes = {k: _array_hash(np.asarray(v, dtype="<i8")) for k, v in split.items()}
    if not allow_noncanonical_fixture:
        if split_hash != EXPECTED_SPLIT_HASH:
            raise ValueError("canonical full adaptive split hash mismatch")
        if any(id_hashes[k] != v for k, v in EXPECTED_ID_HASHES.items()):
            raise ValueError("canonical full fitting/repair ID hash mismatch")
    fit_ids = _select_ids(split["fit"], labels, "fit")
    repair_ids = _select_ids(split["repair_holdout"], labels, "repair")
    excluded = set(split["excluded_discovery"])
    fit_set, repair_set = set(fit_ids.tolist()), set(repair_ids.tolist())
    if fit_set & repair_set or (fit_set | repair_set) & excluded:
        raise AssertionError("selected records overlap or include excluded discovery")
    fit, fit_pixels_hash = _build_selected(images, fit_ids, "fit")
    repair, repair_pixels_hash = _build_selected(images, repair_ids, "repair")
    # No all-fitting, discovery or holdout pixel statistic is computed. Only
    # selected native pixels are hashed/dequantized after split verification.
    del images, labels
    ledger = {
        "schema_version": 1,
        "dataset": "fabricated-cifar-shaped-fixture" if allow_noncanonical_fixture else "cifar10-training",
        "canonical_dataset_verified": not allow_noncanonical_fixture,
        "allow_noncanonical_fixture": allow_noncanonical_fixture,
        "adaptive_history": "repair split is recycled development; original discovery remains excluded",
        "label_usage": "split and class quotas only; no labels returned to models",
        "record_id_convention": "zero-based ordered concat data_batch_1 through data_batch_5",
        "file_allowlist": [p.name for p in paths], "files": file_ledger,
        "opened_files": opened_files, "test_data_accessed": False,
        "full_split_counts": expected_sizes, "full_split_sha256": split_hash,
        "full_partition_id_sha256": id_hashes,
        "fit_size": len(fit_ids), "repair_size": len(repair_ids),
        "fit_per_class": FIT_PER_CLASS, "repair_per_class": REPAIR_PER_CLASS,
        "fit_repair_intersection": 0, "selected_discovery_intersection": 0,
        "selection_salt": SELECTION_SALT, "dequantization_salt": DEQUANTIZATION_SALT,
        "record_key_encoding": "ASCII salt|role|five-digit-zero-padded-record-id; SHA256",
        "dequantization": "PCG64(int.from_bytes(digest,'little')); U=(uint32_integer+0.5)/2**32; X=(pixel+U)/256",
        "dequantization_variability": "one fixed draw per role and record, shared by all methods and training seeds",
        "shape_per_record": [3, 32, 32], "dtype": "float64",
        "fit_ids_sha256": _array_hash(fit_ids), "repair_ids_sha256": _array_hash(repair_ids),
        "fit_selected_uint8_sha256": fit_pixels_hash, "repair_selected_uint8_sha256": repair_pixels_hash,
        "fit_dequantized_sha256": _array_hash(fit), "repair_dequantized_sha256": _array_hash(repair),
        "numpy_version": np.__version__,
    }
    ledger["ledger_sha256"] = stable_json_hash(ledger)
    for array in (fit, repair, fit_ids, repair_ids):
        array.setflags(write=False)
    return ObservedFlowData(fit, repair, fit_ids, repair_ids, ledger)


__all__ = ["ObservedFlowData", "load_observed_flow_data"]
