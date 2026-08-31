"""Validate and aggregate frozen B1-v2 per-seed score artifacts.

This command is intentionally dataset-agnostic: it reads only score and
provenance artifacts written by the five registered seed runs.  It never
imports a dataset loader or refits a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from qalt.observed_block_statistics import (
    BAND_COUNT,
    CLASS_COUNT,
    SITES_PER_BAND,
    evaluate_registered_gates,
    evaluate_registered_routes,
)


REGISTERED_SEEDS = tuple(range(2100, 2105))
ARM_NAMES = (
    "b1",
    "b4",
    "p4",
    "z4",
    "d4",
    "b4_unconditional",
    "b8",
    "o4",
    "a4",
    "a8",
    "i4",
    "i8",
    "e4",
)
SCORE_KEYS = ("record_ids", "labels", *ARM_NAMES)
SUMMARY_FIELDS = (
    "schema_version",
    "status",
    "sites_per_band",
    "seed",
    "source_commit",
    "protocol_hash",
    "execution_protocol_hash",
    "max_iterations",
    "split_hash",
    "fit_input_hash",
    "holdout_input_hash",
    "site_sample_hash",
    "source_hashes",
    "opened_files",
    "official_test_deserialized",
)
COMMON_METADATA_FIELDS = (
    "schema_version",
    "status",
    "sites_per_band",
    "source_commit",
    "protocol_hash",
    "execution_protocol_hash",
    "max_iterations",
    "split_hash",
    "fit_input_hash",
    "holdout_input_hash",
)
EXPECTED_IMAGES_PER_CLASS = 500
E4_TIE_ATOL = 1e-8
SCHEMA_VERSION = "observed_b1_rgb_block_seed_v1"
AGGREGATE_PATH = Path(__file__).resolve()
PROJECT_ROOT = AGGREGATE_PATH.parents[3]
STATISTICS_PATH = PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_block_statistics.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_summary(path: Path, expected_seed: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    missing = set(SUMMARY_FIELDS) - set(value)
    if missing:
        raise ValueError(f"{path} is missing summary fields: {sorted(missing)}")
    if value["seed"] != expected_seed:
        raise ValueError(f"{path} declares seed {value['seed']!r}, expected {expected_seed}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path} must declare schema_version={SCHEMA_VERSION}")
    if value["status"] not in {
        "adaptive_development_no_coverage",
        "exploratory_optimization_child_no_coverage",
    }:
        raise ValueError(f"{path} declares an unknown study status")
    if value["sites_per_band"] != SITES_PER_BAND:
        raise ValueError(f"{path} must declare sites_per_band={SITES_PER_BAND}")
    if value["max_iterations"] not in {200, 1_000}:
        raise ValueError(f"{path} must declare a frozen max_iterations value")
    for field in (*COMMON_METADATA_FIELDS, "site_sample_hash"):
        if field in {"schema_version", "sites_per_band", "max_iterations"}:
            continue
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{path} field {field} must be a nonempty string")
    source_hashes = value["source_hashes"]
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(not isinstance(name, str) or not isinstance(digest, str) for name, digest in source_hashes.items())
        or any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in source_hashes.values())
    ):
        raise ValueError(f"{path} source_hashes must map names to lowercase SHA-256 strings")
    opened_files = value["opened_files"]
    if not isinstance(opened_files, list) or any(not isinstance(item, str) for item in opened_files):
        raise ValueError(f"{path} opened_files must be a list of strings")
    offending = [item for item in opened_files if "test_batch" in item.lower()]
    if offending:
        raise ValueError(f"official CIFAR test_batch appears in opened_files: {offending}")
    if value["official_test_deserialized"] is not False:
        raise ValueError(f"{path} must declare official_test_deserialized=false")
    return value


def _read_scores(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(SCORE_KEYS):
                missing = set(SCORE_KEYS) - set(archive.files)
                extra = set(archive.files) - set(SCORE_KEYS)
                raise ValueError(
                    f"{path} score schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in SCORE_KEYS}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "score schema mismatch" in str(error):
            raise
        raise ValueError(f"cannot read safe numeric arrays from {path}: {error}") from error

    record_ids = arrays["record_ids"]
    labels = arrays["labels"]
    if record_ids.ndim != 1 or record_ids.dtype.kind not in "iuUS":
        raise ValueError(f"{path} record_ids must be a one-dimensional integer or string array")
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError(f"{path} record_ids must be unique")
    if labels.ndim != 1 or labels.shape != record_ids.shape or labels.dtype.kind not in "iu":
        raise ValueError(f"{path} labels must be a one-dimensional integer array aligned to record_ids")
    class_counts = np.bincount(labels.astype(np.int64), minlength=CLASS_COUNT)
    if (
        len(class_counts) != CLASS_COUNT
        or not np.array_equal(np.unique(labels), np.arange(CLASS_COUNT))
        or np.any(class_counts != EXPECTED_IMAGES_PER_CLASS)
    ):
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_IMAGES_PER_CLASS} ordered images per class 0,...,9"
        )
    for arm in ARM_NAMES:
        values = arrays[arm]
        if values.shape != (len(record_ids), BAND_COUNT) or values.dtype.kind not in "fc":
            raise ValueError(f"{path} arm {arm} must be a floating array with shape (images, 3)")
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path} arm {arm} must contain finite real scores")
    return arrays


def load_registered_artifacts(input_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load five seed directories and enforce the frozen cross-seed contract."""
    root = Path(input_root)
    if not root.is_dir():
        raise ValueError(f"input root is not a directory: {root}")
    seed_pattern = re.compile(r"seed_(\d+)")
    found = {
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := seed_pattern.fullmatch(child.name)) is not None
    }
    expected = set(REGISTERED_SEEDS)
    if found != expected:
        raise ValueError(
            f"expected exactly seed directories {list(REGISTERED_SEEDS)}; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )

    summaries: list[dict[str, Any]] = []
    seed_scores: list[dict[str, np.ndarray]] = []
    for seed in REGISTERED_SEEDS:
        directory = root / f"seed_{seed}"
        summaries.append(_read_summary(directory / "summary.json", seed))
        seed_scores.append(_read_scores(directory / "scores.npz"))

    reference_summary = summaries[0]
    for summary in summaries[1:]:
        for field in COMMON_METADATA_FIELDS:
            if summary[field] != reference_summary[field]:
                raise ValueError(f"cross-seed metadata mismatch for {field}")
        if summary["source_hashes"] != reference_summary["source_hashes"]:
            raise ValueError("cross-seed metadata mismatch for source_hashes")
    required_current_hashes = {
        "aggregate": _sha256_file(AGGREGATE_PATH),
        "statistics": _sha256_file(STATISTICS_PATH),
    }
    for name, actual_digest in required_current_hashes.items():
        declared_digest = reference_summary["source_hashes"].get(name)
        if declared_digest != actual_digest:
            raise ValueError(
                f"declared source_hashes[{name!r}] does not match the current audited file"
            )
    reference_ids = seed_scores[0]["record_ids"]
    reference_labels = seed_scores[0]["labels"]
    for scores in seed_scores[1:]:
        if not np.array_equal(scores["record_ids"], reference_ids):
            raise ValueError("record_ids differ in value or order across seeds")
        if not np.array_equal(scores["labels"], reference_labels):
            raise ValueError("labels differ in value or order across seeds")

    stacked = {
        arm: np.stack([scores[arm] for scores in seed_scores], axis=0)
        for arm in ARM_NAMES
    }
    maximum_e4_difference = float(np.max(np.abs(stacked["b4"] - stacked["e4"])))
    if maximum_e4_difference > E4_TIE_ATOL:
        raise ValueError(
            f"B4/E4 samplewise tie failed: max_abs_difference={maximum_e4_difference:.17g} "
            f"> {E4_TIE_ATOL}"
        )
    metadata = {
        "common": {field: reference_summary[field] for field in COMMON_METADATA_FIELDS},
        "source_hashes": reference_summary["source_hashes"],
        "site_sample_hashes": {
            str(seed): summary["site_sample_hash"]
            for seed, summary in zip(REGISTERED_SEEDS, summaries)
        },
        "opened_files": {
            str(seed): summary["opened_files"]
            for seed, summary in zip(REGISTERED_SEEDS, summaries)
        },
        "official_test_deserialized": False,
        "maximum_b4_e4_abs_difference": maximum_e4_difference,
    }
    return {
        "record_ids": reference_ids,
        "labels": reference_labels,
        **stacked,
    }, metadata


def aggregate(input_root: Path, output_dir: Path) -> dict[str, Any]:
    """Validate artifacts, run frozen statistics, and write a new ensemble directory."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    arrays, provenance = load_registered_artifacts(Path(input_root))
    scores = {arm: arrays[arm].astype(np.float64, copy=False) for arm in ARM_NAMES}
    labels = arrays["labels"].astype(np.int64, copy=False)
    gates = evaluate_registered_gates(scores, labels)
    routes = evaluate_registered_routes(scores, labels, expected_per_class=EXPECTED_IMAGES_PER_CLASS)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": provenance["common"]["status"],
        "registered_seeds": list(REGISTERED_SEEDS),
        "image_count": int(len(labels)),
        "images_per_class": EXPECTED_IMAGES_PER_CLASS,
        "score_contract": "raw band-total log scores; each band sums 256 sites x 3 RGB coordinates",
        "provenance": provenance,
        "registered_gates": gates,
        "registered_routes": routes,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "ensemble_scores.npz",
        seeds=np.asarray(REGISTERED_SEEDS, dtype=np.int64),
        **arrays,
    )
    (output / "aggregate_summary.json").write_text(summary_text)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate(args.input_root, args.output)


if __name__ == "__main__":
    main()
