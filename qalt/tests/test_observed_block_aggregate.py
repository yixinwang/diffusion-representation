import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qalt.observed_block_statistics import DETAIL_COUNT


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "observed_b1_rgb_block"
    / "aggregate.py"
)
SPEC = importlib.util.spec_from_file_location("observed_b1_rgb_block_aggregate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_seed_artifacts(root: Path) -> dict[int, dict[str, np.ndarray]]:
    labels = np.repeat(np.arange(10, dtype=np.int64), 500)
    record_ids = np.arange(len(labels), dtype=np.int64) + 90_000
    rng = np.random.default_rng(510)
    shared = rng.normal(0.0, 0.01, len(labels))
    targets = {
        "b1": 1.05,
        "b4": 1.00,
        "p4": 1.05,
        "z4": 1.05,
        "d4": 1.05,
        "b4_unconditional": 1.03,
        "b8": 1.00,
        "o4": 1.05,
        "a4": 1.01,
        "a8": 1.00,
        "i4": 1.00,
        "i8": 1.00,
    }
    written = {}
    for seed in AGGREGATE.REGISTERED_SEEDS:
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True)
        arrays = {"record_ids": record_ids, "labels": labels}
        for arm, target in targets.items():
            image_nll = target + shared + rng.normal(0.0, 0.0002, len(labels))
            band_total = -image_nll * DETAIL_COUNT / 3.0
            arrays[arm] = np.repeat(band_total[:, None], 3, axis=1).astype(np.float64)
        arrays["e4"] = arrays["b4"].copy()
        np.savez_compressed(directory / "scores.npz", **arrays)
        summary = {
            "schema_version": AGGREGATE.SCHEMA_VERSION,
            "sites_per_band": 256,
            "seed": seed,
            "source_commit": "a" * 40,
            "protocol_hash": "protocol",
            "split_hash": "split",
            "fit_input_hash": "fit",
            "holdout_input_hash": "holdout",
            "site_sample_hash": f"sample-{seed}",
            "source_hashes": {
                "aggregate": _sha256(MODULE_PATH),
                "statistics": _sha256(AGGREGATE.STATISTICS_PATH),
            },
            "opened_files": [f"/cifar/data_batch_{index}" for index in range(1, 6)],
            "official_test_deserialized": False,
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        written[seed] = arrays
    return written


def _rewrite_summary(root: Path, seed: int, **changes: object) -> None:
    path = root / f"seed_{seed}" / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(changes)
    path.write_text(json.dumps(summary))


def _rewrite_scores(root: Path, seed: int, transform) -> None:
    path = root / f"seed_{seed}" / "scores.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


def test_aggregate_preserves_raw_seed_scores_and_runs_frozen_statistics(tmp_path: Path) -> None:
    input_root = tmp_path / "seeds"
    written = _write_seed_artifacts(input_root)
    output = tmp_path / "ensemble"
    summary = AGGREGATE.aggregate(input_root, output)

    assert summary["status"] == "adaptive_development_no_coverage"
    assert summary["registered_gates"]["all_gates_pass"]
    assert summary["registered_routes"]["status"] == "diagnostic_pseudo_certificate_no_coverage"
    assert summary["provenance"]["official_test_deserialized"] is False
    disk_summary = json.loads((output / "aggregate_summary.json").read_text())
    assert disk_summary == summary
    with np.load(output / "ensemble_scores.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"seeds", "record_ids", "labels", *AGGREGATE.ARM_NAMES}
        assert archive["b4"].shape == (5, 5_000, 3)
        for index, seed in enumerate(AGGREGATE.REGISTERED_SEEDS):
            assert np.array_equal(archive["b4"][index], written[seed]["b4"])


def test_rejects_missing_or_extra_registered_seed_directory(tmp_path: Path) -> None:
    root = tmp_path / "seeds"
    _write_seed_artifacts(root)
    (root / "seed_2104").rename(root / "seed_9999")
    with pytest.raises(ValueError, match="expected exactly seed directories"):
        AGGREGATE.load_registered_artifacts(root)


def test_rejects_cross_seed_identity_and_common_hash_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "seeds"
    _write_seed_artifacts(root)
    _rewrite_scores(root, 2103, lambda arrays: arrays["record_ids"].__setitem__(0, 1_000_000))
    with pytest.raises(ValueError, match="record_ids differ"):
        AGGREGATE.load_registered_artifacts(root)

    _write_seed_artifacts(tmp_path / "hashes")
    _rewrite_summary(tmp_path / "hashes", 2102, split_hash="changed")
    with pytest.raises(ValueError, match="metadata mismatch for split_hash"):
        AGGREGATE.load_registered_artifacts(tmp_path / "hashes")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"opened_files": ["/cifar/test_batch"]}, "test_batch"),
        ({"official_test_deserialized": True}, "official_test_deserialized=false"),
    ],
)
def test_rejects_test_access_declarations(tmp_path: Path, changes: dict[str, object], match: str) -> None:
    root = tmp_path / "seeds"
    _write_seed_artifacts(root)
    _rewrite_summary(root, 2101, **changes)
    with pytest.raises(ValueError, match=match):
        AGGREGATE.load_registered_artifacts(root)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"schema_version": "wrong"}, "schema_version"),
        ({"sites_per_band": 255}, "sites_per_band=256"),
    ],
)
def test_rejects_schema_version_or_normalization_drift(
    tmp_path: Path,
    changes: dict[str, object],
    match: str,
) -> None:
    root = tmp_path / "seeds"
    _write_seed_artifacts(root)
    _rewrite_summary(root, 2102, **changes)
    with pytest.raises(ValueError, match=match):
        AGGREGATE.load_registered_artifacts(root)


def test_rejects_cross_seed_source_hash_drift_and_current_file_mismatch(tmp_path: Path) -> None:
    drift_root = tmp_path / "drift"
    _write_seed_artifacts(drift_root)
    drift_summary = json.loads((drift_root / "seed_2101" / "summary.json").read_text())
    drift_summary["source_hashes"]["aggregate"] = "0" * 64
    _rewrite_summary(drift_root, 2101, source_hashes=drift_summary["source_hashes"])
    with pytest.raises(ValueError, match="cross-seed metadata mismatch for source_hashes"):
        AGGREGATE.load_registered_artifacts(drift_root)

    mismatch_root = tmp_path / "mismatch"
    _write_seed_artifacts(mismatch_root)
    for seed in AGGREGATE.REGISTERED_SEEDS:
        summary = json.loads((mismatch_root / f"seed_{seed}" / "summary.json").read_text())
        summary["source_hashes"]["aggregate"] = "f" * 64
        _rewrite_summary(mismatch_root, seed, source_hashes=summary["source_hashes"])
    with pytest.raises(ValueError, match="does not match the current audited file"):
        AGGREGATE.load_registered_artifacts(mismatch_root)


def test_rejects_b4_e4_nontie_and_score_schema_drift(tmp_path: Path) -> None:
    root = tmp_path / "nontie"
    _write_seed_artifacts(root)
    _rewrite_scores(root, 2100, lambda arrays: arrays["e4"].__setitem__((0, 0), arrays["e4"][0, 0] + 1e-4))
    with pytest.raises(ValueError, match="B4/E4 samplewise tie failed"):
        AGGREGATE.load_registered_artifacts(root)

    schema_root = tmp_path / "schema"
    _write_seed_artifacts(schema_root)
    _rewrite_scores(schema_root, 2104, lambda arrays: arrays.__setitem__("unexpected", np.zeros(1)))
    with pytest.raises(ValueError, match="score schema mismatch"):
        AGGREGATE.load_registered_artifacts(schema_root)


def test_refuses_to_overwrite_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        AGGREGATE.aggregate(tmp_path / "missing-input", output)
