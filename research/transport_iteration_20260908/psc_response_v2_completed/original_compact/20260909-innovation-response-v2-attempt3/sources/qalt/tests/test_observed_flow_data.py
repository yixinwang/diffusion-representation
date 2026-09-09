"""Only fabricated five-file inputs; no access to real CIFAR or model training."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from qalt import observed_flow_data as module
from qalt.data_integrity import adaptive_cifar_repair_split, stable_json_hash


class _BroadcastFixtureArray:
    """Serialize a scalar and shape, not 30 MB of redundant fabricated pixels."""

    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return np.broadcast_to, (np.array(self.value, dtype=np.uint8), (10_000, 3072))


@pytest.fixture(scope="module")
def fabricated_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("observed-flow-fabricated")
    for index, value in enumerate([0, 63, 127, 191, 255], 1):
        with (root / f"data_batch_{index}").open("wb") as stream:
            pickle.dump({b"data": _BroadcastFixtureArray(value),
                         b"labels": np.tile(np.arange(10), 1000).tolist()}, stream)
    (root / "test_batch").write_bytes(b"POISON: this file must never be opened")
    assert sum(p.stat().st_size for p in root.iterdir()) < 200_000
    return root


@pytest.fixture(scope="module")
def loaded_fixture(fabricated_files):
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.parent == fabricated_files:
            assert path.name != "test_batch"
            opened.append(path.name)
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        result = module.load_observed_flow_data(fabricated_files, allow_noncanonical_fixture=True)
    return result, opened


def test_five_file_loader_preserves_split_and_excludes_discovery(loaded_fixture):
    data, opened = loaded_fixture
    assert set(opened) == {f"data_batch_{i}" for i in range(1, 6)}
    # One hash pass and one deserialize pass over each eligible file.
    assert all(opened.count(f"data_batch_{i}") == 2 for i in range(1, 6))
    assert data.fit.shape == (4000, 3, 32, 32)
    assert data.repair.shape == (1000, 3, 32, 32)
    assert data.fit.dtype == data.repair.dtype == np.float64
    assert not data.fit.flags.writeable and not data.fit_ids.flags.writeable
    labels = np.tile(np.arange(10), 5000)
    split = adaptive_cifar_repair_split(labels)
    assert not set(data.fit_ids) & set(data.repair_ids)
    assert not (set(data.fit_ids) | set(data.repair_ids)) & set(split["excluded_discovery"])
    assert set(data.fit_ids) <= set(split["fit"])
    assert set(data.repair_ids) <= set(split["repair_holdout"])
    assert np.array_equal(np.bincount(labels[data.fit_ids]), np.full(10, 400))
    assert np.array_equal(np.bincount(labels[data.repair_ids]), np.full(10, 100))
    for role, ids, eligible in [("fit", data.fit_ids, split["fit"]),
                                ("repair", data.repair_ids, split["repair_holdout"])]:
        np.testing.assert_array_equal(ids, module._select_ids(list(reversed(eligible)), labels, role))
    assert data.ledger["full_split_counts"] == {"fit": 40000, "repair_holdout": 5000, "excluded_discovery": 5000}
    assert data.ledger["canonical_dataset_verified"] is False
    assert data.ledger["allow_noncanonical_fixture"] is True
    assert data.ledger["test_data_accessed"] is False
    ledger_without_hash = {k: v for k, v in data.ledger.items() if k != "ledger_sha256"}
    assert stable_json_hash(ledger_without_hash) == data.ledger["ledger_sha256"]
    assert module._array_hash(data.fit_ids) == data.ledger["fit_ids_sha256"]
    assert module._array_hash(data.repair_ids) == data.ledger["repair_ids_sha256"]
    assert not hasattr(data, "labels")


def test_dequantization_is_open_native_pixelwise_and_record_keyed(loaded_fixture):
    data, _ = loaded_fixture
    pixel_values = np.array([0, 63, 127, 191, 255], dtype=np.uint8)
    for role, ids, values in [("fit", data.fit_ids, data.fit), ("repair", data.repair_ids, data.repair)]:
        assert np.all(values > 0) and np.all(values < 1)
        for position in [0, len(ids) // 2, len(ids) - 1]:
            record = int(ids[position])
            pixels = np.full((3, 32, 32), pixel_values[record // 10000], dtype=np.uint8)
            np.testing.assert_array_equal(values[position], module._dequantize_record(pixels, record, role))
            np.testing.assert_array_equal(np.floor(values[position] * 256), pixels)
    zero = np.zeros((3, 32, 32), dtype=np.uint8)
    one = module._dequantize_record(zero, 42, "fit")
    np.testing.assert_array_equal(one, module._dequantize_record(zero, 42, "fit"))
    assert not np.array_equal(one, module._dequantize_record(zero, 43, "fit"))
    assert not np.array_equal(one, module._dequantize_record(zero, 42, "repair"))
    assert np.all(module._dequantize_record(np.full_like(zero, 255), 42, "fit") < 1)


def test_default_rejects_noncanonical_files_before_deserialization(fabricated_files, monkeypatch):
    monkeypatch.setattr(module, "load_cifar_training_batches",
                        lambda *args, **kwargs: pytest.fail("must not deserialize unverified files"))
    with pytest.raises(ValueError, match="canonical training batch hash mismatch"):
        module.load_observed_flow_data(fabricated_files)


def test_full_split_hash_checked_before_pixel_subset(fabricated_files, monkeypatch):
    expected = json.loads(module.MANIFEST.read_text())["cifar10"]["allowed_development_batches"]
    monkeypatch.setattr(module, "sha256_file", lambda path: expected[path.name])

    def fake_loader(root, opened_files):
        opened_files.extend(str(p.resolve()) for p in module.cifar_training_batch_paths(root))
        return np.broadcast_to(np.uint8(0), (50000, 3, 32, 32)), np.tile(np.arange(10), 5000)

    monkeypatch.setattr(module, "load_cifar_training_batches", fake_loader)
    monkeypatch.setattr(module, "_build_selected", lambda *args: pytest.fail("must verify full split first"))
    with pytest.raises(ValueError, match="canonical full adaptive split hash mismatch"):
        module.load_observed_flow_data(fabricated_files)


def test_symlink_batch_is_rejected_before_open(tmp_path):
    (tmp_path / "test_batch").write_bytes(b"poison")
    (tmp_path / "data_batch_1").symlink_to(tmp_path / "test_batch")
    with pytest.raises(ValueError, match="nonsymlink"):
        module.load_observed_flow_data(tmp_path, allow_noncanonical_fixture=True)
