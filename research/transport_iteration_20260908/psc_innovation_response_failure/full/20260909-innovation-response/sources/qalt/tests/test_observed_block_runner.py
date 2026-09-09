import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "observed_b1_rgb_block"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("observed_b1_rgb_block_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_frozen_protocol_hash_matches_runner_contract() -> None:
    protocol = RUNNER.PROJECT_ROOT / "qalt" / "theory" / "OBSERVED_B1_RGB_BLOCK_PROTOCOL.md"
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert digest == RUNNER.PROTOCOL_HASH
    child = (
        RUNNER.PROJECT_ROOT
        / "qalt"
        / "theory"
        / "OBSERVED_B1_RGB_BLOCK_OPTIMIZATION_CHILD_PROTOCOL.md"
    )
    child_digest = hashlib.sha256(child.read_bytes()).hexdigest()
    assert child_digest == RUNNER.OPTIMIZATION_CHILD_PROTOCOL_HASH


def test_coefficient_build_is_chunk_invariant_and_explicit() -> None:
    rng = np.random.default_rng(601)
    images = rng.integers(0, 256, size=(5, 3, 32, 32), dtype=np.uint8)
    record_ids = np.array([4, 9, 12, 20, 27], dtype=np.int64)
    first = RUNNER.build_coefficients(images, record_ids, 2100, chunk_images=1, progress=lambda _: None)
    second = RUNNER.build_coefficients(images, record_ids, 2100, chunk_images=5, progress=lambda _: None)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[0].shape == (5, 3, 16, 16)
    assert first[1].shape == (5, 16, 16, 3, 3)


def test_runner_refuses_unregistered_seed_and_existing_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside frozen development seeds"):
        RUNNER.run(9999, tmp_path / "new", tmp_path, "wrong")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        RUNNER.run(2100, existing, tmp_path, "wrong")
    with pytest.raises(ValueError, match="max_iterations must select"):
        RUNNER.run(2100, tmp_path / "new", tmp_path, "wrong", max_iterations=201)
