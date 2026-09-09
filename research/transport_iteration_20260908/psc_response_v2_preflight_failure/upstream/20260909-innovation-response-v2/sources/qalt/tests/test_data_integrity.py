import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from qalt.data_integrity import (
    UCFRecord,
    adaptive_cifar_repair_split,
    assert_group_disjoint,
    cifar_training_batch_paths,
    inspect_ucf_archive,
    parse_ucf_member,
    require_test_confirmation,
    stratified_cifar_split,
)


def _write_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name in names:
            payload = b"not-decoded-video"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_uncompressed_tar_with_gzip_suffix_and_group_identity(tmp_path: Path) -> None:
    archive_path = tmp_path / "subset.tar.gz"
    _write_tar(
        archive_path,
        [
            "subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
            "subset/val/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c01.avi",
            "subset/test/Archery/v_Archery_g01_c01.avi",
        ],
    )
    records = inspect_ucf_archive(archive_path)
    assert len(records) == 3
    assert records[0].group_id != records[2].group_id


@pytest.mark.parametrize(
    "name",
    [
        "../train/UCF101/Archery/v_Archery_g01_c01.avi",
        "x/train/UCF101/Archery/v_Wrong_g01_c01.avi",
        "x/train/UCF101/Archery/not_a_clip.avi",
    ],
)
def test_ucf_parser_rejects_unsafe_or_malformed_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_ucf_member(name, 10)


def test_ucf_group_overlap_is_rejected() -> None:
    rows = [
        UCFRecord("train", "Archery", "Archery/g01", "c01", "a", 1),
        UCFRecord("test", "Archery", "Archery/g01", "c02", "b", 1),
    ]
    with pytest.raises(ValueError, match="group leakage"):
        assert_group_disjoint(rows)


def test_cifar_split_is_stratified_disjoint_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 5_000)
    first = stratified_cifar_split(labels)
    second = stratified_cifar_split(labels)
    assert first == second
    assert len(first["fit"]) == 45_000
    assert len(first["validation"]) == 5_000
    assert not set(first["fit"]) & set(first["validation"])
    for indices, expected in ((first["fit"], 4_500), (first["validation"], 500)):
        counts = np.bincount(labels[indices], minlength=10)
        assert np.all(counts == expected)


def test_adaptive_cifar_repair_split_excludes_discovery_and_is_stratified() -> None:
    labels = np.repeat(np.arange(10), 5_000)
    original = stratified_cifar_split(labels)
    first = adaptive_cifar_repair_split(labels)
    second = adaptive_cifar_repair_split(labels)
    assert first == second
    assert first["excluded_discovery"] == original["validation"]
    expected_counts = {"fit": 4_000, "repair_holdout": 500, "excluded_discovery": 500}
    partitions = []
    for name, expected in expected_counts.items():
        indices = np.asarray(first[name])
        partitions.append(set(indices.tolist()))
        assert np.all(np.bincount(labels[indices], minlength=10) == expected)
    assert not any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3))
    assert set().union(*partitions) == set(range(50_000))


def test_test_phase_guard_requires_exact_frozen_hash() -> None:
    for phase, candidate in (("development", "abc"), ("confirmation", "wrong")):
        with pytest.raises(PermissionError):
            require_test_confirmation(phase, candidate, "abc")
    require_test_confirmation("confirmation", "abc", "abc")


def test_cifar_training_loader_allowlist_excludes_test_batch(tmp_path: Path) -> None:
    paths = cifar_training_batch_paths(tmp_path)
    assert [path.name for path in paths] == [f"data_batch_{index}" for index in range(1, 6)]
    assert all(path.name != "test_batch" for path in paths)
