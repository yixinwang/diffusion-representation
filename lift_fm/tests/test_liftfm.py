from __future__ import annotations

import numpy as np
import pytest

from liftfm.data import fixed_split, load_partition, split_manifest
from liftfm.fiber import JointGSM, fit_joint_gsm
from liftfm.lifting import haar2d_forward, haar2d_inverse, pack_coefficients, unpack_coefficients


def determinant_one_shape() -> np.ndarray:
    raw = np.array([[1.2, 0.6, -0.2], [0.6, 1.0, 0.25], [-0.2, 0.25, 0.9]])
    return raw / np.linalg.det(raw) ** (1.0 / 3.0)


def test_haar_roundtrip_norm_and_packing() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 8, 8))
    z, r = haar2d_forward(x)
    packed = pack_coefficients(z, r)
    recovered_z, recovered_r = unpack_coefficients(packed)
    np.testing.assert_allclose(recovered_z, z)
    np.testing.assert_allclose(recovered_r, r)
    np.testing.assert_allclose(haar2d_inverse(z, r), x, atol=2e-15, rtol=2e-15)
    assert abs(np.sum(x**2) - np.sum(z**2) - np.sum(r**2)) < 1e-10


def test_joint_gsm_density_beats_exact_marginal_product_on_correlated_samples() -> None:
    rng = np.random.default_rng(2)
    truth = JointGSM(np.array([0.35, 0.65]), np.array([0.3, 1.2]), determinant_one_shape())
    sample = truth.sample_joint(rng.normal(size=(80_000, 3)))
    gap = np.mean(truth.log_prob(sample) - truth.product_log_prob(sample))
    assert gap > 0.05


def test_joint_gsm_sampler_uses_only_three_source_normals_and_recovers_moments() -> None:
    rng = np.random.default_rng(3)
    model = JointGSM(np.array([0.2, 0.5, 0.3]), np.array([0.25, 0.8, 1.7]), determinant_one_shape())
    source = rng.normal(size=(150_000, 3))
    sample = model.sample_joint(source)
    expected_cov = np.sum(model.weights * model.scales**2) * model.shape
    np.testing.assert_allclose(np.cov(sample, rowvar=False), expected_cov, atol=0.025, rtol=0.04)
    assert sample.shape == source.shape


def test_gsm_fit_improves_over_product_on_heldout() -> None:
    rng = np.random.default_rng(4)
    truth = JointGSM(np.array([0.4, 0.6]), np.array([0.35, 1.4]), determinant_one_shape())
    train = truth.sample_joint(rng.normal(size=(30_000, 3)))
    test = truth.sample_joint(rng.normal(size=(10_000, 3)))
    fitted = fit_joint_gsm(train, components=4)
    assert np.mean(fitted.log_prob(test) - fitted.product_log_prob(test)) > 0.04


def test_final_partition_is_sealed_by_loader(monkeypatch) -> None:
    manifest = split_manifest()
    split = fixed_split()
    assert len(set(split["train"]) & set(split["validation"])) == 0
    assert len(set(split["train"]) & set(split["test"])) == 0
    assert manifest["sizes"] == {"train": 1078, "validation": 359, "test": 360}

    # The permission check must fire before label or pixel I/O.
    import liftfm.data as data_module

    def forbidden_io():
        raise AssertionError("I/O occurred before the test guard")

    monkeypatch.setattr(data_module, "_load_labels_only", forbidden_io)
    with pytest.raises(PermissionError, match="sealed"):
        data_module.load_partition("test", seed=9, allow_test=False)
