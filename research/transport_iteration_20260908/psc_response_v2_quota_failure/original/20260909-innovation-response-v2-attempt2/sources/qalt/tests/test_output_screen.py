import numpy as np
import json
import pytest
from scipy.spatial.distance import cdist, pdist
from qalt.output_screen import displacement_screen


def score(samples, references):
    return cdist(samples, references).mean() - 0.5 * pdist(samples).mean()


def test_same_sample_score_bound_on_nonlinear_nongaussian_outputs():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(130, 12))
    x = np.tanh(u + 0.3 * u**3)
    candidate = x.copy()
    candidate[:, :3] = np.tanh(u[:, :3] + 0.3 * u[:, :3]**3 + 0.002 * np.sin(u[:, 3:6]))
    # References are synthetic checks, never real holdout data.
    for reference in [rng.standard_t(4, size=(67, 12)), np.zeros((1, 12))]:
        assert abs(score(x, reference) - score(candidate, reference)) <= 2 * np.linalg.norm(x-candidate, axis=1).mean()
    result = displacement_screen(x, candidate, displacement_bound=0.002*np.sqrt(3), margin=0.01)
    assert result["reject_margin"]
    assert not result["quality_established"]
    json.dumps(result, allow_nan=False)


def test_identical_generator_requires_analytic_bound_to_certify_zero():
    x = np.zeros((10, 3))
    assert displacement_screen(x, x, displacement_bound=0, margin=0.001)["reject_margin"]
    assert not displacement_screen(x, x, displacement_bound=1, margin=0.001)["reject_margin"]


def test_units_and_family_penalty():
    x = np.zeros((100, 3))
    y = np.ones((100, 3))*0.01
    a = displacement_screen(x, y, displacement_bound=1, margin=0.1)
    b = displacement_screen(x, y, displacement_bound=1, margin=0.1, family_size=4)
    assert b["mean_displacement_upper"] > a["mean_displacement_upper"]
    c = displacement_screen(x*2, y*2, displacement_bound=1, margin=0.1, distance_scale=2)
    assert c["mean_displacement_upper"] == a["mean_displacement_upper"]


@pytest.mark.parametrize("kwargs", [dict(displacement_bound=-1), dict(margin=0), dict(alpha=1), dict(family_size=0), dict(distance_scale=0)])
def test_invalid_contract(kwargs):
    args = dict(displacement_bound=1, margin=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        displacement_screen(np.zeros((10, 3)), np.ones((10, 3)), **args)


def test_bound_violation_fails_closed():
    with pytest.raises(ValueError, match="violates"):
        displacement_screen(np.zeros((10, 3)), np.ones((10, 3)), displacement_bound=1, margin=0.1)
