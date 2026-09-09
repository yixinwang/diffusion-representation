"""Deterministic world checks only: no estimator fitting or experimental seeds."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import quad

SPEC = importlib.util.spec_from_file_location(
    "conditional_cdf_validation_run",
    Path(__file__).resolve().parents[1] / "experiments/conditional_cdf_validation/run.py",
)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_true_conditional_inverse_and_endpoints():
    contexts = np.array([0., .125, .5, .9, 1.])[:, None]
    uniforms = np.array([1e-8, .01, .25, .5, .99, 1 - 1e-8])[None, :]
    values = RUN.true_inverse(uniforms, contexts)
    np.testing.assert_allclose(RUN.true_cdf(values, contexts),
                               np.broadcast_to(uniforms, values.shape), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(RUN.true_cdf(0., contexts), 0., atol=1e-15)
    np.testing.assert_allclose(RUN.true_cdf(1., contexts), 1., atol=1e-15)


def test_cosine_conditional_normalization_and_positive_density():
    for context in [0., .1, .25, .5, .9]:
        density = lambda t: 1 + RUN.THETA * np.cos(2 * np.pi * context) * np.cos(2 * np.pi * t)
        mass, _ = quad(density, 0., 1.)
        assert abs(mass - 1.) < 1e-12
        assert min(density(t) for t in np.linspace(0, 1, 33)) >= 1 - RUN.THETA - 1e-14


def test_distant_density_ignores_middle_and_local_graph_omits_source():
    first = np.full((2, 8), .25)
    first[:, [0, 7]] = [.1, .2]
    first[1, 1:7] = .8
    np.testing.assert_array_equal(RUN.true_log_prob(first, "distant"),
                                 np.repeat(RUN.true_log_prob(first[:1], "distant"), 2))
    assert RUN.graph(8, "distant")[-1] == 6
