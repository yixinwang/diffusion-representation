import numpy as np
from pathlib import Path
import json
import runpy

from qalt.multiscale import (
    MixtureFiber,
    band_slices,
    fit_mixture_fiber,
    haar_forward,
    haar_inverse,
    token_benchmark,
)


def test_image_and_video_haar_are_exact_and_norm_preserving() -> None:
    rng = np.random.default_rng(31)
    for shape, axes in (((3, 32, 32), (1, 2)), ((2, 16, 32, 32), (1, 2, 3))):
        values = rng.normal(size=shape)
        coefficients = haar_forward(values, axes)
        assert np.max(np.abs(haar_inverse(coefficients, axes) - values)) < 1e-12
        assert abs(np.sum(coefficients**2) - np.sum(values**2)) < 1e-9
        assert sum(np.prod(coefficients[index].shape) for index in band_slices(shape, axes)) == values.size


def test_train_fitted_mixture_recovers_parent_gating() -> None:
    rng = np.random.default_rng(37)
    parent = rng.normal(size=20_000)
    truth = MixtureFiber(np.array([-0.3, 1.1, -0.5]), np.array([0.45, 1.7]))
    detail = truth.sample(parent, rng)
    fitted = fit_mixture_fiber(detail, parent)
    grid = np.linspace(-2.0, 2.0, 101)
    assert np.corrcoef(truth.probabilities(grid), fitted.probabilities(grid))[0, 1] > 0.98
    assert np.max(np.abs(truth.scales - fitted.scales)) < 0.1


def test_token_accounting_and_memory_are_strict() -> None:
    result = token_benchmark(shape=(32, 32), steps=20, repeats=3)
    assert result["qalt_token_updates"] < result["full_token_updates"]
    assert result["memory_ratio"] < 1.0


def test_confirmation_inference_wiring_on_development_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "experiments/multiscale_stage_a/run.py"))
    result_root = root / "results/multiscale_stage_a_corrected_development_20260823"
    units = [json.loads(path.read_text()) for path in sorted(result_root.glob("seed_*.json"))]
    config = namespace["Config"](seeds=tuple(range(700, 705)), confirmation=True)
    summary = namespace["summarize"](config, units)
    assert len(summary["confirmation_inference"]["components"]) == 16
    assert summary["confirmation_inference"]["all_registered_gates_pass"]
