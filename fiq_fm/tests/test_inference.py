import json
from pathlib import Path
import runpy

import numpy as np

from fiqfm.inference import holm_adjust, one_sided_mean_test, student_mean_interval


def test_student_interval_and_directional_test() -> None:
    values = np.array([0.8, 1.0, 1.2, 1.1])
    interval = student_mean_interval(values)
    assert interval["low"] < interval["mean"] < interval["high"]
    assert one_sided_mean_test(values, 0.0, "greater")["raw_p"] < 0.05
    assert one_sided_mean_test(values, 2.0, "less")["raw_p"] < 0.05


def test_zero_variance_directional_boundary_is_not_rejected() -> None:
    values = np.zeros(30)
    assert one_sided_mean_test(values, 0.0, "greater")["raw_p"] == 1.0
    assert one_sided_mean_test(values, -0.1, "greater")["raw_p"] == 0.0


def test_holm_adjustment_is_ordered_and_capped() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.8})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.8}


def test_confirmation_gate_wiring_on_development_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "experiments/residual_rotation/run.py"))
    result_root = root / "results/residual_rotation_development_v3"
    results = [json.loads(path.read_text()) for path in sorted(result_root.glob("seed_*.json"))]
    config = namespace["Config"](seeds=tuple(range(5)), confirmation=True)
    summary = namespace["summarize"](results, config)["summary"]
    inference = summary["confirmation_inference"]
    assert len(inference["components"]) == 14
    assert inference["all_registered_gates_pass"]
