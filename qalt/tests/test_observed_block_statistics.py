import numpy as np
import pytest

from qalt.observed_block_statistics import (
    DETAIL_COUNT,
    REGISTERED_GATES,
    balanced_class_mean,
    construct_route_regrets,
    equivalence_pvalue,
    evaluate_registered_gates,
    evaluate_registered_routes,
    holm_adjust,
    registered_density_log_bounds,
    registered_route_regret_ranges,
    seed_averaged_nll,
    stratified_percentile_intervals,
    superiority_pvalue,
)


def labels_with(counts: list[int]) -> np.ndarray:
    return np.concatenate([np.full(count, class_id, dtype=np.int64) for class_id, count in enumerate(counts)])


def log_score_cube(nll: np.ndarray, seed_offsets: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(nll, dtype=float)
    offsets = np.zeros(5) if seed_offsets is None else np.asarray(seed_offsets, dtype=float)
    per_band = -(values[None, :, None] + offsets[:, None, None]) * DETAIL_COUNT / 3.0
    return np.broadcast_to(per_band, (5, len(values), 3)).copy()


def test_seed_averaging_stays_inside_image() -> None:
    nll = np.array([1.0, 2.0])
    offsets = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    result = seed_averaged_nll({"arm": log_score_cube(nll, offsets)})["arm"]
    assert result.shape == (2,)
    assert np.allclose(result, nll)


def test_normalization_contract_uses_band_sums_over_256_sites() -> None:
    per_coefficient_nll = 2.0
    band_total_log_score = -per_coefficient_nll * 3 * 256
    scores = np.full((5, 4, 3), band_total_log_score)
    result = seed_averaged_nll({"arm": scores})["arm"]
    assert np.array_equal(result, np.full(4, per_coefficient_nll))


def test_balanced_class_mean_is_not_pooled_mean() -> None:
    labels = labels_with([2] * 9 + [20])
    values = (labels == 9).astype(float)
    assert balanced_class_mean(values, labels) == pytest.approx(0.1)
    assert np.mean(values) > 0.5


def test_superiority_and_tost_have_registered_directions() -> None:
    labels = labels_with([40] * 10)
    rng = np.random.default_rng(12)
    superior = 0.03 + rng.normal(0.0, 0.002, len(labels))
    equivalent = rng.normal(0.0, 0.001, len(labels))
    p_superior, summary = superiority_pvalue(superior, labels, margin=0.01)
    p_equivalent, tost = equivalence_pvalue(equivalent, labels, half_width=0.01)
    assert summary["mean"] > 0.01
    assert p_superior < 1e-8
    assert abs(tost["mean"]) < 0.01
    assert p_equivalent < 1e-8


def test_holm_requires_ten_and_is_monotone_in_rank() -> None:
    raw = {f"p{index}": value for index, value in enumerate(np.linspace(0.001, 0.2, 10))}
    adjusted = holm_adjust(raw)
    ordered = [adjusted[name] for name in raw]
    assert np.all(np.diff(ordered) >= -1e-15)
    assert all(adjusted[name] >= raw[name] for name in raw)
    with pytest.raises(ValueError, match="exactly 10"):
        holm_adjust({"only": 0.01})


def test_stratified_percentile_bootstrap_is_paired_and_deterministic() -> None:
    labels = labels_with([5] * 10)
    values = np.column_stack((np.arange(len(labels)), -np.arange(len(labels))))
    first = stratified_percentile_intervals(values, labels, draws=199, seed=77, batch_size=17)
    second = stratified_percentile_intervals(values, labels, draws=199, seed=77, batch_size=17)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 0], -first[::-1, 1])


def test_registered_gate_aggregator_passes_clear_synthetic_effects() -> None:
    labels = labels_with([20] * 10)
    rng = np.random.default_rng(31)
    shared = rng.normal(0.0, 0.02, len(labels))
    arm_noise = {name: rng.normal(0.0, 0.0005, len(labels)) for spec in REGISTERED_GATES for name in (spec.left, spec.right)}
    target = {
        "b4": 1.00,
        "o4": 1.05,
        "p4": 1.05,
        "b1": 1.05,
        "z4": 1.05,
        "d4": 1.05,
        "b4_unconditional": 1.03,
        "b8": 1.00,
        "a8": 1.00,
        "i8": 1.00,
        "i4": 1.00,
    }
    scores = {
        name: log_score_cube(target[name] + shared + arm_noise[name])
        for name in target
    }
    result = evaluate_registered_gates(scores, labels, bootstrap_draws=199, bootstrap_seed=44)
    assert result["status"] == "adaptive_development_no_coverage"
    assert result["all_gates_pass"]
    assert len(result["gates"]) == 10
    assert all(gate["holm_adjusted_pvalue"] <= 0.05 for gate in result["gates"].values())


def test_route_construction_uses_b4_a8_bands_but_direct_i8_regret() -> None:
    shape = (5, 4, 3)
    scores = {
        "b4": np.full(shape, 10.0),
        "a8": np.full(shape, 4.0),
        "i8": np.full(shape, 9.0),
    }
    regret = construct_route_regrets(scores)
    assert np.allclose(regret[0], (27.0 - 12.0) / DETAIL_COUNT)
    assert np.allclose(regret[7], (27.0 - 30.0) / DETAIL_COUNT)
    expected = np.array([(27.0 - (12.0 + 6.0 * mask.bit_count())) / DETAIL_COUNT for mask in range(8)])
    assert np.allclose(regret[:, 0], expected)


def test_registered_range_helper_matches_endpoint_formulas() -> None:
    bounds = registered_density_log_bounds()
    ranges = registered_route_regret_ranges()
    scalar_lower, scalar_upper = bounds["scalar"]
    block_lower, block_upper = bounds["block"]
    assert ranges.shape == (8, 2)
    assert ranges[0, 0] == pytest.approx((3 * scalar_lower - 3 * scalar_upper) / 3)
    assert ranges[0, 1] == pytest.approx((3 * scalar_upper - 3 * scalar_lower) / 3)
    assert ranges[7, 0] == pytest.approx((3 * scalar_lower - block_upper) / 3)
    assert ranges[7, 1] == pytest.approx((3 * scalar_upper - block_lower) / 3)
    assert np.all(ranges[:, 0] < ranges[:, 1])


def test_classwise_pseudo_eb_abstains_when_registered_range_is_vacuous() -> None:
    labels = labels_with([500] * 10)
    shape = (5, len(labels), 3)
    scores = {
        "b4": np.zeros(shape),
        "a8": np.zeros(shape),
        "i8": np.zeros(shape),
    }
    result = evaluate_registered_routes(scores, labels)
    assert result["status"] == "diagnostic_pseudo_certificate_no_coverage"
    assert result["decision"] == "fallback_i8"
    assert result["selected_mask"] is None
    assert all(not route["eligible"] for route in result["routes"])
    assert result["routes"][0]["class_range_radii"][0] == pytest.approx(61.3571727982)
    assert result["routes"][7]["class_range_radii"][0] == pytest.approx(332.8512761466)


def test_classwise_pseudo_eb_selects_lowest_cost_eligible_route() -> None:
    labels = labels_with([500] * 10)
    shape = (5, len(labels), 3)
    bounds = registered_density_log_bounds()
    scalar_lower, scalar_upper = bounds["scalar"]
    _, block_upper = bounds["block"]
    scores = {
        "b4": np.full(shape, 256.0 * block_upper),
        "a8": np.full(shape, 256.0 * 3.0 * scalar_upper),
        "i8": np.full(shape, 256.0 * 3.0 * scalar_lower),
    }
    result = evaluate_registered_routes(scores, labels)
    assert result["decision"] == "route"
    assert result["selected_mask"] == 7
    assert result["selected_route"]["critical_depth"] == 1
    assert result["selected_route"]["unbatched_calls"] == 3
    assert result["selected_route"]["batched_head_invocations"] == 1
    assert all(route["eligible"] for route in result["routes"])


def test_registered_score_validation_rejects_seed_pseudoreplication() -> None:
    with pytest.raises(ValueError, match=r"shape \(5, images, 3\)"):
        seed_averaged_nll({"arm": np.zeros((10, 7, 3))})


def test_route_construction_rejects_scores_outside_registered_density_range() -> None:
    shape = (5, 4, 3)
    scores = {"b4": np.zeros(shape), "a8": np.zeros(shape), "i8": np.zeros(shape)}
    scores["b4"][0, 0, 0] = 1e9
    with pytest.raises(ValueError, match="registered density range"):
        construct_route_regrets(scores)
