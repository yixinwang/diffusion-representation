#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from liftfm.data import load_partition, split_manifest, stable_hash  # noqa: E402
from liftfm.fiber import ConditionalBlockFiber  # noqa: E402
from liftfm.lifting import (  # noqa: E402
    haar2d_forward,
    haar2d_inverse,
    pack_coefficients,
    unpack_coefficients,
)
from liftfm.metrics import all_metrics, balanced_labels, energy_score, sliced_wasserstein  # noqa: E402
from liftfm.models import (  # noqa: E402
    reconstruct_vae,
    sample_rectified_flow,
    sample_vae,
    train_rectified_flow,
    train_vae,
)


@dataclass(frozen=True)
class Config:
    phase: str = "development"
    seeds: tuple[int, ...] = (4099,)
    flow_steps: int = 250
    vae_steps: int = 200
    batch_size: int = 128
    flow_width: int = 40
    flow_depth: int = 2
    flow_heads: int = 4
    flow_lr: float = 1.5e-3
    vae_lr: float = 1.5e-3
    beta_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    nfe_grid: tuple[int, ...] = (4, 8, 16, 32)
    main_nfe: int = 16
    generated_samples: int = 500
    fiber_components: int = 4
    fiber_ridge: float = 1e-2
    energy_margin: float = 0.02
    swd_margin: float = 0.02


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def method_configuration(config: Config) -> dict[str, Any]:
    value = asdict(config)
    value.pop("phase")
    value.pop("seeds")
    return value


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/liftfm/data.py",
        ROOT / "src/liftfm/lifting.py",
        ROOT / "src/liftfm/fiber.py",
        ROOT / "src/liftfm/models.py",
        ROOT / "src/liftfm/metrics.py",
        ROOT.parent / "research/LIFT_FM_PREREGISTRATION.md",
    ]
    return {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def provenance() -> dict[str, object]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "command": sys.argv,
        "source_sha256": source_hashes(),
    }


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def prepare(seed: int, allow_test: bool):
    train_x, train_y, train_ids = load_partition("train", seed, allow_test=False)
    validation_x, validation_y, validation_ids = load_partition("validation", seed, allow_test=False)
    reference_name = "test" if allow_test else "validation"
    reference_x, reference_y, reference_ids = load_partition(reference_name, seed, allow_test=allow_test)

    train_z, train_r = haar2d_forward(train_x)
    validation_z, validation_r = haar2d_forward(validation_x)
    reference_z, reference_r = haar2d_forward(reference_x)
    train_packed = pack_coefficients(train_z, train_r)
    validation_packed = pack_coefficients(validation_z, validation_r)
    reference_packed = pack_coefficients(reference_z, reference_r)

    mean = train_packed.reshape(len(train_packed), -1).mean(axis=0)
    scale = train_packed.reshape(len(train_packed), -1).std(axis=0)
    scale = np.maximum(scale, 1e-3)
    full_train = (train_packed.reshape(len(train_packed), -1) - mean) / scale
    full_validation = (validation_packed.reshape(len(validation_packed), -1) - mean) / scale
    full_reference = (reference_packed.reshape(len(reference_packed), -1) - mean) / scale
    active_mask = np.zeros((8, 8), dtype=bool)
    active_mask[:4, :4] = True
    active_indices = np.flatnonzero(active_mask.reshape(-1))
    active_train = full_train[:, active_indices]
    active_validation = full_validation[:, active_indices]
    active_reference = full_reference[:, active_indices]

    roundtrip = float(np.max(np.abs(haar2d_inverse(train_z[:16], train_r[:16]) - train_x[:16])))
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_ids": train_ids,
        "validation_x": validation_x,
        "validation_y": validation_y,
        "validation_ids": validation_ids,
        "reference_x": reference_x,
        "reference_y": reference_y,
        "reference_ids": reference_ids,
        "train_z": train_z,
        "train_r": train_r,
        "validation_z": validation_z,
        "validation_r": validation_r,
        "reference_z": reference_z,
        "reference_r": reference_r,
        "mean": mean,
        "scale": scale,
        "active_indices": active_indices,
        "full_train": full_train,
        "full_validation": full_validation,
        "full_reference": full_reference,
        "active_train": active_train,
        "active_validation": active_validation,
        "active_reference": active_reference,
        "roundtrip": roundtrip,
        "reference_name": reference_name,
    }


def generated_label_vector(count: int) -> np.ndarray:
    labels = balanced_labels(count)
    return labels[np.argsort(labels, kind="stable")]


def lift_samples(
    active_model,
    fiber: ConditionalBlockFiber,
    labels: np.ndarray,
    nfe: int,
    seed: int,
    full_source: np.ndarray,
    active_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    joint: bool,
    fiber_source: np.ndarray,
) -> np.ndarray:
    active_source = full_source[:, active_indices]
    active_standardized = sample_rectified_flow(active_model, labels, nfe, seed + 11, source=active_source)
    active_raw = active_standardized * scale[active_indices] + mean[active_indices]
    coarse = active_raw.reshape(len(labels), 4, 4)
    details = fiber.sample(coarse, labels, fiber_source, joint=joint)
    return np.clip(haar2d_inverse(coarse, details), 0.0, 1.0)


def full_samples(full_model, labels: np.ndarray, nfe: int, seed: int, source: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standard = sample_rectified_flow(full_model, labels, nfe, seed + 17, source=source)
    packed = (standard * scale + mean).reshape(len(labels), 8, 8)
    coarse, details = unpack_coefficients(packed)
    return np.clip(haar2d_inverse(coarse, details), 0.0, 1.0)


def train_and_evaluate(config: Config, seed: int, allow_test: bool) -> dict[str, object]:
    data = prepare(seed, allow_test)
    started = time.perf_counter()

    full_fit = train_rectified_flow(
        data["full_train"], data["train_y"], seed + 1, config.flow_steps, config.batch_size,
        config.flow_width, config.flow_depth, config.flow_heads, config.flow_lr,
    )
    active_fit = train_rectified_flow(
        data["active_train"], data["train_y"], seed + 2, config.flow_steps, config.batch_size,
        config.flow_width, config.flow_depth, config.flow_heads, config.flow_lr,
    )
    fiber_started = time.perf_counter()
    fiber = ConditionalBlockFiber.fit(
        data["train_z"], data["train_r"], data["train_y"],
        components=config.fiber_components, ridge_penalty=config.fiber_ridge,
    )
    fiber_fit_seconds = time.perf_counter() - fiber_started
    validation_joint_nll = fiber.nll_per_coefficient(data["validation_z"], data["validation_r"], data["validation_y"], joint=True)
    validation_scalar_nll = fiber.nll_per_coefficient(data["validation_z"], data["validation_r"], data["validation_y"], joint=False)

    labels = generated_label_vector(config.generated_samples)
    rng = np.random.default_rng(seed + 300)
    full_source = rng.normal(size=(len(labels), 64))
    fiber_source = rng.normal(size=(len(labels), 4, 4, 3))

    # Frozen validation beta search. Every candidate is trained and counted.
    vae_candidates = []
    vae_search_seconds = 0.0
    for beta_index, beta in enumerate(config.beta_grid):
        fit = train_vae(data["train_x"], data["train_y"], seed + 100 + beta_index, beta, config.vae_steps, config.batch_size, config.vae_lr)
        vae_search_seconds += fit.seconds
        validation_generated = sample_vae(fit.model, labels, seed + 500 + beta_index)
        score = energy_score(data["validation_x"], data["validation_y"], validation_generated, labels)
        vae_candidates.append({"beta": beta, "validation_energy": score, "fit": fit})
    chosen = min(vae_candidates, key=lambda item: (item["validation_energy"], item["beta"]))
    vae_model = chosen["fit"].model

    frontier: dict[str, object] = {}
    generated_cache: dict[str, np.ndarray] = {}
    sampling_seconds: dict[str, dict[str, float]] = {}
    for nfe in config.nfe_grid:
        method_samples: dict[str, np.ndarray] = {}
        method_times: dict[str, float] = {}
        for name, call in {
            "full_flow": lambda: full_samples(full_fit.model, labels, nfe, seed + 700 + nfe, full_source, data["mean"], data["scale"]),
            "lift_joint": lambda: lift_samples(active_fit.model, fiber, labels, nfe, seed + 800 + nfe, full_source, data["active_indices"], data["mean"], data["scale"], True, fiber_source),
            "lift_scalar": lambda: lift_samples(active_fit.model, fiber, labels, nfe, seed + 900 + nfe, full_source, data["active_indices"], data["mean"], data["scale"], False, fiber_source),
        }.items():
            clock = time.perf_counter()
            method_samples[name] = call()
            method_times[name] = time.perf_counter() - clock
        method_samples["split_copy"] = method_samples["lift_joint"].copy()
        method_times["split_copy"] = method_times["lift_joint"]
        if nfe == config.main_nfe:
            frontier[str(nfe)] = {
                name: all_metrics(
                    data["train_x"], data["train_y"], data["reference_x"], data["reference_y"], sample, labels,
                    seed + 1000 + nfe,
                )
                for name, sample in method_samples.items()
            }
        else:
            frontier[str(nfe)] = {
                name: {
                    "energy_score": energy_score(data["reference_x"], data["reference_y"], sample, labels),
                    "swd": sliced_wasserstein(data["reference_x"], sample, seed + 1000 + nfe, projections=128),
                }
                for name, sample in method_samples.items()
            }
        sampling_seconds[str(nfe)] = method_times
        if nfe == config.main_nfe:
            generated_cache = method_samples

    clock = time.perf_counter()
    vae_generated = sample_vae(vae_model, labels, seed + 1200)
    vae_sampling_seconds = time.perf_counter() - clock
    vae_metrics = all_metrics(
        data["train_x"], data["train_y"], data["reference_x"], data["reference_y"], vae_generated, labels, seed + 1300
    )
    vae_reconstruction = reconstruct_vae(vae_model, data["reference_x"], data["reference_y"])
    vae_reconstruction_mse = float(np.mean((vae_reconstruction - data["reference_x"]) ** 2))

    # Exact controls and operation accounting.
    copy_error = float(np.max(np.abs(generated_cache["split_copy"] - generated_cache["lift_joint"])))
    full_per_eval = full_fit.model.flop_proxy()
    active_per_eval = active_fit.model.flop_proxy()
    feature_dimension = 1 + 3 * 16 + 10
    fiber_flops = 2 * feature_dimension * 48 + 1200 + 16 * 32
    inverse_flops = 64 * 12
    operation_proxy = {
        str(nfe): {
            "full_flow": int(nfe * full_per_eval + inverse_flops),
            "lift_joint": int(nfe * active_per_eval + fiber_flops + inverse_flops),
            "lift_scalar": int(nfe * active_per_eval + fiber_flops + inverse_flops),
            "ratio_lift_to_full": float((nfe * active_per_eval + fiber_flops + inverse_flops) / (nfe * full_per_eval + inverse_flops)),
        }
        for nfe in config.nfe_grid
    }
    parameters = {
        "full_flow": count_parameters(full_fit.model),
        "active_flow": count_parameters(active_fit.model),
        "fiber": int(fiber.ridge.size + fiber.location_scale.size + sum(model.weights.size + model.scales.size + model.shape.size for model in fiber.class_models)),
        "vae": count_parameters(vae_model),
    }
    main = frontier[str(config.main_nfe)]
    gates = {
        "joint_fiber_validation_nll_better": validation_scalar_nll > validation_joint_nll,
        "lift_energy_noninferior_full": float(main["lift_joint"]["energy_score"]) <= float(main["full_flow"]["energy_score"]) + config.energy_margin,
        "lift_swd_noninferior_full": float(main["lift_joint"]["swd"]) <= float(main["full_flow"]["swd"]) + config.swd_margin,
        "lift_faster_total": sampling_seconds[str(config.main_nfe)]["lift_joint"] < sampling_seconds[str(config.main_nfe)]["full_flow"],
        "operation_proxy_lower": operation_proxy[str(config.main_nfe)]["ratio_lift_to_full"] < 1.0,
        "copy_control_exact": copy_error == 0.0,
        "haar_roundtrip": data["roundtrip"] < 1e-10,
        "vae_positive_reconstruction_error": vae_reconstruction_mse > 0.0,
    }

    return {
        "seed": seed,
        "phase": config.phase,
        "reference_partition": data["reference_name"],
        "split_ids_sha256": {
            "train": hashlib.sha256(np.asarray(data["train_ids"], dtype="<i8").tobytes()).hexdigest(),
            "validation": hashlib.sha256(np.asarray(data["validation_ids"], dtype="<i8").tobytes()).hexdigest(),
            "reference": hashlib.sha256(np.asarray(data["reference_ids"], dtype="<i8").tobytes()).hexdigest(),
        },
        "fiber_validation_nll_per_coefficient": {
            "joint": validation_joint_nll,
            "scalar_product": validation_scalar_nll,
            "scalar_minus_joint": validation_scalar_nll - validation_joint_nll,
        },
        "frontier": frontier,
        "vae": {
            "selected_beta": chosen["beta"],
            "validation_candidates": [{"beta": item["beta"], "validation_energy": item["validation_energy"]} for item in vae_candidates],
            "metrics": vae_metrics,
            "reconstruction_mse": vae_reconstruction_mse,
            "sampling_seconds": vae_sampling_seconds,
        },
        "sampling_seconds": sampling_seconds,
        "training_seconds": {
            "full_flow": full_fit.seconds,
            "active_flow": active_fit.seconds,
            "fiber": fiber_fit_seconds,
            "vae_grid_total": vae_search_seconds,
            "total": time.perf_counter() - started,
        },
        "operation_proxy": operation_proxy,
        "parameters": parameters,
        "activation_proxy_bytes": {
            "full_flow": full_fit.model.activation_proxy_bytes(),
            "active_flow": active_fit.model.activation_proxy_bytes(),
        },
        "roundtrip_max_error": data["roundtrip"],
        "copy_max_error": copy_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def paired_summary(units: list[dict[str, object]], config: Config) -> dict[str, object]:
    main = str(config.main_nfe)
    contrasts = {}
    definitions = {
        "full_minus_lift_energy": lambda unit: float(unit["frontier"][main]["full_flow"]["energy_score"]) - float(unit["frontier"][main]["lift_joint"]["energy_score"]),
        "scalar_minus_joint_energy": lambda unit: float(unit["frontier"][main]["lift_scalar"]["energy_score"]) - float(unit["frontier"][main]["lift_joint"]["energy_score"]),
        "full_minus_lift_swd": lambda unit: float(unit["frontier"][main]["full_flow"]["swd"]) - float(unit["frontier"][main]["lift_joint"]["swd"]),
        "scalar_minus_joint_nll": lambda unit: float(unit["fiber_validation_nll_per_coefficient"]["scalar_minus_joint"]),
        "time_ratio_lift_full": lambda unit: float(unit["sampling_seconds"][main]["lift_joint"]) / float(unit["sampling_seconds"][main]["full_flow"]),
    }
    for name, function in definitions.items():
        values = np.asarray([function(unit) for unit in units], dtype=float)
        if len(values) == 1:
            low = high = float(values[0])
        else:
            critical = 2.7764451051977987 if len(values) == 5 else 1.96
            half = critical * values.std(ddof=1) / np.sqrt(len(values))
            low, high = float(values.mean() - half), float(values.mean() + half)
        contrasts[name] = {"mean": float(values.mean()), "ci95": [low, high], "values": values.tolist()}
    confirmation = len(units) == 5 and config.phase == "confirmation"
    promotion = {
        "joint_beats_scalar_nll": contrasts["scalar_minus_joint_nll"]["ci95"][0] > 0.0 if confirmation else None,
        "energy_noninferior_full": contrasts["full_minus_lift_energy"]["ci95"][0] > -config.energy_margin if confirmation else None,
        "swd_noninferior_full": contrasts["full_minus_lift_swd"]["ci95"][0] > -config.swd_margin if confirmation else None,
        "time_strictly_lower": contrasts["time_ratio_lift_full"]["ci95"][1] < 1.0 if confirmation else None,
        "all_unit_hard_checks": all(unit["gates"]["copy_control_exact"] and unit["gates"]["haar_roundtrip"] for unit in units),
    }
    return {
        "contrasts": contrasts,
        "promotion": promotion,
        "all_confirmation_promotion_gates_pass": all(value for value in promotion.values() if value is not None) if confirmation else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--frozen-config-hash")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists")
    if args.phase == "confirmation":
        seeds = tuple(range(4100, 4105))
    else:
        seeds = (4099,)
    config = Config(phase=args.phase, seeds=seeds)
    if args.quick:
        if args.phase == "confirmation":
            raise SystemExit("quick mode is development only")
        config = Config(
            phase=args.phase,
            seeds=seeds,
            flow_steps=120,
            vae_steps=100,
            generated_samples=300,
            nfe_grid=(4, 16),
        )
    method_hash = stable_hash(method_configuration(config))
    if args.phase == "confirmation" and args.frozen_config_hash != method_hash:
        raise SystemExit(f"confirmation requires frozen method hash {method_hash}")
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", {
        **asdict(config),
        "method_config_hash": method_hash,
        "split_manifest": split_manifest(),
        "test_data_opened": args.phase == "confirmation",
        "provenance": provenance(),
    })
    units = []
    for seed in seeds:
        print(f"[lift-fm digits] phase={args.phase} seed={seed}", flush=True)
        unit = train_and_evaluate(config, seed, allow_test=args.phase == "confirmation")
        units.append(unit)
        write_json(args.output / f"seed_{seed}.json", unit)
        print(json.dumps({"seed": seed, "gates": unit["gates"]}, indent=2), flush=True)
    summary = paired_summary(units, config)
    summary["method_config_hash"] = method_hash
    summary["phase"] = args.phase
    summary["seeds"] = list(seeds)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
