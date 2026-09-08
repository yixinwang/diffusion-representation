"""Strict CIFAR input-readiness check: no fitting, generation or quality metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

import numpy as np
import scipy
from scipy.special import expit

from qalt.data_integrity import stable_json_hash
from qalt.observed_flow_data import load_observed_flow_data

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "qalt/src/qalt/data_integrity.py",
    "qalt/src/qalt/observed_flow_data.py",
    "qalt/data/observed_manifest_v1.json",
    "qalt/experiments/observed_flow_pilot/PROTOCOL.md",
    "qalt/experiments/observed_flow_pilot/check_data.py",
    "qalt/experiments/observed_flow_pilot/run_data_check.slurm",
)
DATA_ROOT = Path("/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py")
FLOAT64_ROUNDTRIP_LIMIT = 1e-12
FLOAT32_LOGIT_ROUNDTRIP_LIMIT = 1e-6
CHUNK_SIZE = 64
SOFT_DEADLINE_SECONDS = 270
DIMENSION = 3072


def _source_guard(requested):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if requested != commit:
        raise ValueError("HEAD differs from requested source commit")
    hashes = {}
    for relative in SOURCE_FILES:
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        actual = (ROOT / relative).read_bytes()
        if actual != committed:
            raise ValueError(f"registered source differs from committed file: {relative}")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    return commit, hashes


def _atomic_json(path, value):
    if path.exists():
        raise FileExistsError("refusing to replace an existing result")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _timeout(signum, frame):
    raise TimeoutError("readiness check exceeded its fixed 270-second limit")


def check_transformation(values):
    """Check only fixed numerical charts, in chunks; return no image content.

    The float32 check includes sigmoid evaluated on float32 logits. It does
    not send rounded unit-cube values into another logit or a fitted model.
    All methods must receive the same float64-logit-then-float32 pipeline.
    """
    if values.dtype != np.float64 or values.ndim != 4 or values.shape[1:] != (3, 32, 32):
        raise ValueError("expected native float64 NCHW RGB32 observations")
    if math.prod(values.shape[1:]) != DIMENSION:
        raise AssertionError("source-coordinate dimension is not 3072")
    max64 = max32 = max_logit_cast = max_ld_cancellation = 0.
    rounded_sigmoid_boundary_count = 0
    logit64_hash, logit32_hash, jacobian_hash = (hashlib.sha256() for _ in range(3))
    for start in range(0, len(values), CHUNK_SIZE):
        x = values[start:start + CHUNK_SIZE]
        if not np.all(np.isfinite(x)) or not np.all((x > 0) & (x < 1)):
            raise FloatingPointError("nonfinite or boundary unit-cube input")
        log_x, log_one_minus_x = np.log(x), np.log1p(-x)
        y64 = log_x - log_one_minus_x
        jacobian = (-log_x - log_one_minus_x).reshape(len(x), -1).sum(axis=1)
        y32 = y64.astype(np.float32)
        inverse64, inverse32 = expit(y64), expit(y32)
        if inverse32.dtype != np.float32:
            raise TypeError("float32 sigmoid computation unexpectedly changed dtype")
        inverse_jacobian = (-np.logaddexp(0., -y64) - np.logaddexp(0., y64)).reshape(len(x), -1).sum(axis=1)
        arrays = (y64, y32, inverse64, inverse32, jacobian, inverse_jacobian)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise FloatingPointError("nonfinite fixed-chart value or Jacobian")
        max64 = max(max64, float(np.max(np.abs(inverse64-x))))
        max32 = max(max32, float(np.max(np.abs(inverse32-x))))
        max_logit_cast = max(max_logit_cast, float(np.max(np.abs(y32.astype(np.float64)-y64))))
        max_ld_cancellation = max(max_ld_cancellation, float(np.max(np.abs(jacobian+inverse_jacobian))))
        rounded_sigmoid_boundary_count += int(np.count_nonzero((inverse32 <= 0) | (inverse32 >= 1)))
        for digest, array in ((logit64_hash,y64),(logit32_hash,y32),(jacobian_hash,jacobian)):
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    passed = max64 <= FLOAT64_ROUNDTRIP_LIMIT and max32 <= FLOAT32_LOGIT_ROUNDTRIP_LIMIT
    return {
        "record_count": len(values), "coordinates_per_record": DIMENSION,
        "all_checked_tensors_finite": True, "unit_cube_input_strictly_interior": True,
        "float64_logit_sigmoid_roundtrip_max": max64,
        "float64_roundtrip_limit": FLOAT64_ROUNDTRIP_LIMIT,
        "float32_logit_sigmoid_roundtrip_max": max32,
        "float32_logit_roundtrip_limit": FLOAT32_LOGIT_ROUNDTRIP_LIMIT,
        "float32_logit_cast_error_max": max_logit_cast,
        "float64_log_jacobian_cancellation_max": max_ld_cancellation,
        "float32_sigmoid_rounded_boundary_count": rounded_sigmoid_boundary_count,
        "float32_boundary_interpretation": "count is a precision diagnostic; outputs are not reused as strict-unit-cube inputs or clipped",
        "float64_logits_raw_bytes_sha256": logit64_hash.hexdigest(),
        "float32_logits_raw_bytes_sha256": logit32_hash.hexdigest(),
        "float64_per_image_log_jacobian_raw_bytes_sha256": jacobian_hash.hexdigest(),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output directory already exists")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key}=1 is required")
    commit, hashes = _source_guard(args.source_commit)
    args.output.mkdir(parents=True, exist_ok=False)
    started, phase = time.perf_counter(), "strict_loader"
    provenance = {
        "kind": "input_readiness_only_no_generative_results",
        "source_commit": commit, "source_hashes": hashes, "data_root": str(args.data_root),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "configuration": {"fit_count":4000,"repair_count":1000,"coordinate_count":DIMENSION,
                          "float64_roundtrip_limit":FLOAT64_ROUNDTRIP_LIMIT,
                          "float32_logit_roundtrip_limit":FLOAT32_LOGIT_ROUNDTRIP_LIMIT,
                          "chunk_size":CHUNK_SIZE,"soft_deadline_seconds":SOFT_DEADLINE_SECONDS},
        "transformation": "float64 X -> log(X)-log1p(-X) in float64 -> cast logits to float32, identically for all methods",
        "jacobian": "per-image sum[-log(X)-log1p(-X)] in float64; retained explicitly by future likelihood code",
        "training_performed": False, "generation_performed": False,
        "quality_statistics_computed": False, "test_data_accessed": False,
        "pro5_twenty_thousand_split_executed": False,
    }
    _atomic_json(args.output / "provenance.json", provenance)
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, SOFT_DEADLINE_SECONDS)
    try:
        # No fixture opt-out is exposed or supplied: canonical hashes mandatory.
        data = load_observed_flow_data(args.data_root)
        if data.ledger["canonical_dataset_verified"] is not True or data.ledger["allow_noncanonical_fixture"] is not False:
            raise AssertionError("strict canonical dataset verification is required")
        if data.fit.shape != (4000,3,32,32) or data.repair.shape != (1000,3,32,32):
            raise AssertionError("fixed hash-selected subset sizes differ")
        if len(data.fit_ids) != 4000 or len(data.repair_ids) != 1000:
            raise AssertionError("selected record-ID counts differ")
        ledger_without_hash = {k:v for k,v in data.ledger.items() if k != "ledger_sha256"}
        if stable_json_hash(ledger_without_hash) != data.ledger["ledger_sha256"]:
            raise AssertionError("loader ledger hash does not match")
        _atomic_json(args.output / "data_ledger.json", data.ledger)
        _atomic_json(args.output / "record_ids.json", {"fit":data.fit_ids.tolist(),"seen_repair":data.repair_ids.tolist()})
        phase = "fixed_transform_checks"
        checks = {}
        for role, array in (("fit",data.fit),("seen_repair",data.repair)):
            checks[role] = check_transformation(array)
            _atomic_json(args.output / f"{role}_numerical_checks.json", checks[role])
            if not checks[role]["passed"]:
                raise ArithmeticError(f"frozen fixed-chart roundtrip tolerance failed: {role}")
        phase = "complete"
        signal.setitimer(signal.ITIMER_REAL, 0)
        result = {"status":"input_readiness_checks_passed", "source_commit":commit,
                  "numerical_checks":checks,"wall_seconds":time.perf_counter()-started,
                  "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024),
                  "training_performed":False,"generation_performed":False,"quality_statistics_computed":False,
                  "interpretation":"Input identity and fixed-chart numerical validation only; no generative performance evidence."}
        _atomic_json(args.output / "summary.json", result)
        payloads = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*.json"))}
        _atomic_json(args.output / "COMPLETE.json", {"status":result["status"],"source_commit":commit,"payload_hashes":payloads})
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _atomic_json(args.output / "FAILED.json", {"source_commit":commit,"phase":phase,
                     "exception_type":type(exc).__name__,"message":str(exc),"wall_seconds":time.perf_counter()-started,
                     "training_performed":False,"generation_performed":False,"test_data_accessed":False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
