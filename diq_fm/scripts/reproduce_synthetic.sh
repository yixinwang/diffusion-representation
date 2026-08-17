#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
python "$ROOT/experiments/nonlinear_nongaussian/run.py"
python "$ROOT/experiments/nonlinear_nongaussian/run_vae.py"
