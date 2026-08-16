#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -e .
pytest -q
python experiments/synthetic_exact/run.py --seeds 0 1 2 3 4 --output results/synthetic_exact_verified
python experiments/sklearn_digits/run.py --seeds 0 1 2 3 4 --output results/sklearn_digits_verified
