#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
export PYTHONPATH="$PWD/qalt/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python -u qalt/experiments/rgb_codec_readiness/run.py --expected-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT"
