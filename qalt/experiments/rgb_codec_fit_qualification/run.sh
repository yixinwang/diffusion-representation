#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_COMMIT:?required frozen revision}"
: "${RESULT_ROOT:?required exclusive result path}"
export PYTHONPATH="$PWD/qalt/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python -u qalt/experiments/rgb_codec_fit_qualification/run.py --expected-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT"
