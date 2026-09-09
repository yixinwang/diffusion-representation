#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_COMMIT:?required frozen revision}"
: "${RESULT_ROOT:?required exclusive output}"
: "${FIT_FAMILY:?pixel or latent required}"
: "${FIT_SEED:?registered training seed required}"
export PYTHONPATH="$PWD/qalt/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# Stdlib bootstrap stamps the monotonic clock before NumPy/Torch/runner imports.
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c 'import os,sys,time; os.environ["RGB_PROCESS_START_NS"]=str(time.perf_counter_ns()); os.execv(sys.executable,[sys.executable,"-u","qalt/experiments/rgb_full_generation_fit/run.py",*sys.argv[1:]])' --family "$FIT_FAMILY" --seed "$FIT_SEED" --expected-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT"
