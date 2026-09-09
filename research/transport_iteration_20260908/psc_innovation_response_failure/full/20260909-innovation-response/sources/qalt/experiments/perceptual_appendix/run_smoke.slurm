#!/bin/bash
#SBATCH --job-name=inception-fabricated-smoke
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:10:00

# RM-shared permits 2000M/core: reserve two cores for 4000M, computation stays one thread.
set -euo pipefail
# Positional arguments are explicit frozen checkout, revision, and new output.
# Root submits only after review/commit freeze. No dataset/bank argument exists.
if [[ "$#" -ne 3 ]]; then
  echo 'Usage: sbatch run_smoke.slurm FROZEN_CHECKOUT FROZEN_COMMIT NEW_OUTPUT' >&2
  exit 2
fi
EVAL_CHECKOUT="$1"
EVAL_REVISION="$2"
EVAL_OUTPUT="$3"
EVAL_ASSETS=/ocean/projects/mth250006p/ywang26/diffusion-evaluator-20260909
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
/ocean/projects/mth250006p/ywang26/pytorch/bin/python -B \
  "$EVAL_CHECKOUT/qalt/experiments/perceptual_appendix/smoke.py" \
  --expected-commit "$EVAL_REVISION" \
  --inception-source "$EVAL_ASSETS/inception.py" \
  --weights "$EVAL_ASSETS/pt_inception-2015-12-05-6726825d.pth" \
  --output "$EVAL_OUTPUT"
