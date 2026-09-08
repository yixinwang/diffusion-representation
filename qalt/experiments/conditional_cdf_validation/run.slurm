#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00

set -euo pipefail
: "${RESULT_ROOT:?set RESULT_ROOT to a new directory}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact committed revision}"
repo_root=$(git rev-parse --show-toplevel)
export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo_root"
/ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/conditional_cdf_validation/run.py \
  --source-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT"
