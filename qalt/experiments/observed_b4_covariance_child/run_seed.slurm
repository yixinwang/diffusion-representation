#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RESULT_ROOT:?set RESULT_ROOT to a new result directory}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact committed source revision}"

repo_root=$(git rev-parse --show-toplevel)
data_root=${CIFAR_DATA_ROOT:-/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py}
python_bin=/ocean/projects/mth250006p/ywang26/pytorch/bin/python

export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$repo_root"
"$python_bin" qalt/experiments/observed_b4_covariance_child/run.py \
  --seed 2100 \
  --data-root "$data_root" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RESULT_ROOT/seed_2100"
