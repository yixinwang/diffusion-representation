#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=04:00:00
#SBATCH --array=0-4

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
seed=$((2100 + SLURM_ARRAY_TASK_ID))
data_root=${CIFAR_DATA_ROOT:-/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py}
python_bin=/ocean/projects/mth250006p/ywang26/pytorch/bin/python

export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$repo_root"
"$python_bin" qalt/experiments/observed_b1_rgb_block/run.py \
  --seed "$seed" \
  --output "$RESULT_ROOT/seed_$seed" \
  --data-root "$data_root" \
  --source-commit "$SOURCE_COMMIT"
