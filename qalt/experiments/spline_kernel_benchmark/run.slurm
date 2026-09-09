#!/bin/bash
#SBATCH --job-name=checked-spline-kernel
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --nodelist=v024
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo 'Usage: sbatch run.slurm FROZEN_CHECKOUT FROZEN_REVISION NEW_OUTPUT' >&2
  exit 2
fi
BENCH_CHECKOUT="$1"
BENCH_REVISION="$2"
BENCH_OUTPUT="$3"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
/ocean/projects/mth250006p/ywang26/pytorch/bin/python -B \
  "$BENCH_CHECKOUT/qalt/experiments/spline_kernel_benchmark/run.py" \
  --expected-commit "$BENCH_REVISION" --output "$BENCH_OUTPUT"
