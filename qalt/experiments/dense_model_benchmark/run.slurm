#!/usr/bin/env bash
#SBATCH --job-name=dense-full-model-benchmark
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nodelist=v024
#SBATCH --signal=B:USR1@60
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo 'Usage: run.slurm FROZEN_CHECKOUT FROZEN_REVISION NEW_OUTPUT' >&2
    exit 2
fi
BENCH_CHECKOUT="$1"
BENCH_REVISION="$2"
BENCH_OUTPUT="$3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$BENCH_CHECKOUT"
/ocean/projects/mth250006p/ywang26/pytorch/bin/python -B \
  qalt/experiments/dense_model_benchmark/run.py \
  --expected-commit "$BENCH_REVISION" --output "$BENCH_OUTPUT"
