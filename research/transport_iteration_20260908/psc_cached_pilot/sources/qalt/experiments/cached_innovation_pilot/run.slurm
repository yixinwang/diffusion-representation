#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --nodelist=v024
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${RESULT_ROOT:?new output directory required}"
: "${SOURCE_COMMIT:?full frozen source revision required}"
: "${INCEPTION_SOURCE:?pinned local upstream source required}"
: "${INCEPTION_WEIGHTS:?pinned local weights required}"
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TASK_PYTHON=/ocean/projects/mth250006p/ywang26/pytorch/bin/python
exec "$TASK_PYTHON" qalt/experiments/cached_innovation_pilot/run.py \
 --expected-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT" \
 --inception-source "$INCEPTION_SOURCE" --weights "$INCEPTION_WEIGHTS"
