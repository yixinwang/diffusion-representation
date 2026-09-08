#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=v005
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${RESULT_ROOT:?set RESULT_ROOT to a new output directory}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the full frozen source revision}"
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/ocean/projects/mth250006p/ywang26/pytorch/bin/python
exec "$TASK_PYTHON" qalt/experiments/observed_flow_pilot/run_shared_study.py \
  --source-commit "$SOURCE_COMMIT" --output "$RESULT_ROOT"
