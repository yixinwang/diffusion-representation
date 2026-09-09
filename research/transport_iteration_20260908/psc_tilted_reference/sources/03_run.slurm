#!/usr/bin/env bash
#SBATCH --job-name=tilted-normal-reference
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@60
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "Usage: run.slurm REPO PYTHON SOURCE_COMMIT NEW_OUTPUT" >&2
    exit 2
fi
repo_dir="$1"
python_bin="$2"
source_commit="$3"
output_dir="$4"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_dir"
export PYTHONPATH="$repo_dir/qalt/src"
exec "$python_bin" qalt/experiments/tilted_normal_reference/run.py \
  --source-commit "$source_commit" --output "$output_dir"
