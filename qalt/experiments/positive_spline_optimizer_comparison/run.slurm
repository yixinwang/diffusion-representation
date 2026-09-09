#!/usr/bin/env bash
#SBATCH --job-name=positive-spline-optimizer-comparison
#SBATCH --partition=RM-shared
#SBATCH --account=cis260243p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
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
exec "$python_bin" qalt/experiments/positive_spline_optimizer_comparison/run.py \
  --source-commit "$source_commit" --output "$output_dir"
