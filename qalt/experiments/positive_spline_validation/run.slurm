#!/bin/bash
#SBATCH --job-name=positive-spline-validation
#SBATCH --partition=RM-shared
#SBATCH --account=cis260243p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:12:00

set -euo pipefail
if [ "$#" -lt 4 ]; then
    echo "Usage: run.slurm REPO PYTHON SOURCE_COMMIT OUTPUT [--smoke]" >&2
    exit 2
fi
repo_dir="$1"
python_bin="$2"
source_commit="$3"
output_dir="$4"
shift 4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo_dir"
export PYTHONPATH="$repo_dir/qalt/src"
exec "$python_bin" qalt/experiments/positive_spline_validation/run.py \
    --source-commit "$source_commit" --output "$output_dir" "$@"
