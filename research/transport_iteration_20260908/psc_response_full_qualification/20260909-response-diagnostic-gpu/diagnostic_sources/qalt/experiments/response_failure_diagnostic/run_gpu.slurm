#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodelist=v024
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@60
set -euo pipefail
: "${SOURCE_COMMIT:?full frozen revision required}"
: "${OUTPUT_DIR:?new output required}"
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python qalt/experiments/response_failure_diagnostic/run.py --expected-commit "$SOURCE_COMMIT" --output "$OUTPUT_DIR" --device cuda
