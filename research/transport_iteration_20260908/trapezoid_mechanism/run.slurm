#!/bin/bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
: "${SOURCE_COMMIT:?frozen revision required}"
: "${OUTPUT_DIR:?exclusive output required}"
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python research/transport_iteration_20260908/trapezoid_mechanism/run.py --commit "$SOURCE_COMMIT" --output "$OUTPUT_DIR"
