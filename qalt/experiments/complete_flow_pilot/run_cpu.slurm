#!/bin/bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --job-name=diff-complete-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:12:00
set -euo pipefail
: "${RESULT_ROOT:?RESULT_ROOT must be exported}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/qalt/src"
TASK_PYTHON=/ocean/projects/mth250006p/ywang26/pytorch/bin/python
"$TASK_PYTHON" -m pytest -q qalt/tests/test_spline.py qalt/tests/test_multiscale_flow.py qalt/tests/test_flow_matching.py
"$TASK_PYTHON" qalt/experiments/complete_flow_pilot/run.py --device cpu --world local --channels 1 --seed 3100 --steps 100000 --seconds 90 --output "$RESULT_ROOT/local_image"
