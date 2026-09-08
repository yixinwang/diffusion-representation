#!/bin/bash
#SBATCH --account=cis260243p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${RESULT_ROOT:?RESULT_ROOT must be exported}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/qalt/src"
TASK_PYTHON=/ocean/projects/mth250006p/ywang26/pytorch/bin/python
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'
"$TASK_PYTHON" -m pytest -q qalt/tests/test_spline.py qalt/tests/test_multiscale_flow.py qalt/tests/test_flow_matching.py
"$TASK_PYTHON" qalt/experiments/complete_flow_pilot/run.py --device cuda --world local --channels 1 --seed 3100 --steps 100000 --seconds 90 --output "$RESULT_ROOT/local_image"
"$TASK_PYTHON" qalt/experiments/complete_flow_pilot/run.py --device cuda --world distant --channels 1 --seed 3101 --steps 100000 --seconds 90 --output "$RESULT_ROOT/distant_image"
"$TASK_PYTHON" qalt/experiments/complete_flow_pilot/run.py --device cuda --world local --channels 4 --seed 3110 --steps 100000 --seconds 90 --output "$RESULT_ROOT/local_multiframe"
"$TASK_PYTHON" qalt/experiments/complete_flow_pilot/run.py --device cuda --world distant --channels 4 --seed 3111 --steps 100000 --seconds 90 --output "$RESULT_ROOT/distant_multiframe"
