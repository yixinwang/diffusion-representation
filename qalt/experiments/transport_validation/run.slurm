#!/usr/bin/env bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
set -euo pipefail
: "${RESULT_ROOT:?set RESULT_ROOT to a new result directory}"
repo_root=$(git rev-parse --show-toplevel)
python_bin=/ocean/projects/mth250006p/ywang26/pytorch/bin/python
export PYTHONPATH="$repo_root/qalt/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_root"
mkdir -p "$RESULT_ROOT"
"$python_bin" -m pytest -q qalt/tests/test_output_screen.py qalt/tests/test_radial_transport.py
"$python_bin" qalt/experiments/transport_validation/run.py --output "$RESULT_ROOT/fixture.json"
