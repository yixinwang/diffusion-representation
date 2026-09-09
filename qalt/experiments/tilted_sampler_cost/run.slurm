#!/bin/bash
#SBATCH --job-name=tilt-central-cache
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:20:00
set -euo pipefail
# Frozen checkout, full revision and unique output are selected by root.
if [[ "$#" -ne 3 ]]; then
 echo 'Expected frozen-checkout full-revision new-output' >&2
 exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python -B "$1/qalt/experiments/tilted_sampler_cost/run.py" \
 --expected-commit "$2" --output "$3" --mode timing
