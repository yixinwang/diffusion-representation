#!/bin/bash
#SBATCH --account=cis260243p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:20:00
set -euo pipefail
# Arguments supplied only after root freezes a commit:
# full commit, new detached checkout, verified header manifest, exclusive output
exec /ocean/projects/mth250006p/ywang26/pytorch/bin/python /ocean/projects/mth250006p/ywang26/video-diagnostic-launch/launch_video_diagnostic_guarded.py --commit "$1" --checkout "$2" --manifest "$3" --output "$4"
