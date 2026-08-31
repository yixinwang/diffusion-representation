#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RESULT_ROOT" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
result_root=$(realpath -m "$1")
source_commit=$(git rev-parse HEAD)
mkdir -p "$result_root"

sbatch \
  --export="ALL,RESULT_ROOT=$result_root,SOURCE_COMMIT=$source_commit" \
  --output="$result_root/slurm_%A_%a.out" \
  "$repo_root/qalt/experiments/observed_b1_rgb_block/run_seed.slurm"
