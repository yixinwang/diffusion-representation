#!/usr/bin/env bash
set -euo pipefail
STAGE=/ocean/projects/mth260022p/ywang26/rgb-readiness-launch-20260909-84c0cad3
REPO=/ocean/projects/mth260022p/ywang26/diffusion-rgb-readiness-20260909-84c0cad3
RESULT=/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-rgb-readiness-84c0cad3
REV=84c0cad383be64d6632a5fa61e5d1bc4b605ca5f
export SOURCE_COMMIT="$REV" RESULT_ROOT="$RESULT"
export PYTHONPYCACHEPREFIX="$STAGE/pycache" XDG_CACHE_HOME="$STAGE/cache" TMPDIR="$STAGE/tmp"
export TORCH_HOME="$STAGE/torch-cache" CUDA_CACHE_PATH="$STAGE/cuda-cache"
export HF_HOME="$STAGE/hf-cache" TRITON_CACHE_DIR="$STAGE/triton-cache"
test ! -e "$RESULT"
cd "$REPO"
# One allocation, one worker, immediate release when the worker exits. No retries.
exec salloc --account=cis260243p --partition=GPU-shared --qos=gpuinteract --gres=gpu:v100-32:1 --exclude=v005 -N1 -n1 -c4 --mem=16000M -t01:00:00 --job-name=rgb-codec-readiness \
 srun --export=ALL --output="$STAGE/worker-%j.out" --error="$STAGE/worker-%j.err" bash qalt/experiments/rgb_codec_readiness/run.sh
