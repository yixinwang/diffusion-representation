#!/usr/bin/env bash
set -euo pipefail
STAGE=/ocean/projects/mth260022p/ywang26/rgb-fit-qualification-launch-20260909-6ae48a91
REPO=/ocean/projects/mth260022p/ywang26/diffusion-rgb-fit-qualification-20260909-6ae48a91
RESULT=/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-rgb-fit-qualification-6ae48a91
REV=6ae48a91706c97e65c833f94bd51a7b54f74c653
export SOURCE_COMMIT="$REV" RESULT_ROOT="$RESULT"
export PYTHONPYCACHEPREFIX="$STAGE/pycache" XDG_CACHE_HOME="$STAGE/cache" TMPDIR="$STAGE/tmp"
export TORCH_HOME="$STAGE/torch-cache" CUDA_CACHE_PATH="$STAGE/cuda-cache"
export HF_HOME="$STAGE/hf-cache" TRITON_CACHE_DIR="$STAGE/triton-cache"
test ! -e "$RESULT"
cd "$REPO"
# One allocation, one worker, immediate release when the worker exits. No retries.
exec salloc --account=cis260243p --partition=GPU-shared --qos=gpuinteract --gres=gpu:v100-32:1 --exclude=v005 -N1 -n1 -c4 --mem=16000M -t01:00:00 --job-name=rgb-codec-fit-qualification \
 srun --export=ALL --output="$STAGE/worker-%j.out" --error="$STAGE/worker-%j.err" bash qalt/experiments/rgb_codec_fit_qualification/run.sh
