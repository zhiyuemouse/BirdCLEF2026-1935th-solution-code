#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
SEED="${SEED:-2026}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_gm_perch_spectrogram_stage2}"
CACHE_DIR="${CACHE_DIR:-perch_spectrogram_cache_labeled_all}"
MODEL_NAME="${MODEL_NAME:-convnextv2_atto.fcmae_ft_in1k}"
HEAD_TYPE="${HEAD_TYPE:-csiro_conv_v1}"
HEAD_POOL_TYPE="${HEAD_POOL_TYPE:-avg}"
SPEC_PROCESS="${SPEC_PROCESS:-direct_repeat}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-28}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE2_SAMPLES_PER_EPOCH="${STAGE2_SAMPLES_PER_EPOCH:-2048}"
STAGE2_BACKBONE_LR="${STAGE2_BACKBONE_LR:-5e-5}"
STAGE2_HEAD_LR="${STAGE2_HEAD_LR:-5e-4}"
STAGE2_FREEZE_BACKBONE_EPOCHS="${STAGE2_FREEZE_BACKBONE_EPOCHS:-1}"

MIXUP_ALPHA="${MIXUP_ALPHA:-0.0}"
MIXUP_PROB="${MIXUP_PROB:-0.0}"
CUTMIX_ALPHA="${CUTMIX_ALPHA:-0.0}"
CUTMIX_PROB="${CUTMIX_PROB:-0.0}"

cmd=(
  "${PYTHON_BIN}" birdclef2026_gm_train_perch_spectrogram_stage2.py
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --model-name "${MODEL_NAME}"
  --head-type "${HEAD_TYPE}"
  --head-pool-type "${HEAD_POOL_TYPE}"
  --spec-process "${SPEC_PROCESS}"
  --image-height "${IMAGE_HEIGHT}"
  --image-width "${IMAGE_WIDTH}"
  --seed "${SEED}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --stage2-samples-per-epoch "${STAGE2_SAMPLES_PER_EPOCH}"
  --stage2-backbone-lr "${STAGE2_BACKBONE_LR}"
  --stage2-head-lr "${STAGE2_HEAD_LR}"
  --stage2-freeze-backbone-epochs "${STAGE2_FREEZE_BACKBONE_EPOCHS}"
  --mixup-alpha "${MIXUP_ALPHA}"
  --mixup-prob "${MIXUP_PROB}"
  --cutmix-alpha "${CUTMIX_ALPHA}"
  --cutmix-prob "${CUTMIX_PROB}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
