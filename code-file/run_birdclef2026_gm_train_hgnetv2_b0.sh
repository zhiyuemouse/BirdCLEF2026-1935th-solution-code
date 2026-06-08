#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-2026}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_gm_hgnetv2_b0}"
MODEL_NAME="${MODEL_NAME:-hgnetv2_b0.ssld_stage2_ft_in1k}"
HEAD_TYPE="${HEAD_TYPE:-csiro_conv_v1}"
HEAD_POOL_TYPE="${HEAD_POOL_TYPE:-avg}"
SPECTROGRAM_VARIANT="${SPECTROGRAM_VARIANT:-logmel}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-12}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-28}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE1_BACKBONE_LR="${STAGE1_BACKBONE_LR:-1e-4}"
STAGE1_HEAD_LR="${STAGE1_HEAD_LR:-1e-3}"
STAGE2_BACKBONE_LR="${STAGE2_BACKBONE_LR:-5e-5}"
STAGE2_HEAD_LR="${STAGE2_HEAD_LR:-5e-4}"

# Keep this aligned with the current log-mel CNN mainline. This experiment only
# swaps the timm backbone to measure whether HGNetV2-B0 is a better CNN branch.
MIXUP_ALPHA="${MIXUP_ALPHA:-0.0}"
MIXUP_PROB="${MIXUP_PROB:-0.0}"
CUTMIX_ALPHA="${CUTMIX_ALPHA:-0.0}"
CUTMIX_PROB="${CUTMIX_PROB:-0.0}"
MIXUP_DOMAIN="${MIXUP_DOMAIN:-waveform}"

STAGE1_MIXUP_ALPHA="${STAGE1_MIXUP_ALPHA:-0.20}"
STAGE1_MIXUP_PROB="${STAGE1_MIXUP_PROB:-0.10}"
STAGE1_CUTMIX_ALPHA="${STAGE1_CUTMIX_ALPHA:-0.0}"
STAGE1_CUTMIX_PROB="${STAGE1_CUTMIX_PROB:-0.0}"

STAGE2_MIXUP_ALPHA="${STAGE2_MIXUP_ALPHA:-0.0}"
STAGE2_MIXUP_PROB="${STAGE2_MIXUP_PROB:-0.0}"
STAGE2_CUTMIX_ALPHA="${STAGE2_CUTMIX_ALPHA:-0.0}"
STAGE2_CUTMIX_PROB="${STAGE2_CUTMIX_PROB:-0.0}"

cmd=(
  "${PYTHON_BIN}" birdclef2026_gm_train.py
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --head-type "${HEAD_TYPE}"
  --head-pool-type "${HEAD_POOL_TYPE}"
  --spectrogram-variant "${SPECTROGRAM_VARIANT}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --stage1-backbone-lr "${STAGE1_BACKBONE_LR}"
  --stage1-head-lr "${STAGE1_HEAD_LR}"
  --stage2-backbone-lr "${STAGE2_BACKBONE_LR}"
  --stage2-head-lr "${STAGE2_HEAD_LR}"
  --mixup-alpha "${MIXUP_ALPHA}"
  --mixup-prob "${MIXUP_PROB}"
  --cutmix-alpha "${CUTMIX_ALPHA}"
  --cutmix-prob "${CUTMIX_PROB}"
  --mixup-domain "${MIXUP_DOMAIN}"
  --stage1-mixup-alpha "${STAGE1_MIXUP_ALPHA}"
  --stage1-mixup-prob "${STAGE1_MIXUP_PROB}"
  --stage1-cutmix-alpha "${STAGE1_CUTMIX_ALPHA}"
  --stage1-cutmix-prob "${STAGE1_CUTMIX_PROB}"
  --stage2-mixup-alpha "${STAGE2_MIXUP_ALPHA}"
  --stage2-mixup-prob "${STAGE2_MIXUP_PROB}"
  --stage2-cutmix-alpha "${STAGE2_CUTMIX_ALPHA}"
  --stage2-cutmix-prob "${STAGE2_CUTMIX_PROB}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
