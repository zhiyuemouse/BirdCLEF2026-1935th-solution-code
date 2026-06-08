#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-2026}"
PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_gm_sed_att_v1}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MODEL_NAME="${MODEL_NAME:-convnextv2_atto.fcmae_ft_in1k}"
HEAD_TYPE="sed_att_v1"
CLIP_SECONDS="${CLIP_SECONDS:-5}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
SED_FRAME_LOSS_WEIGHT="${SED_FRAME_LOSS_WEIGHT:-0.5}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-12}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-24}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
STAGE1_SAMPLES_PER_EPOCH="${STAGE1_SAMPLES_PER_EPOCH:-24000}"
STAGE2_SAMPLES_PER_EPOCH="${STAGE2_SAMPLES_PER_EPOCH:-2048}"

STAGE1_BACKBONE_LR="${STAGE1_BACKBONE_LR:-1e-4}"
STAGE1_HEAD_LR="${STAGE1_HEAD_LR:-1e-3}"
STAGE2_BACKBONE_LR="${STAGE2_BACKBONE_LR:-5e-5}"
STAGE2_HEAD_LR="${STAGE2_HEAD_LR:-5e-4}"

MIXUP_DOMAIN="${MIXUP_DOMAIN:-image}"
STAGE1_MIXUP_ALPHA="${STAGE1_MIXUP_ALPHA:-0.0}"
STAGE1_MIXUP_PROB="${STAGE1_MIXUP_PROB:-0.0}"
STAGE2_MIXUP_ALPHA="${STAGE2_MIXUP_ALPHA:-0.0}"
STAGE2_MIXUP_PROB="${STAGE2_MIXUP_PROB:-0.0}"

cmd=(
  "${PYTHON_BIN}" birdclef2026_gm_train.py
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --head-type "${HEAD_TYPE}"
  --clip-seconds "${CLIP_SECONDS}"
  --image-height "${IMAGE_HEIGHT}"
  --image-width "${IMAGE_WIDTH}"
  --sed-frame-loss-weight "${SED_FRAME_LOSS_WEIGHT}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --stage1-samples-per-epoch "${STAGE1_SAMPLES_PER_EPOCH}"
  --stage2-samples-per-epoch "${STAGE2_SAMPLES_PER_EPOCH}"
  --stage1-backbone-lr "${STAGE1_BACKBONE_LR}"
  --stage1-head-lr "${STAGE1_HEAD_LR}"
  --stage2-backbone-lr "${STAGE2_BACKBONE_LR}"
  --stage2-head-lr "${STAGE2_HEAD_LR}"
  --mixup-domain "${MIXUP_DOMAIN}"
  --stage1-mixup-alpha "${STAGE1_MIXUP_ALPHA}"
  --stage1-mixup-prob "${STAGE1_MIXUP_PROB}"
  --stage1-cutmix-alpha 0.0
  --stage1-cutmix-prob 0.0
  --stage2-mixup-alpha "${STAGE2_MIXUP_ALPHA}"
  --stage2-mixup-prob "${STAGE2_MIXUP_PROB}"
  --stage2-cutmix-alpha 0.0
  --stage2-cutmix-prob 0.0
)

if (($# > 0)); then
  cmd+=("$@")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
