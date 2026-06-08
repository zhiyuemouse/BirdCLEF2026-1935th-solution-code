#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1
SEED=2026

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MODEL_NAME="convnextv2_atto.fcmae_ft_in1k"
HEAD_TYPE="csiro_conv_v1"
HEAD_POOL_TYPE="avg"
STAGE1_EPOCHS=12
STAGE2_EPOCHS=28
STAGE1_BATCH_SIZE=8
STAGE2_BATCH_SIZE=8
NUM_WORKERS=0
STAGE1_BACKBONE_LR=1e-4 # 1e-4
STAGE1_HEAD_LR=1e-3 # 1e-3
STAGE2_BACKBONE_LR=5e-5 # 5e-5
STAGE2_HEAD_LR=5e-4 # 5e-4
USE_PERCH_DISTILL=0
PERCH_SPATIAL_CACHE_DIR="perch_spatial_cache_labeled_all"
PERCH_SPATIAL_META_PATH=""
PERCH_SPATIAL_ARRAYS_PATH=""
PERCH_DISTILL_WEIGHT=0.05
PERCH_DISTILL_TOKEN_KEY="spatial_tokens"

# Batch-level augmentation knobs.
# Legacy shared defaults stay at zero; we control stage1/stage2 separately below.
MIXUP_ALPHA=0.0
MIXUP_PROB=0.0
CUTMIX_ALPHA=0.0
CUTMIX_PROB=0.0
MIXUP_DOMAIN="waveform"

# Mainline: 20260505_195634, no BirdCLEF2025 external stage1 rows.
# The 2025 external-data trial (20260506_175829) dropped final OOF to 0.7356.

STAGE1_MIXUP_ALPHA=0.20
STAGE1_MIXUP_PROB=0.10
STAGE1_CUTMIX_ALPHA=0.0
STAGE1_CUTMIX_PROB=0.0

# STAGE2_MIXUP_ALPHA=0.10
# STAGE2_MIXUP_PROB=0.05
STAGE2_MIXUP_ALPHA=0.0
STAGE2_MIXUP_PROB=0.0
STAGE2_CUTMIX_ALPHA=0.0
STAGE2_CUTMIX_PROB=0.0

cmd=(
  python birdclef2026_gm_train.py
  --model-name "${MODEL_NAME}"
  --head-type "${HEAD_TYPE}"
  --head-pool-type "${HEAD_POOL_TYPE}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
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

if [[ "${USE_PERCH_DISTILL}" == "1" ]]; then
  cmd+=(
    --use-perch-distill
    --perch-spatial-cache-dir "${PERCH_SPATIAL_CACHE_DIR}"
    --perch-distill-weight "${PERCH_DISTILL_WEIGHT}"
    --perch-distill-token-key "${PERCH_DISTILL_TOKEN_KEY}"
  )
  if [[ -n "${PERCH_SPATIAL_META_PATH}" ]]; then
    cmd+=(--perch-spatial-meta-path "${PERCH_SPATIAL_META_PATH}")
  fi
  if [[ -n "${PERCH_SPATIAL_ARRAYS_PATH}" ]]; then
    cmd+=(--perch-spatial-arrays-path "${PERCH_SPATIAL_ARRAYS_PATH}")
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
