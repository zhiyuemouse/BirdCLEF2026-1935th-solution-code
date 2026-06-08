#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1
SEED=2026

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MODEL_NAME="convnextv2_atto.fcmae_ft_in1k"
HEAD_TYPE="csiro_conv_v1"
STAGE1_EPOCHS=12
STAGE2_EPOCHS=28
STAGE1_BATCH_SIZE=8
STAGE2_BATCH_SIZE=8
STAGE1_BACKBONE_LR=1e-4
STAGE1_HEAD_LR=1e-3
STAGE2_BACKBONE_LR=5e-5
STAGE2_HEAD_LR=5e-4

# Keep the current best augmentation recipe fixed so this run isolates the
# waveform side branch effect.
MIXUP_ALPHA=0.0
MIXUP_PROB=0.0
CUTMIX_ALPHA=0.0
CUTMIX_PROB=0.0
MIXUP_DOMAIN="waveform"

STAGE1_MIXUP_ALPHA=0.20
STAGE1_MIXUP_PROB=0.10
STAGE1_CUTMIX_ALPHA=0.0
STAGE1_CUTMIX_PROB=0.0

STAGE2_MIXUP_ALPHA=0.0
STAGE2_MIXUP_PROB=0.0
STAGE2_CUTMIX_ALPHA=0.0
STAGE2_CUTMIX_PROB=0.0

WAVEFORM_BRANCH_D_MODEL=128
WAVEFORM_BRANCH_LAYERS=1
WAVEFORM_BRANCH_HEADS=4
WAVEFORM_BRANCH_DROPOUT=0.10

cmd=(
  python birdclef2026_gm_train.py
  --model-name "${MODEL_NAME}"
  --head-type "${HEAD_TYPE}"
  --use-waveform-branch
  --waveform-branch-d-model "${WAVEFORM_BRANCH_D_MODEL}"
  --waveform-branch-layers "${WAVEFORM_BRANCH_LAYERS}"
  --waveform-branch-heads "${WAVEFORM_BRANCH_HEADS}"
  --waveform-branch-dropout "${WAVEFORM_BRANCH_DROPOUT}"
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
