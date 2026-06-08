#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-perch_cache_labeled_all}"
SPATIAL_CACHE_DIR="${SPATIAL_CACHE_DIR:-perch_spatial_cache_labeled_all}"
AUDIO_CACHE_DIR="${AUDIO_CACHE_DIR:-perch_audio_spatial_cache_max20}"
LABELS_PATH="${LABELS_PATH:-input/train_soundscapes_labels.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-input/sample_submission.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_spatial_mamba_audio_pretrain_max20_v1}"
N_FOLDS="${N_FOLDS:-3}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
LIMIT_FILES="${LIMIT_FILES:--1}"
TOKEN_PCA_DIM="${TOKEN_PCA_DIM:-1536}"
FREQ_POOL="${FREQ_POOL:-mean}"
HEAD_VARIANT="${HEAD_VARIANT:-perch_mamba_v1}"
PROTOTYPE_PER_CLASS="${PROTOTYPE_PER_CLASS:-5}"
PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-12.0}"
PROTOTYPE_ORTH_WEIGHT="${PROTOTYPE_ORTH_WEIGHT:-0.01}"
TOKEN_MASK_PROB="${TOKEN_MASK_PROB:-0.0}"
TOKEN_MASK_MAX_FRAC="${TOKEN_MASK_MAX_FRAC:-0.15}"
MIXUP_PROB="${MIXUP_PROB:-0.0}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR_STAGE1="${LR_STAGE1:-0.0005}"
LR_STAGE2="${LR_STAGE2:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-10}"
PATIENCE="${PATIENCE:-35}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"

"${PYTHON_BIN}" birdclef2026_perch_spatial_mamba_audio_pretrain_train.py \
  --base-cache-dir "${BASE_CACHE_DIR}" \
  --spatial-cache-dir "${SPATIAL_CACHE_DIR}" \
  --audio-cache-dir "${AUDIO_CACHE_DIR}" \
  --labels-path "${LABELS_PATH}" \
  --sample-submission-path "${SAMPLE_SUBMISSION_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}" \
  --limit-files "${LIMIT_FILES}" \
  --token-pca-dim "${TOKEN_PCA_DIM}" \
  --freq-pool "${FREQ_POOL}" \
  --head-variant "${HEAD_VARIANT}" \
  --prototype-per-class "${PROTOTYPE_PER_CLASS}" \
  --prototype-temperature "${PROTOTYPE_TEMPERATURE}" \
  --prototype-orth-weight "${PROTOTYPE_ORTH_WEIGHT}" \
  --token-mask-prob "${TOKEN_MASK_PROB}" \
  --token-mask-max-frac "${TOKEN_MASK_MAX_FRAC}" \
  --mixup-prob "${MIXUP_PROB}" \
  --mixup-alpha "${MIXUP_ALPHA}" \
  --stage1-epochs "${STAGE1_EPOCHS}" \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr-stage1 "${LR_STAGE1}" \
  --lr-stage2 "${LR_STAGE2}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --pos-weight-power "${POS_WEIGHT_POWER}" \
  --pos-weight-max "${POS_WEIGHT_MAX}" \
  --inner-val-files "${INNER_VAL_FILES}" \
  --patience "${PATIENCE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
