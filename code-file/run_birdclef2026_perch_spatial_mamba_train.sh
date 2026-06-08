#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-perch_cache_labeled_all}"
SPATIAL_CACHE_DIR="${SPATIAL_CACHE_DIR:-perch_spatial_cache_labeled_all}"
PSEUDO_ROOT="${PSEUDO_ROOT:-}"
PSEUDO_SPATIAL_CACHE_DIR="${PSEUDO_SPATIAL_CACHE_DIR:-}"
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-1.0}"
MIN_PSEUDO_MAX_PROB="${MIN_PSEUDO_MAX_PROB:-0.0}"
MAX_PSEUDO_ROWS="${MAX_PSEUDO_ROWS:--1}"
LABELS_PATH="${LABELS_PATH:-input/train_soundscapes_labels.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-input/sample_submission.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_spatial_mamba_labeled_all_v1}"
N_FOLDS="${N_FOLDS:-5}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-}"
LIMIT_FILES="${LIMIT_FILES:--1}"
TOKEN_PCA_DIM="${TOKEN_PCA_DIM:-256}"
FREQ_POOL="${FREQ_POOL:-mean}"
USE_POS_EMBED="${USE_POS_EMBED:-0}"
INCLUDE_RAW_SCORES="${INCLUDE_RAW_SCORES:-1}"
RAW_PROJ_DIM="${RAW_PROJ_DIM:-128}"
HEAD_VARIANT="${HEAD_VARIANT:-generic}"
PROTOTYPE_PER_CLASS="${PROTOTYPE_PER_CLASS:-5}"
PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-12.0}"
PROTOTYPE_ORTH_WEIGHT="${PROTOTYPE_ORTH_WEIGHT:-0.01}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
TOKEN_MASK_PROB="${TOKEN_MASK_PROB:-0.0}"
TOKEN_MASK_MAX_FRAC="${TOKEN_MASK_MAX_FRAC:-0.15}"
MIXUP_PROB="${MIXUP_PROB:-0.0}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.4}"
MLP_MIN_POS="${MLP_MIN_POS:-4}"
EPOCHS="${EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-10}"
PATIENCE="${PATIENCE:-35}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"

ARGS=(
  --base-cache-dir "${BASE_CACHE_DIR}"
  --spatial-cache-dir "${SPATIAL_CACHE_DIR}"
  --pseudo-loss-weight "${PSEUDO_LOSS_WEIGHT}"
  --min-pseudo-max-prob "${MIN_PSEUDO_MAX_PROB}"
  --max-pseudo-rows "${MAX_PSEUDO_ROWS}"
  --labels-path "${LABELS_PATH}"
  --sample-submission-path "${SAMPLE_SUBMISSION_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --n-folds "${N_FOLDS}"
  --limit-files "${LIMIT_FILES}"
  --token-pca-dim "${TOKEN_PCA_DIM}"
  --freq-pool "${FREQ_POOL}"
  --raw-proj-dim "${RAW_PROJ_DIM}"
  --head-variant "${HEAD_VARIANT}"
  --prototype-per-class "${PROTOTYPE_PER_CLASS}"
  --prototype-temperature "${PROTOTYPE_TEMPERATURE}"
  --prototype-orth-weight "${PROTOTYPE_ORTH_WEIGHT}"
  --num-blocks "${NUM_BLOCKS}"
  --kernel-size "${KERNEL_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --token-mask-prob "${TOKEN_MASK_PROB}"
  --token-mask-max-frac "${TOKEN_MASK_MAX_FRAC}"
  --mixup-prob "${MIXUP_PROB}"
  --mixup-alpha "${MIXUP_ALPHA}"
  --mlp-min-pos "${MLP_MIN_POS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --pos-weight-power "${POS_WEIGHT_POWER}"
  --pos-weight-max "${POS_WEIGHT_MAX}"
  --inner-val-files "${INNER_VAL_FILES}"
  --patience "${PATIENCE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ "${INCLUDE_RAW_SCORES}" == "1" ]]; then
  ARGS+=(--include-raw-scores)
fi
if [[ "${USE_POS_EMBED}" == "1" ]]; then
  ARGS+=(--use-pos-embed)
fi
if [[ -n "${PSEUDO_ROOT}" ]]; then
  ARGS+=(--pseudo-root "${PSEUDO_ROOT}")
fi
if [[ -n "${PSEUDO_SPATIAL_CACHE_DIR}" ]]; then
  ARGS+=(--pseudo-spatial-cache-dir "${PSEUDO_SPATIAL_CACHE_DIR}")
fi
if [[ -n "${FOLD_ASSIGNMENT_PATH}" ]]; then
  ARGS+=(--fold-assignment-path "${FOLD_ASSIGNMENT_PATH}")
fi

"${PYTHON_BIN}" birdclef2026_perch_spatial_mamba_train.py "${ARGS[@]}"
