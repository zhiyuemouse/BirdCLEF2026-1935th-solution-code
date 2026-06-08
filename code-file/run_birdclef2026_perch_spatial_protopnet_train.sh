#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-perch_cache_labeled_all}"
SPATIAL_CACHE_DIR="${SPATIAL_CACHE_DIR:-perch_spatial_cache_labeled_all_flat64}"
LABELS_PATH="${LABELS_PATH:-input/train_soundscapes_labels.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-input/sample_submission.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_spatial_protopnet_labeled_all_cnn195634folds_nopca_noraw_v1}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
N_FOLDS="${N_FOLDS:-3}"
LIMIT_FILES="${LIMIT_FILES:--1}"
TOKEN_PCA_DIM="${TOKEN_PCA_DIM:-0}"
PROTOTYPE_PER_CLASS="${PROTOTYPE_PER_CLASS:-5}"
PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-12.0}"
PROTOTYPE_ORTH_WEIGHT="${PROTOTYPE_ORTH_WEIGHT:-0.01}"
PROTOTYPE_INIT_SOURCE="${PROTOTYPE_INIT_SOURCE:-random}"
PROTOTYPE_INIT_AUDIO_CACHE_DIR="${PROTOTYPE_INIT_AUDIO_CACHE_DIR:-perch_audio_spatial_cache_max100_flat64}"
PROTOTYPE_INIT_AUDIO_META_PATH="${PROTOTYPE_INIT_AUDIO_META_PATH:-}"
PROTOTYPE_INIT_AUDIO_ARRAYS_PATH="${PROTOTYPE_INIT_AUDIO_ARRAYS_PATH:-}"
PROTOTYPE_INIT_MAX_ROWS_PER_CLASS="${PROTOTYPE_INIT_MAX_ROWS_PER_CLASS:-80}"
PROTOTYPE_INIT_CANDIDATE_TOKENS="${PROTOTYPE_INIT_CANDIDATE_TOKENS:-2048}"
MLP_MIN_POS="${MLP_MIN_POS:-4}"
EPOCHS="${EPOCHS:-260}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-10}"
PATIENCE="${PATIENCE:-35}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"

"${PYTHON_BIN}" birdclef2026_perch_spatial_mamba_train.py \
  --base-cache-dir "${BASE_CACHE_DIR}" \
  --spatial-cache-dir "${SPATIAL_CACHE_DIR}" \
  --labels-path "${LABELS_PATH}" \
  --sample-submission-path "${SAMPLE_SUBMISSION_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}" \
  --limit-files "${LIMIT_FILES}" \
  --token-pca-dim "${TOKEN_PCA_DIM}" \
  --freq-pool flat64 \
  --head-variant prototype_pooling \
  --prototype-per-class "${PROTOTYPE_PER_CLASS}" \
  --prototype-temperature "${PROTOTYPE_TEMPERATURE}" \
  --prototype-orth-weight "${PROTOTYPE_ORTH_WEIGHT}" \
  --prototype-init-source "${PROTOTYPE_INIT_SOURCE}" \
  --prototype-init-audio-cache-dir "${PROTOTYPE_INIT_AUDIO_CACHE_DIR}" \
  --prototype-init-audio-meta-path "${PROTOTYPE_INIT_AUDIO_META_PATH}" \
  --prototype-init-audio-arrays-path "${PROTOTYPE_INIT_AUDIO_ARRAYS_PATH}" \
  --prototype-init-max-rows-per-class "${PROTOTYPE_INIT_MAX_ROWS_PER_CLASS}" \
  --prototype-init-candidate-tokens "${PROTOTYPE_INIT_CANDIDATE_TOKENS}" \
  --mlp-min-pos "${MLP_MIN_POS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --pos-weight-power "${POS_WEIGHT_POWER}" \
  --pos-weight-max "${POS_WEIGHT_MAX}" \
  --inner-val-files "${INNER_VAL_FILES}" \
  --patience "${PATIENCE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
