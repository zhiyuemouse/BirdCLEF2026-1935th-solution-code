#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
CACHE_DIR="${CACHE_DIR:-audiomae_soundscape_cache_cnn195634folds_v1}"
LABELS_PATH="${LABELS_PATH:-input/train_soundscapes_labels.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-input/sample_submission.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/audiomae_mlp_labeled_cnn195634folds_h512_256_v1}"
N_FOLDS="${N_FOLDS:-3}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
LIMIT_FILES="${LIMIT_FILES:--1}"
MLP_MIN_POS="${MLP_MIN_POS:-4}"
FALLBACK_PROB="${FALLBACK_PROB:-0.5}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512,256}"
DROPOUT="${DROPOUT:-0.35}"
EPOCHS="${EPOCHS:-220}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-10}"
PATIENCE="${PATIENCE:-35}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"

"${PYTHON_BIN}" birdclef2026_audiomae_mlp_train.py \
  --cache-dir "${CACHE_DIR}" \
  --labels-path "${LABELS_PATH}" \
  --sample-submission-path "${SAMPLE_SUBMISSION_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}" \
  --limit-files "${LIMIT_FILES}" \
  --mlp-min-pos "${MLP_MIN_POS}" \
  --fallback-prob "${FALLBACK_PROB}" \
  --hidden-dims "${HIDDEN_DIMS}" \
  --dropout "${DROPOUT}" \
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
