#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-perch_cache_labeled_all}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_context_mlp_labeled_all_v1}"
N_FOLDS="${N_FOLDS:-5}"
EMBEDDING_PCA_DIM="${EMBEDDING_PCA_DIM:-64}"
FEATURE_SET="${FEATURE_SET:-base}"
CONTEXT_MODE="${CONTEXT_MODE:-core}"
MLP_MIN_POS="${MLP_MIN_POS:-4}"
HIDDEN_DIMS="${HIDDEN_DIMS:-192}"
DROPOUT="${DROPOUT:-0.4}"
EPOCHS="${EPOCHS:-220}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0008}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0015}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-10}"
PATIENCE="${PATIENCE:-30}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"
PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"

"${PYTHON_BIN}" birdclef2026_perch_context_mlp_train.py \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --embedding-pca-dim "${EMBEDDING_PCA_DIM}" \
  --feature-set "${FEATURE_SET}" \
  --context-mode "${CONTEXT_MODE}" \
  --mlp-min-pos "${MLP_MIN_POS}" \
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
  --device "${DEVICE}" \
  --seed "${SEED}"
