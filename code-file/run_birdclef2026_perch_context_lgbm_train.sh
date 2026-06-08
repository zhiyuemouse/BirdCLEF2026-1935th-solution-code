#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-perch_cache_labeled_all}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_context_lgbm_labeled_all_v1}"
N_FOLDS="${N_FOLDS:-5}"
EMBEDDING_PCA_DIM="${EMBEDDING_PCA_DIM:-128}"
FEATURE_SET="${FEATURE_SET:-full}"
LGBM_MIN_POS="${LGBM_MIN_POS:-8}"
LGBM_N_ESTIMATORS="${LGBM_N_ESTIMATORS:-120}"
LGBM_LEARNING_RATE="${LGBM_LEARNING_RATE:-0.035}"
LGBM_NUM_LEAVES="${LGBM_NUM_LEAVES:-7}"
LGBM_MAX_DEPTH="${LGBM_MAX_DEPTH:-3}"
LGBM_MIN_CHILD_SAMPLES="${LGBM_MIN_CHILD_SAMPLES:-12}"
LGBM_SUBSAMPLE="${LGBM_SUBSAMPLE:-0.85}"
LGBM_COLSAMPLE_BYTREE="${LGBM_COLSAMPLE_BYTREE:-0.65}"
LGBM_REG_ALPHA="${LGBM_REG_ALPHA:-0.05}"
LGBM_REG_LAMBDA="${LGBM_REG_LAMBDA:-2.0}"
LGBM_CLASS_WEIGHT="${LGBM_CLASS_WEIGHT:-balanced}"
LGBM_N_JOBS="${LGBM_N_JOBS:-2}"
SEED="${SEED:-2026}"
PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"

"${PYTHON_BIN}" birdclef2026_perch_context_lgbm_train.py \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --embedding-pca-dim "${EMBEDDING_PCA_DIM}" \
  --feature-set "${FEATURE_SET}" \
  --lgbm-min-pos "${LGBM_MIN_POS}" \
  --lgbm-n-estimators "${LGBM_N_ESTIMATORS}" \
  --lgbm-learning-rate "${LGBM_LEARNING_RATE}" \
  --lgbm-num-leaves "${LGBM_NUM_LEAVES}" \
  --lgbm-max-depth "${LGBM_MAX_DEPTH}" \
  --lgbm-min-child-samples "${LGBM_MIN_CHILD_SAMPLES}" \
  --lgbm-subsample "${LGBM_SUBSAMPLE}" \
  --lgbm-colsample-bytree "${LGBM_COLSAMPLE_BYTREE}" \
  --lgbm-reg-alpha "${LGBM_REG_ALPHA}" \
  --lgbm-reg-lambda "${LGBM_REG_LAMBDA}" \
  --lgbm-class-weight "${LGBM_CLASS_WEIGHT}" \
  --lgbm-n-jobs "${LGBM_N_JOBS}" \
  --seed "${SEED}"
