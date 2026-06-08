#!/usr/bin/env bash
set -euo pipefail

# Train Perch LogReg folds aligned to the CNN stage2 fold assignment.
# This is the safe teacher for fold-specific Perch+CNN pseudo labels.

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/perch/bin/python}"
CACHE_DIR="${CACHE_DIR:-perch_cache_labeled_all}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
N_FOLDS="${N_FOLDS:-3}"
EMBEDDING_PCA_DIM="${EMBEDDING_PCA_DIM:-128}"
LOGREG_C="${LOGREG_C:-0.25}"
LOGREG_MAX_ITER="${LOGREG_MAX_ITER:-1000}"
LOGREG_MIN_POS="${LOGREG_MIN_POS:-8}"
SEED="${SEED:-2026}"

"${PYTHON_BIN}" birdclef2026_perch_context_train.py \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}" \
  --embedding-pca-dim "${EMBEDDING_PCA_DIM}" \
  --logreg-c "${LOGREG_C}" \
  --logreg-max-iter "${LOGREG_MAX_ITER}" \
  --logreg-min-pos "${LOGREG_MIN_POS}" \
  --seed "${SEED}"
