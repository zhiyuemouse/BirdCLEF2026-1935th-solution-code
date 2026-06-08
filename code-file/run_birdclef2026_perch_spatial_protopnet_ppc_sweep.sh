#!/usr/bin/env bash
set -euo pipefail

PPCS="${PPCS:-2 3 8 12}"
LOG_DIR="${LOG_DIR:-outputs}"
RUN_TAG="${RUN_TAG:-20260520}"
EPOCHS="${EPOCHS:-260}"
PATIENCE="${PATIENCE:-35}"
BASE_OUTPUT_PREFIX="${BASE_OUTPUT_PREFIX:-outputs/perch_spatial_protopnet_labeled_all_cnn195634folds_nopca_noraw_ppc}"

mkdir -p "${LOG_DIR}"

for PPC in ${PPCS}; do
  OUTPUT_DIR="${BASE_OUTPUT_PREFIX}${PPC}_v1"
  LOG_PATH="${LOG_DIR}/perch_spatial_protopnet_ppc${PPC}_${RUN_TAG}.log"

  echo "[SWEEP] prototype_per_class=${PPC}"
  echo "[SWEEP] output_dir=${OUTPUT_DIR}"
  echo "[SWEEP] log_path=${LOG_PATH}"

  PROTOTYPE_PER_CLASS="${PPC}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  ./run_birdclef2026_perch_spatial_protopnet_train.sh 2>&1 | tee "${LOG_PATH}"
done

