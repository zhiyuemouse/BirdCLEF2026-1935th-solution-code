#!/usr/bin/env bash
set -euo pipefail

COMPETITION_ROOT="${COMPETITION_ROOT:-/kaggle/input/competitions/birdclef-2026}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_SOUNDSCAPES_DIR="${TEST_SOUNDSCAPES_DIR:-${SOUNDSCAPES_DIR:-}}"
PERCH_DIR="${PERCH_DIR:-/kaggle/input/birdclef2026-perch/Perch}"
PERCH_ONNX_PATH="${PERCH_ONNX_PATH:-}"
MODEL_PATH="${MODEL_PATH:-/kaggle/input/birdclef2026-perch-spatial-mamba/perch_spatial_mamba_artifacts.joblib}"
OUTPUT_PATH="${OUTPUT_PATH:-/kaggle/working/submission.csv}"
FEATURES_NPZ="${FEATURES_NPZ:-}"
FEATURES_META_PATH="${FEATURES_META_PATH:-}"
BATCH_FILES="${BATCH_FILES:-16}"
RUNTIME_NUM_THREADS="${RUNTIME_NUM_THREADS:-4}"
SEED="${SEED:-2026}"
DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-4}"

ARGS=(
  --competition-root "${COMPETITION_ROOT}"
  --perch-dir "${PERCH_DIR}"
  --model-path "${MODEL_PATH}"
  --output-path "${OUTPUT_PATH}"
  --batch-files "${BATCH_FILES}"
  --runtime-num-threads "${RUNTIME_NUM_THREADS}"
  --seed "${SEED}"
)

if [[ -n "${TEST_SOUNDSCAPES_DIR}" ]]; then
  ARGS+=(--soundscapes-dir "${TEST_SOUNDSCAPES_DIR}")
fi
if [[ -n "${PERCH_ONNX_PATH}" ]]; then
  ARGS+=(--perch-onnx-path "${PERCH_ONNX_PATH}")
fi
if [[ -n "${FEATURES_NPZ}" ]]; then
  ARGS+=(--features-npz "${FEATURES_NPZ}")
fi
if [[ -n "${FEATURES_META_PATH}" ]]; then
  ARGS+=(--features-meta-path "${FEATURES_META_PATH}")
fi
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

"${PYTHON_BIN}" birdclef2026_perch_kaggle_infer_spatial_mamba.py "${ARGS[@]}"
