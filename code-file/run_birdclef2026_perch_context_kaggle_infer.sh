#!/usr/bin/env bash
set -euo pipefail

COMPETITION_ROOT="${COMPETITION_ROOT:-/kaggle/input/competitions/birdclef-2026}"
TEST_SOUNDSCAPES_DIR="${TEST_SOUNDSCAPES_DIR:-${SOUNDSCAPES_DIR:-}}"
PERCH_DIR="${PERCH_DIR:-/kaggle/input/birdclef2026-perch/Perch}"
PERCH_BACKEND="${PERCH_BACKEND:-auto}"
PERCH_ONNX_PATH="${PERCH_ONNX_PATH:-}"
PERCH_TFLITE_PATH="${PERCH_TFLITE_PATH:-}"
MODEL_PATH="${MODEL_PATH:-/kaggle/input/birdclef2026-perch-context-logreg/perch_context_logreg_artifacts.joblib}"
OUTPUT_PATH="${OUTPUT_PATH:-/kaggle/working/submission.csv}"
BATCH_FILES="${BATCH_FILES:-32}"
RUNTIME_NUM_THREADS="${RUNTIME_NUM_THREADS:-4}"
SEED="${SEED:-2026}"

ARGS=(
  --competition-root "${COMPETITION_ROOT}"
  --perch-dir "${PERCH_DIR}"
  --perch-backend "${PERCH_BACKEND}"
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
if [[ -n "${PERCH_TFLITE_PATH}" ]]; then
  ARGS+=(--perch-tflite-path "${PERCH_TFLITE_PATH}")
fi

python birdclef2026_perch_kaggle_infer_context_logreg.py "${ARGS[@]}"
