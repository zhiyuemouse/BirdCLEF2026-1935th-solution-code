#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/perch/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-birdclef2026}"
INPUT_DIR="${INPUT_DIR:-input}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-}"
LABELS_PATH="${LABELS_PATH:-}"
ONNX_PATH="${ONNX_PATH:-PerchV2Onnx/perch_v2.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-perch_spectrogram_cache_labeled_all}"
FILE_SCOPE="${FILE_SCOPE:-labeled}"
LIMIT_FILES="${LIMIT_FILES:--1}"
BATCH_FILES="${BATCH_FILES:-1}"
NUM_THREADS="${NUM_THREADS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

args=(
  --input-dir "${INPUT_DIR}"
  --onnx-path "${ONNX_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --file-scope "${FILE_SCOPE}"
  --limit-files "${LIMIT_FILES}"
  --batch-files "${BATCH_FILES}"
  --num-threads "${NUM_THREADS}"
)

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  args+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi
if [[ -n "${LABELS_PATH}" ]]; then
  args+=(--labels-path "${LABELS_PATH}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi

"${PYTHON_BIN}" birdclef2026_cache_perch_spectrogram_onnx.py "${args[@]}"
