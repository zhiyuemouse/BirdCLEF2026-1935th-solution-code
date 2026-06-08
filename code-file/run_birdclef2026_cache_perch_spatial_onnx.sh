#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/perch/bin/python}"
INPUT_DIR="${INPUT_DIR:-input}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-}"
LABELS_PATH="${LABELS_PATH:-}"
ONNX_PATH="${ONNX_PATH:-PerchV2Onnx/perch_v2.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-perch_spatial_cache_labeled_all}"
TARGET_ROWS_PATH="${TARGET_ROWS_PATH:-}"
PSEUDO_ROOT="${PSEUDO_ROOT:-}"
FILE_SCOPE="${FILE_SCOPE:-labeled}"
LIMIT_FILES="${LIMIT_FILES:--1}"
NUM_THREADS="${NUM_THREADS:-4}"
BATCH_FILES="${BATCH_FILES:-1}"
CLIP_OFFSET_SECONDS="${CLIP_OFFSET_SECONDS:-0.0}"
SAVE_FREQ_MAX="${SAVE_FREQ_MAX:-0}"
SAVE_FLAT64="${SAVE_FLAT64:-0}"

ARGS=(
  --input-dir "${INPUT_DIR}"
  --onnx-path "${ONNX_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --file-scope "${FILE_SCOPE}"
  --limit-files "${LIMIT_FILES}"
  --num-threads "${NUM_THREADS}"
  --batch-files "${BATCH_FILES}"
  --clip-offset-seconds "${CLIP_OFFSET_SECONDS}"
)

if [[ "${SAVE_FREQ_MAX}" == "1" ]]; then
  ARGS+=(--save-freq-max)
fi
if [[ "${SAVE_FLAT64}" == "1" ]]; then
  ARGS+=(--save-flat64)
fi
if [[ -n "${TARGET_ROWS_PATH}" ]]; then
  ARGS+=(--target-rows-path "${TARGET_ROWS_PATH}")
fi
if [[ -n "${PSEUDO_ROOT}" ]]; then
  ARGS+=(--pseudo-root "${PSEUDO_ROOT}")
fi

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  ARGS+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi
if [[ -n "${LABELS_PATH}" ]]; then
  ARGS+=(--labels-path "${LABELS_PATH}")
fi

"${PYTHON_BIN}" birdclef2026_cache_perch_spatial_onnx.py "${ARGS[@]}"
