#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
INPUT_DIR="${INPUT_DIR:-input}"
TRAIN_CSV_PATH="${TRAIN_CSV_PATH:-}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-}"
ONNX_PATH="${ONNX_PATH:-PerchV2Onnx/perch_v2.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-perch_audio_embedding_cache_max100}"
MAX_PER_CLASS="${MAX_PER_CLASS:-100}"
MIN_RATING="${MIN_RATING:--1}"
LIMIT_ROWS="${LIMIT_ROWS:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_THREADS="${NUM_THREADS:-4}"
SEED="${SEED:-2026}"
INCLUDE_SECONDARY_LABELS="${INCLUDE_SECONDARY_LABELS:-0}"

ARGS=(
  --input-dir "${INPUT_DIR}"
  --onnx-path "${ONNX_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-per-class "${MAX_PER_CLASS}"
  --min-rating "${MIN_RATING}"
  --limit-rows "${LIMIT_ROWS}"
  --batch-size "${BATCH_SIZE}"
  --num-threads "${NUM_THREADS}"
  --seed "${SEED}"
)

if [[ -n "${TRAIN_CSV_PATH}" ]]; then
  ARGS+=(--train-csv-path "${TRAIN_CSV_PATH}")
fi
if [[ -n "${TRAIN_AUDIO_DIR}" ]]; then
  ARGS+=(--train-audio-dir "${TRAIN_AUDIO_DIR}")
fi
if [[ -n "${SAMPLE_SUBMISSION_PATH}" ]]; then
  ARGS+=(--sample-submission-path "${SAMPLE_SUBMISSION_PATH}")
fi
if [[ "${INCLUDE_SECONDARY_LABELS}" == "1" ]]; then
  ARGS+=(--include-secondary-labels)
fi

"${PYTHON_BIN}" birdclef2026_cache_perch_audio_embedding_onnx.py "${ARGS[@]}"
