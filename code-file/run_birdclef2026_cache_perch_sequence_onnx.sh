#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-input}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-}"
LABELS_PATH="${LABELS_PATH:-}"
TAXONOMY_PATH="${TAXONOMY_PATH:-}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-}"
PERCH_DIR="${PERCH_DIR:-Perch}"
ONNX_PATH="${ONNX_PATH:-PerchV2Onnx/perch_v2.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-perch_sequence_cache_labeled_all_full}"
FILE_SCOPE="${FILE_SCOPE:-labeled}"
LIMIT_FILES="${LIMIT_FILES:--1}"
BATCH_FILES="${BATCH_FILES:-1}"
NUM_THREADS="${NUM_THREADS:-4}"
PROXY_REDUCE="${PROXY_REDUCE:-max}"

ARGS=(
  --input-dir "${INPUT_DIR}"
  --perch-dir "${PERCH_DIR}"
  --onnx-path "${ONNX_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --file-scope "${FILE_SCOPE}"
  --limit-files "${LIMIT_FILES}"
  --batch-files "${BATCH_FILES}"
  --num-threads "${NUM_THREADS}"
  --proxy-reduce "${PROXY_REDUCE}"
)

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  ARGS+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi
if [[ -n "${LABELS_PATH}" ]]; then
  ARGS+=(--labels-path "${LABELS_PATH}")
fi
if [[ -n "${TAXONOMY_PATH}" ]]; then
  ARGS+=(--taxonomy-path "${TAXONOMY_PATH}")
fi
if [[ -n "${SAMPLE_SUBMISSION_PATH}" ]]; then
  ARGS+=(--sample-submission-path "${SAMPLE_SUBMISSION_PATH}")
fi

/home/hjs/anaconda3/envs/transformers/bin/python birdclef2026_cache_perch_sequence_onnx.py "${ARGS[@]}"
