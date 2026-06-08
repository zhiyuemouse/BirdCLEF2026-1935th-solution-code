#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COMPETITION_ROOT="${COMPETITION_ROOT:-/kaggle/input/competitions/birdclef-2026}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-${TEST_SOUNDSCAPES_DIR:-}}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-}"
AUDIOMAE_CKPT_DIR="${AUDIOMAE_CKPT_DIR:-/kaggle/input/birdclef2026-audiomae-hf/AudioMAE-HF}"
MODEL_PATH="${MODEL_PATH:-/kaggle/input/birdclef2026-audiomae-token-attention/audiomae_token_head_artifacts.joblib}"
OUTPUT_PATH="${OUTPUT_PATH:-/kaggle/working/submission.csv}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
FILE_SCALE_TOPK="${FILE_SCALE_TOPK:-2}"
DISABLE_FILE_SCALE="${DISABLE_FILE_SCALE:-0}"
DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-10}"
SEED="${SEED:-2026}"

ARGS=(
  --competition-root "${COMPETITION_ROOT}"
  --audiomae-ckpt-dir "${AUDIOMAE_CKPT_DIR}"
  --model-path "${MODEL_PATH}"
  --output-path "${OUTPUT_PATH}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --file-scale-topk "${FILE_SCALE_TOPK}"
  --seed "${SEED}"
)

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  ARGS+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi
if [[ -n "${SAMPLE_SUBMISSION_PATH}" ]]; then
  ARGS+=(--sample-submission-path "${SAMPLE_SUBMISSION_PATH}")
fi
if [[ "${DISABLE_FILE_SCALE}" == "1" ]]; then
  ARGS+=(--disable-file-scale)
fi
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

"${PYTHON_BIN}" birdclef2026_kaggle_infer_audiomae_token.py "${ARGS[@]}"
