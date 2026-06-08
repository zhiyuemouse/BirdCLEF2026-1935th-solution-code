#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-input/train_soundscapes}"
TARGET_ROWS_PATH="${TARGET_ROWS_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
CKPT_DIR="${CKPT_DIR:-ckpt/AudioMAE-HF}"
OUTPUT_DIR="${OUTPUT_DIR:-audiomae_soundscape_token_cache_cnn195634folds_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
LIMIT_ROWS="${LIMIT_ROWS:--1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ARGS=(
  --soundscapes-dir "${SOUNDSCAPES_DIR}"
  --target-rows-path "${TARGET_ROWS_PATH}"
  --ckpt-dir "${CKPT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --limit-rows "${LIMIT_ROWS}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

"${PYTHON_BIN}" birdclef2026_cache_audiomae_soundscape_tokens.py "${ARGS[@]}"
