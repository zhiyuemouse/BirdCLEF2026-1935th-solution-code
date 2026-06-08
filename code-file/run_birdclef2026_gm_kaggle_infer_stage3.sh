#!/usr/bin/env bash
set -euo pipefail

# Stage3-only Kaggle submission template.
# This intentionally does not blend Perch or apply extra post-processing.

PYTHON_CMD=(python)

COMPETITION_ROOT="${COMPETITION_ROOT:-/kaggle/input/competitions/birdclef-2026}"
MODEL_ROOT="${MODEL_ROOT:-/kaggle/input/birdclef2026-stage3-perchcnn-white-v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo}"
OUTPUT_PATH="${OUTPUT_PATH:-/kaggle/working/submission.csv}"
SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-}"
SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-12}"
SEED="${SEED:-2026}"
DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-4}"

cmd=(
  "${PYTHON_CMD[@]}"
  birdclef2026_gm_kaggle_infer_stage3.py
  --competition-root "${COMPETITION_ROOT}"
  --model-root "${MODEL_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --segment-batch-size "${SEGMENT_BATCH_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  cmd+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi

if [[ "${DEBUG}" == "1" ]]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
