#!/usr/bin/env bash
set -euo pipefail

# Dual-run Stage 2 ensemble submission template.
# Current pair:
# - old strong inference-side baseline
# - new stronger training-side stage1-only-cutmix run

PYTHON_CMD=(python)

COMPETITION_ROOT="/kaggle/input/competitions/birdclef-2026"
OUTPUT_PATH="/kaggle/working/submission.csv"
SOUNDSCAPES_DIR=""

MODEL_ROOTS=(
  "outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k"
  "outputs/birdclef2026_gm/20260501_165002_convnextv2_atto.fcmae_ft_in1k"
)

SEGMENT_BATCH_SIZE=12
TTA_OFFSETS="0,-1.25,1.25"
SMOOTHING_KERNEL="0.1,0.8,0.1"
SOUNDSCAPE_TOP_K=1

DEBUG=0
DEBUG_LIMIT=4

cmd=(
  "${PYTHON_CMD[@]}"
  birdclef2026_gm_kaggle_infer_ensemble.py
  --competition-root "${COMPETITION_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --segment-batch-size "${SEGMENT_BATCH_SIZE}"
  --tta-offsets "${TTA_OFFSETS}"
)

if [[ -n "${SMOOTHING_KERNEL}" ]]; then
  cmd+=(--smoothing-kernel "${SMOOTHING_KERNEL}")
fi

if [[ "${SOUNDSCAPE_TOP_K}" -gt 0 ]]; then
  cmd+=(--soundscape-top-k "${SOUNDSCAPE_TOP_K}")
fi

if [[ -n "${SOUNDSCAPES_DIR}" ]]; then
  cmd+=(--soundscapes-dir "${SOUNDSCAPES_DIR}")
fi

if [[ "${DEBUG}" == "1" ]]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

for model_root in "${MODEL_ROOTS[@]}"; do
  cmd+=(--model-root "${model_root}")
done

printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
