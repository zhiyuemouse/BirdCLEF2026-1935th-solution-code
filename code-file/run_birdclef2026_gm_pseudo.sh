#!/usr/bin/env bash
set -euo pipefail

# Conservative pseudo-label template:
# - use multiple strong teachers offline
# - keep only high-confidence pseudo rows
# - keep only top 1-2 labels instead of many weak labels

CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="outputs/pseudo_labels"
OUTPUT_NAME="conservative_fold_specific_multi_teacher_convnextv2"

TEACHER_RUN_DIRS=(
  "outputs/birdclef2026_gm_stage3_pseudo/20260429_033312_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo"
  "outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k"
)

# Optional diversity teacher:
# TEACHER_RUN_DIRS+=(
#   "outputs/birdclef2026_gm/20260427_160037_convnext_atto.d2_in1k"
# )

SOUNDSCAPES_DIR="input/train_soundscapes"
LABELS_CSV="input/train_soundscapes_labels.csv"

SEGMENT_BATCH_SIZE=12
TTA_OFFSETS="0"
SMOOTHING_KERNEL=""
SOUNDSCAPE_TOP_K=0

# Conservative thresholds.
PROB_THRESHOLD=0.35
ROW_MIN_MAX_PROB=0.85
TOP_K_LABELS=2

PSEUDO_SCOPE="fold-specific"
TEACHER_FOLDS=""

cmd=(
  python birdclef2026_gm_make_pseudo_labels.py
  --output-dir "${OUTPUT_DIR}"
  --output-name "${OUTPUT_NAME}"
  --soundscapes-dir "${SOUNDSCAPES_DIR}"
  --labels-csv "${LABELS_CSV}"
  --segment-batch-size "${SEGMENT_BATCH_SIZE}"
  --tta-offsets "${TTA_OFFSETS}"
  --prob-threshold "${PROB_THRESHOLD}"
  --row-min-max-prob "${ROW_MIN_MAX_PROB}"
  --top-k-labels "${TOP_K_LABELS}"
  --pseudo-scope "${PSEUDO_SCOPE}"
)

for teacher_run_dir in "${TEACHER_RUN_DIRS[@]}"; do
  cmd+=(--model-root "${teacher_run_dir}")
done

if [[ -n "${SMOOTHING_KERNEL}" ]]; then
  cmd+=(--smoothing-kernel "${SMOOTHING_KERNEL}")
fi

if [[ "${SOUNDSCAPE_TOP_K}" -gt 0 ]]; then
  cmd+=(--soundscape-top-k "${SOUNDSCAPE_TOP_K}")
fi

if [[ -n "${TEACHER_FOLDS}" ]]; then
  cmd+=(--teacher-folds "${TEACHER_FOLDS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
