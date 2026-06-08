#!/usr/bin/env bash
set -euo pipefail

# Fold-specific pseudo labels from the leak-safe Perch+CNN white-list blend.
#
# Recommended usage:
#   MODE=perch-cache ./run_birdclef2026_make_pseudo_perch_cnn_blend.sh
#   MODE=pseudo-from-cache ./run_birdclef2026_make_pseudo_perch_cnn_blend.sh
#
# Use the `perch` env for MODE=perch-cache and the `transformers` env for
# MODE=pseudo-from-cache. The split avoids requiring TensorFlow and PyTorch in
# the same environment.

MODE="${MODE:-perch-cache}"  # perch-cache | pseudo-from-cache
PYTHON_CMD="${PYTHON_CMD:-python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"

ROOT="${ROOT:-.}"
INPUT_DIR="${INPUT_DIR:-input}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pseudo_labels}"
OUTPUT_NAME="${OUTPUT_NAME:-perch_cnn_blend_white_v1}"
PERCH_CACHE_DIR="${PERCH_CACHE_DIR:-outputs/pseudo_labels/perch_cnn_blend_white_v1_perch_cache}"

SOUNDSCAPES_DIR="${SOUNDSCAPES_DIR:-input/train_soundscapes}"
LABELS_CSV="${LABELS_CSV:-input/train_soundscapes_labels.csv}"
CNN_MODEL_ROOT="${CNN_MODEL_ROOT:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k}"
PERCH_MODEL_PATH="${PERCH_MODEL_PATH:-outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib}"
PERCH_DIR="${PERCH_DIR:-Perch}"

TEACHER_FOLDS="${TEACHER_FOLDS:-0,1,2}"
PSEUDO_SCOPE="${PSEUDO_SCOPE:-fold-specific}"
PERCH_BACKEND="${PERCH_BACKEND:-onnx}"
PERCH_ONNX_PATH="${PERCH_ONNX_PATH:-PerchV2Onnx/perch_v2.onnx}"
PERCH_TFLITE_PATH="${PERCH_TFLITE_PATH:-}"
RUNTIME_NUM_THREADS="${RUNTIME_NUM_THREADS:-4}"
BATCH_FILES="${BATCH_FILES:-16}"
SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-12}"

PERCH_WEIGHT="${PERCH_WEIGHT:-0.83}"
FILE_SCALE_MODE="${FILE_SCALE_MODE:-topk_mean}"
FILE_SCALE_VALUE="${FILE_SCALE_VALUE:-2.0}"
SMOOTH_MODE="${SMOOTH_MODE:-adaptive}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.10}"

PROB_THRESHOLD="${PROB_THRESHOLD:-0.35}"
ROW_MIN_MAX_PROB="${ROW_MIN_MAX_PROB:-0.85}"
TOP_K_LABELS="${TOP_K_LABELS:-2}"
MIN_TOP1_TOP2_MARGIN="${MIN_TOP1_TOP2_MARGIN:-0.0}"
MAX_TOPK_ENTROPY="${MAX_TOPK_ENTROPY:--1.0}"
ENTROPY_TOP_K="${ENTROPY_TOP_K:-5}"

DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-16}"
INCLUDE_LABELED="${INCLUDE_LABELED:-0}"
SEED="${SEED:-2026}"

ARGS=(
  --mode "${MODE}"
  --root "${ROOT}"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --output-name "${OUTPUT_NAME}"
  --perch-cache-dir "${PERCH_CACHE_DIR}"
  --soundscapes-dir "${SOUNDSCAPES_DIR}"
  --labels-csv "${LABELS_CSV}"
  --cnn-model-root "${CNN_MODEL_ROOT}"
  --perch-model-path "${PERCH_MODEL_PATH}"
  --perch-dir "${PERCH_DIR}"
  --perch-backend "${PERCH_BACKEND}"
  --runtime-num-threads "${RUNTIME_NUM_THREADS}"
  --batch-files "${BATCH_FILES}"
  --segment-batch-size "${SEGMENT_BATCH_SIZE}"
  --pseudo-scope "${PSEUDO_SCOPE}"
  --teacher-folds "${TEACHER_FOLDS}"
  --perch-weight "${PERCH_WEIGHT}"
  --file-scale-mode "${FILE_SCALE_MODE}"
  --file-scale-value "${FILE_SCALE_VALUE}"
  --smooth-mode "${SMOOTH_MODE}"
  --smooth-alpha "${SMOOTH_ALPHA}"
  --prob-threshold "${PROB_THRESHOLD}"
  --row-min-max-prob "${ROW_MIN_MAX_PROB}"
  --top-k-labels "${TOP_K_LABELS}"
  --min-top1-top2-margin "${MIN_TOP1_TOP2_MARGIN}"
  --max-topk-entropy "${MAX_TOPK_ENTROPY}"
  --entropy-top-k "${ENTROPY_TOP_K}"
  --seed "${SEED}"
)

if [[ -n "${PERCH_ONNX_PATH}" ]]; then
  ARGS+=(--perch-onnx-path "${PERCH_ONNX_PATH}")
fi
if [[ -n "${PERCH_TFLITE_PATH}" ]]; then
  ARGS+=(--perch-tflite-path "${PERCH_TFLITE_PATH}")
fi
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi
if [[ "${INCLUDE_LABELED}" == "1" ]]; then
  ARGS+=(--include-labeled)
fi

echo "[RUN] MODE=${MODE}"
echo "[RUN] PYTHON_CMD=${PYTHON_CMD}"
echo "[RUN] PERCH_CACHE_DIR=${PERCH_CACHE_DIR}"
printf '[RUN]'
printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${PYTHON_CMD}" birdclef2026_make_pseudo_perch_cnn_blend.py "${ARGS[@]}"
echo

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${PYTHON_CMD}" birdclef2026_make_pseudo_perch_cnn_blend.py "${ARGS[@]}"
