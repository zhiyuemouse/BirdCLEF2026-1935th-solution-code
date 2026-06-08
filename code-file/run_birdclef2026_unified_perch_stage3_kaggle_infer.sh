#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COMPETITION_ROOT="${COMPETITION_ROOT:-/kaggle/input/competitions/birdclef-2026}"
TEST_SOUNDSCAPES_DIR="${TEST_SOUNDSCAPES_DIR:-${SOUNDSCAPES_DIR:-}}"
PERCH_DIR="${PERCH_DIR:-/kaggle/input/birdclef2026-perch/Perch}"
PERCH_ONNX_PATH="${PERCH_ONNX_PATH:-}"
PERCH_LR_MODEL_PATH="${PERCH_LR_MODEL_PATH:-/kaggle/input/birdclef2026-perch-lr-cnn195634folds/perch_context_logreg_artifacts.joblib}"
MAMBA_MODEL_PATH="${MAMBA_MODEL_PATH:-/kaggle/input/birdclef2026-perch-mamba-conservative093-w025/perch_spatial_mamba_artifacts.joblib}"
ATTENTION_MODEL_PATH="${ATTENTION_MODEL_PATH:-/kaggle/input/birdclef2026-perch-attention-flat64/perch_spatial_mamba_artifacts.joblib}"
TEMPORAL_MODEL_PATH="${TEMPORAL_MODEL_PATH:-}"
SSM_MODEL_PATH="${SSM_MODEL_PATH:-}"
STAGE3_MODEL_ROOT="${STAGE3_MODEL_ROOT:-/kaggle/input/birdclef2026-stage3-perchcnn-white}"
BASE_CNN_MODEL_ROOT="${BASE_CNN_MODEL_ROOT:-}"
RAW_WAVE_MODEL_ROOT="${RAW_WAVE_MODEL_ROOT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-/kaggle/working/submission.csv}"
BATCH_FILES="${BATCH_FILES:-16}"
RUNTIME_NUM_THREADS="${RUNTIME_NUM_THREADS:-4}"
STAGE3_BACKEND="${STAGE3_BACKEND:-torch}"
STAGE3_SEGMENT_BATCH_SIZE="${STAGE3_SEGMENT_BATCH_SIZE:-12}"
BASE_CNN_SEGMENT_BATCH_SIZE="${BASE_CNN_SEGMENT_BATCH_SIZE:-12}"
RAW_WAVE_BACKEND="${RAW_WAVE_BACKEND:-torch}"
RAW_WAVE_SEGMENT_BATCH_SIZE="${RAW_WAVE_SEGMENT_BATCH_SIZE:-12}"
FILE_SCALE_TOPK="${FILE_SCALE_TOPK:-2}"
FILE_SCALE_MODE="${FILE_SCALE_MODE:-topk_mean}"
FILE_SCALE_VALUE="${FILE_SCALE_VALUE:-${FILE_SCALE_TOPK}}"
SMOOTH_MODE="${SMOOTH_MODE:-none}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.10}"
MAMBA_TTA_OFFSETS="${MAMBA_TTA_OFFSETS:-}"
STAGE3_TTA_OFFSETS="${STAGE3_TTA_OFFSETS:-}"
BASE_CNN_TTA_OFFSETS="${BASE_CNN_TTA_OFFSETS:-}"
BLEND_MODE="${BLEND_MODE:-${FAMILY_BLEND_MODE:-logit}}"
RANK_BLEND_ALPHA_LOGIT="${RANK_BLEND_ALPHA_LOGIT:-0.70}"
if [[ -n "${SSM_MODEL_PATH}" ]]; then
  PERCH_LR_WEIGHT="${PERCH_LR_WEIGHT:-0.22625}"
  MAMBA_WEIGHT="${MAMBA_WEIGHT:-0.1465}"
  STAGE3_WEIGHT="${STAGE3_WEIGHT:-0.13575}"
  ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-0.1465}"
  RAW_WAVE_WEIGHT="${RAW_WAVE_WEIGHT:-0.095}"
  TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.0}"
  SSM_WEIGHT="${SSM_WEIGHT:-0.25}"
  BASE_CNN_WEIGHT="${BASE_CNN_WEIGHT:-0.0}"
elif [[ -n "${TEMPORAL_MODEL_PATH}" ]]; then
  PERCH_LR_WEIGHT="${PERCH_LR_WEIGHT:-0.2275}"
  MAMBA_WEIGHT="${MAMBA_WEIGHT:-0.273}"
  STAGE3_WEIGHT="${STAGE3_WEIGHT:-0.1365}"
  ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-0.193}"
  RAW_WAVE_WEIGHT="${RAW_WAVE_WEIGHT:-0.09}"
  TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.08}"
  SSM_WEIGHT="${SSM_WEIGHT:-0.0}"
  BASE_CNN_WEIGHT="${BASE_CNN_WEIGHT:-0.0}"
else
  PERCH_LR_WEIGHT="${PERCH_LR_WEIGHT:-0.2275}"
  MAMBA_WEIGHT="${MAMBA_WEIGHT:-0.273}"
  STAGE3_WEIGHT="${STAGE3_WEIGHT:-0.1365}"
  ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-0.273}"
  RAW_WAVE_WEIGHT="${RAW_WAVE_WEIGHT:-0.09}"
  TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.0}"
  SSM_WEIGHT="${SSM_WEIGHT:-0.0}"
  BASE_CNN_WEIGHT="${BASE_CNN_WEIGHT:-0.0}"
fi
SEED="${SEED:-2026}"
DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-4}"
SAVE_BRANCH_SUBMISSIONS="${SAVE_BRANCH_SUBMISSIONS:-0}"

ARGS=(
  --competition-root "${COMPETITION_ROOT}"
  --perch-dir "${PERCH_DIR}"
  --perch-lr-model-path "${PERCH_LR_MODEL_PATH}"
  --mamba-model-path "${MAMBA_MODEL_PATH}"
  --attention-model-path "${ATTENTION_MODEL_PATH}"
  --stage3-model-root "${STAGE3_MODEL_ROOT}"
  --base-cnn-model-root "${BASE_CNN_MODEL_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --batch-files "${BATCH_FILES}"
  --runtime-num-threads "${RUNTIME_NUM_THREADS}"
  --stage3-backend "${STAGE3_BACKEND}"
  --stage3-segment-batch-size "${STAGE3_SEGMENT_BATCH_SIZE}"
  --base-cnn-segment-batch-size "${BASE_CNN_SEGMENT_BATCH_SIZE}"
  --raw-wave-backend "${RAW_WAVE_BACKEND}"
  --raw-wave-segment-batch-size "${RAW_WAVE_SEGMENT_BATCH_SIZE}"
  --perch-lr-weight "${PERCH_LR_WEIGHT}"
  --mamba-weight "${MAMBA_WEIGHT}"
  --mamba-tta-offsets="${MAMBA_TTA_OFFSETS}"
  --stage3-weight "${STAGE3_WEIGHT}"
  --stage3-tta-offsets="${STAGE3_TTA_OFFSETS}"
  --attention-weight "${ATTENTION_WEIGHT}"
  --base-cnn-weight "${BASE_CNN_WEIGHT}"
  --base-cnn-tta-offsets="${BASE_CNN_TTA_OFFSETS}"
  --raw-wave-weight "${RAW_WAVE_WEIGHT}"
  --temporal-weight "${TEMPORAL_WEIGHT}"
  --ssm-weight "${SSM_WEIGHT}"
  --file-scale-mode "${FILE_SCALE_MODE}"
  --file-scale-value "${FILE_SCALE_VALUE}"
  --file-scale-topk "${FILE_SCALE_TOPK}"
  --smooth-mode "${SMOOTH_MODE}"
  --smooth-alpha "${SMOOTH_ALPHA}"
  --blend-mode "${BLEND_MODE}"
  --rank-blend-alpha-logit "${RANK_BLEND_ALPHA_LOGIT}"
  --seed "${SEED}"
)

if [[ -n "${TEST_SOUNDSCAPES_DIR}" ]]; then
  ARGS+=(--soundscapes-dir "${TEST_SOUNDSCAPES_DIR}")
fi
if [[ -n "${PERCH_ONNX_PATH}" ]]; then
  ARGS+=(--perch-onnx-path "${PERCH_ONNX_PATH}")
fi
if [[ -n "${RAW_WAVE_MODEL_ROOT}" ]]; then
  ARGS+=(--raw-wave-model-root "${RAW_WAVE_MODEL_ROOT}")
fi
if [[ -n "${TEMPORAL_MODEL_PATH}" ]]; then
  ARGS+=(--temporal-model-path "${TEMPORAL_MODEL_PATH}")
fi
if [[ -n "${SSM_MODEL_PATH}" ]]; then
  ARGS+=(--ssm-model-path "${SSM_MODEL_PATH}")
fi
if [[ "${DEBUG}" == "1" ]]; then
  ARGS+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi
if [[ "${SAVE_BRANCH_SUBMISSIONS}" == "1" ]]; then
  ARGS+=(--save-branch-submissions)
fi

"${PYTHON_BIN}" birdclef2026_kaggle_infer_unified_perch_stage3.py "${ARGS[@]}"
