#!/usr/bin/env bash
set -euo pipefail

BASE_CACHE_DIR="${BASE_CACHE_DIR:-perch_cache_labeled_all}"
SPATIAL_CACHE_DIR="${SPATIAL_CACHE_DIR:-perch_spatial_cache_labeled_all}"
SPATIAL_TOKEN_KEY="${SPATIAL_TOKEN_KEY:-spatial_tokens}"
LABELS_PATH="${LABELS_PATH:-input/train_soundscapes_labels.csv}"
SAMPLE_SUBMISSION_PATH="${SAMPLE_SUBMISSION_PATH:-input/sample_submission.csv}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/perch_temporal_head_labeled_all_cnn195634folds_v1}"
N_FOLDS="${N_FOLDS:-3}"
LIMIT_FILES="${LIMIT_FILES:--1}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
LOCAL_BLOCKS="${LOCAL_BLOCKS:-0}"
LOCAL_KERNEL_SIZE="${LOCAL_KERNEL_SIZE:-5}"
LOCAL_ON_RAW_TOKENS="${LOCAL_ON_RAW_TOKENS:-0}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.25}"
MLP_MIN_POS="${MLP_MIN_POS:-4}"
EPOCHS="${EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-12.0}"
INNER_VAL_FILES="${INNER_VAL_FILES:-8}"
PATIENCE="${PATIENCE:-35}"
TEACHER_TARGET_PATH="${TEACHER_TARGET_PATH:-}"
TEACHER_LOSS_WEIGHT="${TEACHER_LOSS_WEIGHT:-0.0}"
TEACHER_USE_UNLABELED_WINDOWS="${TEACHER_USE_UNLABELED_WINDOWS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"

ARGS=(
  --base-cache-dir "${BASE_CACHE_DIR}" \
  --spatial-cache-dir "${SPATIAL_CACHE_DIR}" \
  --spatial-token-key "${SPATIAL_TOKEN_KEY}" \
  --labels-path "${LABELS_PATH}" \
  --sample-submission-path "${SAMPLE_SUBMISSION_PATH}" \
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-folds "${N_FOLDS}" \
  --limit-files "${LIMIT_FILES}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --local-blocks "${LOCAL_BLOCKS}" \
  --local-kernel-size "${LOCAL_KERNEL_SIZE}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --dropout "${DROPOUT}" \
  --mlp-min-pos "${MLP_MIN_POS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --pos-weight-power "${POS_WEIGHT_POWER}" \
  --pos-weight-max "${POS_WEIGHT_MAX}" \
  --inner-val-files "${INNER_VAL_FILES}" \
  --patience "${PATIENCE}" \
  --teacher-target-path "${TEACHER_TARGET_PATH}" \
  --teacher-loss-weight "${TEACHER_LOSS_WEIGHT}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
)

if [[ "${LOCAL_ON_RAW_TOKENS}" == "1" ]]; then
  ARGS+=(--local-on-raw-tokens)
fi
if [[ "${TEACHER_USE_UNLABELED_WINDOWS}" == "1" ]]; then
  ARGS+=(--teacher-use-unlabeled-windows)
fi

/home/hjs/anaconda3/envs/transformers/bin/python birdclef2026_perch_temporal_head_train.py "${ARGS[@]}"
