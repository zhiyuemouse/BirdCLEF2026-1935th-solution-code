#!/usr/bin/env bash
set -euo pipefail

# Generate fold-specific pseudo labels from the leak-safe 0.916-style teacher:
# Perch aligned fold_k + Stage3 CNN fold_k + OOF-selected post-processing.
# Then train a conservative Stage3 student from the original Stage2 run.

TRANSFORMERS_PY="${TRANSFORMERS_PY:-/home/hjs/anaconda3/envs/transformers/bin/python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-1}"

PSEUDO_OUTPUT_DIR="${PSEUDO_OUTPUT_DIR:-outputs/pseudo_labels}"
PSEUDO_OUTPUT_NAME="${PSEUDO_OUTPUT_NAME:-perch_stage3_teacher_conservative_v1}"
PERCH_CACHE_DIR="${PERCH_CACHE_DIR:-outputs/pseudo_labels/perch_cnn_blend_white_v1_perch_cache}"
TEACHER_FOLDS="${TEACHER_FOLDS:-0,1,2}"

CNN_MODEL_ROOT="${CNN_MODEL_ROOT:-outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo}"
PERCH_MODEL_PATH="${PERCH_MODEL_PATH:-outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib}"

PERCH_WEIGHT="${PERCH_WEIGHT:-0.74}"
FILE_SCALE_MODE="${FILE_SCALE_MODE:-max_power}"
FILE_SCALE_VALUE="${FILE_SCALE_VALUE:-0.4}"
SMOOTH_MODE="${SMOOTH_MODE:-plain}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.15}"

PROB_THRESHOLD="${PROB_THRESHOLD:-0.50}"
ROW_MIN_MAX_PROB="${ROW_MIN_MAX_PROB:-0.93}"
TOP_K_LABELS="${TOP_K_LABELS:-1}"
MIN_TOP1_TOP2_MARGIN="${MIN_TOP1_TOP2_MARGIN:-0.25}"
MAX_TOPK_ENTROPY="${MAX_TOPK_ENTROPY:--1.0}"
ENTROPY_TOP_K="${ENTROPY_TOP_K:-5}"

STUDENT_RUN_DIR="${STUDENT_RUN_DIR:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k}"
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-outputs/birdclef2026_gm_stage3_perchstage3_teacher_conservative_v1}"
STAGE3_FOLDS="${STAGE3_FOLDS:-0,1,2}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE3_SAMPLES_PER_EPOCH="${STAGE3_SAMPLES_PER_EPOCH:-4096}"
STAGE3_BACKBONE_LR="${STAGE3_BACKBONE_LR:-2e-5}"
STAGE3_HEAD_LR="${STAGE3_HEAD_LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PATIENCE="${PATIENCE:-3}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-0.20}"
PSEUDO_SAMPLER_WEIGHT="${PSEUDO_SAMPLER_WEIGHT:-0.20}"
MIN_PSEUDO_MAX_PROB="${MIN_PSEUDO_MAX_PROB:-0.93}"
MAX_PSEUDO_ROWS="${MAX_PSEUDO_ROWS:--1}"

DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-16}"
STAGE3_SMOKE_TEST="${STAGE3_SMOKE_TEST:-0}"

echo "[RUN] Generate conservative fold-specific pseudo from Perch+Stage3 teacher"
MODE=pseudo-from-cache \
PYTHON_CMD="${TRANSFORMERS_PY}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
OUTPUT_DIR="${PSEUDO_OUTPUT_DIR}" \
OUTPUT_NAME="${PSEUDO_OUTPUT_NAME}" \
PERCH_CACHE_DIR="${PERCH_CACHE_DIR}" \
CNN_MODEL_ROOT="${CNN_MODEL_ROOT}" \
PERCH_MODEL_PATH="${PERCH_MODEL_PATH}" \
TEACHER_FOLDS="${TEACHER_FOLDS}" \
PERCH_WEIGHT="${PERCH_WEIGHT}" \
FILE_SCALE_MODE="${FILE_SCALE_MODE}" \
FILE_SCALE_VALUE="${FILE_SCALE_VALUE}" \
SMOOTH_MODE="${SMOOTH_MODE}" \
SMOOTH_ALPHA="${SMOOTH_ALPHA}" \
PROB_THRESHOLD="${PROB_THRESHOLD}" \
ROW_MIN_MAX_PROB="${ROW_MIN_MAX_PROB}" \
TOP_K_LABELS="${TOP_K_LABELS}" \
MIN_TOP1_TOP2_MARGIN="${MIN_TOP1_TOP2_MARGIN}" \
MAX_TOPK_ENTROPY="${MAX_TOPK_ENTROPY}" \
ENTROPY_TOP_K="${ENTROPY_TOP_K}" \
DEBUG="${DEBUG}" \
DEBUG_LIMIT="${DEBUG_LIMIT}" \
./run_birdclef2026_make_pseudo_perch_cnn_blend.sh

PSEUDO_ROOT="$(find "${PSEUDO_OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${PSEUDO_OUTPUT_NAME}" | sort | tail -n 1)"
if [[ -z "${PSEUDO_ROOT}" ]]; then
  echo "[ERROR] Could not find generated pseudo root matching *_${PSEUDO_OUTPUT_NAME}"
  exit 1
fi
echo "[INFO] Using pseudo root: ${PSEUDO_ROOT}"

STAGE3_ARGS=(
  birdclef2026_gm_train_stage3_pseudo.py
  --root .
  --input-dir input
  --output-dir "${STAGE3_OUTPUT_DIR}"
  --student-run-dir "${STUDENT_RUN_DIR}"
  --pseudo-root "${PSEUDO_ROOT}"
  --num-workers "${NUM_WORKERS}"
  --stage3-epochs "${STAGE3_EPOCHS}"
  --stage3-batch-size "${STAGE3_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --stage3-samples-per-epoch "${STAGE3_SAMPLES_PER_EPOCH}"
  --stage3-backbone-lr "${STAGE3_BACKBONE_LR}"
  --stage3-head-lr "${STAGE3_HEAD_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --patience "${PATIENCE}"
  --freeze-backbone-epochs "${FREEZE_BACKBONE_EPOCHS}"
  --pseudo-loss-weight "${PSEUDO_LOSS_WEIGHT}"
  --pseudo-sampler-weight "${PSEUDO_SAMPLER_WEIGHT}"
  --min-pseudo-max-prob "${MIN_PSEUDO_MAX_PROB}"
  --max-pseudo-rows "${MAX_PSEUDO_ROWS}"
  --folds "${STAGE3_FOLDS}"
)

if [[ "${STAGE3_SMOKE_TEST}" == "1" ]]; then
  STAGE3_ARGS+=(--smoke-test)
fi

echo "[RUN] Conservative Stage3 with fold-specific Perch+Stage3 pseudo"
printf '[RUN]'
printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${TRANSFORMERS_PY}" "${STAGE3_ARGS[@]}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${TRANSFORMERS_PY}" "${STAGE3_ARGS[@]}"
