#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEED="${SEED:-2026}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

STUDENT_RUN_DIR="${STUDENT_RUN_DIR:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k}"
TEACHER_OOF_PATH="${TEACHER_OOF_PATH:-outputs/whitelist_blend_unified_raw_waveform_20260512/best_oof_predictions.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_gm_stage3_oof_teacher}"
TEACHER_LOSS_WEIGHT="${TEACHER_LOSS_WEIGHT:-0.25}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-6}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
STAGE3_SAMPLES_PER_EPOCH="${STAGE3_SAMPLES_PER_EPOCH:-4096}"
STAGE3_BACKBONE_LR="${STAGE3_BACKBONE_LR:-2e-5}"
STAGE3_HEAD_LR="${STAGE3_HEAD_LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
FOLDS="${FOLDS:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SMOKE="${SMOKE:-0}"
LOG_PATH="${LOG_PATH:-outputs/gm_stage3_oof_teacher_$(date +%Y%m%d_%H%M%S).log}"

cmd=(
  /home/hjs/anaconda3/envs/transformers/bin/python birdclef2026_gm_train_stage3_oof_teacher.py
  --student-run-dir "${STUDENT_RUN_DIR}"
  --teacher-oof-path "${TEACHER_OOF_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --teacher-loss-weight "${TEACHER_LOSS_WEIGHT}"
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
)

if [[ -n "${FOLDS}" ]]; then
  cmd+=(--folds "${FOLDS}")
fi

if [[ "${SMOKE}" == "1" ]]; then
  cmd+=(--smoke-test)
fi

mkdir -p "$(dirname "${LOG_PATH}")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"
