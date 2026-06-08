#!/usr/bin/env bash
set -euo pipefail

# Fold-safe Noisy Student Stage3:
# each sampled real labeled soundscape window is waveform-mixed with one
# fold-specific pseudo-labeled train_soundscape window at a fixed lambda.

TRANSFORMERS_PY="${TRANSFORMERS_PY:-/home/hjs/anaconda3/envs/transformers/bin/python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-1}"

STUDENT_RUN_DIR="${STUDENT_RUN_DIR:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k}"
PSEUDO_ROOT="${PSEUDO_ROOT:-outputs/pseudo_labels/20260507_165105_perch_cnn_blend_white_v1}"
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-outputs/birdclef2026_gm_stage3_noisy_student_v1}"
STAGE3_FOLDS="${STAGE3_FOLDS:-0,1,2}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-6}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE3_SAMPLES_PER_EPOCH="${STAGE3_SAMPLES_PER_EPOCH:-4096}"
STAGE3_BACKBONE_LR="${STAGE3_BACKBONE_LR:-2e-5}"
STAGE3_HEAD_LR="${STAGE3_HEAD_LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"

NOISY_STUDENT_LAMBDA="${NOISY_STUDENT_LAMBDA:-0.5}"
NOISY_STUDENT_PSEUDO_SAMPLE_POWER="${NOISY_STUDENT_PSEUDO_SAMPLE_POWER:-1.0}"
MIN_PSEUDO_MAX_PROB="${MIN_PSEUDO_MAX_PROB:-0.85}"
MAX_PSEUDO_ROWS="${MAX_PSEUDO_ROWS:--1}"
STAGE3_SMOKE_TEST="${STAGE3_SMOKE_TEST:-0}"

ARGS=(
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
  --training-mode noisy-student
  --noisy-student-lambda "${NOISY_STUDENT_LAMBDA}"
  --noisy-student-pseudo-sample-power "${NOISY_STUDENT_PSEUDO_SAMPLE_POWER}"
  --min-pseudo-max-prob "${MIN_PSEUDO_MAX_PROB}"
  --max-pseudo-rows "${MAX_PSEUDO_ROWS}"
  --folds "${STAGE3_FOLDS}"
)

if [[ "${STAGE3_SMOKE_TEST}" == "1" ]]; then
  ARGS+=(--smoke-test)
fi

echo "[RUN] Fold-safe Noisy Student Stage3"
printf '[RUN]'
printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" TQDM_DISABLE=1 "${TRANSFORMERS_PY}" "${ARGS[@]}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" TQDM_DISABLE=1 "${TRANSFORMERS_PY}" "${ARGS[@]}"
