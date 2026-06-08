#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
SEED="${SEED:-1086}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_gm_hgnet_v8}"
MODEL_NAME="${MODEL_NAME:-hgnetv2_b0.ssld_stage2_ft_in1k}"
HEAD_TYPE="${HEAD_TYPE:-lse_head_v1}"
HEAD_POOL_TYPE="${HEAD_POOL_TYPE:-lse}"
N_FOLDS="${N_FOLDS:-4}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-128}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE1_SAMPLES_PER_EPOCH="${STAGE1_SAMPLES_PER_EPOCH:-24000}"
STAGE2_SAMPLES_PER_EPOCH="${STAGE2_SAMPLES_PER_EPOCH:-4096}"

STAGE1_BACKBONE_LR="${STAGE1_BACKBONE_LR:-1e-3}"
STAGE1_HEAD_LR="${STAGE1_HEAD_LR:-1e-3}"
STAGE2_BACKBONE_LR="${STAGE2_BACKBONE_LR:-1e-3}"
STAGE2_HEAD_LR="${STAGE2_HEAD_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-onecycle}"

SPEC_FREQ_MASK="${SPEC_FREQ_MASK:-30}"
SPEC_TIME_MASK="${SPEC_TIME_MASK:-50}"
MIXUP_ALPHA="${MIXUP_ALPHA:-1.0}"
MIXUP_PROB="${MIXUP_PROB:-0.8}"
MIXUP_START_EPOCH="${MIXUP_START_EPOCH:-5}"

DROPOUT="${DROPOUT:-0.5}"
DROP_PATH="${DROP_PATH:-0.0}"
PATIENCE="${PATIENCE:-8}"
STAGE2_FREEZE_BACKBONE_EPOCHS="${STAGE2_FREEZE_BACKBONE_EPOCHS:-0}"
AMP_MODE="${AMP_MODE:-fp16}"

cmd=(
  "${PYTHON_BIN}" birdclef2026_gm_train.py
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --head-type "${HEAD_TYPE}"
  --head-pool-type "${HEAD_POOL_TYPE}"
  --n-folds "${N_FOLDS}"
  --seed "${SEED}"
  --image-height 256
  --image-width 256
  --input-channels 1
  --spectrogram-variant logmel_v8
  --image-normalize zero_one
  --scheduler-type "${SCHEDULER_TYPE}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --stage1-samples-per-epoch "${STAGE1_SAMPLES_PER_EPOCH}"
  --stage2-samples-per-epoch "${STAGE2_SAMPLES_PER_EPOCH}"
  --stage1-backbone-lr "${STAGE1_BACKBONE_LR}"
  --stage1-head-lr "${STAGE1_HEAD_LR}"
  --stage2-backbone-lr "${STAGE2_BACKBONE_LR}"
  --stage2-head-lr "${STAGE2_HEAD_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --dropout "${DROPOUT}"
  --drop-path "${DROP_PATH}"
  --specaug-freq-mask "${SPEC_FREQ_MASK}"
  --specaug-time-mask "${SPEC_TIME_MASK}"
  --mixup-domain image
  --stage1-mixup-alpha "${MIXUP_ALPHA}"
  --stage1-mixup-prob "${MIXUP_PROB}"
  --stage1-mixup-start-epoch "${MIXUP_START_EPOCH}"
  --stage1-cutmix-alpha 0.0
  --stage1-cutmix-prob 0.0
  --stage2-mixup-alpha "${MIXUP_ALPHA}"
  --stage2-mixup-prob "${MIXUP_PROB}"
  --stage2-mixup-start-epoch "${MIXUP_START_EPOCH}"
  --stage2-cutmix-alpha 0.0
  --stage2-cutmix-prob 0.0
  --stage2-freeze-backbone-epochs "${STAGE2_FREEZE_BACKBONE_EPOCHS}"
  --amp-mode "${AMP_MODE}"
  --patience "${PATIENCE}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" "$@"
else
  unset CUDA_VISIBLE_DEVICES
  "${cmd[@]}" "$@"
fi
