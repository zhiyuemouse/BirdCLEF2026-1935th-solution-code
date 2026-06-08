#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
SEED="${SEED:-2026}"

if [[ "${EXPORT_PYTHONHASHSEED:-0}" == "1" ]]; then
  export PYTHONHASHSEED="${SEED}"
fi
if [[ -n "${CUBLAS_WORKSPACE_CONFIG:-}" ]]; then
  export CUBLAS_WORKSPACE_CONFIG
fi

MODEL_NAME="${MODEL_NAME:-raw_wave_conv_tokenizer_base_long_n32_d768}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_raw_waveform_transformer}"
FOLD_ASSIGNMENT_PATH="${FOLD_ASSIGNMENT_PATH:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-}"
TEACHER_OOF_PATH="${TEACHER_OOF_PATH:-}"
TEACHER_LOSS_WEIGHT="${TEACHER_LOSS_WEIGHT:-0.0}"

NUM_TOKENS="${NUM_TOKENS:-32}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-conv_stack}"
WAVEFORM_MODEL_VARIANT="${WAVEFORM_MODEL_VARIANT:-base}"
D_MODEL="${D_MODEL:-768}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-8}"
TRANSFORMER_FF_MULT="${TRANSFORMER_FF_MULT:-4}"
DROPOUT="${DROPOUT:-0.20}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-25}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
STAGE1_SAMPLES_PER_EPOCH="${STAGE1_SAMPLES_PER_EPOCH:-24000}"
STAGE2_SAMPLES_PER_EPOCH="${STAGE2_SAMPLES_PER_EPOCH:-2048}"
STAGE1_LR="${STAGE1_LR:-3e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PATIENCE="${PATIENCE:-10}"

STAGE1_MIXUP_ALPHA="${STAGE1_MIXUP_ALPHA:-0.20}"
STAGE1_MIXUP_PROB="${STAGE1_MIXUP_PROB:-0.10}"
STAGE2_MIXUP_ALPHA="${STAGE2_MIXUP_ALPHA:-0.0}"
STAGE2_MIXUP_PROB="${STAGE2_MIXUP_PROB:-0.0}"

RAW_STRONG_AUG="${RAW_STRONG_AUG:-0}"
RAW_GAIN_MIN="${RAW_GAIN_MIN:-0.65}"
RAW_GAIN_MAX="${RAW_GAIN_MAX:-1.50}"
RAW_POLARITY_PROB="${RAW_POLARITY_PROB:-0.20}"
RAW_TIME_SHIFT_PROB="${RAW_TIME_SHIFT_PROB:-0.50}"
RAW_TIME_SHIFT_MAX_SEC="${RAW_TIME_SHIFT_MAX_SEC:-0.35}"
RAW_NOISE_PROB="${RAW_NOISE_PROB:-0.50}"
RAW_NOISE_MIN="${RAW_NOISE_MIN:-0.001}"
RAW_NOISE_MAX="${RAW_NOISE_MAX:-0.020}"
RAW_FILTER_PROB="${RAW_FILTER_PROB:-0.35}"

SMOKE="${SMOKE:-0}"
USE_TEE="${USE_TEE:-0}"
LOG_PATH="${LOG_PATH:-outputs/raw_waveform_transformer_$(date +%Y%m%d_%H%M%S).log}"

cmd=(
  /home/hjs/anaconda3/envs/transformers/bin/python birdclef2026_raw_waveform_transformer_train.py
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --fold-assignment-path "${FOLD_ASSIGNMENT_PATH}"
  --teacher-loss-weight "${TEACHER_LOSS_WEIGHT}"
  --seed "${SEED}"
  --num-tokens "${NUM_TOKENS}"
  --tokenizer-type "${TOKENIZER_TYPE}"
  --waveform-model-variant "${WAVEFORM_MODEL_VARIANT}"
  --d-model "${D_MODEL}"
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-heads "${TRANSFORMER_HEADS}"
  --transformer-ff-mult "${TRANSFORMER_FF_MULT}"
  --dropout "${DROPOUT}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-batch-size "${STAGE1_BATCH_SIZE}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --stage1-samples-per-epoch "${STAGE1_SAMPLES_PER_EPOCH}"
  --stage2-samples-per-epoch "${STAGE2_SAMPLES_PER_EPOCH}"
  --stage1-lr "${STAGE1_LR}"
  --stage2-lr "${STAGE2_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --patience "${PATIENCE}"
  --stage1-mixup-alpha "${STAGE1_MIXUP_ALPHA}"
  --stage1-mixup-prob "${STAGE1_MIXUP_PROB}"
  --stage2-mixup-alpha "${STAGE2_MIXUP_ALPHA}"
  --stage2-mixup-prob "${STAGE2_MIXUP_PROB}"
  --raw-gain-min "${RAW_GAIN_MIN}"
  --raw-gain-max "${RAW_GAIN_MAX}"
  --raw-polarity-prob "${RAW_POLARITY_PROB}"
  --raw-time-shift-prob "${RAW_TIME_SHIFT_PROB}"
  --raw-time-shift-max-sec "${RAW_TIME_SHIFT_MAX_SEC}"
  --raw-noise-prob "${RAW_NOISE_PROB}"
  --raw-noise-min "${RAW_NOISE_MIN}"
  --raw-noise-max "${RAW_NOISE_MAX}"
  --raw-filter-prob "${RAW_FILTER_PROB}"
)

if [[ -n "${STAGE1_CHECKPOINT_PATH}" ]]; then
  cmd+=(--stage1-checkpoint-path "${STAGE1_CHECKPOINT_PATH}")
fi

if [[ -n "${TEACHER_OOF_PATH}" ]]; then
  cmd+=(--teacher-oof-path "${TEACHER_OOF_PATH}")
fi

if [[ "${RAW_STRONG_AUG}" == "1" ]]; then
  cmd+=(--raw-strong-aug)
fi

if [[ "${SMOKE}" == "1" ]]; then
  cmd+=(--smoke-test)
fi

mkdir -p "$(dirname "${LOG_PATH}")"
if [[ "${USE_TEE}" == "1" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"
  else
    "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"
  fi
else
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
fi
