#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/hjs/anaconda3/envs/transformers/bin/python}"
PSEUDO_MAX_FILES="${PSEUDO_MAX_FILES:-1024}"
PSEUDO_OUTPUT_NAME="${PSEUDO_OUTPUT_NAME:-unified_teacher_0921_softpseudo_1024files_r070}"
PSEUDO_ROW_MIN_MAX_PROB="${PSEUDO_ROW_MIN_MAX_PROB:-0.70}"
PSEUDO_MIN_MARGIN="${PSEUDO_MIN_MARGIN:-0.0}"
PSEUDO_MAX_ENTROPY="${PSEUDO_MAX_ENTROPY:--1.0}"
PSEUDO_ENTROPY_TOP_K="${PSEUDO_ENTROPY_TOP_K:-5}"
PSEUDO_ZERO_OUT_NONTOPK="${PSEUDO_ZERO_OUT_NONTOPK:-0}"
PSEUDO_BATCH_FILES="${PSEUDO_BATCH_FILES:-16}"
PSEUDO_RUNTIME_NUM_THREADS="${PSEUDO_RUNTIME_NUM_THREADS:-4}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/birdclef2026_raw_waveform_transformer_unified_teacher_pseudo}"
MODEL_NAME="${MODEL_NAME:-raw_wave_unified_teacher_softpseudo_1024_r070_w100_pf050}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-outputs/birdclef2026_raw_waveform_transformer/20260512_013731_raw_wave_conv_tokenizer_base_long_n32_d768/stage1_audio/stage1_audio_best.pth}"
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-1.0}"
PSEUDO_SAMPLER_FRACTION="${PSEUDO_SAMPLER_FRACTION:-0.5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"
STAGE2_SAMPLES_PER_EPOCH="${STAGE2_SAMPLES_PER_EPOCH:-2048}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-2026}"

export CUDA_VISIBLE_DEVICES
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED=1

mkdir -p outputs

"${PYTHON_BIN}" birdclef2026_make_pseudo_unified_teacher.py \
  --root "${ROOT}" \
  --max-files "${PSEUDO_MAX_FILES}" \
  --output-name "${PSEUDO_OUTPUT_NAME}" \
  --row-min-max-prob "${PSEUDO_ROW_MIN_MAX_PROB}" \
  --min-top1-top2-margin "${PSEUDO_MIN_MARGIN}" \
  --max-topk-entropy "${PSEUDO_MAX_ENTROPY}" \
  --entropy-top-k "${PSEUDO_ENTROPY_TOP_K}" \
  --zero-out-nontopk "${PSEUDO_ZERO_OUT_NONTOPK}" \
  --batch-files "${PSEUDO_BATCH_FILES}" \
  --runtime-num-threads "${PSEUDO_RUNTIME_NUM_THREADS}" \
  --seed "${SEED}"

PSEUDO_DIR="$(ls -dt outputs/pseudo_labels/*_${PSEUDO_OUTPUT_NAME} | head -1)"
echo "[INFO] Using pseudo dir: ${PSEUDO_DIR}"

"${PYTHON_BIN}" birdclef2026_raw_waveform_transformer_train.py \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --stage1-checkpoint-path "${STAGE1_CHECKPOINT_PATH}" \
  --stage1-epochs 0 \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  --stage2-samples-per-epoch "${STAGE2_SAMPLES_PER_EPOCH}" \
  --stage2-batch-size "${STAGE2_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --patience "${PATIENCE}" \
  --pseudo-dir "${PSEUDO_DIR}" \
  --pseudo-loss-weight "${PSEUDO_LOSS_WEIGHT}" \
  --pseudo-sampler-fraction "${PSEUDO_SAMPLER_FRACTION}" \
  --raw-strong-aug \
  --seed "${SEED}"
