#!/usr/bin/env bash
set -euo pipefail

# Wait for Perch cache, generate fold-specific Perch+CNN pseudo labels, then
# launch Stage3 training. This script intentionally uses fold-specific pseudo
# only; it does not pass --allow-global-pseudo.

TRANSFORMERS_PY="${TRANSFORMERS_PY:-/home/hjs/anaconda3/envs/transformers/bin/python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-1}"

PSEUDO_OUTPUT_DIR="${PSEUDO_OUTPUT_DIR:-outputs/pseudo_labels}"
PSEUDO_OUTPUT_NAME="${PSEUDO_OUTPUT_NAME:-perch_cnn_blend_white_v1}"
PERCH_CACHE_DIR="${PERCH_CACHE_DIR:-outputs/pseudo_labels/perch_cnn_blend_white_v1_perch_cache}"
TEACHER_FOLDS="${TEACHER_FOLDS:-0,1,2}"

STUDENT_RUN_DIR="${STUDENT_RUN_DIR:-outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k}"
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-outputs/birdclef2026_gm_stage3_perchcnn_white_v1}"
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
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-0.35}"
PSEUDO_SAMPLER_WEIGHT="${PSEUDO_SAMPLER_WEIGHT:-0.5}"
MIN_PSEUDO_MAX_PROB="${MIN_PSEUDO_MAX_PROB:-0.85}"
MAX_PSEUDO_ROWS="${MAX_PSEUDO_ROWS:--1}"
STAGE3_SMOKE_TEST="${STAGE3_SMOKE_TEST:-0}"

WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
WAIT_MAX_MINUTES="${WAIT_MAX_MINUTES:-720}"

cache_ready() {
  [[ -s "${PERCH_CACHE_DIR}/perch_meta.csv" ]] &&
  [[ -s "${PERCH_CACHE_DIR}/perch_arrays.npz" ]] &&
  [[ -s "${PERCH_CACHE_DIR}/perch_fold_preds.npz" ]] &&
  [[ -s "${PERCH_CACHE_DIR}/perch_cache_summary.json" ]]
}

echo "[INFO] Waiting for Perch cache: ${PERCH_CACHE_DIR}"
deadline=$((SECONDS + WAIT_MAX_MINUTES * 60))
while ! cache_ready; do
  if (( SECONDS >= deadline )); then
    echo "[ERROR] Timed out waiting for Perch cache after ${WAIT_MAX_MINUTES} minutes."
    exit 1
  fi
  echo "[INFO] Cache not ready yet; sleeping ${WAIT_INTERVAL_SECONDS}s..."
  sleep "${WAIT_INTERVAL_SECONDS}"
done
echo "[INFO] Perch cache is ready."

MODE=pseudo-from-cache \
PYTHON_CMD="${TRANSFORMERS_PY}" \
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE}" \
OUTPUT_DIR="${PSEUDO_OUTPUT_DIR}" \
OUTPUT_NAME="${PSEUDO_OUTPUT_NAME}" \
PERCH_CACHE_DIR="${PERCH_CACHE_DIR}" \
TEACHER_FOLDS="${TEACHER_FOLDS}" \
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

echo "[RUN] Stage3 with fold-specific Perch+CNN pseudo"
printf '[RUN]'
printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${TRANSFORMERS_PY}" "${STAGE3_ARGS[@]}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${TRANSFORMERS_PY}" "${STAGE3_ARGS[@]}"
