#!/usr/bin/env bash

set -euo pipefail

# Run this script inside the `transformers` conda env by default.
# If you prefer calling conda explicitly, replace this with:
# PYTHON_CMD=(conda run -n transformers python)
PYTHON_CMD=(python)

# =========================
# Basic switches
# =========================
CUDA_VISIBLE_DEVICES_VALUE=0
MODE="both"                # pseudo | stage3 | both
ROOT="."
INPUT_DIR="input"

# =========================
# Teacher / pseudo config
# =========================
TEACHER_MODEL_ROOTS=(
  "outputs/birdclef2026_gm/20260427_160037_convnext_atto.d2_in1k"
)

PSEUDO_OUTPUT_DIR="outputs/pseudo_labels"
PSEUDO_OUTPUT_NAME="convnext_atto_teacher_v1"
EXISTING_PSEUDO_ROOT=""    # If MODE=stage3, you can fill this directly.

SOUNDSCAPES_DIR="input/train_soundscapes"
LABELS_CSV="input/train_soundscapes_labels.csv"
PSEUDO_SCOPE="fold-specific"   # fold-specific | global
TEACHER_FOLDS=""               # e.g. "0,1,2", empty means all available folds

SEGMENT_BATCH_SIZE=12
TTA_OFFSETS="0"
SMOOTHING_KERNEL=""
SOUNDSCAPE_TOP_K=0
PROB_THRESHOLD=0.15
ROW_MIN_MAX_PROB=0.55
TOP_K_LABELS=6
INCLUDE_LABELED=0              # 0 | 1

PSEUDO_DEBUG=0                 # 0 | 1
PSEUDO_DEBUG_LIMIT=16

# =========================
# Stage3 config
# =========================
STUDENT_RUN_DIR="outputs/birdclef2026_gm/20260427_160037_convnext_atto.d2_in1k"
STAGE3_OUTPUT_DIR="outputs/birdclef2026_gm_stage3_pseudo"
STAGE3_FOLDS=""                # e.g. "0,1,2", empty means all folds

STAGE3_EPOCHS=6
STAGE3_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
NUM_WORKERS=4
STAGE3_SAMPLES_PER_EPOCH=4096
STAGE3_BACKBONE_LR=2e-5
STAGE3_HEAD_LR=2e-4
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=1
PATIENCE=5
FREEZE_BACKBONE_EPOCHS=0

PSEUDO_LOSS_WEIGHT=0.5
PSEUDO_SAMPLER_WEIGHT=0.5
MIN_PSEUDO_MAX_PROB=0.55
MAX_PSEUDO_ROWS=-1
ALLOW_GLOBAL_PSEUDO=0          # 0 | 1

STAGE3_SMOKE_TEST=0            # 0 | 1


find_latest_pseudo_root() {
  local latest_dir
  latest_dir="$(find "${PSEUDO_OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${PSEUDO_OUTPUT_NAME}" | sort | tail -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    echo ""
  else
    echo "${latest_dir}"
  fi
}


run_pseudo() {
  local cmd=(
    "${PYTHON_CMD[@]}"
    birdclef2026_gm_make_pseudo_labels.py
    --root "${ROOT}"
    --input-dir "${INPUT_DIR}"
    --output-dir "${PSEUDO_OUTPUT_DIR}"
    --soundscapes-dir "${SOUNDSCAPES_DIR}"
    --labels-csv "${LABELS_CSV}"
    --segment-batch-size "${SEGMENT_BATCH_SIZE}"
    --tta-offsets "${TTA_OFFSETS}"
    --smoothing-kernel "${SMOOTHING_KERNEL}"
    --soundscape-top-k "${SOUNDSCAPE_TOP_K}"
    --prob-threshold "${PROB_THRESHOLD}"
    --row-min-max-prob "${ROW_MIN_MAX_PROB}"
    --top-k-labels "${TOP_K_LABELS}"
    --pseudo-scope "${PSEUDO_SCOPE}"
    --output-name "${PSEUDO_OUTPUT_NAME}"
  )

  local teacher_root
  for teacher_root in "${TEACHER_MODEL_ROOTS[@]}"; do
    cmd+=(--model-root "${teacher_root}")
  done

  if [[ -n "${TEACHER_FOLDS}" ]]; then
    cmd+=(--teacher-folds "${TEACHER_FOLDS}")
  fi

  if [[ "${INCLUDE_LABELED}" == "1" ]]; then
    cmd+=(--include-labeled)
  fi

  if [[ "${PSEUDO_DEBUG}" == "1" ]]; then
    cmd+=(--debug --debug-limit "${PSEUDO_DEBUG_LIMIT}")
  fi

  echo "[RUN] Generating pseudo labels"
  printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
}


run_stage3() {
  local pseudo_root="$1"

  local cmd=(
    "${PYTHON_CMD[@]}"
    birdclef2026_gm_train_stage3_pseudo.py
    --root "${ROOT}"
    --input-dir "${INPUT_DIR}"
    --output-dir "${STAGE3_OUTPUT_DIR}"
    --student-run-dir "${STUDENT_RUN_DIR}"
    --pseudo-root "${pseudo_root}"
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
  )

  if [[ -n "${STAGE3_FOLDS}" ]]; then
    cmd+=(--folds "${STAGE3_FOLDS}")
  fi

  if [[ "${ALLOW_GLOBAL_PSEUDO}" == "1" ]]; then
    cmd+=(--allow-global-pseudo)
  fi

  if [[ "${STAGE3_SMOKE_TEST}" == "1" ]]; then
    cmd+=(--smoke-test)
  fi

  echo "[RUN] Training stage3 with pseudo labels"
  printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
}


main() {
  local pseudo_root="${EXISTING_PSEUDO_ROOT}"

  case "${MODE}" in
    pseudo)
      run_pseudo
      ;;
    stage3)
      if [[ -z "${pseudo_root}" ]]; then
        pseudo_root="$(find_latest_pseudo_root)"
      fi
      if [[ -z "${pseudo_root}" ]]; then
        echo "[ERROR] No pseudo root found. Please set EXISTING_PSEUDO_ROOT or run MODE=both/pseudo first."
        exit 1
      fi
      echo "[INFO] Using pseudo root: ${pseudo_root}"
      run_stage3 "${pseudo_root}"
      ;;
    both)
      run_pseudo
      pseudo_root="$(find_latest_pseudo_root)"
      if [[ -z "${pseudo_root}" ]]; then
        echo "[ERROR] Pseudo generation finished but no pseudo root matching *_${PSEUDO_OUTPUT_NAME} was found."
        exit 1
      fi
      echo "[INFO] Using pseudo root: ${pseudo_root}"
      run_stage3 "${pseudo_root}"
      ;;
    *)
      echo "[ERROR] Unsupported MODE: ${MODE}"
      echo "        Expected one of: pseudo | stage3 | both"
      exit 1
      ;;
  esac
}


main "$@"
