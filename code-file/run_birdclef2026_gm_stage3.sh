#!/usr/bin/env bash
set -euo pipefail

# Conservative Stage 3 template:
# - use only a small amount of high-confidence pseudo labels
# - treat pseudo data as a light finetune signal, not the main training source

CUDA_VISIBLE_DEVICES=1

STUDENT_RUN_DIR="outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k"
OUTPUT_DIR="outputs/birdclef2026_gm_stage3_pseudo"
PSEUDO_ROOT="outputs/pseudo_labels/20260429_195217_conservative_fold_specific_multi_teacher_convnextv2"
PSEUDO_SUFFIX="conservative_fold_specific_multi_teacher_convnextv2"

FOLDS=""
NUM_WORKERS=4

STAGE3_EPOCHS=3
STAGE3_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
STAGE3_SAMPLES_PER_EPOCH=2048

STAGE3_BACKBONE_LR=1e-5
STAGE3_HEAD_LR=1e-4
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=1
PATIENCE=5
FREEZE_BACKBONE_EPOCHS=1

# Batch-level augmentation knobs.
# Recommended first trial for stage3:
#   MIXUP_ALPHA=0.2
#   MIXUP_PROB=0.5
#   CUTMIX_ALPHA=0.0
#   CUTMIX_PROB=0.0
MIXUP_ALPHA=0.0
MIXUP_PROB=0.0
CUTMIX_ALPHA=0.0
CUTMIX_PROB=0.0

PSEUDO_LOSS_WEIGHT=0.15
PSEUDO_SAMPLER_WEIGHT=0.05
MIN_PSEUDO_MAX_PROB=0.85
MAX_PSEUDO_ROWS=5000

if [[ -z "${PSEUDO_ROOT}" ]]; then
  latest_pseudo_dir="$(find outputs/pseudo_labels -maxdepth 1 -mindepth 1 -type d -name "*_${PSEUDO_SUFFIX}" | sort | tail -n 1)"
  if [[ -z "${latest_pseudo_dir}" ]]; then
    echo "Could not find pseudo-label directory matching suffix: ${PSEUDO_SUFFIX}"
    echo "Run bash run_birdclef2026_gm_pseudo.sh first, or set PSEUDO_ROOT manually."
    exit 1
  fi
  PSEUDO_ROOT="${latest_pseudo_dir}"
fi

cmd=(
  python birdclef2026_gm_train_stage3_pseudo.py
  --student-run-dir "${STUDENT_RUN_DIR}"
  --pseudo-root "${PSEUDO_ROOT}"
  --output-dir "${OUTPUT_DIR}"
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
  --mixup-alpha "${MIXUP_ALPHA}"
  --mixup-prob "${MIXUP_PROB}"
  --cutmix-alpha "${CUTMIX_ALPHA}"
  --cutmix-prob "${CUTMIX_PROB}"
  --pseudo-loss-weight "${PSEUDO_LOSS_WEIGHT}"
  --pseudo-sampler-weight "${PSEUDO_SAMPLER_WEIGHT}"
  --min-pseudo-max-prob "${MIN_PSEUDO_MAX_PROB}"
  --max-pseudo-rows "${MAX_PSEUDO_ROWS}"
)

if [[ -n "${FOLDS}" ]]; then
  cmd+=(--folds "${FOLDS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
