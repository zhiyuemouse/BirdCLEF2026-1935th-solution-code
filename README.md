# BirdCLEF+ 2026 1935th Place Solution

This repository contains our code and notes for the Kaggle **BirdCLEF+ 2026** competition.

Final rank: **1935 / 4084**

Final practical Public LB line: **0.922**

## Final Safe Ensemble

```text
PerchLR + PerchMamba + PerchAttention + PerchSSM + Stage3 CNN
BLEND_MODE=family3
FILE_SCALE_MODE=topk_mean
FILE_SCALE_VALUE=2
RawWave disabled
Mamba TTA disabled
Stage3 TTA disabled
```

The final submission prioritized CPU runtime stability. Heavier local-CV configurations using RawWave and TTA were not used because they timed out or were too risky online.

## Main Files
- `birdclef2026_kaggle_infer_unified_perch_stage3.py`: final unified Kaggle inference script.
- `run_birdclef2026_unified_perch_stage3_kaggle_infer.sh`: final inference wrapper.
- `birdclef2026_gm_train.py`: CNN training script.
- `birdclef2026_gm_train_stage3_pseudo.py`: Stage3 pseudo training script.
- `birdclef2026_perch_context_train.py`: Perch context LogReg training.
- `birdclef2026_perch_spatial_mamba_train.py`: Perch spatial Mamba training.
- `birdclef2026_perch_sequence_ssm_train.py`: Perch 60s sequence SSM training.
- `birdclef2026_final_oof_gating.py`: final OOF gating experiments.
- `birdclef2026_safe_oof_branch_probe.py`: safe branch replacement diagnostics.
- `CV-LB.md`: local CV and Public LB tracking.
- `MEMORY.md`: detailed experiment memory.

## Not Included

This code package does not include Kaggle input data, trained model weights, Perch model files, ONNX files, caches, or `outputs/` artifacts.

## Leakage Note

The clean final submitted line was the runtime-safe `0.922` Public LB ensemble. Some late post-competition diagnostics reached higher local CV, but one `0.945506` line had indirect leakage / optimistic CV risk through a non-strict OOF teacher. The solution write-up explains this distinction.
