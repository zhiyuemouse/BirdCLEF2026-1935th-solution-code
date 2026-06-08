# BirdCLEF+ 2026 1935th Place Code Package

This package contains the main code and experiment notes for our BirdCLEF+ 2026 solution.

Final rank: **1935 / 4084**

Final practical public LB line: **0.922**

Final safe ensemble:

```text
PerchLR + PerchMamba + PerchAttention + PerchSSM + Stage3 CNN
BLEND_MODE=family3
FILE_SCALE_MODE=topk_mean
FILE_SCALE_VALUE=2
RawWave disabled
Mamba TTA disabled
Stage3 TTA disabled
```

## Important Files

### Solution Notes

- `我的其他的比赛编写的解决方案/BirdCLEF2026-1935th-solution.md`
- `CV-LB.md`
- `MEMORY.md`
- `history.txt`
- `比赛介绍.txt`

### Final Unified Inference

- `birdclef2026_kaggle_infer_unified_perch_stage3.py`
- `run_birdclef2026_unified_perch_stage3_kaggle_infer.sh`

### CNN / Stage3

- `birdclef2026_gm_train.py`
- `birdclef2026_gm_train_stage3_pseudo.py`
- `birdclef2026_gm_train_stage3_oof_teacher.py`
- `birdclef2026_gm_kaggle_infer.py`
- `birdclef2026_gm_kaggle_infer_stage3.py`
- `birdclef2026_make_pseudo_perch_cnn_blend.py`
- `run_birdclef2026_gm_train.sh`
- `run_birdclef2026_gm_stage3.sh`
- `run_birdclef2026_gm_stage3_oof_teacher.sh`
- `run_birdclef2026_make_pseudo_perch_cnn_blend.sh`

### Perch Branches

- `birdclef2026_run_perch_local.py`
- `birdclef2026_cache_perch_spatial_onnx.py`
- `birdclef2026_cache_perch_sequence_onnx.py`
- `birdclef2026_perch_context_train.py`
- `birdclef2026_perch_context_mlp_train.py`
- `birdclef2026_perch_context_lgbm_train.py`
- `birdclef2026_perch_kaggle_infer_context_logreg.py`
- `birdclef2026_perch_spatial_mamba_train.py`
- `birdclef2026_perch_sequence_ssm_train.py`
- `birdclef2026_perch_temporal_head_train.py`
- `birdclef2026_perch_kaggle_infer_spatial_mamba.py`
- `run_birdclef2026_perch_context_aligned_train.sh`
- `run_birdclef2026_perch_spatial_mamba_train.sh`
- `run_birdclef2026_perch_sequence_ssm_train.sh`

### Ensembling / OOF / Diagnostics

- `birdclef2026_blend_submissions_postprocess.py`
- `birdclef2026_whitelist_blend_grid.py`
- `birdclef2026_whitelist_blend_grid_threeway.py`
- `birdclef2026_whitelist_blend_unified_ssm.py`
- `birdclef2026_final_oof_gating.py`
- `birdclef2026_final_oof_tricks.py`
- `birdclef2026_safe_oof_branch_probe.py`
- `birdclef2026_teacher_oof.py`
- `birdclef2026_make_strict_fold_teacher.py`

### Extra Branches / Ablations

These were useful for experiments, but not part of the final safe online submission:

- `birdclef2026_raw_waveform_transformer_train.py`
- `waveform_model.py`
- `birdclef2026_audiomae_mlp_train.py`
- `birdclef2026_audiomae_token_head_train.py`
- `birdclef2026_kaggle_infer_audiomae_token.py`
- `birdclef2026_eval_cnn_shift_tta.py`
- `birdclef2026_eval_perch_shift_tta.py`

## Not Included

This zip is a **code package**, not a full reproducibility package. It does not include:

- Kaggle input data
- Perch model weights
- ONNX weights
- trained `.pth` / `.joblib` artifacts
- `outputs/`
- local caches such as `perch_*_cache*`
- old external reference solutions

Those files were intentionally excluded to keep the GitHub package small and clean.

## Final Inference Idea

The final inference script expects external Kaggle datasets for:

- Perch v2 directory with `assets/labels.csv`
- Perch ONNX model
- PerchLR artifact
- PerchMamba artifact
- PerchAttention artifact
- PerchSSM artifact
- Stage3 CNN fold checkpoints

The final safe online configuration disabled RawWave and all TTA for runtime stability.

## Leakage Notes

The local CV used grouped soundscape folds. All 5-second windows from the same 1-minute soundscape were kept in the same fold.

Late post-competition diagnostic experiments reached higher local CV, for example `0.945506`, but one of those lines had indirect leakage / optimistic CV risk because the Stage3 OOF teacher used ordinary OOF `pred` instead of strict `pred_by_fold`. The clean solution note treats that result as diagnostic, not as the final clean CV.

## Final Lesson

The strongest practical improvement came from combining several frozen Perch heads with one complementary CNN branch, while keeping CPU inference simple enough to finish.
