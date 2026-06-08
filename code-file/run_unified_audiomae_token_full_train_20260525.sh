#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate transformers
python -u birdclef2026_kaggle_infer_unified_perch_stage3.py \
  --competition-root input \
  --soundscapes-dir input/train_soundscapes \
  --sample-submission-path input/sample_submission.csv \
  --taxonomy-path input/taxonomy.csv \
  --perch-dir Perch \
  --perch-onnx-path PerchV2Onnx/perch_v2.onnx \
  --perch-lr-model-path outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib \
  --mamba-model-path outputs/perch_spatial_mamba_audio_pretrain_max100_stage1e15_cnn195634folds_v1/perch_spatial_mamba_artifacts.joblib \
  --attention-model-path outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1/perch_spatial_mamba_artifacts.joblib \
  --stage3-model-root outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo \
  --raw-wave-model-root outputs/birdclef2026_raw_waveform_transformer/20260512_013731_raw_wave_conv_tokenizer_base_long_n32_d768 \
  --audiomae-ckpt-dir ckpt/AudioMAE-HF \
  --audiomae-token-model-path outputs/audiomae_token_attention_labeled_cnn195634folds_h384_v1/audiomae_token_head_artifacts.joblib \
  --output-path outputs/unified_audiomae_token_train_soundscapes_20260525.csv \
  --batch-files 8 \
  --runtime-num-threads 4 \
  --stage3-segment-batch-size 32 \
  --raw-wave-segment-batch-size 32 \
  --audiomae-token-batch-size 32 \
  --perch-lr-weight 0 \
  --mamba-weight 0.275 \
  --attention-weight 0.075 \
  --stage3-weight 0.075 \
  --raw-wave-weight 0.15 \
  --audiomae-token-weight 0.425 \
  --file-scale-mode topk_mean \
  --file-scale-value 2 \
  --save-branch-submissions
