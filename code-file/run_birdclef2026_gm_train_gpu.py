#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)

# Force CUDA visibility before birdclef2026_gm_train.py imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTHONHASHSEED"] = "2026"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.argv = [
    "birdclef2026_gm_train.py",
    "--model-name",
    "convnextv2_atto.fcmae_ft_in1k",
    "--head-type",
    "csiro_conv_v1",
    "--head-pool-type",
    "avg",
    "--seed",
    "2026",
    "--stage1-epochs",
    "12",
    "--stage2-epochs",
    "28",
    "--stage1-batch-size",
    "8",
    "--stage2-batch-size",
    "8",
    "--num-workers",
    "0",
    "--stage1-backbone-lr",
    "1e-4",
    "--stage1-head-lr",
    "1e-3",
    "--stage2-backbone-lr",
    "5e-5",
    "--stage2-head-lr",
    "5e-4",
    "--mixup-alpha",
    "0.0",
    "--mixup-prob",
    "0.0",
    "--cutmix-alpha",
    "0.0",
    "--cutmix-prob",
    "0.0",
    "--mixup-domain",
    "waveform",
    "--stage1-mixup-alpha",
    "0.20",
    "--stage1-mixup-prob",
    "0.10",
    "--stage1-cutmix-alpha",
    "0.0",
    "--stage1-cutmix-prob",
    "0.0",
    "--stage2-mixup-alpha",
    "0.0",
    "--stage2-mixup-prob",
    "0.0",
    "--stage2-cutmix-alpha",
    "0.0",
    "--stage2-cutmix-prob",
    "0.0",
]

runpy.run_path(str(REPO_ROOT / "birdclef2026_gm_train.py"), run_name="__main__")
