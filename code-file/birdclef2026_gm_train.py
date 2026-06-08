from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Failed to load image Python extension:.*")

ML_IMPORT_ERROR = None


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()

try:
    import soundfile as sf
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except (ModuleNotFoundError, ImportError, OSError) as exc:
    ML_IMPORT_ERROR = exc
    sf = None
    timm = None
    torch = None
    F = None
    AdamW = None
    CosineAnnealingLR = None
    LinearLR = None
    SequentialLR = None
    DataLoader = object
    WeightedRandomSampler = object

    class Dataset:  # type: ignore[override]
        pass

    class _NNFallback:
        Module = object

    nn = _NNFallback()

    class GradScaler:  # type: ignore[override]
        def __init__(self, enabled: bool = False):
            self.enabled = enabled

    def autocast(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("AMP is unavailable because torch is not installed.")


@dataclass
class Config:
    root: str = "."
    input_dir: str = "input"
    ckpt_dir: str = "ckpt"
    output_dir: str = "outputs/birdclef2026_gm"
    model_name: str = "convnext_atto.d2_in1k"
    n_folds: int = 3
    seed: int = 2026
    sample_rate: int = 32000
    clip_seconds: float = 5.0
    image_height: int = 256
    image_width: int = 320
    input_channels: int = 3
    spectrogram_variant: str = "logmel"
    image_normalize: str = "minus_one_one"
    multi_context_num_slots: int = 3
    multi_context_global_loss_weight: float = 0.35
    stage1_epochs: int = 6
    stage2_epochs: int = 14
    stage1_batch_size: int = 16
    stage2_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = max(2, (os.cpu_count() or 4) // 2)
    stage1_samples_per_epoch: int = 24000
    stage2_samples_per_epoch: int = 2048
    stage1_backbone_lr: float = 1e-4
    stage1_head_lr: float = 1e-3
    stage2_backbone_lr: float = 5e-5
    stage2_head_lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    scheduler_type: str = "linear_cosine"
    stage2_freeze_backbone_epochs: int = 1
    dropout: float = 0.2
    drop_path: float = 0.1
    head_type: str = "linear"
    head_pool_type: str = "avg"
    lse_temperature: float = 1.0
    sed_frame_loss_weight: float = 0.5
    sed_center_context: bool = False
    use_waveform_branch: bool = False
    waveform_branch_d_model: int = 128
    waveform_branch_layers: int = 1
    waveform_branch_heads: int = 4
    waveform_branch_dropout: float = 0.1
    use_perch_distill: bool = False
    perch_spatial_cache_dir: str = "perch_spatial_cache_labeled_all"
    perch_spatial_meta_path: str = ""
    perch_spatial_arrays_path: str = ""
    perch_distill_weight: float = 0.0
    perch_distill_token_key: str = "spatial_tokens"
    specaug_time_mask: int = 32
    specaug_freq_mask: int = 24
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.0
    mixup_domain: str = "image"
    stage1_mixup_alpha: float = 0.0
    stage1_mixup_prob: float = 0.0
    stage1_mixup_start_epoch: int = 1
    stage1_cutmix_alpha: float = 0.0
    stage1_cutmix_prob: float = 0.0
    stage1_cutmix_start_epoch: int = 1
    stage2_mixup_alpha: float = 0.0
    stage2_mixup_prob: float = 0.0
    stage2_mixup_start_epoch: int = 1
    stage2_cutmix_alpha: float = 0.0
    stage2_cutmix_prob: float = 0.0
    stage2_cutmix_start_epoch: int = 1
    clip_valid_fraction: float = 0.05
    use_secondary_labels: bool = True
    patience: int = 5
    use_amp: bool = True
    amp_mode: str = "auto"
    grad_clip_norm: float = 0.0
    smoke_test: bool = False
    build_folds_only: bool = False
    max_train_audio_rows: int = -1
    max_soundscape_segments: int = -1
    use_birdclef2025_stage1: bool = False
    birdclef2025_root: str = "BirdCLEF2025-Dataset"
    birdclef2025_stage1_max_rows: int = -1
    birdclef2025_stage1_max_per_label: int = -1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 training pipeline with leakage-aware local CV.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm")
    parser.add_argument("--model-name", type=str, default="convnext_atto.d2_in1k")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--spectrogram-variant", type=str, choices=["logmel", "pcen", "logmel_v8"], default="logmel")
    parser.add_argument("--image-normalize", type=str, choices=["minus_one_one", "zero_one"], default="minus_one_one")
    parser.add_argument(
        "--head-type",
        type=str,
        choices=["linear", "csiro_conv_v1", "csiro_multicontext_v1", "sed_att_v1", "lse_head_v1"],
        default="linear",
    )
    parser.add_argument("--head-pool-type", type=str, choices=["avg", "gem", "lse", "avg_max"], default="avg")
    parser.add_argument("--lse-temperature", type=float, default=1.0)
    parser.add_argument("--sed-frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--sed-center-context", action="store_true")
    parser.add_argument("--multi-context-num-slots", type=int, default=3)
    parser.add_argument("--multi-context-global-loss-weight", type=float, default=0.35)
    parser.add_argument("--use-waveform-branch", action="store_true")
    parser.add_argument("--waveform-branch-d-model", type=int, default=128)
    parser.add_argument("--waveform-branch-layers", type=int, default=1)
    parser.add_argument("--waveform-branch-heads", type=int, default=4)
    parser.add_argument("--waveform-branch-dropout", type=float, default=0.1)
    parser.add_argument("--use-perch-distill", action="store_true")
    parser.add_argument("--perch-spatial-cache-dir", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--perch-spatial-meta-path", type=str, default="")
    parser.add_argument("--perch-spatial-arrays-path", type=str, default="")
    parser.add_argument("--perch-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--perch-distill-token-key",
        type=str,
        choices=["spatial_tokens", "spatial_tokens_max", "spatial_tokens_64"],
        default="spatial_tokens",
    )
    parser.add_argument("--stage1-epochs", type=int, default=6)
    parser.add_argument("--stage2-epochs", type=int, default=14)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--stage1-samples-per-epoch", type=int, default=24000)
    parser.add_argument("--stage2-samples-per-epoch", type=int, default=2048)
    parser.add_argument("--stage1-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-head-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-backbone-lr", type=float, default=5e-5)
    parser.add_argument("--stage2-head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--scheduler-type", type=str, choices=["linear_cosine", "onecycle"], default="linear_cosine")
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--mixup-domain", type=str, choices=["image", "waveform"], default="image")
    parser.add_argument("--stage1-mixup-alpha", type=float, default=None)
    parser.add_argument("--stage1-mixup-prob", type=float, default=None)
    parser.add_argument("--stage1-mixup-start-epoch", type=int, default=1)
    parser.add_argument("--stage1-cutmix-alpha", type=float, default=None)
    parser.add_argument("--stage1-cutmix-prob", type=float, default=None)
    parser.add_argument("--stage1-cutmix-start-epoch", type=int, default=1)
    parser.add_argument("--stage2-mixup-alpha", type=float, default=None)
    parser.add_argument("--stage2-mixup-prob", type=float, default=None)
    parser.add_argument("--stage2-mixup-start-epoch", type=int, default=1)
    parser.add_argument("--stage2-cutmix-alpha", type=float, default=None)
    parser.add_argument("--stage2-cutmix-prob", type=float, default=None)
    parser.add_argument("--stage2-cutmix-start-epoch", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp-mode", type=str, choices=["auto", "fp16", "bf16", "off"], default="auto")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--stage2-freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--specaug-time-mask", type=int, default=32)
    parser.add_argument("--specaug-freq-mask", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--build-folds-only", action="store_true")
    parser.add_argument("--max-train-audio-rows", type=int, default=-1)
    parser.add_argument("--max-soundscape-segments", type=int, default=-1)
    parser.add_argument("--use-birdclef2025-stage1", action="store_true")
    parser.add_argument("--birdclef2025-root", type=str, default="BirdCLEF2025-Dataset")
    parser.add_argument("--birdclef2025-stage1-max-rows", type=int, default=-1)
    parser.add_argument("--birdclef2025-stage1-max-per-label", type=int, default=-1)
    args = parser.parse_args()

    def resolve_stage_aug(stage_value: Optional[float], global_value: float) -> float:
        if stage_value is None:
            return float(global_value)
        return float(stage_value)

    cfg = Config(
        root=args.root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        image_height=args.image_height,
        image_width=args.image_width,
        input_channels=args.input_channels,
        spectrogram_variant=args.spectrogram_variant,
        image_normalize=args.image_normalize,
        head_type=args.head_type,
        head_pool_type=args.head_pool_type,
        lse_temperature=args.lse_temperature,
        sed_frame_loss_weight=args.sed_frame_loss_weight,
        sed_center_context=args.sed_center_context,
        multi_context_num_slots=args.multi_context_num_slots,
        multi_context_global_loss_weight=args.multi_context_global_loss_weight,
        use_waveform_branch=args.use_waveform_branch,
        waveform_branch_d_model=args.waveform_branch_d_model,
        waveform_branch_layers=args.waveform_branch_layers,
        waveform_branch_heads=args.waveform_branch_heads,
        waveform_branch_dropout=args.waveform_branch_dropout,
        use_perch_distill=args.use_perch_distill,
        perch_spatial_cache_dir=args.perch_spatial_cache_dir,
        perch_spatial_meta_path=args.perch_spatial_meta_path,
        perch_spatial_arrays_path=args.perch_spatial_arrays_path,
        perch_distill_weight=args.perch_distill_weight,
        perch_distill_token_key=args.perch_distill_token_key,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_batch_size=args.stage1_batch_size,
        stage2_batch_size=args.stage2_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        stage1_samples_per_epoch=args.stage1_samples_per_epoch,
        stage2_samples_per_epoch=args.stage2_samples_per_epoch,
        stage1_backbone_lr=args.stage1_backbone_lr,
        stage1_head_lr=args.stage1_head_lr,
        stage2_backbone_lr=args.stage2_backbone_lr,
        stage2_head_lr=args.stage2_head_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler_type,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        mixup_domain=args.mixup_domain,
        stage1_mixup_alpha=resolve_stage_aug(args.stage1_mixup_alpha, args.mixup_alpha),
        stage1_mixup_prob=resolve_stage_aug(args.stage1_mixup_prob, args.mixup_prob),
        stage1_mixup_start_epoch=args.stage1_mixup_start_epoch,
        stage1_cutmix_alpha=resolve_stage_aug(args.stage1_cutmix_alpha, args.cutmix_alpha),
        stage1_cutmix_prob=resolve_stage_aug(args.stage1_cutmix_prob, args.cutmix_prob),
        stage1_cutmix_start_epoch=args.stage1_cutmix_start_epoch,
        stage2_mixup_alpha=resolve_stage_aug(args.stage2_mixup_alpha, args.mixup_alpha),
        stage2_mixup_prob=resolve_stage_aug(args.stage2_mixup_prob, args.mixup_prob),
        stage2_mixup_start_epoch=args.stage2_mixup_start_epoch,
        stage2_cutmix_alpha=resolve_stage_aug(args.stage2_cutmix_alpha, args.cutmix_alpha),
        stage2_cutmix_prob=resolve_stage_aug(args.stage2_cutmix_prob, args.cutmix_prob),
        stage2_cutmix_start_epoch=args.stage2_cutmix_start_epoch,
        use_amp=not args.disable_amp,
        amp_mode=args.amp_mode,
        grad_clip_norm=args.grad_clip_norm,
        stage2_freeze_backbone_epochs=args.stage2_freeze_backbone_epochs,
        dropout=args.dropout,
        drop_path=args.drop_path,
        specaug_time_mask=args.specaug_time_mask,
        specaug_freq_mask=args.specaug_freq_mask,
        patience=args.patience,
        smoke_test=args.smoke_test,
        build_folds_only=args.build_folds_only,
        max_train_audio_rows=args.max_train_audio_rows,
        max_soundscape_segments=args.max_soundscape_segments,
        use_birdclef2025_stage1=args.use_birdclef2025_stage1,
        birdclef2025_root=args.birdclef2025_root,
        birdclef2025_stage1_max_rows=args.birdclef2025_stage1_max_rows,
        birdclef2025_stage1_max_per_label=args.birdclef2025_stage1_max_per_label,
    )

    if cfg.smoke_test:
        cfg.stage1_epochs = 1
        cfg.stage2_epochs = 1
        cfg.stage1_samples_per_epoch = min(cfg.stage1_samples_per_epoch, 256)
        cfg.stage2_samples_per_epoch = min(cfg.stage2_samples_per_epoch, 128)
        cfg.max_train_audio_rows = 256 if cfg.max_train_audio_rows < 0 else min(cfg.max_train_audio_rows, 256)
        cfg.max_soundscape_segments = 96 if cfg.max_soundscape_segments < 0 else min(cfg.max_soundscape_segments, 96)
        cfg.num_workers = min(cfg.num_workers, 2)
        cfg.patience = 1
    if cfg.input_channels == 1 and cfg.mixup_domain == "waveform":
        # Waveform mixup is still valid with one-channel spectrograms, but the V8
        # recipe uses image-level mixup after mel rendering.
        pass
    if cfg.use_waveform_branch and cfg.waveform_branch_d_model % cfg.waveform_branch_heads != 0:
        raise ValueError("--waveform-branch-d-model must be divisible by --waveform-branch-heads")
    if not 0.0 <= float(cfg.sed_frame_loss_weight) <= 1.0:
        raise ValueError("--sed-frame-loss-weight must be in [0, 1].")
    if cfg.sed_center_context and cfg.head_type != "sed_att_v1":
        raise ValueError("--sed-center-context is only supported with --head-type sed_att_v1.")
    if cfg.sed_center_context and float(cfg.clip_seconds) < 5.0:
        raise ValueError("--sed-center-context expects --clip-seconds >= 5.")
    if cfg.head_type == "csiro_multicontext_v1":
        if cfg.multi_context_num_slots < 2:
            raise ValueError("--multi-context-num-slots must be >= 2 for csiro_multicontext_v1")
        expected_seconds = 5.0 * float(cfg.multi_context_num_slots)
        if abs(float(cfg.clip_seconds) - expected_seconds) > 1e-6:
            raise ValueError(
                "csiro_multicontext_v1 expects --clip-seconds to equal "
                f"5 * --multi-context-num-slots ({expected_seconds:g}s)."
            )
    return cfg


def format_batch_aug_summary(mixup_alpha: float, mixup_prob: float, cutmix_alpha: float, cutmix_prob: float) -> str:
    return (
        f"mixup(alpha={mixup_alpha}, prob={mixup_prob}) | "
        f"cutmix(alpha={cutmix_alpha}, prob={cutmix_prob})"
    )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    if torch is None:
        return
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def is_vit_like_model(model_name: str) -> bool:
    name = str(model_name).lower()
    vit_keywords = (
        "vit",
        "deit",
        "beit",
        "eva",
        "swin",
        "xcit",
        "mvit",
    )
    return any(keyword in name for keyword in vit_keywords)


def cuda_supports_bf16() -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_amp_settings(cfg: Config, device: torch.device) -> Dict[str, object]:
    if torch is None or device.type != "cuda" or not cfg.use_amp:
        return {
            "enabled": False,
            "dtype": None,
            "use_grad_scaler": False,
            "description": "disabled",
        }

    amp_mode = str(cfg.amp_mode).lower()
    if amp_mode == "off":
        return {
            "enabled": False,
            "dtype": None,
            "use_grad_scaler": False,
            "description": "disabled",
        }

    if amp_mode == "auto":
        if is_vit_like_model(cfg.model_name):
            if cuda_supports_bf16():
                return {
                    "enabled": True,
                    "dtype": torch.bfloat16,
                    "use_grad_scaler": False,
                    "description": "bf16(auto-vit)",
                }
            return {
                "enabled": False,
                "dtype": None,
                "use_grad_scaler": False,
                "description": "disabled(auto-vit-no-bf16)",
            }
        amp_mode = "fp16"

    if amp_mode == "bf16":
        if cuda_supports_bf16():
            return {
                "enabled": True,
                "dtype": torch.bfloat16,
                "use_grad_scaler": False,
                "description": "bf16",
            }
        print("[WARN] bf16 AMP requested but current GPU does not support bf16. Falling back to fp32.")
        return {
            "enabled": False,
            "dtype": None,
            "use_grad_scaler": False,
            "description": "disabled(bf16-unsupported)",
        }

    return {
        "enabled": True,
        "dtype": torch.float16,
        "use_grad_scaler": True,
        "description": "fp16",
    }


def maybe_autocast(enabled: bool, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return autocast(enabled=enabled)
    return autocast(enabled=enabled, dtype=amp_dtype)


def scaler_is_enabled(scaler: GradScaler) -> bool:
    return bool(getattr(scaler, "is_enabled", lambda: getattr(scaler, "enabled", False))())


def backoff_grad_scaler(scaler: GradScaler) -> None:
    if not scaler_is_enabled(scaler):
        return
    if hasattr(scaler, "get_scale") and hasattr(scaler, "update"):
        current_scale = float(scaler.get_scale())
        try:
            scaler.update(max(current_scale / 2.0, 1.0))
            return
        except TypeError:
            pass
    scaler.update()


def gradients_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def resolve_grad_clip_norm(cfg: Config) -> float:
    if cfg.grad_clip_norm > 0:
        return float(cfg.grad_clip_norm)
    if is_vit_like_model(cfg.model_name):
        return 1.0
    return 0.0


def macro_auc_skip_missing(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for class_idx in range(y_true.shape[1]):
        auc = binary_auc(y_true[:, class_idx], y_pred[:, class_idx])
        if auc is not None:
            scores.append(auc)
    if not scores:
        raise ValueError("No positive classes in validation data; cannot compute macro ROC-AUC.")
    return float(np.mean(scores))


def macro_auc_for_class_indices(y_true: np.ndarray, y_pred: np.ndarray, class_indices: np.ndarray) -> float | None:
    scores = []
    for class_idx in np.asarray(class_indices, dtype=np.int64):
        auc = binary_auc(y_true[:, class_idx], y_pred[:, class_idx])
        if auc is not None:
            scores.append(auc)
    if not scores:
        return None
    return float(np.mean(scores))


def auc_frequency_slices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_max_pos: int = 4,
    common_min_pos: int = 10,
) -> Dict[str, Optional[float]]:
    pos_counts = (np.asarray(y_true) > 0.5).sum(axis=0)
    valid_classes = np.where((pos_counts > 0) & (pos_counts < y_true.shape[0]))[0]
    rare_classes = valid_classes[pos_counts[valid_classes] <= rare_max_pos]
    common_classes = valid_classes[pos_counts[valid_classes] >= common_min_pos]
    mid_classes = np.setdiff1d(valid_classes, np.union1d(rare_classes, common_classes), assume_unique=False)
    return {
        "auc_rare": macro_auc_for_class_indices(y_true, y_pred, rare_classes),
        "auc_mid": macro_auc_for_class_indices(y_true, y_pred, mid_classes),
        "auc_common": macro_auc_for_class_indices(y_true, y_pred, common_classes),
        "n_rare_classes": int(len(rare_classes)),
        "n_mid_classes": int(len(mid_classes)),
        "n_common_classes": int(len(common_classes)),
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos_mask = y_true > 0.5
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())

    # Skip classes without positives like the competition metric,
    # and also skip degenerate all-positive folds where ROC-AUC is undefined.
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    n = len(sorted_scores)
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    pos_ranks = ranks[sorted_true > 0.5].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def time_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def parse_secondary_labels(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    text = text.strip("[]")
    if not text:
        return []
    labels = []
    for part in text.split(","):
        label = part.strip().strip("'").strip('"')
        if label:
            labels.append(label)
    return labels


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class TeeStream:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.text_buffer = ""
        self.progress_buffer = None

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())

    def fileno(self):
        return self.console_stream.fileno()

    def write(self, text):
        if not text:
            return 0
        self.console_stream.write(text)
        self._write_to_log(text)
        return len(text)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def flush_pending(self):
        if self.progress_buffer:
            self.log_stream.write(self.progress_buffer.rstrip() + "\n")
            self.progress_buffer = None
        elif self.text_buffer:
            self.log_stream.write(self.text_buffer)
            self.text_buffer = ""
        self.log_stream.flush()

    def _write_to_log(self, text: str) -> None:
        for char in text:
            if char == "\r":
                if self.text_buffer:
                    self.log_stream.write(self.text_buffer)
                    self.text_buffer = ""
                self.progress_buffer = ""
            elif char == "\n":
                if self.progress_buffer is not None:
                    self.log_stream.write(self.progress_buffer.rstrip())
                    self.progress_buffer = None
                else:
                    self.log_stream.write(self.text_buffer)
                    self.text_buffer = ""
                self.log_stream.write("\n")
                self.log_stream.flush()
            else:
                if self.progress_buffer is not None:
                    self.progress_buffer += char
                else:
                    self.text_buffer += char


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_stream = None
        self.stdout_proxy = None
        self.stderr_proxy = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.log_stream = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_proxy = TeeStream(self.original_stdout, self.log_stream)
        self.stderr_proxy = TeeStream(self.original_stderr, self.log_stream)
        sys.stdout = self.stdout_proxy
        sys.stderr = self.stderr_proxy
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stdout_proxy is not None:
            self.stdout_proxy.flush_pending()
        if self.stderr_proxy is not None:
            self.stderr_proxy.flush_pending()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_stream is not None:
            self.log_stream.close()


def require_training_dependencies() -> None:
    if ML_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Training/inference dependencies are missing. "
            "This script needs at least torch, timm, and soundfile."
        ) from ML_IMPORT_ERROR


def load_class_names(input_dir: Path) -> List[str]:
    sample_submission = pd.read_csv(input_dir / "sample_submission.csv", nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def labels_to_indices(labels: Sequence[str], label_to_idx: Dict[str, int]) -> List[int]:
    return sorted({label_to_idx[label] for label in labels if label in label_to_idx})


def indices_to_multihot(label_indices: Sequence[int], num_classes: int) -> np.ndarray:
    target = np.zeros(num_classes, dtype=np.float32)
    if label_indices:
        target[np.asarray(label_indices, dtype=np.int64)] = 1.0
    return target


def build_clip_holdout_split(df: pd.DataFrame, valid_fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    valid_indices: List[int] = []

    for _, group in df.groupby("primary_label"):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices.tolist())
            continue
        n_valid = max(1, int(round(len(indices) * valid_fraction)))
        n_valid = min(n_valid, len(indices) - 1)
        valid_indices.extend(indices[:n_valid].tolist())
        train_indices.extend(indices[n_valid:].tolist())

    split_df = df.copy()
    split_df["clip_split"] = "train"
    split_df.loc[valid_indices, "clip_split"] = "valid"
    return split_df


def load_train_audio_metadata(cfg: Config, input_dir: Path, label_to_idx: Dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "train.csv")
    if cfg.max_train_audio_rows > 0:
        sample_size = min(cfg.max_train_audio_rows, len(df))
        df = df.sample(sample_size, random_state=cfg.seed).reset_index(drop=True)
    df["audio_path"] = df["filename"].map(lambda x: str(input_dir / "train_audio" / x))
    df["secondary_list"] = df["secondary_labels"].apply(parse_secondary_labels)
    if cfg.use_secondary_labels:
        df["labels"] = df.apply(
            lambda row: sorted({str(row["primary_label"])} | set(row["secondary_list"])),
            axis=1,
        )
    else:
        df["labels"] = df["primary_label"].astype(str).map(lambda x: [x])
    df["label_indices"] = df["labels"].apply(lambda x: labels_to_indices(x, label_to_idx))
    df = df[df["label_indices"].map(len) > 0].reset_index(drop=True)
    df = build_clip_holdout_split(df, valid_fraction=cfg.clip_valid_fraction, seed=cfg.seed)
    df["source_dataset"] = "birdclef2026"
    df["recording_id"] = df["filename"].map(lambda x: Path(str(x)).stem)
    if cfg.use_birdclef2025_stage1:
        df = append_birdclef2025_stage1_audio(
            df=df,
            cfg=cfg,
            root_dir=input_dir.parent,
            label_to_idx=label_to_idx,
        )
    return df


def append_birdclef2025_stage1_audio(
    df: pd.DataFrame,
    cfg: Config,
    root_dir: Path,
    label_to_idx: Dict[str, int],
) -> pd.DataFrame:
    external_root = Path(cfg.birdclef2025_root)
    if not external_root.is_absolute():
        external_root = root_dir / external_root
    train_csv = external_root / "train.csv"
    audio_dir = external_root / "train_audio"
    if not train_csv.exists() or not audio_dir.exists():
        print(f"[WARN] BirdCLEF2025 Stage 1 data not found under {external_root}. Skipping external Stage 1 data.")
        return df

    ext = pd.read_csv(train_csv)
    ext["primary_label"] = ext["primary_label"].astype(str)
    ext = ext[ext["primary_label"].isin(label_to_idx)].copy()
    if ext.empty:
        print("[WARN] BirdCLEF2025 Stage 1 data has no labels overlapping the 2026 taxonomy. Skipping.")
        return df

    ext["recording_id"] = ext["filename"].map(lambda x: Path(str(x)).stem)
    existing_ids = set(df["recording_id"].astype(str))
    existing_filenames = set(df["filename"].astype(str))
    existing_urls = set(df["url"].astype(str)) if "url" in df.columns else set()
    ext = ext[
        ~ext["recording_id"].astype(str).isin(existing_ids)
        & ~ext["filename"].astype(str).isin(existing_filenames)
        & ~ext["url"].astype(str).isin(existing_urls)
    ].copy()
    if ext.empty:
        print("[WARN] BirdCLEF2025 Stage 1 data was fully removed by duplicate filtering. Skipping.")
        return df

    if cfg.birdclef2025_stage1_max_per_label > 0:
        ext = (
            ext.groupby("primary_label", group_keys=False)
            .apply(lambda group: group.sample(min(len(group), cfg.birdclef2025_stage1_max_per_label), random_state=cfg.seed))
            .reset_index(drop=True)
        )
    if cfg.birdclef2025_stage1_max_rows > 0 and len(ext) > cfg.birdclef2025_stage1_max_rows:
        ext = ext.sample(cfg.birdclef2025_stage1_max_rows, random_state=cfg.seed).reset_index(drop=True)

    ext["audio_path"] = ext["filename"].map(lambda x: str(audio_dir / x))
    ext["secondary_list"] = ext["secondary_labels"].apply(parse_secondary_labels)
    if cfg.use_secondary_labels:
        ext["labels"] = ext.apply(
            lambda row: sorted({str(row["primary_label"])} | set(row["secondary_list"])),
            axis=1,
        )
    else:
        ext["labels"] = ext["primary_label"].astype(str).map(lambda x: [x])
    ext["label_indices"] = ext["labels"].apply(lambda x: labels_to_indices(x, label_to_idx))
    ext = ext[ext["label_indices"].map(len) > 0].reset_index(drop=True)
    ext["clip_split"] = "train"
    ext["source_dataset"] = "birdclef2025"

    missing_audio = ~ext["audio_path"].map(lambda x: Path(x).exists())
    if bool(missing_audio.any()):
        print(f"[WARN] Dropping {int(missing_audio.sum())} BirdCLEF2025 rows with missing audio files.")
        ext = ext[~missing_audio].reset_index(drop=True)
    if ext.empty:
        print("[WARN] BirdCLEF2025 Stage 1 data has no usable rows after file checks. Skipping.")
        return df

    added_label_count = int(ext["primary_label"].nunique())
    print(
        f"[INFO] Added BirdCLEF2025 Stage 1 train_audio rows: {len(ext)} "
        f"across {added_label_count} overlapping labels after duplicate filtering."
    )
    for column in df.columns:
        if column not in ext.columns:
            ext[column] = np.nan
    return pd.concat([df, ext[df.columns]], axis=0, ignore_index=True)


def load_soundscape_segments(cfg: Config, input_dir: Path, label_to_idx: Dict[str, int]) -> pd.DataFrame:
    raw = pd.read_csv(input_dir / "train_soundscapes_labels.csv")
    grouped = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda values: sorted({label for value in values.astype(str) for label in value.split(";") if label}))
        .reset_index()
    )
    grouped["audio_path"] = grouped["filename"].map(lambda x: str(input_dir / "train_soundscapes" / x))
    grouped["site"] = grouped["filename"].str.extract(r"_(S\d+)_")
    grouped["start_sec"] = grouped["start"].map(time_to_seconds)
    grouped["end_sec"] = grouped["end"].map(time_to_seconds)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)
    grouped["label_indices"] = grouped["primary_label"].apply(lambda x: labels_to_indices(x, label_to_idx))
    grouped = grouped[grouped["label_indices"].map(len) > 0].reset_index(drop=True)
    if cfg.max_soundscape_segments > 0:
        sample_size = min(cfg.max_soundscape_segments, len(grouped))
        grouped = grouped.sample(sample_size, random_state=cfg.seed).reset_index(drop=True)
    grouped = grouped.rename(columns={"primary_label": "labels"})
    return grouped


def resolve_perch_spatial_paths_from_args(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates: List[Path] = []
    arrays_candidates: List[Path] = []
    if meta_path_arg:
        meta_candidates.append(Path(meta_path_arg))
    else:
        meta_candidates.extend([cache_dir / "perch_spatial_meta.parquet", cache_dir / "perch_spatial_meta.csv"])
    if arrays_path_arg:
        arrays_candidates.append(Path(arrays_path_arg))
    else:
        arrays_candidates.append(cache_dir / "perch_spatial_arrays.npz")
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find spatial meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find spatial arrays under {cache_dir}")
    return meta_path, arrays_path


def load_perch_spatial_cache_from_paths(
    cache_dir_arg: str,
    meta_path_arg: str = "",
    arrays_path_arg: str = "",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    meta_path, arrays_path = resolve_perch_spatial_paths_from_args(
        cache_dir_arg=cache_dir_arg,
        meta_path_arg=meta_path_arg,
        arrays_path_arg=arrays_path_arg,
    )
    meta_df = pd.read_parquet(meta_path) if meta_path.suffix.lower() == ".parquet" else pd.read_csv(meta_path)
    arrays = np.load(arrays_path)
    if "spatial_tokens" not in arrays:
        raise KeyError(f"{arrays_path} must contain spatial_tokens")
    spatial_tokens = arrays["spatial_tokens"].astype(np.float32, copy=False)
    if spatial_tokens.ndim != 3:
        raise ValueError(f"Expected spatial_tokens [rows,tokens,dim], got {spatial_tokens.shape}")
    spatial_tokens_max = None
    if "spatial_tokens_max" in arrays:
        spatial_tokens_max = arrays["spatial_tokens_max"].astype(np.float32, copy=False)
        if spatial_tokens_max.shape != spatial_tokens.shape:
            raise ValueError(
                "spatial_tokens_max shape must match spatial_tokens: "
                f"{spatial_tokens_max.shape} vs {spatial_tokens.shape}"
            )
    spatial_tokens_64 = None
    if "spatial_tokens_64" in arrays:
        spatial_tokens_64 = arrays["spatial_tokens_64"].astype(np.float32, copy=False)
        if spatial_tokens_64.ndim != 3 or spatial_tokens_64.shape[1:] != (64, spatial_tokens.shape[2]):
            raise ValueError(f"Unexpected spatial_tokens_64 shape: {spatial_tokens_64.shape}")
    return meta_df, spatial_tokens, spatial_tokens_max, spatial_tokens_64


def align_array_by_row_id(
    source_meta: pd.DataFrame,
    source_array: np.ndarray,
    target_row_ids: Sequence[str],
    name: str,
) -> np.ndarray:
    source_pos = pd.Series(np.arange(len(source_meta), dtype=np.int64), index=source_meta["row_id"].astype(str))
    indices = pd.Series(list(target_row_ids), dtype=str).map(source_pos)
    if indices.isna().any():
        missing = pd.Series(list(target_row_ids), dtype=str).loc[indices.isna()].head(5).tolist()
        raise ValueError(f"{name} cache missing {indices.isna().sum()} rows. Examples: {missing}")
    return source_array[indices.to_numpy(dtype=np.int64)]


def build_long_context_windows(segment_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build 15s-style windows with 5s slot labels for multi-context training.

    Each row is one contiguous window. Its target is a list of per-slot label
    indices plus a mask, because a few training soundscapes have partial labels.
    """
    slots = int(cfg.multi_context_num_slots)
    if slots < 2:
        raise ValueError("multi_context_num_slots must be >= 2")
    rows = []
    for filename, file_df in segment_df.groupby("filename", sort=True):
        file_df = file_df.sort_values("end_sec").reset_index(drop=True)
        audio_path = str(file_df["audio_path"].iloc[0])
        site = file_df["site"].iloc[0]
        fold = int(file_df["fold"].iloc[0]) if "fold" in file_df.columns else -1
        stem = Path(filename).stem
        label_by_end = {int(row["end_sec"]): row["label_indices"] for _, row in file_df.iterrows()}
        row_id_by_end = {int(row["end_sec"]): row["row_id"] for _, row in file_df.iterrows()}

        # BirdCLEF soundscapes are 60s, scored every 5s. A 15s/3-slot model
        # therefore sees starts 0..45s and emits slots for the next 3 rows.
        for start_slot in range(0, 12 - slots + 1):
            slot_label_indices = []
            slot_mask = []
            slot_row_ids = []
            union_labels = set()
            for slot in range(slots):
                end_sec = int((start_slot + slot + 1) * 5)
                labels = list(label_by_end.get(end_sec, []))
                has_label = end_sec in label_by_end
                slot_label_indices.append(labels)
                slot_mask.append(bool(has_label))
                slot_row_ids.append(row_id_by_end.get(end_sec, f"{stem}_{end_sec}"))
                if has_label:
                    union_labels.update(labels)

            if not any(slot_mask) or not union_labels:
                continue
            rows.append(
                {
                    "filename": filename,
                    "audio_path": audio_path,
                    "site": site,
                    "fold": fold,
                    "window_start_sec": float(start_slot * 5),
                    "window_id": f"{stem}_ctx_{start_slot * 5:02d}",
                    "slot_label_indices": slot_label_indices,
                    "slot_mask": slot_mask,
                    "slot_row_ids": slot_row_ids,
                    "label_indices": sorted(union_labels),
                }
            )

    return pd.DataFrame(rows)


def _evaluate_fold_assignment(
    label_counts: np.ndarray,
    segment_counts: np.ndarray,
    file_counts: np.ndarray,
    site_fold_counts: np.ndarray,
    label_totals: np.ndarray,
    site_totals: np.ndarray,
) -> float:
    zero_penalty = 0.0
    spread_penalty = 0.0
    for class_idx, total in enumerate(label_totals):
        if total <= 0:
            continue
        fold_values = label_counts[:, class_idx]
        zero_penalty += float((fold_values == 0).sum()) / math.sqrt(float(total))
        spread_penalty += float(np.std(fold_values / float(total)))

    segment_penalty = float(np.std(segment_counts / max(segment_counts.mean(), 1.0)))
    file_penalty = float(np.std(file_counts / max(file_counts.mean(), 1.0)))

    site_penalty = 0.0
    for site_idx, total in enumerate(site_totals):
        if total <= 0:
            continue
        site_penalty += float(np.std(site_fold_counts[:, site_idx] / float(total)))

    return (100.0 * zero_penalty) + (25.0 * spread_penalty) + (2.0 * segment_penalty) + file_penalty + (0.5 * site_penalty)


def build_soundscape_folds(segment_df: pd.DataFrame, num_classes: int, n_folds: int, seed: int) -> pd.DataFrame:
    file_df = (
        segment_df.groupby("filename")
        .agg(
            site=("site", "first"),
            n_segments=("row_id", "count"),
            labels=("label_indices", lambda values: sorted({index for row in values for index in row})),
        )
        .reset_index()
    )

    label_matrix = np.zeros((len(file_df), num_classes), dtype=np.float32)
    for row_idx, indices in enumerate(file_df["labels"]):
        label_matrix[row_idx, indices] = 1.0

    label_totals = label_matrix.sum(axis=0)
    label_rarity = np.zeros_like(label_totals, dtype=np.float32)
    np.divide(1.0, label_totals, out=label_rarity, where=label_totals > 0)
    target_segments_per_fold = file_df["n_segments"].sum() / n_folds
    target_files_per_fold = len(file_df) / n_folds
    sites = sorted(file_df["site"].dropna().unique().tolist())
    site_to_idx = {site: idx for idx, site in enumerate(sites)}
    file_site_idx = file_df["site"].map(site_to_idx).to_numpy()
    site_totals = np.zeros(len(sites), dtype=np.float32)
    for idx in file_site_idx:
        site_totals[idx] += 1.0

    base_order_score = (label_matrix * label_rarity).sum(axis=1) + 0.05 * label_matrix.sum(axis=1)
    n_trials = 256 if len(file_df) >= 40 else 96
    rng = np.random.default_rng(seed)
    best_assignments = None
    best_objective = None

    for _ in range(n_trials):
        noise = rng.normal(0.0, 1e-3, size=len(file_df))
        order = np.argsort(-(base_order_score + noise))

        fold_label_counts = np.zeros((n_folds, num_classes), dtype=np.float32)
        fold_segment_counts = np.zeros(n_folds, dtype=np.float32)
        fold_file_counts = np.zeros(n_folds, dtype=np.float32)
        fold_site_counts = np.zeros((n_folds, len(sites)), dtype=np.float32)
        trial_assignments = np.full(len(file_df), -1, dtype=np.int64)

        for file_idx in order:
            label_indices = file_df.iloc[file_idx]["labels"]
            n_segments = float(file_df.iloc[file_idx]["n_segments"])
            site_idx = int(file_site_idx[file_idx])
            best_fold = None
            best_score = None

            for fold in range(n_folds):
                if label_indices:
                    coverage_gain = float(label_rarity[label_indices][fold_label_counts[fold, label_indices] == 0].sum())
                    label_balance = float(np.mean(fold_label_counts[fold, label_indices] / np.maximum(label_totals[label_indices], 1.0)))
                else:
                    coverage_gain = 0.0
                    label_balance = 0.0
                segment_balance = ((fold_segment_counts[fold] + n_segments) - target_segments_per_fold) / max(target_segments_per_fold, 1.0)
                file_balance = ((fold_file_counts[fold] + 1.0) - target_files_per_fold) / max(target_files_per_fold, 1.0)
                site_target = site_totals[site_idx] / n_folds
                site_balance = ((fold_site_counts[fold, site_idx] + 1.0) - site_target) / max(site_target, 1.0)
                score = (-40.0 * coverage_gain) + (4.0 * label_balance) + (0.2 * segment_balance ** 2) + (0.05 * file_balance ** 2) + (0.02 * site_balance ** 2)

                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold

            trial_assignments[file_idx] = int(best_fold)
            if label_indices:
                fold_label_counts[best_fold, label_indices] += 1.0
            fold_segment_counts[best_fold] += n_segments
            fold_file_counts[best_fold] += 1.0
            fold_site_counts[best_fold, site_idx] += 1.0

        objective = _evaluate_fold_assignment(
            label_counts=fold_label_counts,
            segment_counts=fold_segment_counts,
            file_counts=fold_file_counts,
            site_fold_counts=fold_site_counts,
            label_totals=label_totals,
            site_totals=site_totals,
        )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_assignments = trial_assignments.copy()

    assignments = {file_df.iloc[idx]["filename"]: int(best_assignments[idx]) for idx in range(len(file_df))}
    output = segment_df.copy()
    output["fold"] = output["filename"].map(assignments).astype(int)
    return output


def summarize_soundscape_folds(segment_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    summary_rows = []
    for fold, fold_df in segment_df.groupby("fold"):
        y = np.stack(fold_df["label_indices"].map(lambda x: indices_to_multihot(x, num_classes)).to_numpy())
        summary_rows.append(
            {
                "fold": int(fold),
                "files": fold_df["filename"].nunique(),
                "segments": len(fold_df),
                "sites": fold_df["site"].nunique(),
                "scored_classes": int((y.sum(axis=0) > 0).sum()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("fold").reset_index(drop=True)


@lru_cache(maxsize=65536)
def cached_sf_info(path: str):
    return sf.info(path)


def linear_resample(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    if len(audio) <= 1:
        return np.zeros(int(round(len(audio) * target_sr / max(original_sr, 1))), dtype=np.float32)

    duration = (len(audio) - 1) / float(original_sr)
    target_len = max(1, int(round(len(audio) * target_sr / float(original_sr))))
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=True, dtype=np.float64)
    new_times = np.linspace(0.0, duration, num=target_len, endpoint=True, dtype=np.float64)
    resampled = np.interp(new_times, old_times, audio).astype(np.float32)
    return resampled


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq / 700.0))


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    norm: str = "",
) -> torch.Tensor:
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, num=(n_fft // 2) + 1, dtype=np.float32)
    mel_edges = np.linspace(hz_to_mel(np.array([f_min], dtype=np.float32))[0], hz_to_mel(np.array([f_max], dtype=np.float32))[0], num=n_mels + 2, dtype=np.float32)
    hz_edges = mel_to_hz(mel_edges)

    filterbank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for mel_idx in range(n_mels):
        left = hz_edges[mel_idx]
        center = hz_edges[mel_idx + 1]
        right = hz_edges[mel_idx + 2]

        if center <= left or right <= center:
            continue

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)

        filterbank[mel_idx, left_mask] = (fft_freqs[left_mask] - left) / max(center - left, 1e-8)
        filterbank[mel_idx, right_mask] = (right - fft_freqs[right_mask]) / max(right - center, 1e-8)

    if str(norm).lower() == "slaney":
        enorm = 2.0 / np.maximum(hz_edges[2 : n_mels + 2] - hz_edges[:n_mels], 1e-8)
        filterbank *= enorm[:, np.newaxis].astype(np.float32)

    return torch.from_numpy(filterbank)


def power_to_db(spec: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    spec = torch.clamp(spec, min=1e-10)
    db = 10.0 * torch.log10(spec)
    max_db = db.amax()
    return torch.clamp(db, min=max_db - top_db)


def pcen_transform(
    spec: torch.Tensor,
    eps: float = 1e-6,
    alpha: float = 0.98,
    delta: float = 2.0,
    root: float = 0.5,
    smooth_kernel: int = 31,
) -> torch.Tensor:
    """Apply a lightweight PCEN-style dynamic range compression along time."""

    spec = torch.clamp(spec, min=0.0)
    time_steps = int(spec.shape[-1])
    kernel = min(int(smooth_kernel), time_steps if time_steps % 2 == 1 else max(time_steps - 1, 1))
    if kernel <= 1:
        smooth = spec
    else:
        pad = kernel // 2
        original_shape = spec.shape
        x = spec.reshape(-1, original_shape[-2], original_shape[-1])
        x = F.pad(x, (pad, pad), mode="replicate")
        smooth = F.avg_pool1d(x, kernel_size=kernel, stride=1)
        smooth = smooth.reshape(original_shape)
    return (spec / torch.pow(eps + smooth, alpha) + delta).pow(root) - (delta**root)


def apply_frequency_mask(image: torch.Tensor, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return image
    width = random.randint(0, min(max_width, image.shape[1]))
    if width == 0:
        return image
    start = random.randint(0, image.shape[1] - width)
    image[:, start : start + width, :] = image.mean()
    return image


def apply_time_mask(image: torch.Tensor, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return image
    width = random.randint(0, min(max_width, image.shape[2]))
    if width == 0:
        return image
    start = random.randint(0, image.shape[2] - width)
    image[:, :, start : start + width] = image.mean()
    return image


def load_audio_clip(path: str, target_seconds: float, sample_rate: int, train_mode: bool, start_sec: float = None) -> np.ndarray:
    info = cached_sf_info(path)
    native_sr = info.samplerate
    target_native_frames = int(round(target_seconds * native_sr))

    if start_sec is None:
        if info.frames > target_native_frames:
            if train_mode:
                max_start = info.frames - target_native_frames
                start_frame = random.randint(0, max_start)
            else:
                start_frame = max(0, (info.frames - target_native_frames) // 2)
            stop_frame = start_frame + target_native_frames
            audio, sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float32")
        else:
            audio, sr = sf.read(path, dtype="float32")
    else:
        start_frame = max(0, int(round(start_sec * native_sr)))
        stop_frame = start_frame + target_native_frames
        audio, sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        audio = np.zeros(1, dtype=np.float32)

    if sr != sample_rate:
        audio = linear_resample(audio, sr, sample_rate)

    target_len = int(round(target_seconds * sample_rate))
    if len(audio) < target_len:
        repeats = int(math.ceil(target_len / max(len(audio), 1)))
        audio = np.tile(audio, repeats)[:target_len]
    elif len(audio) > target_len:
        if start_sec is None and train_mode:
            start = random.randint(0, len(audio) - target_len)
        else:
            start = 0
        audio = audio[start : start + target_len]

    return np.asarray(audio, dtype=np.float32)


def load_audio_window_with_padding(path: str, target_seconds: float, sample_rate: int, start_sec: float) -> np.ndarray:
    info = cached_sf_info(path)
    native_sr = info.samplerate
    target_native_frames = int(round(target_seconds * native_sr))
    requested_start = int(round(start_sec * native_sr))
    read_start = max(0, requested_start)
    read_stop = min(info.frames, requested_start + target_native_frames)

    if read_stop > read_start:
        audio, sr = sf.read(path, start=read_start, stop=read_stop, dtype="float32")
    else:
        audio = np.zeros(0, dtype=np.float32)
        sr = native_sr

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = linear_resample(audio, sr, sample_rate)

    target_len = int(round(target_seconds * sample_rate))
    left_pad = max(0, int(round(-min(float(start_sec), 0.0) * sample_rate)))
    out = np.zeros(target_len, dtype=np.float32)
    if len(audio) > 0 and left_pad < target_len:
        copy_len = min(len(audio), target_len - left_pad)
        out[left_pad:left_pad + copy_len] = audio[:copy_len]
    return out


def augment_waveform(audio: np.ndarray) -> np.ndarray:
    gain = random.uniform(0.8, 1.2)
    audio = audio * gain
    if random.random() < 0.3:
        noise_scale = random.uniform(0.001, 0.01) * max(float(audio.std()), 1e-4)
        audio = audio + np.random.normal(0.0, noise_scale, size=audio.shape).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def sample_mix_lambda(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def choose_mix_mode(mixup_prob: float, cutmix_prob: float) -> Optional[str]:
    candidates = []
    if mixup_prob > 0:
        candidates.append(("mixup", float(mixup_prob)))
    if cutmix_prob > 0:
        candidates.append(("cutmix", float(cutmix_prob)))
    if not candidates:
        return None

    total_prob = sum(weight for _, weight in candidates)
    if total_prob <= 0:
        return None
    if random.random() >= min(total_prob, 1.0):
        return None

    draw = random.random() * total_prob
    running = 0.0
    for mode, weight in candidates:
        running += weight
        if draw <= running:
            return mode
    return candidates[-1][0]


def sample_cutmix_box(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h = max(1, int(round(height * cut_ratio)))
    cut_w = max(1, int(round(width * cut_ratio)))
    center_y = random.randint(0, height - 1)
    center_x = random.randint(0, width - 1)

    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)
    return y1, y2, x1, x2


def apply_batch_mix_augmentation(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    mixup_prob: float,
    cutmix_alpha: float,
    cutmix_prob: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
    if images.ndim != 4 or images.size(0) < 2:
        return images, targets, sample_weights, None

    mode = choose_mix_mode(mixup_prob=mixup_prob, cutmix_prob=cutmix_prob)
    if mode is None:
        return images, targets, sample_weights, None

    perm = torch.randperm(images.size(0), device=images.device)
    partner_images = images[perm]
    partner_targets = targets[perm]

    if mode == "mixup":
        lam = sample_mix_lambda(mixup_alpha)
        mixed_images = (images * lam) + (partner_images * (1.0 - lam))
    else:
        lam = sample_mix_lambda(cutmix_alpha)
        _, _, height, width = images.shape
        y1, y2, x1, x2 = sample_cutmix_box(height=height, width=width, lam=lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = partner_images[:, :, y1:y2, x1:x2]
        cut_area = float(max(y2 - y1, 0) * max(x2 - x1, 0))
        lam = 1.0 - (cut_area / float(height * width))

    mixed_targets = (targets * lam) + (partner_targets * (1.0 - lam))

    mixed_weights = sample_weights
    if sample_weights is not None:
        mixed_weights = (sample_weights * lam) + (sample_weights[perm] * (1.0 - lam))

    return mixed_images, mixed_targets, mixed_weights, {"mode": mode, "lambda": float(lam)}


class SpectrogramRenderer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.top_db = 80.0
        self.spectrogram_variant = str(cfg.spectrogram_variant).lower()
        if self.spectrogram_variant not in {"logmel", "pcen", "logmel_v8"}:
            raise ValueError(f"Unsupported spectrogram_variant: {cfg.spectrogram_variant}")
        if self.spectrogram_variant == "logmel_v8":
            self.transforms = [
                self._build_spec(
                    n_fft=2048,
                    hop_length=313,
                    n_mels=256,
                    f_min=20.0,
                    f_max=16000.0,
                    win_length=626,
                    mel_norm="slaney",
                )
            ]
        else:
            self.transforms = [
                self._build_spec(n_fft=1024, hop_length=256, n_mels=128, f_min=20.0, f_max=16000.0),
                self._build_spec(n_fft=2048, hop_length=512, n_mels=128, f_min=20.0, f_max=12000.0),
                self._build_spec(n_fft=4096, hop_length=1024, n_mels=128, f_min=20.0, f_max=8000.0),
            ]

    def _build_spec(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        win_length: Optional[int] = None,
        mel_norm: str = "",
    ) -> Dict[str, torch.Tensor]:
        win_length = int(win_length or n_fft)
        return {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": torch.hann_window(win_length),
            "mel_filter": build_mel_filterbank(
                sample_rate=self.cfg.sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                norm=mel_norm,
            ),
        }

    def _mel_spectrogram(self, waveform: torch.Tensor, spec_cfg: Dict[str, torch.Tensor]) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=int(spec_cfg["n_fft"]),
            hop_length=int(spec_cfg["hop_length"]),
            win_length=int(spec_cfg.get("win_length", spec_cfg["n_fft"])),
            window=spec_cfg["window"],
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power_spec = stft.abs().pow(2.0)
        mel_spec = torch.matmul(spec_cfg["mel_filter"], power_spec)
        if self.spectrogram_variant == "pcen":
            return pcen_transform(mel_spec)
        return power_to_db(mel_spec, top_db=self.top_db)

    def __call__(self, waveform: np.ndarray, train_mode: bool) -> torch.Tensor:
        x = torch.from_numpy(waveform).float()
        channels = []
        for transform in self.transforms:
            mel = self._mel_spectrogram(x, transform)
            mel = mel.unsqueeze(0).unsqueeze(0)
            mel = F.interpolate(
                mel,
                size=(self.cfg.image_height, self.cfg.image_width),
                mode="bilinear",
                align_corners=False,
            )
            channels.append(mel.squeeze(0))
        image = torch.cat(channels, dim=0)
        if self.cfg.input_channels == 1 and image.shape[0] != 1:
            image = image.mean(dim=0, keepdim=True)
        elif self.cfg.input_channels == 3 and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if train_mode:
            if random.random() < 0.75:
                image = apply_frequency_mask(image, self.cfg.specaug_freq_mask)
            if random.random() < 0.75:
                image = apply_time_mask(image, self.cfg.specaug_time_mask)
        image = image - image.amin(dim=(-2, -1), keepdim=True)
        image = image / (image.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        if self.cfg.image_normalize == "minus_one_one":
            image = (image - 0.5) / 0.5
        return image


class TrainAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        renderer: SpectrogramRenderer,
        num_classes: int,
        train_mode: bool,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        mixup_domain: str = "image",
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.mixup_domain = str(mixup_domain)

    def _load_row_audio(self, row: pd.Series) -> np.ndarray:
        audio = load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=self.train_mode,
        )
        if self.train_mode:
            audio = augment_waveform(audio)
        return np.asarray(audio, dtype=np.float32)

    def _maybe_waveform_mixup(
        self,
        idx: int,
        audio: np.ndarray,
        target: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, bool]:
        if not self.train_mode or self.mixup_domain != "waveform":
            return audio, target, False
        if self.mixup_prob <= 0 or self.mixup_alpha <= 0 or len(self.df) < 2:
            return audio, target, False
        if random.random() >= min(self.mixup_prob, 1.0):
            return audio, target, False

        lam = sample_mix_lambda(self.mixup_alpha)
        if lam >= 1.0:
            return audio, target, False

        partner_idx = random.randrange(len(self.df) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner_row = self.df.iloc[partner_idx]
        partner_audio = self._load_row_audio(partner_row)
        partner_target = torch.from_numpy(indices_to_multihot(partner_row["label_indices"], self.num_classes)).float()

        mixed_audio = (audio * lam) + (partner_audio * (1.0 - lam))
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)
        mixed_target = (target * lam) + (partner_target * (1.0 - lam))
        return mixed_audio, mixed_target, True

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = self._load_row_audio(row)
        target = torch.from_numpy(indices_to_multihot(row["label_indices"], self.num_classes)).float()
        audio, target, waveform_mixup_applied = self._maybe_waveform_mixup(idx=idx, audio=audio, target=target)
        image = self.renderer(audio, train_mode=self.train_mode)
        item = {
            "image": image,
            "target": target,
            "row_id": row["filename"],
            "waveform_mixup_applied": torch.tensor(waveform_mixup_applied, dtype=torch.bool),
        }
        if self.cfg.use_waveform_branch:
            item["waveform"] = torch.from_numpy(audio).float()
        return item


class SoundscapeSegmentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        renderer: SpectrogramRenderer,
        num_classes: int,
        train_mode: bool,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        mixup_domain: str = "image",
        perch_teacher_tokens: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.mixup_domain = str(mixup_domain)
        self.perch_teacher_tokens = perch_teacher_tokens

    def _load_row_audio(self, row: pd.Series) -> np.ndarray:
        audio = load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(row["start_sec"]),
        )
        if self.train_mode:
            audio = augment_waveform(audio)
        return np.asarray(audio, dtype=np.float32)

    def _maybe_waveform_mixup(
        self,
        idx: int,
        audio: np.ndarray,
        target: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, bool]:
        if not self.train_mode or self.mixup_domain != "waveform":
            return audio, target, False
        if self.mixup_prob <= 0 or self.mixup_alpha <= 0 or len(self.df) < 2:
            return audio, target, False
        if random.random() >= min(self.mixup_prob, 1.0):
            return audio, target, False

        lam = sample_mix_lambda(self.mixup_alpha)
        if lam >= 1.0:
            return audio, target, False

        partner_idx = random.randrange(len(self.df) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner_row = self.df.iloc[partner_idx]
        partner_audio = self._load_row_audio(partner_row)
        partner_target = torch.from_numpy(indices_to_multihot(partner_row["label_indices"], self.num_classes)).float()

        mixed_audio = (audio * lam) + (partner_audio * (1.0 - lam))
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)
        mixed_target = (target * lam) + (partner_target * (1.0 - lam))
        return mixed_audio, mixed_target, True

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = self._load_row_audio(row)
        target = torch.from_numpy(indices_to_multihot(row["label_indices"], self.num_classes)).float()
        audio, target, waveform_mixup_applied = self._maybe_waveform_mixup(idx=idx, audio=audio, target=target)
        image = self.renderer(audio, train_mode=self.train_mode)
        item = {
            "image": image,
            "target": target,
            "row_id": row["row_id"],
            "site": row["site"],
            "waveform_mixup_applied": torch.tensor(waveform_mixup_applied, dtype=torch.bool),
        }
        if self.perch_teacher_tokens is not None:
            item["perch_teacher_tokens"] = torch.from_numpy(self.perch_teacher_tokens[idx]).float()
        if self.cfg.use_waveform_branch:
            item["waveform"] = torch.from_numpy(audio).float()
        return item


class SoundscapeCenterContextDataset(SoundscapeSegmentDataset):
    def _load_row_audio(self, row: pd.Series) -> np.ndarray:
        context_seconds = float(self.cfg.clip_seconds)
        center_start_sec = float(row["start_sec"])
        start_sec = center_start_sec - max((context_seconds - 5.0) * 0.5, 0.0)
        audio = load_audio_window_with_padding(
            path=row["audio_path"],
            target_seconds=context_seconds,
            sample_rate=self.cfg.sample_rate,
            start_sec=start_sec,
        )
        if self.train_mode:
            audio = augment_waveform(audio)
        return np.asarray(audio, dtype=np.float32)


class SoundscapeLongContextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        renderer: SpectrogramRenderer,
        num_classes: int,
        train_mode: bool,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        mixup_domain: str = "image",
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.mixup_domain = str(mixup_domain)

    def _load_row_audio(self, row: pd.Series) -> np.ndarray:
        audio = load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(row["window_start_sec"]),
        )
        if self.train_mode:
            audio = augment_waveform(audio)
        return np.asarray(audio, dtype=np.float32)

    def _build_slot_target(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        targets = np.zeros((self.cfg.multi_context_num_slots, self.num_classes), dtype=np.float32)
        masks = np.zeros(self.cfg.multi_context_num_slots, dtype=np.float32)
        for slot, label_indices in enumerate(row["slot_label_indices"]):
            if slot >= self.cfg.multi_context_num_slots:
                break
            if bool(row["slot_mask"][slot]):
                masks[slot] = 1.0
            if label_indices:
                targets[slot, np.asarray(label_indices, dtype=np.int64)] = 1.0
        return torch.from_numpy(targets).float(), torch.from_numpy(masks).float()

    def _maybe_waveform_mixup(
        self,
        idx: int,
        audio: np.ndarray,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, bool]:
        if not self.train_mode or self.mixup_domain != "waveform":
            return audio, target, target_mask, False
        if self.mixup_prob <= 0 or self.mixup_alpha <= 0 or len(self.df) < 2:
            return audio, target, target_mask, False
        if random.random() >= min(self.mixup_prob, 1.0):
            return audio, target, target_mask, False

        lam = sample_mix_lambda(self.mixup_alpha)
        if lam >= 1.0:
            return audio, target, target_mask, False

        partner_idx = random.randrange(len(self.df) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner_row = self.df.iloc[partner_idx]
        partner_audio = self._load_row_audio(partner_row)
        partner_target, partner_mask = self._build_slot_target(partner_row)

        mixed_audio = (audio * lam) + (partner_audio * (1.0 - lam))
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)
        mixed_target = (target * lam) + (partner_target * (1.0 - lam))
        mixed_mask = torch.maximum(target_mask, partner_mask)
        return mixed_audio, mixed_target, mixed_mask, True

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = self._load_row_audio(row)
        target, target_mask = self._build_slot_target(row)
        audio, target, target_mask, waveform_mixup_applied = self._maybe_waveform_mixup(
            idx=idx,
            audio=audio,
            target=target,
            target_mask=target_mask,
        )
        image = self.renderer(audio, train_mode=self.train_mode)
        item = {
            "image": image,
            "target": target,
            "target_mask": target_mask,
            "row_id": row["window_id"],
            "slot_row_ids": "|".join(str(value) for value in row["slot_row_ids"]),
            "site": row["site"],
            "waveform_mixup_applied": torch.tensor(waveform_mixup_applied, dtype=torch.bool),
        }
        if self.cfg.use_waveform_branch:
            item["waveform"] = torch.from_numpy(audio).float()
        return item


def build_train_audio_sampler(
    df: pd.DataFrame,
    samples_per_epoch: int,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    primary_counts = df["primary_label"].value_counts().to_dict()
    weights = df["primary_label"].map(lambda x: 1.0 / math.sqrt(primary_counts[x])).astype(np.float64).to_numpy()
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )


def build_soundscape_sampler(
    df: pd.DataFrame,
    num_classes: int,
    samples_per_epoch: int,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for indices in df["label_indices"]:
        if indices:
            class_counts[np.asarray(indices, dtype=np.int64)] += 1.0

    weights = []
    for indices in df["label_indices"]:
        if indices:
            label_counts = class_counts[np.asarray(indices, dtype=np.int64)]
            weight = float(np.max(1.0 / np.sqrt(np.maximum(label_counts, 1.0))))
        else:
            weight = 1.0
        weights.append(weight)
    weights = np.asarray(weights, dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = False,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(seed),
    )


class WaveformTransformerBranch(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, dropout: float, max_tokens: int = 512):
        super().__init__()
        stem_hidden = max(d_model // 2, 32)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_hidden, kernel_size=321, stride=160, padding=160, bias=False),
            nn.GroupNorm(1, stem_hidden),
            nn.GELU(),
            nn.Conv1d(stem_hidden, d_model, kernel_size=5, stride=4, padding=2, bias=False),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        self.pool = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _position_embedding(self, tokens: int) -> torch.Tensor:
        if tokens <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :tokens]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.ndim != 3:
            raise ValueError(f"Unsupported waveform tensor shape: {tuple(waveform.shape)}")
        x = self.stem(waveform)
        x = x.transpose(1, 2)
        x = x + self._position_embedding(x.shape[1]).to(dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        return self.pool(torch.cat([mean_pool, max_pool], dim=1))


class BirdCLEFNet(nn.Module):
    def __init__(self, cfg: Config, num_classes: int, backbone_weight_path: Path):
        super().__init__()
        self.head_type = str(cfg.head_type)
        self.use_waveform_branch = bool(cfg.use_waveform_branch)
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=int(cfg.input_channels),
            num_classes=0,
            global_pool="avg" if self.head_type == "linear" else "",
            drop_path_rate=cfg.drop_path,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        if self.head_type == "linear":
            self.head = nn.Linear(self.backbone.num_features, num_classes)
        elif self.head_type == "csiro_conv_v1":
            self.head = CSIROHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=cfg.dropout,
                pool_type=cfg.head_pool_type,
            )
        elif self.head_type == "lse_head_v1":
            self.head = LSEHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=cfg.dropout,
                temperature=cfg.lse_temperature,
            )
        elif self.head_type == "csiro_multicontext_v1":
            self.head = CSIROMultiContextHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=cfg.dropout,
                num_slots=cfg.multi_context_num_slots,
                pool_type=cfg.head_pool_type,
            )
        elif self.head_type == "sed_att_v1":
            self.head = SEDAttentionHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")
        if self.use_waveform_branch:
            self.waveform_branch = WaveformTransformerBranch(
                d_model=cfg.waveform_branch_d_model,
                num_layers=cfg.waveform_branch_layers,
                num_heads=cfg.waveform_branch_heads,
                dropout=cfg.waveform_branch_dropout,
            )
            self.waveform_projection = nn.Linear(cfg.waveform_branch_d_model, self.backbone.num_features)
            self.waveform_gate = nn.Sequential(
                nn.LayerNorm(self.backbone.num_features * 2),
                nn.Linear(self.backbone.num_features * 2, self.backbone.num_features),
                nn.Sigmoid(),
            )
        self.use_perch_distill = bool(getattr(cfg, "use_perch_distill", False))
        self.perch_distill_token_key = str(getattr(cfg, "perch_distill_token_key", "spatial_tokens"))
        self.perch_distill_proj: Optional[nn.Module] = None
        if self.use_perch_distill:
            self.perch_distill_proj = nn.Linear(self.backbone.num_features, 1536)
        self._load_backbone_weights(backbone_weight_path)

    def _load_backbone_weights(self, path: Path) -> None:
        if not path.exists():
            print(f"[WARN] Backbone weight file not found: {path}. Continuing with timm init.")
            return
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ["state_dict", "model", "net"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        state = {key.replace("module.", ""): value for key, value in state.items()}
        model_state = self.backbone.state_dict()
        adapted_keys = []
        for key, value in list(state.items()):
            if key not in model_state:
                continue
            target = model_state[key]
            if (
                isinstance(value, torch.Tensor)
                and isinstance(target, torch.Tensor)
                and value.ndim == 4
                and target.ndim == 4
                and value.shape[0] == target.shape[0]
                and value.shape[2:] == target.shape[2:]
                and value.shape[1] == 3
                and target.shape[1] == 1
            ):
                # Match timm's grayscale adaptation: summing RGB filters makes a
                # single-channel input equivalent to feeding the same image to all
                # three RGB channels.
                state[key] = value.sum(dim=1, keepdim=True)
                adapted_keys.append(key)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if adapted_keys:
            print(f"[INFO] Adapted RGB backbone conv weights to 1ch: {adapted_keys[:3]}")
        print(f"[INFO] Loaded backbone weights from {path.name} | missing={len(missing)} unexpected={len(unexpected)}")

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            features = features[-1]
        if isinstance(features, dict):
            for key in ["x", "feat", "features", "last_hidden_state"]:
                if key in features:
                    features = features[key]
                    break
            else:
                raise TypeError(f"Unsupported backbone feature dict keys: {list(features.keys())}")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Unsupported backbone feature type: {type(features)}")
        return features

    def _flatten_feature_sequence(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3:
            if features.shape[-1] == self.backbone.num_features:
                return features
            if features.shape[1] == self.backbone.num_features:
                return features.transpose(1, 2)
            raise ValueError(f"Unsupported 3D backbone feature shape: {tuple(features.shape)}")
        if features.ndim == 2:
            return features.unsqueeze(1)
        raise ValueError(f"Unsupported backbone feature shape: {tuple(features.shape)}")

    def _time_feature_sequence(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if features.shape[1] == self.backbone.num_features:
                return features.mean(dim=2).transpose(1, 2)
            if features.shape[-1] == self.backbone.num_features:
                return features.mean(dim=1)
            raise ValueError(f"Unsupported 4D backbone feature shape: {tuple(features.shape)}")
        return self._flatten_feature_sequence(features)

    def _fuse_waveform_feature(self, features: torch.Tensor, waveform: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_waveform_branch:
            return features
        if waveform is None:
            raise ValueError("Waveform branch is enabled, but the batch does not contain a waveform tensor.")

        time_feature = self.waveform_projection(self.waveform_branch(waveform))
        if features.ndim == 3:
            freq_context = features.mean(dim=1)
            gate = self.waveform_gate(torch.cat([freq_context, time_feature], dim=1))
            return features + gate.unsqueeze(1) * time_feature.unsqueeze(1)
        if features.ndim == 2:
            gate = self.waveform_gate(torch.cat([features, time_feature], dim=1))
            return features + gate * time_feature
        raise ValueError(f"Unsupported feature shape for waveform fusion: {tuple(features.shape)}")

    def project_cnn_token_features(self, feature_sequence: torch.Tensor) -> torch.Tensor:
        if self.perch_distill_proj is None:
            raise RuntimeError("Perch distillation projector is not enabled.")
        if feature_sequence.ndim == 4:
            token_sequence = self._flatten_feature_sequence(feature_sequence)
        elif feature_sequence.ndim == 3:
            token_sequence = feature_sequence
        elif feature_sequence.ndim == 2:
            token_sequence = feature_sequence.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported feature shape for distillation: {tuple(feature_sequence.shape)}")
        if token_sequence.shape[-1] == self.backbone.num_features:
            return self.perch_distill_proj(token_sequence)
        if token_sequence.shape[1] == self.backbone.num_features:
            return self.perch_distill_proj(token_sequence.transpose(1, 2))
        raise ValueError(f"Unexpected CNN token shape for distillation: {tuple(token_sequence.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        waveform: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        if self.head_type == "linear":
            features = self.backbone(x)
            features = self._fuse_waveform_feature(features, waveform)
            logits = self.head(self.dropout(features))
            if not return_features:
                return logits
            feature_sequence = self._flatten_feature_sequence(self._forward_backbone_features(x))
            feature_sequence = self._fuse_waveform_feature(feature_sequence, waveform)
            return logits, feature_sequence

        features = self._forward_backbone_features(x)
        if self.head_type in {"csiro_multicontext_v1", "sed_att_v1"}:
            feature_sequence = self._time_feature_sequence(features)
        else:
            feature_sequence = self._flatten_feature_sequence(features)
        feature_sequence = self._fuse_waveform_feature(feature_sequence, waveform)
        logits = self.head(feature_sequence)
        if return_features:
            return logits, feature_sequence
        return logits


class LocalSequenceBlock(nn.Module):
    def __init__(self, in_features: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(in_features)
        self.pointwise_in = nn.Linear(in_features, in_features)
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_features,
            bias=True,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.pointwise_out = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pointwise_in(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pointwise_out(x)
        return residual + x


class SequencePooling(nn.Module):
    def __init__(self, pool_type: str, p: float = 3.0, lse_temperature: float = 1.0):
        super().__init__()
        self.pool_type = str(pool_type)
        self.p = float(p)
        self.lse_temperature = float(lse_temperature)
        if self.pool_type not in {"avg", "gem", "lse", "avg_max"}:
            raise ValueError(f"Unsupported head_pool_type: {self.pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return x.mean(dim=1)
        if self.pool_type == "avg_max":
            return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
        if self.pool_type == "lse":
            return torch.logsumexp(x / self.lse_temperature, dim=1) * self.lse_temperature - math.log(x.shape[1])
        shifted = x - x.amin(dim=1, keepdim=True)
        pooled = shifted.clamp_min(1e-6).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return pooled + x.amin(dim=1)


class CSIROHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float, pool_type: str = "avg"):
        super().__init__()
        hidden_features = max(in_features // 2, 64)
        self.fusion = nn.Sequential(
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
        )
        self.pool = SequencePooling(pool_type=pool_type)
        self.out_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)
        x = self.pool(x)
        return self.out_head(x)


class LSEHead(nn.Module):
    """Lightweight LSE pooling head used by the HGNet V8-style experiment."""

    def __init__(self, in_features: int, num_classes: int, dropout: float, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.logsumexp(x / self.temperature, dim=1) * self.temperature - math.log(x.shape[1])
        return self.classifier(self.dropout(pooled))


class SEDAttentionHead(nn.Module):
    """Framewise SED head with per-class attention pooling."""

    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        hidden_features = max(in_features // 2, 64)
        self.fusion = nn.Sequential(
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
        )
        self.frame_head = nn.Linear(hidden_features, num_classes)
        self.attention_head = nn.Linear(hidden_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.fusion(x)
        x = self.proj(x)
        framewise_logits = self.frame_head(x)
        attention_logits = torch.clamp(self.attention_head(x), -10.0, 10.0)
        attention = torch.softmax(attention_logits, dim=1)
        clipwise_logits = torch.sum(framewise_logits * attention, dim=1)
        return {
            "clipwise_logits": clipwise_logits,
            "framewise_logits": framewise_logits,
            "attention": attention,
        }


class CSIROMultiContextHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float, num_slots: int, pool_type: str = "avg"):
        super().__init__()
        self.num_slots = int(num_slots)
        hidden_features = max(in_features // 2, 64)
        self.fusion = nn.Sequential(
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
        )
        self.global_pool = SequencePooling(pool_type=pool_type)
        self.slot_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )
        self.global_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def _slot_pool(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.shape[1]
        pooled = []
        for slot in range(self.num_slots):
            start = int(math.floor(slot * n_tokens / self.num_slots))
            end = int(math.floor((slot + 1) * n_tokens / self.num_slots))
            end = max(end, start + 1)
            pooled.append(x[:, start:end].mean(dim=1))
        return torch.stack(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.fusion(x)
        slot_features = self._slot_pool(x)
        global_features = self.global_pool(x)
        return {
            "slot_logits": self.slot_head(slot_features),
            "global_logits": self.global_head(global_features),
        }


def set_backbone_trainable(model: BirdCLEFNet, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def build_optimizer(model: BirdCLEFNet, backbone_lr: float, head_lr: float, weight_decay: float):
    head_params = list(model.head.parameters())
    if getattr(model, "use_waveform_branch", False):
        head_params.extend(model.waveform_branch.parameters())
        head_params.extend(model.waveform_projection.parameters())
        head_params.extend(model.waveform_gate.parameters())
    if getattr(model, "perch_distill_proj", None) is not None:
        head_params.extend(model.perch_distill_proj.parameters())
    return AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int, scheduler_type: str = "linear_cosine"):
    total_steps = max(steps_per_epoch * epochs, 1)
    if str(scheduler_type).lower() == "onecycle":
        pct_start = min(max(float(warmup_epochs) / max(float(epochs), 1.0), 1e-3), 0.95)
        max_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        return OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )
    warmup_steps = min(max(steps_per_epoch * warmup_epochs, 1), total_steps - 1) if total_steps > 1 else 1
    cosine_steps = max(total_steps - warmup_steps, 1)
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return scheduler


def model_metric_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "clipwise_logits" in outputs:
            return outputs["clipwise_logits"]
        if "slot_logits" in outputs:
            return outputs["slot_logits"]
        return outputs["global_logits"]
    return outputs


def compute_training_loss(
    outputs,
    targets: torch.Tensor,
    criterion: nn.Module,
    target_mask: Optional[torch.Tensor] = None,
    global_loss_weight: float = 0.0,
    sed_frame_loss_weight: float = 0.5,
) -> torch.Tensor:
    if not isinstance(outputs, dict):
        return criterion(outputs, targets)

    if "clipwise_logits" in outputs and "framewise_logits" in outputs:
        clip_loss = criterion(outputs["clipwise_logits"], targets)
        frame_logits = outputs["framewise_logits"].amax(dim=1)
        frame_loss = criterion(frame_logits, targets)
        weight = float(sed_frame_loss_weight)
        return (clip_loss * (1.0 - weight)) + (frame_loss * weight)

    global_logits = outputs["global_logits"]
    if targets.ndim == 2:
        return criterion(global_logits, targets)

    slot_logits = outputs["slot_logits"]
    slot_loss_raw = F.binary_cross_entropy_with_logits(slot_logits, targets, reduction="none")
    if target_mask is not None:
        mask = target_mask.to(dtype=slot_loss_raw.dtype, device=slot_loss_raw.device).unsqueeze(-1)
        denom = torch.clamp(mask.sum() * targets.shape[-1], min=1.0)
        slot_loss = (slot_loss_raw * mask).sum() / denom
        global_target = (targets * mask).amax(dim=1)
    else:
        slot_loss = slot_loss_raw.mean()
        global_target = targets.amax(dim=1)

    if global_loss_weight <= 0:
        return slot_loss
    global_loss = criterion(global_logits, global_target)
    return slot_loss + (float(global_loss_weight) * global_loss)


def pool_token_sequence(sequence: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(1)
    if sequence.ndim != 3:
        raise ValueError(f"Expected token sequence [B,T,C], got {tuple(sequence.shape)}")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if sequence.shape[1] == target_tokens:
        return sequence
    sequence_t = sequence.transpose(1, 2)
    if sequence.shape[1] > target_tokens:
        pooled = F.adaptive_avg_pool1d(sequence_t, target_tokens)
    else:
        pooled = F.interpolate(sequence_t, size=target_tokens, mode="linear", align_corners=False)
    return pooled.transpose(1, 2)


def compute_perch_distill_loss(
    model: BirdCLEFNet,
    student_features: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> torch.Tensor:
    if not getattr(model, "use_perch_distill", False) or getattr(model, "perch_distill_proj", None) is None:
        raise RuntimeError("Perch distillation is not enabled on this model.")
    student_tokens = model.project_cnn_token_features(student_features)
    teacher_tokens = teacher_tokens.to(device=student_tokens.device, dtype=student_tokens.dtype, non_blocking=True)
    student_tokens = pool_token_sequence(student_tokens, int(teacher_tokens.shape[1]))
    student_tokens = F.normalize(student_tokens, dim=-1)
    teacher_tokens = F.normalize(teacher_tokens, dim=-1)
    return F.mse_loss(student_tokens, teacher_tokens)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    train_mode: bool,
    scaler: GradScaler,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    grad_clip_norm: float,
    progress_desc: str,
    mixup_domain: str = "image",
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    global_loss_weight: float = 0.0,
    sed_frame_loss_weight: float = 0.5,
    perch_distill_weight: float = 0.0,
):
    model.train(train_mode)
    running_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []
    row_ids = []
    sites = []

    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        desc=progress_desc,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    skipped_batches = 0
    waveform_mixup_batches = 0
    waveform_mixup_samples = 0
    mixup_batches = 0
    cutmix_batches = 0
    logit_sum = 0.0
    logit_sq_sum = 0.0
    logit_count = 0
    distill_loss_sum = 0.0
    distill_batches = 0
    for batch_idx, batch in enumerate(progress, start=1):
        batch_size = len(batch["image"])
        images = batch["image"].to(device, non_blocking=True)
        waveforms = batch["waveform"].to(device, non_blocking=True) if "waveform" in batch else None
        targets = batch["target"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True) if "target_mask" in batch else None

        if train_mode:
            if "waveform_mixup_applied" in batch:
                waveform_mixup_tensor = batch["waveform_mixup_applied"]
                if isinstance(waveform_mixup_tensor, torch.Tensor):
                    waveform_mixup_count = int(waveform_mixup_tensor.sum().item())
                else:
                    waveform_mixup_count = int(sum(bool(value) for value in waveform_mixup_tensor))
                waveform_mixup_samples += waveform_mixup_count
                if waveform_mixup_count > 0:
                    waveform_mixup_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if target_mask is None:
                images, targets, _, mix_info = apply_batch_mix_augmentation(
                    images=images,
                    targets=targets,
                    mixup_alpha=mixup_alpha,
                    mixup_prob=mixup_prob if mixup_domain != "waveform" else 0.0,
                    cutmix_alpha=cutmix_alpha,
                    cutmix_prob=cutmix_prob,
                )
                if mix_info is not None:
                    if mix_info["mode"] == "mixup":
                        mixup_batches += 1
                    elif mix_info["mode"] == "cutmix":
                        cutmix_batches += 1

        grad_context = torch.enable_grad() if train_mode else torch.inference_mode()
        with grad_context:
            with maybe_autocast(enabled=amp_enabled, amp_dtype=amp_dtype):
                output = model(images, waveform=waveforms, return_features=perch_distill_weight > 0 and train_mode)
                if isinstance(output, tuple):
                    logits, feature_sequence = output
                else:
                    logits = output
                    feature_sequence = None
                loss = compute_training_loss(
                    outputs=logits,
                    targets=targets,
                    criterion=criterion,
                    target_mask=target_mask,
                    global_loss_weight=global_loss_weight,
                    sed_frame_loss_weight=sed_frame_loss_weight,
                )
                if train_mode and perch_distill_weight > 0 and feature_sequence is not None and "perch_teacher_tokens" in batch:
                    distill_loss = compute_perch_distill_loss(
                        model=model,
                        student_features=feature_sequence,
                        teacher_tokens=batch["perch_teacher_tokens"].to(device, non_blocking=True),
                    )
                    distill_loss_sum += float(distill_loss.item()) * batch_size
                    distill_batches += 1
                    loss = loss + (float(perch_distill_weight) * distill_loss)

        if not torch.isfinite(loss).all():
            if train_mode:
                skipped_batches += 1
                backoff_grad_scaler(scaler)
                if skipped_batches <= 3 or skipped_batches % 20 == 0:
                    print(
                        f"[WARN] {progress_desc}: non-finite loss at batch {batch_idx}. "
                        f"Skipping batch and reducing AMP scale."
                    )
                postfix = {"loss": f"{running_loss / max(sample_count, 1):.4f}"}
                if skipped_batches > 0:
                    postfix["skipped"] = skipped_batches
                progress.set_postfix(**postfix)
                del images, waveforms, targets, target_mask, logits, feature_sequence, output, loss
                continue
            raise FloatingPointError(f"{progress_desc}: encountered non-finite validation loss at batch {batch_idx}.")

        step_taken = False
        if train_mode:
            if scaler_is_enabled(scaler):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grads_finite = gradients_are_finite(model)
                if grads_finite:
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    step_taken = True
                else:
                    skipped_batches += 1
                    if skipped_batches <= 3 or skipped_batches % 20 == 0:
                        print(
                            f"[WARN] {progress_desc}: non-finite gradients at batch {batch_idx}. "
                            f"Skipping optimizer step so AMP can recover."
                        )
                scaler.update()
            else:
                loss.backward()
                grads_finite = gradients_are_finite(model)
                if not grads_finite:
                    raise FloatingPointError(
                        f"{progress_desc}: encountered non-finite gradients without AMP at batch {batch_idx}."
                    )
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                step_taken = True

            if step_taken and scheduler is not None:
                scheduler.step()

        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        metric_logits = model_metric_logits(logits).float().detach()
        logit_sum += float(metric_logits.sum().cpu())
        logit_sq_sum += float(metric_logits.square().sum().cpu())
        logit_count += int(metric_logits.numel())
        logit_mean = logit_sum / max(logit_count, 1)
        logit_var = max((logit_sq_sum / max(logit_count, 1)) - (logit_mean ** 2), 0.0)
        logit_std = math.sqrt(logit_var)
        postfix = {"loss": f"{running_loss / max(sample_count, 1):.4f}", "logit_std": f"{logit_std:.3f}"}
        if train_mode and skipped_batches > 0:
            postfix["skipped"] = skipped_batches
        if train_mode and distill_batches > 0:
            postfix["distill"] = f"{distill_loss_sum / max(sample_count, 1):.4f}"
        progress.set_postfix(**postfix)

        if not train_mode:
            if isinstance(logits, dict) and targets.ndim == 3:
                probs = torch.sigmoid(logits["slot_logits"].float()).detach().cpu().numpy()
                target_np = targets.detach().cpu().numpy()
                mask_np = target_mask.detach().cpu().numpy() if target_mask is not None else np.ones(target_np.shape[:2], dtype=np.float32)
                slot_row_ids = batch["slot_row_ids"]
                batch_sites = batch.get("site", ["unknown"] * len(slot_row_ids))
                for sample_idx, row_id_text in enumerate(slot_row_ids):
                    row_id_parts = str(row_id_text).split("|")
                    for slot_idx, slot_row_id in enumerate(row_id_parts[: probs.shape[1]]):
                        if mask_np[sample_idx, slot_idx] <= 0:
                            continue
                        y_true.append(target_np[sample_idx, slot_idx][None, :])
                        y_pred.append(probs[sample_idx, slot_idx][None, :])
                        row_ids.append(slot_row_id)
                        sites.append(batch_sites[sample_idx] if sample_idx < len(batch_sites) else "unknown")
            else:
                if isinstance(logits, dict):
                    metric_output = model_metric_logits(logits)
                else:
                    metric_output = logits
                y_true.append(targets.detach().cpu().numpy())
                y_pred.append(torch.sigmoid(metric_output.float()).detach().cpu().numpy())
                row_ids.extend(batch["row_id"])
                if "site" in batch:
                    sites.extend(batch["site"])

        del images, waveforms, targets, target_mask, logits, feature_sequence, output, loss

    avg_loss = running_loss / max(sample_count, 1)
    result = {
        "loss": avg_loss,
        "y_true": None,
        "y_pred": None,
        "row_ids": row_ids,
        "sites": sites,
        "skipped_batches": skipped_batches,
        "waveform_mixup_batches": waveform_mixup_batches,
        "waveform_mixup_samples": waveform_mixup_samples,
        "mixup_batches": mixup_batches,
        "cutmix_batches": cutmix_batches,
        "logit_std": math.sqrt(max((logit_sq_sum / max(logit_count, 1)) - ((logit_sum / max(logit_count, 1)) ** 2), 0.0)),
        "distill_loss": distill_loss_sum / max(sample_count, 1) if distill_batches > 0 else 0.0,
    }
    if not train_mode and y_true:
        result["y_true"] = np.concatenate(y_true, axis=0)
        result["y_pred"] = np.concatenate(y_pred, axis=0)
    return result


def fit_one_stage(
    stage_name: str,
    model: BirdCLEFNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int,
    scheduler_type: str,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    use_grad_scaler: bool,
    grad_clip_norm: float,
    patience: int,
    freeze_backbone_epochs: int = 0,
    mixup_domain: str = "image",
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    mixup_start_epoch: int = 1,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    cutmix_start_epoch: int = 1,
    global_loss_weight: float = 0.0,
    sed_frame_loss_weight: float = 0.5,
    perch_distill_weight: float = 0.0,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        scheduler_type=scheduler_type,
    )
    scaler = GradScaler(enabled=use_grad_scaler)

    best_metric = -np.inf
    best_path = output_dir / f"{stage_name}_best.pth"
    history = []
    patience_left = patience

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=epoch > freeze_backbone_epochs)
        epoch_mixup_alpha = mixup_alpha if epoch >= int(mixup_start_epoch) else 0.0
        epoch_mixup_prob = mixup_prob if epoch >= int(mixup_start_epoch) else 0.0
        epoch_cutmix_alpha = cutmix_alpha if epoch >= int(cutmix_start_epoch) else 0.0
        epoch_cutmix_prob = cutmix_prob if epoch >= int(cutmix_start_epoch) else 0.0

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_mode=True,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip_norm=grad_clip_norm,
            progress_desc=f"{stage_name}/train/e{epoch:02d}",
            mixup_domain=mixup_domain,
            mixup_alpha=epoch_mixup_alpha,
            mixup_prob=epoch_mixup_prob,
            cutmix_alpha=epoch_cutmix_alpha,
            cutmix_prob=epoch_cutmix_prob,
            global_loss_weight=global_loss_weight,
            sed_frame_loss_weight=sed_frame_loss_weight,
            perch_distill_weight=perch_distill_weight,
        )
        valid_result = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            train_mode=False,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip_norm=grad_clip_norm,
            progress_desc=f"{stage_name}/valid/e{epoch:02d}",
            global_loss_weight=global_loss_weight,
            sed_frame_loss_weight=sed_frame_loss_weight,
        )
        valid_metric = macro_auc_skip_missing(valid_result["y_true"], valid_result["y_pred"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "valid_loss": valid_result["loss"],
                "valid_auc": valid_metric,
                "train_logit_std": train_result["logit_std"],
                "valid_logit_std": valid_result["logit_std"],
                "train_distill_loss": train_result["distill_loss"],
            }
        )
        print(
            f"[{stage_name}] epoch={epoch:02d} "
            f"train_loss={train_result['loss']:.4f} "
            f"valid_loss={valid_result['loss']:.4f} "
            f"valid_auc={valid_metric:.5f} "
            f"train_logit_std={train_result['logit_std']:.4f} "
            f"valid_logit_std={valid_result['logit_std']:.4f} "
            f"train_distill_loss={train_result['distill_loss']:.4f} "
            f"skipped_batches={train_result['skipped_batches']} "
            f"waveform_mixup_batches={train_result['waveform_mixup_batches']} "
            f"waveform_mixup_samples={train_result['waveform_mixup_samples']} "
            f"mixup_batches={train_result['mixup_batches']} "
            f"cutmix_batches={train_result['cutmix_batches']}"
        )

        if valid_metric > best_metric:
            best_metric = valid_metric
            patience_left = patience
            torch.save({"model": model.state_dict(), "history": history}, best_path)
            print(f"[{stage_name}] saved best checkpoint -> {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{stage_name}] early stopping triggered.")
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{stage_name}_history.csv", index=False)
    return model, best_path


def evaluate_soundscape_model(
    model: BirdCLEFNet,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    sed_frame_loss_weight: float = 0.5,
) -> pd.DataFrame:
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=False)
    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
        device=device,
        train_mode=False,
        scaler=scaler,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        grad_clip_norm=0.0,
        progress_desc="soundscape_eval",
        global_loss_weight=0.0,
        sed_frame_loss_weight=sed_frame_loss_weight,
    )
    pred_long = pd.DataFrame(result["y_pred"])
    pred_long.insert(0, "row_id", result["row_ids"])
    pred_long["site"] = result["sites"]
    target_long = pd.DataFrame(result["y_true"])
    target_long.insert(0, "row_id", result["row_ids"])
    target_long["site"] = result["sites"]
    prediction_df = pred_long.groupby("row_id", sort=False).mean(numeric_only=True).reset_index()
    site_map = pred_long.groupby("row_id", sort=False)["site"].first()
    prediction_df["site"] = prediction_df["row_id"].map(site_map)
    target_df = target_long.groupby("row_id", sort=False).max(numeric_only=True).reset_index()
    pred_cols = [column for column in prediction_df.columns if isinstance(column, int)]
    target_cols = [column for column in target_df.columns if isinstance(column, int)]
    score = macro_auc_skip_missing(target_df[target_cols].to_numpy(dtype=np.float32), prediction_df[pred_cols].to_numpy(dtype=np.float32))
    slices = auc_frequency_slices(
        target_df[target_cols].to_numpy(dtype=np.float32),
        prediction_df[pred_cols].to_numpy(dtype=np.float32),
    )
    print(
        f"[stage2-valid] fold_auc={score:.5f} "
        f"rare={slices['auc_rare']} mid={slices['auc_mid']} common={slices['auc_common']} "
        f"logit_std={result['logit_std']:.4f}"
    )
    prediction_df["fold_auc"] = score
    return prediction_df


def prepare_stage1_loaders(cfg: Config, train_audio_df: pd.DataFrame, renderer: SpectrogramRenderer, num_classes: int):
    train_df = train_audio_df[train_audio_df["clip_split"] == "train"].reset_index(drop=True)
    valid_df = train_audio_df[train_audio_df["clip_split"] == "valid"].reset_index(drop=True)

    train_dataset = TrainAudioDataset(
        train_df,
        cfg=cfg,
        renderer=renderer,
        num_classes=num_classes,
        train_mode=True,
        mixup_alpha=cfg.stage1_mixup_alpha,
        mixup_prob=cfg.stage1_mixup_prob,
        mixup_domain=cfg.mixup_domain,
    )
    valid_dataset = TrainAudioDataset(
        valid_df,
        cfg=cfg,
        renderer=renderer,
        num_classes=num_classes,
        train_mode=False,
    )
    train_sampler = build_train_audio_sampler(
        train_df,
        samples_per_epoch=cfg.stage1_samples_per_epoch,
        generator=build_torch_generator(cfg.seed + 101),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.stage1_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 102,
        sampler=train_sampler,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 103,
        shuffle=False,
    )
    return train_loader, valid_loader


def prepare_stage2_loaders(
    cfg: Config,
    soundscape_df: pd.DataFrame,
    fold: int,
    renderer: SpectrogramRenderer,
    num_classes: int,
    perch_teacher_tokens: Optional[np.ndarray] = None,
):
    train_mask = soundscape_df["fold"].to_numpy() != fold
    valid_mask = ~train_mask
    train_df = soundscape_df.loc[train_mask].reset_index(drop=True)
    valid_df = soundscape_df.loc[valid_mask].reset_index(drop=True)
    seed_base = cfg.seed + (1000 * (fold + 1))

    if cfg.head_type == "csiro_multicontext_v1":
        if cfg.use_perch_distill:
            raise RuntimeError("Perch distillation is not implemented for csiro_multicontext_v1.")
        train_window_df = build_long_context_windows(train_df, cfg=cfg)
        valid_window_df = build_long_context_windows(valid_df, cfg=cfg)
        print(
            f"[INFO] Multi-context fold {fold}: "
            f"train_windows={len(train_window_df)} valid_windows={len(valid_window_df)} "
            f"train_rows={len(train_df)} valid_rows={len(valid_df)}"
        )
        train_dataset = SoundscapeLongContextDataset(
            train_window_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=True,
            mixup_alpha=cfg.stage2_mixup_alpha,
            mixup_prob=cfg.stage2_mixup_prob,
            mixup_domain=cfg.mixup_domain,
        )
        valid_dataset = SoundscapeLongContextDataset(
            valid_window_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=False,
        )
        sampler_df = train_window_df
    elif cfg.head_type == "sed_att_v1" and cfg.sed_center_context:
        print(
            f"[INFO] SED center-context fold {fold}: "
            f"clip_seconds={cfg.clip_seconds:g} train_rows={len(train_df)} valid_rows={len(valid_df)}"
        )
        train_dataset = SoundscapeCenterContextDataset(
            train_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=True,
            mixup_alpha=cfg.stage2_mixup_alpha,
            mixup_prob=cfg.stage2_mixup_prob,
            mixup_domain=cfg.mixup_domain,
        )
        valid_dataset = SoundscapeCenterContextDataset(
            valid_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=False,
        )
        sampler_df = train_df
    else:
        perch_teacher_tokens_train = None
        if perch_teacher_tokens is not None:
            perch_teacher_tokens_train = perch_teacher_tokens[train_mask]
        train_dataset = SoundscapeSegmentDataset(
            train_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=True,
            mixup_alpha=cfg.stage2_mixup_alpha,
            mixup_prob=cfg.stage2_mixup_prob,
            mixup_domain=cfg.mixup_domain,
            perch_teacher_tokens=perch_teacher_tokens_train,
        )
        valid_dataset = SoundscapeSegmentDataset(
            valid_df,
            cfg=cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=False,
            perch_teacher_tokens=perch_teacher_tokens[valid_mask] if perch_teacher_tokens is not None else None,
        )
        sampler_df = train_df
    train_sampler = build_soundscape_sampler(
        sampler_df,
        num_classes=num_classes,
        samples_per_epoch=cfg.stage2_samples_per_epoch,
        generator=build_torch_generator(seed_base + 1),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.stage2_batch_size,
        num_workers=cfg.num_workers,
        seed=seed_base + 2,
        sampler=train_sampler,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=seed_base + 3,
        shuffle=False,
    )
    return train_loader, valid_loader, valid_df


def load_perch_teacher_tokens_for_soundscapes(
    cfg: Config,
    soundscape_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    if not cfg.use_perch_distill or float(cfg.perch_distill_weight) <= 0.0:
        return None
    meta_df, spatial_tokens, spatial_tokens_max, spatial_tokens_64 = load_perch_spatial_cache_from_paths(
        cache_dir_arg=cfg.perch_spatial_cache_dir,
        meta_path_arg=cfg.perch_spatial_meta_path,
        arrays_path_arg=cfg.perch_spatial_arrays_path,
    )
    token_source = {
        "spatial_tokens": spatial_tokens,
        "spatial_tokens_max": spatial_tokens_max,
        "spatial_tokens_64": spatial_tokens_64,
    }[cfg.perch_distill_token_key]
    if token_source is None:
        raise ValueError(
            f"Perch distill token key={cfg.perch_distill_token_key} is missing in {cfg.perch_spatial_cache_dir}"
        )
    row_ids = soundscape_df["row_id"].astype(str).tolist()
    aligned = align_array_by_row_id(
        source_meta=meta_df,
        source_array=token_source,
        target_row_ids=row_ids,
        name="Perch spatial teacher",
    )
    return aligned.astype(np.float32, copy=False)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def run_pipeline(cfg: Config, input_dir: Path, ckpt_dir: Path, run_dir: Path) -> None:
    class_names = load_class_names(input_dir)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    num_classes = len(class_names)

    print("[INFO] Loading metadata...")
    train_audio_df = load_train_audio_metadata(cfg, input_dir=input_dir, label_to_idx=label_to_idx)
    soundscape_df = load_soundscape_segments(cfg, input_dir=input_dir, label_to_idx=label_to_idx)
    soundscape_df = build_soundscape_folds(soundscape_df, num_classes=num_classes, n_folds=cfg.n_folds, seed=cfg.seed)
    perch_teacher_tokens = load_perch_teacher_tokens_for_soundscapes(cfg, soundscape_df)
    if perch_teacher_tokens is not None:
        print(
            f"[INFO] Loaded Perch distill teacher tokens: {perch_teacher_tokens.shape} "
            f"key={cfg.perch_distill_token_key}"
        )
    soundscape_summary = summarize_soundscape_folds(soundscape_df, num_classes=num_classes)
    soundscape_df.to_csv(run_dir / "soundscape_segments_with_folds.csv", index=False)
    soundscape_summary.to_csv(run_dir / "soundscape_fold_summary.csv", index=False)

    print("[INFO] Soundscape fold summary:")
    print(soundscape_summary.to_string(index=False))
    print(
        "[INFO] Leakage policy: folds are assigned at full soundscape filename level, "
        "and local CV is computed only on manually labeled soundscape windows."
    )
    deterministic_algorithms = False
    cudnn_deterministic = False
    cudnn_benchmark = False
    if torch is not None:
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
        if hasattr(torch.backends, "cudnn"):
            cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    print(
        f"[INFO] Seed={cfg.seed} | deterministic_algorithms={deterministic_algorithms} | "
        f"cudnn_deterministic={cudnn_deterministic} | cudnn_benchmark={cudnn_benchmark}"
    )

    if cfg.build_folds_only:
        print(f"[INFO] Folds saved under {run_dir}")
        return

    require_training_dependencies()
    backbone_weight_path = ckpt_dir / f"{cfg.model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_settings = resolve_amp_settings(cfg, device=device)
    effective_grad_clip_norm = resolve_grad_clip_norm(cfg)
    renderer = SpectrogramRenderer(cfg)
    print(
        f"[INFO] Device={device} | AMP={amp_settings['description']} | "
        f"grad_clip_norm={effective_grad_clip_norm:.2f}"
    )
    print(
        f"[INFO] Spectrogram variant: {cfg.spectrogram_variant} | "
        f"input_channels={cfg.input_channels} | image={cfg.image_height}x{cfg.image_width} | "
        f"normalize={cfg.image_normalize}"
    )
    print(f"[INFO] Scheduler: {cfg.scheduler_type} | warmup_epochs={cfg.warmup_epochs}")
    print(
        f"[INFO] Waveform branch: enabled={cfg.use_waveform_branch} | "
        f"d_model={cfg.waveform_branch_d_model} | "
        f"layers={cfg.waveform_branch_layers} | "
        f"heads={cfg.waveform_branch_heads} | "
        f"dropout={cfg.waveform_branch_dropout}"
    )
    print(
        f"[INFO] Perch distill: enabled={cfg.use_perch_distill} | "
        f"weight={cfg.perch_distill_weight} | token_key={cfg.perch_distill_token_key} | "
        f"cache_dir={cfg.perch_spatial_cache_dir}"
    )
    print(f"[INFO] Head type: {cfg.head_type} | head_pool_type={cfg.head_pool_type}")
    if cfg.head_type == "sed_att_v1":
        print(
            f"[INFO] SED: frame_loss_weight={cfg.sed_frame_loss_weight} "
            f"center_context={cfg.sed_center_context}"
        )
    if cfg.head_type == "csiro_multicontext_v1":
        print(
            f"[INFO] Multi-context: clip_seconds={cfg.clip_seconds} | "
            f"slots={cfg.multi_context_num_slots} | "
            f"global_loss_weight={cfg.multi_context_global_loss_weight}"
        )
    print(
        f"[INFO] Batch aug legacy default: "
        f"{format_batch_aug_summary(cfg.mixup_alpha, cfg.mixup_prob, cfg.cutmix_alpha, cfg.cutmix_prob)}"
    )
    print(f"[INFO] Mixup domain: {cfg.mixup_domain}")
    print(
        f"[INFO] Stage 1 aug: "
        f"{format_batch_aug_summary(cfg.stage1_mixup_alpha, cfg.stage1_mixup_prob, cfg.stage1_cutmix_alpha, cfg.stage1_cutmix_prob)} | "
        f"start_epochs mixup={cfg.stage1_mixup_start_epoch} cutmix={cfg.stage1_cutmix_start_epoch}"
    )
    print(
        f"[INFO] Stage 2 aug: "
        f"{format_batch_aug_summary(cfg.stage2_mixup_alpha, cfg.stage2_mixup_prob, cfg.stage2_cutmix_alpha, cfg.stage2_cutmix_prob)} | "
        f"start_epochs mixup={cfg.stage2_mixup_start_epoch} cutmix={cfg.stage2_cutmix_start_epoch}"
    )
    if "source_dataset" in train_audio_df.columns:
        source_summary = (
            train_audio_df.groupby(["source_dataset", "clip_split"])
            .agg(rows=("filename", "count"), labels=("primary_label", "nunique"))
            .reset_index()
            .sort_values(["source_dataset", "clip_split"])
        )
        source_summary.to_csv(run_dir / "stage1_train_audio_source_summary.csv", index=False)
        print("[INFO] Stage 1 train_audio source summary:")
        print(source_summary.to_string(index=False))

    print("[INFO] Stage 1: train_audio pretraining...")
    stage1_dir = run_dir / "stage1_audio"
    ensure_dir(stage1_dir)
    stage1_train_loader, stage1_valid_loader = prepare_stage1_loaders(
        cfg=cfg,
        train_audio_df=train_audio_df,
        renderer=renderer,
        num_classes=num_classes,
    )
    base_model = BirdCLEFNet(cfg=cfg, num_classes=num_classes, backbone_weight_path=backbone_weight_path).to(device)
    base_model, stage1_best_path = fit_one_stage(
        stage_name="stage1_audio",
        model=base_model,
        train_loader=stage1_train_loader,
        valid_loader=stage1_valid_loader,
        device=device,
        output_dir=stage1_dir,
        backbone_lr=cfg.stage1_backbone_lr,
        head_lr=cfg.stage1_head_lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.stage1_epochs,
        warmup_epochs=cfg.warmup_epochs,
        scheduler_type=cfg.scheduler_type,
        amp_enabled=bool(amp_settings["enabled"]),
        amp_dtype=amp_settings["dtype"],
        use_grad_scaler=bool(amp_settings["use_grad_scaler"]),
        grad_clip_norm=effective_grad_clip_norm,
        patience=cfg.patience,
        mixup_domain=cfg.mixup_domain,
        mixup_alpha=cfg.stage1_mixup_alpha,
        mixup_prob=cfg.stage1_mixup_prob,
        mixup_start_epoch=cfg.stage1_mixup_start_epoch,
        cutmix_alpha=cfg.stage1_cutmix_alpha,
        cutmix_prob=cfg.stage1_cutmix_prob,
        cutmix_start_epoch=cfg.stage1_cutmix_start_epoch,
        global_loss_weight=0.0,
    )
    del base_model
    torch.cuda.empty_cache()

    oof_frames = []
    fold_scores = []

    for fold in range(cfg.n_folds):
        print(f"[INFO] Stage 2: fold {fold + 1}/{cfg.n_folds}")
        fold_dir = run_dir / f"fold_{fold}"
        ensure_dir(fold_dir)

        model = BirdCLEFNet(cfg=cfg, num_classes=num_classes, backbone_weight_path=backbone_weight_path).to(device)
        stage1_checkpoint = torch.load(stage1_best_path, map_location="cpu")
        model.load_state_dict(stage1_checkpoint["model"], strict=True)

        train_loader, valid_loader, valid_df = prepare_stage2_loaders(
            cfg=cfg,
            soundscape_df=soundscape_df,
            fold=fold,
            renderer=renderer,
            num_classes=num_classes,
            perch_teacher_tokens=perch_teacher_tokens,
        )

        model, best_path = fit_one_stage(
            stage_name=f"stage2_fold{fold}",
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            output_dir=fold_dir,
            backbone_lr=cfg.stage2_backbone_lr,
            head_lr=cfg.stage2_head_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.stage2_epochs,
            warmup_epochs=cfg.warmup_epochs,
            scheduler_type=cfg.scheduler_type,
            amp_enabled=bool(amp_settings["enabled"]),
            amp_dtype=amp_settings["dtype"],
            use_grad_scaler=bool(amp_settings["use_grad_scaler"]),
            grad_clip_norm=effective_grad_clip_norm,
            patience=cfg.patience,
            freeze_backbone_epochs=cfg.stage2_freeze_backbone_epochs,
            mixup_domain=cfg.mixup_domain,
            mixup_alpha=cfg.stage2_mixup_alpha,
            mixup_prob=cfg.stage2_mixup_prob,
            mixup_start_epoch=cfg.stage2_mixup_start_epoch,
            cutmix_alpha=cfg.stage2_cutmix_alpha,
            cutmix_prob=cfg.stage2_cutmix_prob,
            cutmix_start_epoch=cfg.stage2_cutmix_start_epoch,
            global_loss_weight=cfg.multi_context_global_loss_weight if cfg.head_type == "csiro_multicontext_v1" else 0.0,
            sed_frame_loss_weight=cfg.sed_frame_loss_weight,
            perch_distill_weight=cfg.perch_distill_weight if cfg.use_perch_distill else 0.0,
        )

        prediction_df = evaluate_soundscape_model(
            model,
            valid_loader,
            device=device,
            amp_enabled=bool(amp_settings["enabled"]),
            amp_dtype=amp_settings["dtype"],
            sed_frame_loss_weight=cfg.sed_frame_loss_weight,
        )
        prediction_df.insert(1, "fold", fold)
        prediction_df.to_csv(fold_dir / "valid_predictions.csv", index=False)
        fold_scores.append(float(prediction_df["fold_auc"].iloc[0]))

        prediction_df = prediction_df.rename(columns={i: class_names[i] for i in range(num_classes)})
        truth_df = valid_df[["row_id", "site", "label_indices"]].copy()
        truth_matrix = np.stack(valid_df["label_indices"].map(lambda x: indices_to_multihot(x, num_classes)).to_numpy())
        truth_frame = pd.DataFrame(truth_matrix, columns=[f"target_{label}" for label in class_names])
        truth_full = pd.concat([truth_df.reset_index(drop=True), truth_frame.reset_index(drop=True)], axis=1)
        pred_full = prediction_df[["row_id"] + class_names].copy()
        merged = truth_full.merge(pred_full, on="row_id", how="left", validate="one_to_one")
        if merged[class_names].isna().any().any():
            missing = merged.loc[merged[class_names].isna().any(axis=1), "row_id"].head(10).tolist()
            raise RuntimeError(f"Missing predictions for validation rows: {missing}")
        oof_frames.append(merged)

        del model
        torch.cuda.empty_cache()

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    y_true = oof_df[[f"target_{label}" for label in class_names]].to_numpy(dtype=np.float32)
    y_pred = oof_df[class_names].to_numpy(dtype=np.float32)
    final_cv = macro_auc_skip_missing(y_true, y_pred)
    print(f"[INFO] Final OOF local CV = {final_cv:.6f}")

    oof_df.to_csv(run_dir / "soundscape_oof_predictions.csv", index=False)
    pd.DataFrame({"fold": list(range(cfg.n_folds)), "fold_auc": fold_scores}).to_csv(run_dir / "fold_scores.csv", index=False)
    save_json(
        run_dir / "metrics.json",
        {
            "final_oof_cv": final_cv,
            "fold_scores": fold_scores,
            "n_soundscape_segments": int(len(soundscape_df)),
            "n_soundscape_files": int(soundscape_df["filename"].nunique()),
        },
    )
    print(f"[INFO] Run artifacts saved to {run_dir}")


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    root = Path(cfg.root).resolve()
    input_dir = root / cfg.input_dir
    ckpt_dir = root / cfg.ckpt_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / cfg.output_dir / f"{timestamp}_{cfg.model_name.replace('/', '_')}"
    ensure_dir(run_dir)
    save_json(run_dir / "config.json", asdict(cfg))

    log_path = run_dir / "train.log"
    with RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        run_pipeline(cfg=cfg, input_dir=input_dir, ckpt_dir=ckpt_dir, run_dir=run_dir)


if __name__ == "__main__":
    main()
