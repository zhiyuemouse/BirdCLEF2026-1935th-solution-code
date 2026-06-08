from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import birdclef2026_gm_train as gm


if gm.ML_IMPORT_ERROR is not None:
    gm.require_training_dependencies()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, WeightedRandomSampler

from birdclef2026_teacher_oof import load_teacher_oof_predictions, load_teacher_predictions_for_fold
from waveform_model import RawWaveTransformerMixerModel, RawWaveTransformerModel


@dataclass
class Config:
    root: str = "."
    input_dir: str = "input"
    output_dir: str = "outputs/birdclef2026_raw_waveform_transformer"
    model_name: str = "raw_wave_conv_tokenizer_base_long_n32_d768"
    fold_assignment_path: str = (
        "outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/"
        "soundscape_segments_with_folds.csv"
    )
    n_folds: int = 3
    seed: int = 2026
    sample_rate: int = 32000
    clip_seconds: float = 5.0
    num_tokens: int = 32
    tokenizer_type: str = "conv_stack"
    waveform_model_variant: str = "base"
    d_model: int = 768
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_mult: int = 4
    dropout: float = 0.20
    stage1_epochs: int = 25
    stage2_epochs: int = 40
    stage1_batch_size: int = 16
    stage2_batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = max(2, (os.cpu_count() or 4) // 2)
    stage1_samples_per_epoch: int = 24000
    stage2_samples_per_epoch: int = 2048
    stage1_lr: float = 3e-4
    stage2_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 10
    stage1_mixup_alpha: float = 0.20
    stage1_mixup_prob: float = 0.10
    stage2_mixup_alpha: float = 0.0
    stage2_mixup_prob: float = 0.0
    raw_strong_aug: bool = False
    raw_gain_min: float = 0.65
    raw_gain_max: float = 1.50
    raw_polarity_prob: float = 0.20
    raw_time_shift_prob: float = 0.50
    raw_time_shift_max_sec: float = 0.35
    raw_noise_prob: float = 0.50
    raw_noise_min: float = 0.001
    raw_noise_max: float = 0.020
    raw_filter_prob: float = 0.35
    clip_valid_fraction: float = 0.05
    use_secondary_labels: bool = True
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    smoke_test: bool = False
    build_folds_only: bool = False
    max_train_audio_rows: int = -1
    max_soundscape_segments: int = -1
    use_birdclef2025_stage1: bool = False
    birdclef2025_root: str = "BirdCLEF2025-Dataset"
    birdclef2025_stage1_max_rows: int = -1
    birdclef2025_stage1_max_per_label: int = -1
    stage1_checkpoint_path: str = ""
    teacher_oof_path: str = ""
    teacher_loss_weight: float = 0.0
    pseudo_dir: str = ""
    pseudo_loss_weight: float = 0.0
    pseudo_sampler_fraction: float = -1.0
    pseudo_min_max_prob: float = -1.0
    pseudo_max_topk_entropy: float = -1.0
    pseudo_max_rows: int = -1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Raw waveform Transformer training for BirdCLEF 2026.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_raw_waveform_transformer")
    parser.add_argument("--model-name", type=str, default="raw_wave_conv_tokenizer_base_long_n32_d768")
    parser.add_argument("--fold-assignment-path", type=str, default=Config.fold_assignment_path)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--tokenizer-type", type=str, choices=["conv_stack", "sinc_stack", "patch_stem"], default="conv_stack")
    parser.add_argument("--waveform-model-variant", type=str, choices=["base", "mixer"], default="base")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--stage1-epochs", type=int, default=25)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--stage1-samples-per-epoch", type=int, default=24000)
    parser.add_argument("--stage2-samples-per-epoch", type=int, default=2048)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stage1-mixup-alpha", type=float, default=0.20)
    parser.add_argument("--stage1-mixup-prob", type=float, default=0.10)
    parser.add_argument("--stage2-mixup-alpha", type=float, default=0.0)
    parser.add_argument("--stage2-mixup-prob", type=float, default=0.0)
    parser.add_argument("--raw-strong-aug", action="store_true")
    parser.add_argument("--disable-raw-strong-aug", action="store_true")
    parser.add_argument("--raw-gain-min", type=float, default=0.65)
    parser.add_argument("--raw-gain-max", type=float, default=1.50)
    parser.add_argument("--raw-polarity-prob", type=float, default=0.20)
    parser.add_argument("--raw-time-shift-prob", type=float, default=0.50)
    parser.add_argument("--raw-time-shift-max-sec", type=float, default=0.35)
    parser.add_argument("--raw-noise-prob", type=float, default=0.50)
    parser.add_argument("--raw-noise-min", type=float, default=0.001)
    parser.add_argument("--raw-noise-max", type=float, default=0.020)
    parser.add_argument("--raw-filter-prob", type=float, default=0.35)
    parser.add_argument("--clip-valid-fraction", type=float, default=0.05)
    parser.add_argument("--disable-secondary-labels", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--build-folds-only", action="store_true")
    parser.add_argument("--max-train-audio-rows", type=int, default=-1)
    parser.add_argument("--max-soundscape-segments", type=int, default=-1)
    parser.add_argument("--use-birdclef2025-stage1", action="store_true")
    parser.add_argument("--birdclef2025-root", type=str, default="BirdCLEF2025-Dataset")
    parser.add_argument("--birdclef2025-stage1-max-rows", type=int, default=-1)
    parser.add_argument("--birdclef2025-stage1-max-per-label", type=int, default=-1)
    parser.add_argument("--stage1-checkpoint-path", type=str, default="")
    parser.add_argument("--teacher-oof-path", type=str, default="")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--pseudo-dir",
        type=str,
        default="",
        help="Directory containing pseudo_segments.csv and pseudo_probs.npy generated on unlabeled soundscapes.",
    )
    parser.add_argument(
        "--pseudo-loss-weight",
        type=float,
        default=0.0,
        help="BCE soft-target weight for pseudo rows. Pseudo rows never contribute hard-label loss.",
    )
    parser.add_argument(
        "--pseudo-sampler-fraction",
        type=float,
        default=-1.0,
        help="Target fraction of stage2 sampled rows coming from pseudo data. Negative uses natural sampler weights.",
    )
    parser.add_argument("--pseudo-min-max-prob", type=float, default=-1.0)
    parser.add_argument("--pseudo-max-topk-entropy", type=float, default=-1.0)
    parser.add_argument("--pseudo-max-rows", type=int, default=-1)
    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fold_assignment_path=args.fold_assignment_path,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        num_tokens=args.num_tokens,
        tokenizer_type=args.tokenizer_type,
        waveform_model_variant=args.waveform_model_variant,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_mult=args.transformer_ff_mult,
        dropout=args.dropout,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_batch_size=args.stage1_batch_size,
        stage2_batch_size=args.stage2_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        stage1_samples_per_epoch=args.stage1_samples_per_epoch,
        stage2_samples_per_epoch=args.stage2_samples_per_epoch,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        stage1_mixup_alpha=args.stage1_mixup_alpha,
        stage1_mixup_prob=args.stage1_mixup_prob,
        stage2_mixup_alpha=args.stage2_mixup_alpha,
        stage2_mixup_prob=args.stage2_mixup_prob,
        raw_strong_aug=bool(args.raw_strong_aug and not args.disable_raw_strong_aug),
        raw_gain_min=args.raw_gain_min,
        raw_gain_max=args.raw_gain_max,
        raw_polarity_prob=args.raw_polarity_prob,
        raw_time_shift_prob=args.raw_time_shift_prob,
        raw_time_shift_max_sec=args.raw_time_shift_max_sec,
        raw_noise_prob=args.raw_noise_prob,
        raw_noise_min=args.raw_noise_min,
        raw_noise_max=args.raw_noise_max,
        raw_filter_prob=args.raw_filter_prob,
        clip_valid_fraction=args.clip_valid_fraction,
        use_secondary_labels=not args.disable_secondary_labels,
        use_amp=not args.disable_amp,
        grad_clip_norm=args.grad_clip_norm,
        smoke_test=args.smoke_test,
        build_folds_only=args.build_folds_only,
        max_train_audio_rows=args.max_train_audio_rows,
        max_soundscape_segments=args.max_soundscape_segments,
        use_birdclef2025_stage1=args.use_birdclef2025_stage1,
        birdclef2025_root=args.birdclef2025_root,
        birdclef2025_stage1_max_rows=args.birdclef2025_stage1_max_rows,
        birdclef2025_stage1_max_per_label=args.birdclef2025_stage1_max_per_label,
        stage1_checkpoint_path=args.stage1_checkpoint_path,
        teacher_oof_path=args.teacher_oof_path,
        teacher_loss_weight=args.teacher_loss_weight,
        pseudo_dir=args.pseudo_dir,
        pseudo_loss_weight=args.pseudo_loss_weight,
        pseudo_sampler_fraction=args.pseudo_sampler_fraction,
        pseudo_min_max_prob=args.pseudo_min_max_prob,
        pseudo_max_topk_entropy=args.pseudo_max_topk_entropy,
        pseudo_max_rows=args.pseudo_max_rows,
    )

    if cfg.smoke_test:
        cfg.stage1_epochs = 1
        cfg.stage2_epochs = 1
        cfg.stage1_samples_per_epoch = min(cfg.stage1_samples_per_epoch, 128)
        cfg.stage2_samples_per_epoch = min(cfg.stage2_samples_per_epoch, 96)
        cfg.max_train_audio_rows = 192 if cfg.max_train_audio_rows < 0 else min(cfg.max_train_audio_rows, 192)
        cfg.max_soundscape_segments = 96 if cfg.max_soundscape_segments < 0 else min(cfg.max_soundscape_segments, 96)
        cfg.num_workers = min(cfg.num_workers, 2)
        cfg.patience = 1
    if cfg.d_model % cfg.transformer_heads != 0:
        raise ValueError("--d-model must be divisible by --transformer-heads")
    target_samples = int(round(cfg.sample_rate * cfg.clip_seconds))
    if target_samples % cfg.num_tokens != 0:
        raise ValueError("--sample-rate * --clip-seconds must be divisible by --num-tokens")
    if cfg.tokenizer_type in {"conv_stack", "sinc_stack"} and cfg.num_tokens not in {16, 32, 64}:
        raise ValueError("tokenizer_type=conv_stack/sinc_stack currently expects --num-tokens 16, 32, or 64")
    if cfg.teacher_loss_weight < 0:
        raise ValueError("--teacher-loss-weight must be non-negative")
    if cfg.pseudo_loss_weight < 0:
        raise ValueError("--pseudo-loss-weight must be non-negative")
    if cfg.pseudo_sampler_fraction > 1:
        raise ValueError("--pseudo-sampler-fraction must be <= 1")
    return cfg


def apply_raw_filter(audio: np.ndarray) -> np.ndarray:
    mode = random.choice(["lowpass", "highpass", "bandstop"])
    if mode == "lowpass":
        kernel = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)
        return np.convolve(audio, kernel, mode="same").astype(np.float32)

    smooth_kernel = np.array([0.20, 0.60, 0.20], dtype=np.float32)
    smooth = np.convolve(audio, smooth_kernel, mode="same").astype(np.float32)
    if mode == "highpass":
        return (audio - random.uniform(0.35, 0.75) * smooth).astype(np.float32)

    return (audio - random.uniform(0.20, 0.45) * smooth).astype(np.float32)


def augment_raw_waveform(audio: np.ndarray, cfg: Config) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not cfg.raw_strong_aug:
        return gm.augment_waveform(audio)

    gain = random.uniform(cfg.raw_gain_min, cfg.raw_gain_max)
    audio = audio * gain

    if random.random() < cfg.raw_polarity_prob:
        audio = -audio

    if cfg.raw_time_shift_prob > 0 and cfg.raw_time_shift_max_sec > 0:
        if random.random() < min(cfg.raw_time_shift_prob, 1.0):
            max_shift = int(round(cfg.raw_time_shift_max_sec * cfg.sample_rate))
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift > 0:
                    audio = np.pad(audio, (shift, 0), mode="constant")[: len(audio)]
                elif shift < 0:
                    audio = np.pad(audio[-shift:], (0, -shift), mode="constant")

    if random.random() < cfg.raw_filter_prob:
        audio = apply_raw_filter(audio)

    if random.random() < cfg.raw_noise_prob:
        noise_scale = random.uniform(cfg.raw_noise_min, cfg.raw_noise_max) * max(float(audio.std()), 1e-4)
        audio = audio + np.random.normal(0.0, noise_scale, size=audio.shape).astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


class RawTrainAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        num_classes: int,
        train_mode: bool,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        audio = gm.load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=self.train_mode,
        )
        if self.train_mode:
            audio = augment_raw_waveform(audio, self.cfg)
        return np.asarray(audio, dtype=np.float32)

    def _maybe_mixup(self, idx: int, audio: np.ndarray, target: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        if not self.train_mode or self.mixup_alpha <= 0 or self.mixup_prob <= 0 or len(self.df) < 2:
            return audio, target
        if random.random() >= min(self.mixup_prob, 1.0):
            return audio, target
        lam = gm.sample_mix_lambda(self.mixup_alpha)
        if lam >= 1.0:
            return audio, target
        partner_idx = random.randrange(len(self.df) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner = self.df.iloc[partner_idx]
        partner_audio = self._load_audio(partner)
        partner_target = torch.from_numpy(gm.indices_to_multihot(partner["label_indices"], self.num_classes)).float()
        mixed_audio = np.clip((audio * lam) + (partner_audio * (1.0 - lam)), -1.0, 1.0).astype(np.float32)
        mixed_target = (target * lam) + (partner_target * (1.0 - lam))
        return mixed_audio, mixed_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = self._load_audio(row)
        target = torch.from_numpy(gm.indices_to_multihot(row["label_indices"], self.num_classes)).float()
        audio, target = self._maybe_mixup(idx, audio, target)
        return {
            "waveform": torch.from_numpy(audio).float(),
            "target": target,
            "row_id": str(row["filename"]),
            "site": "train_audio",
        }


class RawSoundscapeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        num_classes: int,
        train_mode: bool,
        teacher_probs: Optional[np.ndarray] = None,
        teacher_weights: Optional[np.ndarray] = None,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.teacher_probs = None if teacher_probs is None else np.asarray(teacher_probs, dtype=np.float32)
        if teacher_weights is None:
            self.teacher_weights = None
        else:
            self.teacher_weights = np.asarray(teacher_weights, dtype=np.float32)
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        if self.teacher_probs is not None and len(self.teacher_probs) != len(self.df):
            raise ValueError("teacher_probs must have the same number of rows as df")
        if self.teacher_weights is not None and len(self.teacher_weights) != len(self.df):
            raise ValueError("teacher_weights must have the same number of rows as df")

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        audio = gm.load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(row["start_sec"]),
        )
        if self.train_mode:
            audio = augment_raw_waveform(audio, self.cfg)
        return np.asarray(audio, dtype=np.float32)

    def _maybe_mixup(
        self,
        idx: int,
        audio: np.ndarray,
        target: torch.Tensor,
        hard_weight: torch.Tensor,
        teacher_target: torch.Tensor,
        has_teacher: torch.Tensor,
        teacher_weight: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.train_mode or self.mixup_alpha <= 0 or self.mixup_prob <= 0 or len(self.df) < 2:
            return audio, target, hard_weight, teacher_target, has_teacher, teacher_weight
        if random.random() >= min(self.mixup_prob, 1.0):
            return audio, target, hard_weight, teacher_target, has_teacher, teacher_weight
        lam = gm.sample_mix_lambda(self.mixup_alpha)
        if lam >= 1.0:
            return audio, target, hard_weight, teacher_target, has_teacher, teacher_weight
        partner_idx = random.randrange(len(self.df) - 1)
        if partner_idx >= idx:
            partner_idx += 1
        partner = self.df.iloc[partner_idx]
        partner_audio = self._load_audio(partner)
        partner_target = torch.from_numpy(gm.indices_to_multihot(partner["label_indices"], self.num_classes)).float()
        partner_hard_weight = torch.tensor(float(partner.get("hard_weight", 1.0)), dtype=torch.float32)
        if self.teacher_probs is not None:
            partner_teacher = torch.from_numpy(self.teacher_probs[partner_idx].astype(np.float32, copy=False)).float()
            if self.teacher_weights is None:
                partner_teacher_weight_value = 1.0
            else:
                partner_teacher_weight_value = float(self.teacher_weights[partner_idx])
            partner_teacher_weight = torch.tensor(partner_teacher_weight_value, dtype=torch.float32)
            partner_has_teacher = torch.tensor(partner_teacher_weight_value > 0, dtype=torch.bool)
        else:
            partner_teacher = torch.zeros(self.num_classes, dtype=torch.float32)
            partner_teacher_weight = torch.tensor(0.0, dtype=torch.float32)
            partner_has_teacher = torch.tensor(False, dtype=torch.bool)
        mixed_audio = np.clip((audio * lam) + (partner_audio * (1.0 - lam)), -1.0, 1.0).astype(np.float32)
        mixed_target = (target * lam) + (partner_target * (1.0 - lam))
        mixed_hard_weight = (hard_weight * lam) + (partner_hard_weight * (1.0 - lam))
        mixed_teacher = (teacher_target * lam) + (partner_teacher * (1.0 - lam))
        mixed_teacher_weight = (teacher_weight * lam) + (partner_teacher_weight * (1.0 - lam))
        mixed_has_teacher = (has_teacher | partner_has_teacher) & (mixed_teacher_weight > 0)
        return mixed_audio, mixed_target, mixed_hard_weight, mixed_teacher, mixed_has_teacher, mixed_teacher_weight

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = self._load_audio(row)
        target = torch.from_numpy(gm.indices_to_multihot(row["label_indices"], self.num_classes)).float()
        hard_weight = torch.tensor(float(row.get("hard_weight", 1.0)), dtype=torch.float32)
        if self.teacher_probs is not None:
            teacher_target = torch.from_numpy(self.teacher_probs[idx].astype(np.float32, copy=False)).float()
            if self.teacher_weights is None:
                teacher_weight_value = 1.0
            else:
                teacher_weight_value = float(self.teacher_weights[idx])
            teacher_weight = torch.tensor(teacher_weight_value, dtype=torch.float32)
            has_teacher = torch.tensor(teacher_weight_value > 0, dtype=torch.bool)
        else:
            teacher_target = torch.zeros(self.num_classes, dtype=torch.float32)
            teacher_weight = torch.tensor(0.0, dtype=torch.float32)
            has_teacher = torch.tensor(False, dtype=torch.bool)
        audio, target, hard_weight, teacher_target, has_teacher, teacher_weight = self._maybe_mixup(
            idx,
            audio,
            target,
            hard_weight,
            teacher_target,
            has_teacher,
            teacher_weight,
        )
        return {
            "waveform": torch.from_numpy(audio).float(),
            "target": target,
            "hard_weight": hard_weight,
            "teacher_target": teacher_target,
            "has_teacher": has_teacher,
            "teacher_weight": teacher_weight,
            "row_id": str(row["row_id"]),
            "site": str(row.get("site", "unknown")),
        }


class RawWaveformTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: int,
        clip_seconds: float,
        num_tokens: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        total_samples = int(round(sample_rate * clip_seconds))
        if total_samples % num_tokens != 0:
            raise ValueError("total_samples must be divisible by num_tokens")
        self.total_samples = total_samples
        self.num_tokens = int(num_tokens)
        self.token_samples = total_samples // int(num_tokens)
        stem_hidden = max(d_model // 4, 128)
        stem_mid = max(d_model // 2, 256)
        self.token_stem = nn.Sequential(
            nn.Conv1d(1, stem_hidden, kernel_size=401, stride=80, padding=200, bias=False),
            nn.GroupNorm(1, stem_hidden),
            nn.GELU(),
            nn.Conv1d(stem_hidden, stem_mid, kernel_size=11, stride=4, padding=5, bias=False),
            nn.GroupNorm(1, stem_mid),
            nn.GELU(),
            nn.Conv1d(stem_mid, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [B,T], got {tuple(waveform.shape)}")
        if waveform.shape[1] < self.total_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.total_samples - waveform.shape[1]))
        elif waveform.shape[1] > self.total_samples:
            waveform = waveform[:, : self.total_samples]
        batch = waveform.shape[0]
        x = waveform.reshape(batch, self.num_tokens, self.token_samples)
        x = x.reshape(batch * self.num_tokens, 1, self.token_samples)
        x = self.token_stem(x).flatten(1)
        x = x.reshape(batch, self.num_tokens, -1)
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.head(pooled)


def apply_fold_assignment(soundscape_df: pd.DataFrame, cfg: Config, num_classes: int) -> pd.DataFrame:
    assignment_path = Path(cfg.fold_assignment_path)
    if not assignment_path.is_absolute():
        assignment_path = Path(cfg.root) / assignment_path
    if assignment_path.exists():
        fold_df = pd.read_csv(assignment_path)
        if "row_id" not in fold_df.columns:
            if {"filename", "end_sec"}.issubset(fold_df.columns):
                fold_df["row_id"] = fold_df["filename"].str.replace(".ogg", "", regex=False) + "_" + fold_df["end_sec"].astype(str)
            elif {"filename", "end"}.issubset(fold_df.columns):
                end_sec = pd.to_timedelta(fold_df["end"]).dt.total_seconds().astype(int)
                fold_df["row_id"] = fold_df["filename"].str.replace(".ogg", "", regex=False) + "_" + end_sec.astype(str)
        fold_col = "fold" if "fold" in fold_df.columns else "fold_id" if "fold_id" in fold_df.columns else None
        if "row_id" in fold_df.columns and fold_col is not None:
            aligned = soundscape_df.merge(
                fold_df[["row_id", fold_col]].drop_duplicates("row_id"),
                on="row_id",
                how="left",
                validate="one_to_one",
            )
            if aligned[fold_col].notna().all():
                aligned["fold"] = aligned[fold_col].astype(int)
                if fold_col != "fold":
                    aligned = aligned.drop(columns=[fold_col])
                print(f"[INFO] Using external fold assignment: {assignment_path}")
                return aligned
            missing = int(aligned[fold_col].isna().sum())
            print(f"[WARN] External fold assignment missing {missing} rows. Rebuilding folds.")
    print("[WARN] External fold assignment not found/usable. Rebuilding folds from soundscape labels.")
    return gm.build_soundscape_folds(soundscape_df, num_classes=num_classes, n_folds=cfg.n_folds, seed=cfg.seed)


def load_raw_wave_pseudo_package(
    cfg: Config,
    input_dir: Path,
    class_names: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    pseudo_dir = Path(cfg.pseudo_dir)
    if not pseudo_dir.is_absolute():
        pseudo_dir = Path(cfg.root) / pseudo_dir
    pseudo_csv = pseudo_dir / "pseudo_segments.csv"
    pseudo_probs_path = pseudo_dir / "pseudo_probs.npy"
    if not pseudo_csv.exists() or not pseudo_probs_path.exists():
        raise FileNotFoundError(
            "Pseudo package must contain pseudo_segments.csv and pseudo_probs.npy under "
            f"{pseudo_dir}"
        )

    pseudo_df = pd.read_csv(pseudo_csv)
    pseudo_probs = np.load(pseudo_probs_path).astype(np.float32, copy=False)
    if len(pseudo_df) != len(pseudo_probs):
        raise RuntimeError(
            f"Pseudo row mismatch: metadata={len(pseudo_df)} probs={len(pseudo_probs)} under {pseudo_dir}"
        )
    if pseudo_probs.shape[1] != len(class_names):
        raise RuntimeError(
            f"Pseudo class mismatch: probs={pseudo_probs.shape[1]} classes={len(class_names)} under {pseudo_dir}"
        )

    required = {"row_id", "filename", "start_sec", "end_sec"}
    missing = sorted(required - set(pseudo_df.columns))
    if missing:
        raise KeyError(f"Pseudo metadata missing columns: {missing}")

    labels_path = input_dir / "train_soundscapes_labels.csv"
    labeled_files = set(pd.read_csv(labels_path)["filename"].astype(str).unique().tolist())
    before = len(pseudo_df)
    unlabeled_mask = ~pseudo_df["filename"].astype(str).isin(labeled_files)
    pseudo_df = pseudo_df.loc[unlabeled_mask].reset_index(drop=True)
    pseudo_probs = pseudo_probs[unlabeled_mask.to_numpy()]
    removed_labeled = before - len(pseudo_df)
    if removed_labeled:
        print(f"[WARN] Dropped {removed_labeled} pseudo rows from labeled files to avoid leakage.")

    if cfg.pseudo_min_max_prob >= 0 and "max_prob" in pseudo_df.columns:
        mask = pseudo_df["max_prob"].astype(float).to_numpy() >= float(cfg.pseudo_min_max_prob)
        pseudo_df = pseudo_df.loc[mask].reset_index(drop=True)
        pseudo_probs = pseudo_probs[mask]
    if cfg.pseudo_max_topk_entropy >= 0 and "topk_entropy" in pseudo_df.columns:
        mask = pseudo_df["topk_entropy"].astype(float).to_numpy() <= float(cfg.pseudo_max_topk_entropy)
        pseudo_df = pseudo_df.loc[mask].reset_index(drop=True)
        pseudo_probs = pseudo_probs[mask]
    if cfg.pseudo_max_rows > 0 and len(pseudo_df) > cfg.pseudo_max_rows:
        rng = np.random.default_rng(cfg.seed + 909)
        selected = np.sort(rng.choice(len(pseudo_df), size=int(cfg.pseudo_max_rows), replace=False))
        pseudo_df = pseudo_df.iloc[selected].reset_index(drop=True)
        pseudo_probs = pseudo_probs[selected]

    if len(pseudo_df) == 0:
        raise ValueError(f"No pseudo rows remain after filtering under {pseudo_dir}")

    audio_root = input_dir / "train_soundscapes"
    pseudo_df["audio_path"] = pseudo_df["filename"].astype(str).map(lambda name: str(audio_root / name))
    missing_audio = [path for path in pseudo_df["audio_path"].head(50).tolist() if not Path(path).exists()]
    if missing_audio:
        raise FileNotFoundError(f"Pseudo audio files missing, examples: {missing_audio[:5]}")

    pseudo_df["label_indices"] = [[] for _ in range(len(pseudo_df))]
    pseudo_df["hard_weight"] = 0.0
    pseudo_df["site"] = pseudo_df.get("site", "pseudo")
    pseudo_df["is_pseudo"] = True
    print(
        f"[INFO] Loaded pseudo package: {pseudo_dir} | rows={len(pseudo_df)} | "
        f"files={pseudo_df['filename'].nunique()} | loss_weight={cfg.pseudo_loss_weight}"
    )
    return pseudo_df, pseudo_probs.astype(np.float32, copy=False)


def build_soundscape_sampler_with_pseudo(
    df: pd.DataFrame,
    num_classes: int,
    samples_per_epoch: int,
    pseudo_sampler_fraction: float,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for _, row in df.iterrows():
        if bool(row.get("is_pseudo", False)):
            continue
        indices = row["label_indices"]
        if indices:
            class_counts[np.asarray(indices, dtype=np.int64)] += 1.0

    weights = []
    for _, row in df.iterrows():
        indices = row["label_indices"]
        if bool(row.get("is_pseudo", False)):
            weight = 1.0
        elif indices:
            label_counts = class_counts[np.asarray(indices, dtype=np.int64)]
            weight = float(np.max(1.0 / np.sqrt(np.maximum(label_counts, 1.0))))
        else:
            weight = 1.0
        weights.append(weight)
    weights = np.asarray(weights, dtype=np.float64)

    pseudo_mask = df.get("is_pseudo", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    if 0 <= pseudo_sampler_fraction <= 1 and pseudo_mask.any() and (~pseudo_mask).any():
        pseudo_sum = float(weights[pseudo_mask].sum())
        real_sum = float(weights[~pseudo_mask].sum())
        target = float(pseudo_sampler_fraction)
        if target <= 0:
            weights[pseudo_mask] = 0.0
        elif target >= 1:
            weights[~pseudo_mask] = 0.0
        elif pseudo_sum > 0 and real_sum > 0:
            current_ratio = pseudo_sum / real_sum
            target_ratio = target / max(1.0 - target, 1e-8)
            weights[pseudo_mask] *= target_ratio / max(current_ratio, 1e-12)

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )


def build_stage1_loaders(cfg: Config, train_audio_df: pd.DataFrame, num_classes: int):
    train_df = train_audio_df[train_audio_df["clip_split"] == "train"].reset_index(drop=True)
    valid_df = train_audio_df[train_audio_df["clip_split"] == "valid"].reset_index(drop=True)
    train_dataset = RawTrainAudioDataset(
        train_df,
        cfg=cfg,
        num_classes=num_classes,
        train_mode=True,
        mixup_alpha=cfg.stage1_mixup_alpha,
        mixup_prob=cfg.stage1_mixup_prob,
    )
    valid_dataset = RawTrainAudioDataset(valid_df, cfg=cfg, num_classes=num_classes, train_mode=False)
    train_sampler = gm.build_train_audio_sampler(
        train_df,
        samples_per_epoch=cfg.stage1_samples_per_epoch,
        generator=gm.build_torch_generator(cfg.seed + 101),
    )
    train_loader = gm.create_dataloader(
        train_dataset,
        batch_size=cfg.stage1_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 102,
        sampler=train_sampler,
    )
    valid_loader = gm.create_dataloader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed + 103,
        shuffle=False,
    )
    return train_loader, valid_loader


def build_stage2_loaders(
    cfg: Config,
    soundscape_df: pd.DataFrame,
    fold: int,
    num_classes: int,
    teacher_probs: Optional[np.ndarray] = None,
    pseudo_df: Optional[pd.DataFrame] = None,
    pseudo_probs: Optional[np.ndarray] = None,
):
    train_mask = (soundscape_df["fold"] != fold).to_numpy()
    valid_mask = (soundscape_df["fold"] == fold).to_numpy()
    train_df = soundscape_df.loc[train_mask].reset_index(drop=True)
    valid_df = soundscape_df.loc[valid_mask].reset_index(drop=True)
    train_teacher_probs = teacher_probs[train_mask] if teacher_probs is not None else None
    if "hard_weight" not in train_df.columns:
        train_df["hard_weight"] = 1.0
    train_df["is_pseudo"] = False

    train_teacher_weights = None
    if train_teacher_probs is not None:
        train_teacher_weights = np.full(len(train_df), float(cfg.teacher_loss_weight), dtype=np.float32)
    if pseudo_df is not None and pseudo_probs is not None and len(pseudo_df) > 0:
        pseudo_part = pseudo_df.copy().reset_index(drop=True)
        train_df = pd.concat([train_df, pseudo_part], axis=0, ignore_index=True)
        if train_teacher_probs is None:
            train_teacher_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
            train_teacher_probs[-len(pseudo_part):] = pseudo_probs
            train_teacher_weights = np.zeros(len(train_df), dtype=np.float32)
            train_teacher_weights[-len(pseudo_part):] = float(cfg.pseudo_loss_weight)
        else:
            train_teacher_probs = np.concatenate([train_teacher_probs, pseudo_probs], axis=0).astype(np.float32, copy=False)
            base_weights = train_teacher_weights
            if base_weights is None:
                base_weights = np.full(len(train_df) - len(pseudo_part), float(cfg.teacher_loss_weight), dtype=np.float32)
            pseudo_weights = np.full(len(pseudo_part), float(cfg.pseudo_loss_weight), dtype=np.float32)
            train_teacher_weights = np.concatenate([base_weights, pseudo_weights], axis=0).astype(np.float32, copy=False)

    seed_base = cfg.seed + (1000 * (fold + 1))
    train_dataset = RawSoundscapeDataset(
        train_df,
        cfg=cfg,
        num_classes=num_classes,
        train_mode=True,
        teacher_probs=train_teacher_probs,
        teacher_weights=train_teacher_weights,
        mixup_alpha=cfg.stage2_mixup_alpha,
        mixup_prob=cfg.stage2_mixup_prob,
    )
    valid_dataset = RawSoundscapeDataset(valid_df, cfg=cfg, num_classes=num_classes, train_mode=False)
    train_sampler = build_soundscape_sampler_with_pseudo(
        train_df,
        num_classes=num_classes,
        samples_per_epoch=cfg.stage2_samples_per_epoch,
        pseudo_sampler_fraction=cfg.pseudo_sampler_fraction,
        generator=gm.build_torch_generator(seed_base + 1),
    )
    train_loader = gm.create_dataloader(
        train_dataset,
        batch_size=cfg.stage2_batch_size,
        num_workers=cfg.num_workers,
        seed=seed_base + 2,
        sampler=train_sampler,
    )
    valid_loader = gm.create_dataloader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=seed_base + 3,
        shuffle=False,
    )
    return train_loader, valid_loader, valid_df


def build_model(cfg: Config, num_classes: int) -> RawWaveformTransformer:
    if cfg.tokenizer_type in {"conv_stack", "sinc_stack"}:
        model_cls = RawWaveTransformerMixerModel if cfg.waveform_model_variant == "mixer" else RawWaveTransformerModel
        return model_cls(
            num_classes=num_classes,
            embed_dim=cfg.d_model,
            depth=cfg.transformer_layers,
            num_heads=cfg.transformer_heads,
            mlp_ratio=cfg.transformer_ff_mult,
            dropout=cfg.dropout,
            num_tokens=cfg.num_tokens,
            tokenizer_type=cfg.tokenizer_type,
        )
    if cfg.waveform_model_variant != "base":
        print("[WARN] --waveform-model-variant is ignored when tokenizer_type=patch_stem.")
    return RawWaveformTransformer(
        num_classes=num_classes,
        sample_rate=cfg.sample_rate,
        clip_seconds=cfg.clip_seconds,
        num_tokens=cfg.num_tokens,
        d_model=cfg.d_model,
        num_layers=cfg.transformer_layers,
        num_heads=cfg.transformer_heads,
        ff_mult=cfg.transformer_ff_mult,
        dropout=cfg.dropout,
    )


def build_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int):
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = min(max(steps_per_epoch * warmup_epochs, 1), total_steps - 1) if total_steps > 1 else 1
    cosine_steps = max(total_steps - warmup_steps, 1)
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    train_mode: bool,
    scaler: GradScaler,
    amp_enabled: bool,
    grad_clip_norm: float,
    progress_desc: str,
    teacher_loss_weight: float = 0.0,
):
    model.train(train_mode)
    running_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []
    row_ids = []
    sites = []
    logit_sum = 0.0
    logit_sq_sum = 0.0
    logit_count = 0
    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        desc=progress_desc,
        dynamic_ncols=True,
        disable=gm.should_disable_tqdm(),
    )
    for batch in progress:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        hard_weight = batch.get("hard_weight")
        teacher_targets = batch.get("teacher_target")
        has_teacher = batch.get("has_teacher")
        teacher_weight = batch.get("teacher_weight")
        if hard_weight is not None:
            hard_weight = hard_weight.to(device, non_blocking=True).float()
        if teacher_targets is not None:
            teacher_targets = teacher_targets.to(device, non_blocking=True)
        if has_teacher is not None:
            has_teacher = has_teacher.to(device, non_blocking=True).bool()
        if teacher_weight is not None:
            teacher_weight = teacher_weight.to(device, non_blocking=True).float()
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if train_mode else torch.inference_mode()
        with context:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(waveforms)
                if hard_weight is None:
                    loss = criterion(logits, targets)
                else:
                    hard_loss_per_sample = F.binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        reduction="none",
                    ).mean(dim=1)
                    hard_den = hard_weight.sum().clamp_min(1e-6)
                    loss = (hard_loss_per_sample * hard_weight).sum() / hard_den
                if train_mode and teacher_targets is not None and has_teacher is not None:
                    if teacher_weight is None:
                        teacher_weight = torch.full_like(has_teacher.float(), float(teacher_loss_weight))
                    active_teacher = has_teacher & (teacher_weight > 0)
                    if bool(active_teacher.any().item()):
                        teacher_loss_per_sample = F.binary_cross_entropy_with_logits(
                            logits[active_teacher],
                            teacher_targets[active_teacher],
                            reduction="none",
                        ).mean(dim=1)
                        active_weight = teacher_weight[active_teacher]
                        teacher_loss = (teacher_loss_per_sample * active_weight).sum() / active_weight.sum().clamp_min(1e-6)
                        loss = loss + teacher_loss

        if train_mode:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = int(waveforms.size(0))
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        metric_logits = logits.float().detach()
        logit_sum += float(metric_logits.sum().cpu())
        logit_sq_sum += float(metric_logits.square().sum().cpu())
        logit_count += int(metric_logits.numel())
        logit_mean = logit_sum / max(logit_count, 1)
        logit_std = math.sqrt(max((logit_sq_sum / max(logit_count, 1)) - (logit_mean**2), 0.0))
        progress.set_postfix(loss=f"{running_loss / max(sample_count, 1):.4f}", logit_std=f"{logit_std:.3f}")

        if not train_mode:
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(torch.sigmoid(metric_logits).detach().cpu().numpy())
            row_ids.extend(batch["row_id"])
            sites.extend(batch.get("site", ["unknown"] * batch_size))

        del waveforms, targets, hard_weight, teacher_targets, has_teacher, teacher_weight, logits, loss

    result = {
        "loss": running_loss / max(sample_count, 1),
        "y_true": np.concatenate(y_true, axis=0) if y_true else None,
        "y_pred": np.concatenate(y_pred, axis=0) if y_pred else None,
        "row_ids": row_ids,
        "sites": sites,
        "logit_std": math.sqrt(max((logit_sq_sum / max(logit_count, 1)) - ((logit_sum / max(logit_count, 1)) ** 2), 0.0)),
    }
    return result


def fit_stage(
    stage_name: str,
    model: nn.Module,
    train_loader,
    valid_loader,
    device: torch.device,
    output_dir: Path,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int,
    amp_enabled: bool,
    grad_clip_norm: float,
    patience: int,
    teacher_loss_weight: float = 0.0,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader), epochs=epochs, warmup_epochs=warmup_epochs)
    scaler = GradScaler(enabled=amp_enabled)
    best_metric = -np.inf
    best_path = output_dir / f"{stage_name}_best.pth"
    history = []
    patience_left = patience

    for epoch in range(1, epochs + 1):
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
            grad_clip_norm=grad_clip_norm,
            progress_desc=f"{stage_name}/train/e{epoch:02d}",
            teacher_loss_weight=teacher_loss_weight,
        )
        valid_result = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=device,
            train_mode=False,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=0.0,
            progress_desc=f"{stage_name}/valid/e{epoch:02d}",
            teacher_loss_weight=0.0,
        )
        valid_metric = gm.macro_auc_skip_missing(valid_result["y_true"], valid_result["y_pred"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "valid_loss": valid_result["loss"],
                "valid_auc": valid_metric,
                "train_logit_std": train_result["logit_std"],
                "valid_logit_std": valid_result["logit_std"],
            }
        )
        print(
            f"[{stage_name}] epoch={epoch:02d} "
            f"train_loss={train_result['loss']:.4f} valid_loss={valid_result['loss']:.4f} "
            f"valid_auc={valid_metric:.5f} "
            f"train_logit_std={train_result['logit_std']:.4f} valid_logit_std={valid_result['logit_std']:.4f}"
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
    pd.DataFrame(history).to_csv(output_dir / f"{stage_name}_history.csv", index=False)
    return model, best_path


def evaluate_soundscape_model(model: nn.Module, loader, device: torch.device, amp_enabled: bool) -> pd.DataFrame:
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
        grad_clip_norm=0.0,
        progress_desc="soundscape_eval",
        teacher_loss_weight=0.0,
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
    score = gm.macro_auc_skip_missing(
        target_df[target_cols].to_numpy(dtype=np.float32),
        prediction_df[pred_cols].to_numpy(dtype=np.float32),
    )
    slices = gm.auc_frequency_slices(
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


def run_pipeline(cfg: Config, input_dir: Path, run_dir: Path) -> None:
    class_names = gm.load_class_names(input_dir)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    num_classes = len(class_names)

    print("[INFO] Loading metadata...")
    train_audio_df = gm.load_train_audio_metadata(cfg, input_dir=input_dir, label_to_idx=label_to_idx)
    soundscape_df = gm.load_soundscape_segments(cfg, input_dir=input_dir, label_to_idx=label_to_idx)
    soundscape_df = apply_fold_assignment(soundscape_df, cfg=cfg, num_classes=num_classes)
    soundscape_summary = gm.summarize_soundscape_folds(soundscape_df, num_classes=num_classes)
    soundscape_df.to_csv(run_dir / "soundscape_segments_with_folds.csv", index=False)
    soundscape_summary.to_csv(run_dir / "soundscape_fold_summary.csv", index=False)
    teacher_probs = None
    teacher_path = None
    pseudo_df = None
    pseudo_probs = None
    expected_targets = np.stack(
        soundscape_df["label_indices"].map(lambda x: gm.indices_to_multihot(x, num_classes)).to_numpy()
    ).astype(np.float32)
    if cfg.teacher_oof_path:
        teacher_path = Path(cfg.teacher_oof_path)
        if not teacher_path.is_absolute():
            teacher_path = Path(cfg.root) / teacher_path
        teacher_probs = load_teacher_oof_predictions(
            teacher_path=teacher_path,
            row_ids=soundscape_df["row_id"].astype(str).tolist(),
            class_names=class_names,
            expected_targets=expected_targets,
        )
        print(
            f"[INFO] Loaded fold-safe teacher OOF: {teacher_path} | "
            f"shape={teacher_probs.shape} | teacher_loss_weight={cfg.teacher_loss_weight}"
        )
        if teacher_path.suffix.lower() == ".npz" and "pred_by_fold" in np.load(teacher_path, allow_pickle=True).files:
            print("[INFO] Teacher package contains pred_by_fold; each stage2 fold will use its matching strict teacher slice.")
    if cfg.pseudo_dir:
        if cfg.pseudo_loss_weight <= 0:
            raise ValueError("--pseudo-loss-weight must be positive when --pseudo-dir is provided")
        pseudo_df, pseudo_probs = load_raw_wave_pseudo_package(
            cfg=cfg,
            input_dir=input_dir,
            class_names=class_names,
        )
        pseudo_df.to_csv(run_dir / "pseudo_segments_used.csv", index=False)
        np.save(run_dir / "pseudo_probs_used.npy", pseudo_probs.astype(np.float16))

    print("[INFO] Soundscape fold summary:")
    print(soundscape_summary.to_string(index=False))
    print("[INFO] Leakage policy: folds are assigned at full soundscape filename level and aligned to CNN 20260505_195634 when available.")
    if pseudo_df is not None:
        print(
            "[INFO] Pseudo leakage policy: pseudo rows are from unlabeled soundscape files only, "
            "are added to stage2 train loaders only, and never enter validation folds."
        )
    if cfg.build_folds_only:
        print(f"[INFO] Folds saved under {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(
        f"[INFO] CUDA diagnostic: available={torch.cuda.is_available()} "
        f"count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    print(
        f"[INFO] Device={device} | AMP={'enabled' if amp_enabled else 'disabled'} | "
        f"tokenizer={cfg.tokenizer_type} | "
        f"waveform_variant={cfg.waveform_model_variant} | "
        f"tokens={cfg.num_tokens} token_samples={int(round(cfg.sample_rate * cfg.clip_seconds)) // cfg.num_tokens} "
        f"d_model={cfg.d_model} layers={cfg.transformer_layers} heads={cfg.transformer_heads}"
    )
    print(
        f"[INFO] Stage1: epochs={cfg.stage1_epochs} samples_per_epoch={cfg.stage1_samples_per_epoch} "
        f"batch={cfg.stage1_batch_size} lr={cfg.stage1_lr} mixup={cfg.stage1_mixup_alpha}/{cfg.stage1_mixup_prob}"
    )
    print(
        f"[INFO] Stage2: epochs={cfg.stage2_epochs} samples_per_epoch={cfg.stage2_samples_per_epoch} "
        f"batch={cfg.stage2_batch_size} lr={cfg.stage2_lr} mixup={cfg.stage2_mixup_alpha}/{cfg.stage2_mixup_prob}"
    )
    print(
        f"[INFO] Raw strong aug: {cfg.raw_strong_aug} | "
        f"gain=({cfg.raw_gain_min},{cfg.raw_gain_max}) polarity={cfg.raw_polarity_prob} "
        f"shift={cfg.raw_time_shift_prob}/{cfg.raw_time_shift_max_sec}s "
        f"noise={cfg.raw_noise_prob}/({cfg.raw_noise_min},{cfg.raw_noise_max}) "
        f"filter={cfg.raw_filter_prob}"
    )

    stage1_dir = run_dir / "stage1_audio"
    gm.ensure_dir(stage1_dir)
    if cfg.stage1_checkpoint_path:
        stage1_best_path = Path(cfg.stage1_checkpoint_path)
        if not stage1_best_path.is_absolute():
            stage1_best_path = Path(cfg.root) / stage1_best_path
        if not stage1_best_path.exists():
            raise FileNotFoundError(f"Stage1 checkpoint does not exist: {stage1_best_path}")
        print(f"[INFO] Reusing stage1 checkpoint: {stage1_best_path}")
    else:
        stage1_train_loader, stage1_valid_loader = build_stage1_loaders(cfg, train_audio_df=train_audio_df, num_classes=num_classes)
        base_model = build_model(cfg, num_classes=num_classes).to(device)
        base_model, stage1_best_path = fit_stage(
            stage_name="stage1_audio",
            model=base_model,
            train_loader=stage1_train_loader,
            valid_loader=stage1_valid_loader,
            device=device,
            output_dir=stage1_dir,
            lr=cfg.stage1_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.stage1_epochs,
            warmup_epochs=cfg.warmup_epochs,
            amp_enabled=amp_enabled,
            grad_clip_norm=cfg.grad_clip_norm,
            patience=cfg.patience,
            teacher_loss_weight=0.0,
        )
        del base_model
        torch.cuda.empty_cache()

    oof_frames = []
    fold_scores = []
    for fold in range(cfg.n_folds):
        print(f"[INFO] Stage 2: fold {fold + 1}/{cfg.n_folds}")
        fold_dir = run_dir / f"fold_{fold}"
        gm.ensure_dir(fold_dir)
        model = build_model(cfg, num_classes=num_classes).to(device)
        stage1_checkpoint = torch.load(stage1_best_path, map_location="cpu")
        model.load_state_dict(stage1_checkpoint["model"], strict=True)
        fold_teacher_probs = teacher_probs
        if teacher_path is not None:
            fold_teacher_probs = load_teacher_predictions_for_fold(
                teacher_path=teacher_path,
                row_ids=soundscape_df["row_id"].astype(str).tolist(),
                fold=fold,
                class_names=class_names,
                expected_targets=expected_targets,
            )
        train_loader, valid_loader, valid_df = build_stage2_loaders(
            cfg,
            soundscape_df=soundscape_df,
            fold=fold,
            num_classes=num_classes,
            teacher_probs=fold_teacher_probs,
            pseudo_df=pseudo_df,
            pseudo_probs=pseudo_probs,
        )
        model, best_path = fit_stage(
            stage_name=f"stage2_fold{fold}",
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            output_dir=fold_dir,
            lr=cfg.stage2_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.stage2_epochs,
            warmup_epochs=cfg.warmup_epochs,
            amp_enabled=amp_enabled,
            grad_clip_norm=cfg.grad_clip_norm,
            patience=cfg.patience,
            teacher_loss_weight=cfg.teacher_loss_weight if fold_teacher_probs is not None else 0.0,
        )
        prediction_df = evaluate_soundscape_model(model, valid_loader, device=device, amp_enabled=amp_enabled)
        prediction_df.insert(1, "fold", fold)
        prediction_df.to_csv(fold_dir / "valid_predictions.csv", index=False)
        fold_scores.append(float(prediction_df["fold_auc"].iloc[0]))

        prediction_df = prediction_df.rename(columns={i: class_names[i] for i in range(num_classes)})
        truth_df = valid_df[["row_id", "site", "label_indices"]].copy()
        truth_matrix = np.stack(valid_df["label_indices"].map(lambda x: gm.indices_to_multihot(x, num_classes)).to_numpy())
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
    final_cv = gm.macro_auc_skip_missing(y_true, y_pred)
    print(f"[INFO] Final OOF local CV = {final_cv:.6f}")
    oof_df.to_csv(run_dir / "soundscape_oof_predictions.csv", index=False)
    pd.DataFrame({"fold": list(range(cfg.n_folds)), "fold_auc": fold_scores}).to_csv(run_dir / "fold_scores.csv", index=False)
    gm.save_json(
        run_dir / "metrics.json",
        {
            "final_oof_cv": final_cv,
            "fold_scores": fold_scores,
            "n_soundscape_segments": int(len(soundscape_df)),
            "n_soundscape_files": int(soundscape_df["filename"].nunique()),
            "teacher_oof_path": cfg.teacher_oof_path,
            "teacher_loss_weight": cfg.teacher_loss_weight,
            "pseudo_dir": cfg.pseudo_dir,
            "pseudo_loss_weight": cfg.pseudo_loss_weight,
            "pseudo_sampler_fraction": cfg.pseudo_sampler_fraction,
            "n_pseudo_segments": int(len(pseudo_df)) if pseudo_df is not None else 0,
            "n_pseudo_files": int(pseudo_df["filename"].nunique()) if pseudo_df is not None else 0,
        },
    )
    print(f"[INFO] Run artifacts saved to {run_dir}")


def main() -> None:
    cfg = parse_args()
    gm.seed_everything(cfg.seed)
    if os.environ.get("BIRDCLEF_KEEP_CUBLAS_WORKSPACE_CONFIG") != "1":
        # This older torch/CUDA stack can lose CUDA visibility when this env is
        # set before the first CUDA check. Raw waveform training needs GPU more
        # than strict deterministic cublas kernels.
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    root = Path(cfg.root).resolve()
    input_dir = root / cfg.input_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / cfg.output_dir / f"{timestamp}_{cfg.model_name.replace('/', '_')}"
    gm.ensure_dir(run_dir)
    gm.save_json(run_dir / "config.json", asdict(cfg))
    log_path = run_dir / "train.log"
    with gm.RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        run_pipeline(cfg=cfg, input_dir=input_dir, run_dir=run_dir)


if __name__ == "__main__":
    main()
