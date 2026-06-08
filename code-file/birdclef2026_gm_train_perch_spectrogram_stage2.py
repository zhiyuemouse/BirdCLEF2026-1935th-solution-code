#!/usr/bin/env python3
"""Train a CNN on cached Perch frontend spectrograms.

This is a quick fold-safe probe for two Perch spectrogram image views:

1. direct_repeat:
       Perch ONNX spectrogram [B, 500, 128]
       -> repeat channels -> [B, 3, 500, 128]

2. transpose_resize:
       Perch ONNX spectrogram [B, time=500, freq=128]
       -> transpose to [B, freq=128, time=500]
       -> bilinear resize to [B, 3, image_height, image_width]

Both variants keep fold assignment at full soundscape-file level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import birdclef2026_gm_train as gm


@dataclass
class Config:
    root: str = "."
    input_dir: str = "input"
    ckpt_dir: str = "ckpt"
    output_dir: str = "outputs/birdclef2026_gm_perch_spectrogram_stage2"
    cache_dir: str = "perch_spectrogram_cache_labeled_all"
    meta_path: str = ""
    arrays_path: str = ""
    model_name: str = "convnextv2_atto.fcmae_ft_in1k"
    head_type: str = "csiro_conv_v1"
    head_pool_type: str = "avg"
    spec_process: str = "direct_repeat"
    image_height: int = 256
    image_width: int = 320
    n_folds: int = 3
    seed: int = 2026
    stage2_epochs: int = 28
    stage2_batch_size: int = 8
    eval_batch_size: int = 16
    num_workers: int = max(2, (os.cpu_count() or 4) // 2)
    stage2_samples_per_epoch: int = 2048
    stage2_backbone_lr: float = 5e-5
    stage2_head_lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    stage2_freeze_backbone_epochs: int = 1
    dropout: float = 0.2
    drop_path: float = 0.1
    specaug_time_mask: int = 32
    specaug_freq_mask: int = 24
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.0
    patience: int = 5
    use_amp: bool = True
    amp_mode: str = "auto"
    grad_clip_norm: float = 0.0
    smoke_test: bool = False
    max_soundscape_segments: int = -1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train CNN on cached Perch spectrograms.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--ckpt-dir", type=str, default="ckpt")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_perch_spectrogram_stage2")
    parser.add_argument("--cache-dir", type=str, default="perch_spectrogram_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="convnextv2_atto.fcmae_ft_in1k")
    parser.add_argument("--head-type", type=str, choices=["linear", "csiro_conv_v1"], default="csiro_conv_v1")
    parser.add_argument("--head-pool-type", type=str, choices=["avg", "gem", "lse", "avg_max"], default="avg")
    parser.add_argument("--spec-process", type=str, choices=["direct_repeat", "transpose_resize"], default="direct_repeat")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage2-epochs", type=int, default=28)
    parser.add_argument("--stage2-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--stage2-samples-per-epoch", type=int, default=2048)
    parser.add_argument("--stage2-backbone-lr", type=float, default=5e-5)
    parser.add_argument("--stage2-head-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp-mode", type=str, choices=["auto", "fp16", "bf16", "off"], default="auto")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-soundscape-segments", type=int, default=-1)
    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        input_dir=args.input_dir,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        meta_path=args.meta_path,
        arrays_path=args.arrays_path,
        model_name=args.model_name,
        head_type=args.head_type,
        head_pool_type=args.head_pool_type,
        spec_process=args.spec_process,
        image_height=args.image_height,
        image_width=args.image_width,
        n_folds=args.n_folds,
        seed=args.seed,
        stage2_epochs=args.stage2_epochs,
        stage2_batch_size=args.stage2_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        stage2_samples_per_epoch=args.stage2_samples_per_epoch,
        stage2_backbone_lr=args.stage2_backbone_lr,
        stage2_head_lr=args.stage2_head_lr,
        stage2_freeze_backbone_epochs=args.stage2_freeze_backbone_epochs,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        use_amp=not args.disable_amp,
        amp_mode=args.amp_mode,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        smoke_test=args.smoke_test,
        max_soundscape_segments=args.max_soundscape_segments,
    )
    if cfg.smoke_test:
        cfg.stage2_epochs = 1
        cfg.stage2_samples_per_epoch = min(cfg.stage2_samples_per_epoch, 128)
        cfg.max_soundscape_segments = 96 if cfg.max_soundscape_segments < 0 else min(cfg.max_soundscape_segments, 96)
        cfg.num_workers = min(cfg.num_workers, 2)
        cfg.patience = 1
    return cfg


class PerchSpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spectrograms: np.ndarray, num_classes: int, train_mode: bool, cfg: Config):
        self.df = df.reset_index(drop=True)
        self.spectrograms = spectrograms
        self.num_classes = int(num_classes)
        self.train_mode = bool(train_mode)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec = torch.from_numpy(self.spectrograms[int(row["cache_idx"])]).float()
        if self.cfg.spec_process == "direct_repeat":
            image = spec.unsqueeze(0).repeat(3, 1, 1)
        elif self.cfg.spec_process == "transpose_resize":
            image = spec.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            image = F.interpolate(
                image,
                size=(self.cfg.image_height, self.cfg.image_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            image = image.repeat(3, 1, 1)
        else:
            raise ValueError(f"Unsupported spec_process: {self.cfg.spec_process}")
        if self.train_mode:
            if random.random() < 0.75:
                image = gm.apply_frequency_mask(image, self.cfg.specaug_freq_mask)
            if random.random() < 0.75:
                image = gm.apply_time_mask(image, self.cfg.specaug_time_mask)
        image = image - image.amin(dim=(-2, -1), keepdim=True)
        image = image / (image.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        image = (image - 0.5) / 0.5
        target = torch.from_numpy(gm.indices_to_multihot(row["label_indices"], self.num_classes)).float()
        return {
            "image": image,
            "target": target,
            "row_id": row["row_id"],
            "site": row["site"],
        }


class PerchSpectrogramCNN(nn.Module):
    def __init__(self, cfg: Config, num_classes: int, backbone_weight_path: Path):
        super().__init__()
        self.head_type = str(cfg.head_type)
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="avg" if self.head_type == "linear" else "",
            drop_path_rate=cfg.drop_path,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        if self.head_type == "linear":
            self.head = nn.Linear(self.backbone.num_features, num_classes)
        elif self.head_type == "csiro_conv_v1":
            self.head = gm.CSIROHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=cfg.dropout,
                pool_type=cfg.head_pool_type,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")
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
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
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

    def forward(self, x: torch.Tensor, waveform: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.head_type == "linear":
            features = self.backbone(x)
            return self.head(self.dropout(features))
        features = self._flatten_feature_sequence(self._forward_backbone_features(x))
        return self.head(features)


def resolve_cache_paths(cfg: Config, root: Path) -> tuple[Path, Path]:
    cache_dir = Path(cfg.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    meta_path = Path(cfg.meta_path) if cfg.meta_path else cache_dir / "perch_spectrogram_meta.parquet"
    arrays_path = Path(cfg.arrays_path) if cfg.arrays_path else cache_dir / "perch_spectrogram_arrays.npz"
    if not meta_path.is_absolute():
        meta_path = root / meta_path
    if not arrays_path.is_absolute():
        arrays_path = root / arrays_path
    return meta_path, arrays_path


def load_cache(cfg: Config, root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta_path, arrays_path = resolve_cache_paths(cfg, root)
    if not meta_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Missing Perch spectrogram cache:\n  {meta_path}\n  {arrays_path}")
    if meta_path.suffix.lower() == ".parquet":
        meta_df = pd.read_parquet(meta_path)
    else:
        meta_df = pd.read_csv(meta_path)
    arrays = np.load(arrays_path)
    if "spectrogram" not in arrays:
        raise KeyError(f"{arrays_path} does not contain `spectrogram`; keys={arrays.files}")
    spectrograms = arrays["spectrogram"].astype(np.float32, copy=False)
    if spectrograms.shape[1:] != (500, 128):
        raise ValueError(f"Unexpected cached spectrogram shape: {spectrograms.shape}")
    meta_df = meta_df.reset_index(drop=True).copy()
    meta_df["cache_idx"] = np.arange(len(meta_df), dtype=np.int64)
    return meta_df, spectrograms


def build_soundscape_df(cfg: Config, input_dir: Path, cache_meta: pd.DataFrame, num_classes: int, label_to_idx: Dict[str, int]) -> pd.DataFrame:
    soundscape_df = gm.load_soundscape_segments(
        gm.Config(max_soundscape_segments=cfg.max_soundscape_segments, seed=cfg.seed),
        input_dir=input_dir,
        label_to_idx=label_to_idx,
    )
    cache_cols = cache_meta[["row_id", "cache_idx"]].drop_duplicates("row_id")
    soundscape_df = soundscape_df.merge(cache_cols, on="row_id", how="inner")
    if len(soundscape_df) == 0:
        raise ValueError("No labeled soundscape rows matched the Perch spectrogram cache.")
    if soundscape_df["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id after cache merge.")
    return gm.build_soundscape_folds(soundscape_df, num_classes=num_classes, n_folds=cfg.n_folds, seed=cfg.seed)


def build_loader(cfg: Config, df: pd.DataFrame, spectrograms: np.ndarray, num_classes: int, train_mode: bool, fold: int) -> DataLoader:
    dataset = PerchSpectrogramDataset(df=df, spectrograms=spectrograms, num_classes=num_classes, train_mode=train_mode, cfg=cfg)
    seed_base = cfg.seed + (1000 * (fold + 1))
    sampler = None
    shuffle = False
    if train_mode:
        sampler = gm.build_soundscape_sampler(
            df,
            num_classes=num_classes,
            samples_per_epoch=cfg.stage2_samples_per_epoch,
            generator=gm.build_torch_generator(seed_base + 1),
        )
    return gm.create_dataloader(
        dataset,
        batch_size=cfg.stage2_batch_size if train_mode else cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=seed_base + (2 if train_mode else 3),
        shuffle=shuffle,
        sampler=sampler,
    )


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def run_pipeline(cfg: Config, input_dir: Path, ckpt_dir: Path, run_dir: Path) -> None:
    class_names = gm.load_class_names(input_dir)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    num_classes = len(class_names)
    cache_meta, spectrograms = load_cache(cfg, root=Path(cfg.root).resolve())
    soundscape_df = build_soundscape_df(
        cfg=cfg,
        input_dir=input_dir,
        cache_meta=cache_meta,
        num_classes=num_classes,
        label_to_idx=label_to_idx,
    )
    soundscape_summary = gm.summarize_soundscape_folds(soundscape_df, num_classes=num_classes)
    soundscape_df.to_csv(run_dir / "soundscape_segments_with_folds.csv", index=False)
    soundscape_summary.to_csv(run_dir / "soundscape_fold_summary.csv", index=False)

    print("[INFO] Perch spectrogram CNN stage2-only")
    if cfg.spec_process == "direct_repeat":
        print("[INFO] Input image: cached Perch spectrogram [500,128] repeated to [3,500,128], no resize")
    else:
        print(
            "[INFO] Input image: cached Perch spectrogram [500,128] -> transpose [128,500] "
            f"-> resize [{cfg.image_height},{cfg.image_width}] -> repeat to 3 channels"
        )
    print(f"[INFO] Cache rows: {len(cache_meta)} | matched labeled rows: {len(soundscape_df)}")
    print("[INFO] Soundscape fold summary:")
    print(soundscape_summary.to_string(index=False))
    print(
        "[INFO] Leakage policy: folds are assigned at full soundscape filename level, "
        "and local CV is computed only on manually labeled soundscape windows."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gm.seed_everything(cfg.seed)
    amp_settings = gm.resolve_amp_settings(cfg, device=device)
    effective_grad_clip_norm = cfg.grad_clip_norm if cfg.grad_clip_norm > 0 else 0.0
    backbone_weight_path = ckpt_dir / f"{cfg.model_name}.pth"
    print(
        f"[INFO] Device={device} | AMP={amp_settings['description']} | "
        f"grad_clip_norm={effective_grad_clip_norm:.2f}"
    )
    print(f"[INFO] Model={cfg.model_name} | head={cfg.head_type} | pool={cfg.head_pool_type}")

    oof_frames = []
    fold_scores = []
    for fold in range(cfg.n_folds):
        print(f"[INFO] Stage 2: fold {fold + 1}/{cfg.n_folds}")
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df = soundscape_df[soundscape_df["fold"] != fold].reset_index(drop=True)
        valid_df = soundscape_df[soundscape_df["fold"] == fold].reset_index(drop=True)
        train_loader = build_loader(cfg, train_df, spectrograms, num_classes, train_mode=True, fold=fold)
        valid_loader = build_loader(cfg, valid_df, spectrograms, num_classes, train_mode=False, fold=fold)

        model = PerchSpectrogramCNN(cfg=cfg, num_classes=num_classes, backbone_weight_path=backbone_weight_path).to(device)
        model, _ = gm.fit_one_stage(
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
            amp_enabled=bool(amp_settings["enabled"]),
            amp_dtype=amp_settings["dtype"],
            use_grad_scaler=bool(amp_settings["use_grad_scaler"]),
            grad_clip_norm=effective_grad_clip_norm,
            patience=cfg.patience,
            freeze_backbone_epochs=cfg.stage2_freeze_backbone_epochs,
            mixup_domain="image",
            mixup_alpha=cfg.mixup_alpha,
            mixup_prob=cfg.mixup_prob,
            cutmix_alpha=cfg.cutmix_alpha,
            cutmix_prob=cfg.cutmix_prob,
        )

        pred_df = gm.evaluate_soundscape_model(
            model=model,
            loader=valid_loader,
            device=device,
            amp_enabled=bool(amp_settings["enabled"]),
            amp_dtype=amp_settings["dtype"],
        )
        pred_df["fold"] = fold
        pred_df.to_csv(fold_dir / "valid_predictions.csv", index=False)
        oof_frames.append(pred_df)
        fold_scores.append(float(pred_df["fold_auc"].iloc[0]))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    oof_df.to_csv(run_dir / "soundscape_oof_predictions.csv", index=False)
    target_map = soundscape_df[["row_id", "label_indices"]].drop_duplicates("row_id")
    target = np.stack(target_map["label_indices"].map(lambda x: gm.indices_to_multihot(x, num_classes)).to_numpy())
    pred_cols = [column for column in oof_df.columns if isinstance(column, int)]
    pred = oof_df.drop_duplicates("row_id").set_index("row_id").loc[target_map["row_id"], pred_cols].to_numpy(dtype=np.float32)
    final_cv = gm.macro_auc_skip_missing(target.astype(np.float32), pred)
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
    print(f"[INFO] Final OOF local CV = {final_cv:.6f}")
    print(f"[INFO] Run artifacts saved to {run_dir}")


def main() -> None:
    cfg = parse_args()
    root = Path(cfg.root).resolve()
    input_dir = root / cfg.input_dir
    ckpt_dir = root / cfg.ckpt_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / cfg.output_dir / f"{timestamp}_{cfg.model_name.replace('/', '_')}_perchspec_stage2"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", asdict(cfg))
    log_path = run_dir / "train.log"
    with gm.RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        run_pipeline(cfg=cfg, input_dir=input_dir, ckpt_dir=ckpt_dir, run_dir=run_dir)


if __name__ == "__main__":
    main()
