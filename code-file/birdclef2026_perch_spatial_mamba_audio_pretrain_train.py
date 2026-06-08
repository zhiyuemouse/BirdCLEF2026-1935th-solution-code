#!/usr/bin/env python3
"""Stage1 train_audio pretrain + fold-safe soundscape finetune for Perch Mamba head."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import birdclef2026_perch_spatial_mamba_train as mamba_train
from birdclef2026_perch_context_train import (
    build_aligned_labels,
    limit_by_files,
    load_cache as load_base_cache,
    load_class_names,
    load_meta,
    macro_auc_skip_empty,
    seed_everything,
    sigmoid_np,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Perch Mamba head on train_audio, finetune on soundscapes.")
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--base-meta-path", type=str, default="")
    parser.add_argument("--base-arrays-path", type=str, default="")
    parser.add_argument("--spatial-cache-dir", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--spatial-meta-path", type=str, default="")
    parser.add_argument("--spatial-arrays-path", type=str, default="")
    parser.add_argument("--audio-cache-dir", type=str, default="perch_audio_spatial_cache_max20")
    parser.add_argument("--audio-meta-path", type=str, default="")
    parser.add_argument("--audio-arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_spatial_mamba_audio_pretrain_max20_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--fold-assignment-path", type=str, default="")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--token-pca-dim", type=int, default=1536)
    parser.add_argument("--freq-pool", type=str, choices=["mean", "flat64"], default="mean")
    parser.add_argument(
        "--head-variant",
        type=str,
        choices=["perch_mamba_v1", "attention_pooling", "multihead_attention_pooling", "prototype_pooling"],
        default="perch_mamba_v1",
    )
    parser.add_argument("--prototype-per-class", type=int, default=5)
    parser.add_argument("--prototype-temperature", type=float, default=12.0)
    parser.add_argument("--prototype-orth-weight", type=float, default=0.01)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--token-mask-prob", type=float, default=0.0)
    parser.add_argument("--token-mask-max-frac", type=float, default=0.15)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-stage1", type=float, default=5e-4)
    parser.add_argument("--lr-stage2", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=12.0)
    parser.add_argument("--inner-val-files", type=int, default=10)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_cache_paths(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates = [Path(meta_path_arg)] if meta_path_arg else [
        cache_dir / "perch_audio_spatial_meta.parquet",
        cache_dir / "perch_audio_spatial_meta.csv",
    ]
    arrays_candidates = [Path(arrays_path_arg)] if arrays_path_arg else [cache_dir / "perch_audio_spatial_arrays.npz"]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find audio meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find audio arrays under {cache_dir}")
    return meta_path, arrays_path


def load_audio_cache(
    cache_dir_arg: str,
    meta_path_arg: str,
    arrays_path_arg: str,
    freq_pool: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path, arrays_path = resolve_cache_paths(cache_dir_arg, meta_path_arg, arrays_path_arg)
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    token_key = "spatial_tokens_64" if freq_pool == "flat64" else "spatial_tokens"
    if token_key not in arrays or "y" not in arrays:
        raise KeyError(f"{arrays_path} must contain {token_key} and y. Available keys: {arrays.files}")
    return (
        meta_df,
        arrays[token_key].astype(np.float32, copy=False),
        arrays["y"].astype(np.float32, copy=False),
    )


def build_loader(
    tokens: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(tokens.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        generator=generator if shuffle else None,
    )


def make_inner_split(train_idx: np.ndarray, groups: np.ndarray, inner_val_files: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_files = pd.Index(groups[train_idx]).unique().to_numpy()
    if inner_val_files <= 0 or len(train_files) <= 2:
        return train_idx, np.asarray([], dtype=np.int64)
    n_val = min(int(inner_val_files), max(1, len(train_files) // 5))
    rng = np.random.default_rng(seed)
    val_files = set(rng.choice(train_files, size=n_val, replace=False).tolist())
    inner_val_mask = np.asarray([name in val_files for name in groups[train_idx]], dtype=bool)
    inner_val_idx = train_idx[inner_val_mask]
    inner_train_idx = train_idx[~inner_val_mask]
    if len(inner_train_idx) == 0 or len(inner_val_idx) == 0:
        return train_idx, np.asarray([], dtype=np.int64)
    return inner_train_idx, inner_val_idx


def compute_pos_weight(y: np.ndarray, fitted_indices: np.ndarray, power: float, max_value: float) -> np.ndarray:
    pos = y.sum(axis=0).astype(np.float32)
    neg = len(y) - pos
    pos_weight = np.ones(y.shape[1], dtype=np.float32)
    valid = pos > 0
    pos_weight[valid] = np.power(neg[valid] / np.maximum(pos[valid], 1.0), float(power))
    pos_weight = np.clip(pos_weight, 1.0, float(max_value)).astype(np.float32)
    fitted_mask = np.zeros(y.shape[1], dtype=bool)
    fitted_mask[fitted_indices] = True
    pos_weight[~fitted_mask] = 1.0
    return pos_weight


def train_epochs(
    model: nn.Module,
    tokens: np.ndarray,
    targets: np.ndarray,
    fitted_indices: np.ndarray,
    args: argparse.Namespace,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    val_tokens: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    fitted_mask_np = np.zeros(targets.shape[1], dtype=bool)
    fitted_mask_np[fitted_indices] = True
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight_np = compute_pos_weight(
        targets,
        fitted_indices=fitted_indices,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )
    pos_weight = torch.from_numpy(pos_weight_np).to(device)
    loader = build_loader(
        tokens=tokens,
        targets=targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(args.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    if val_tokens is not None and val_targets is not None and len(val_tokens) > 0:
        val_tokens_t = torch.from_numpy(val_tokens.astype(np.float32, copy=False)).to(device)
        val_targets_t = torch.from_numpy(val_targets.astype(np.float32, copy=False)).to(device)
    else:
        val_tokens_t = None
        val_targets_t = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: List[float] = []
        for token_batch, y_batch in loader:
            token_batch = token_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            sample_weights = torch.ones(token_batch.shape[0], device=device, dtype=token_batch.dtype)
            token_batch, _, y_batch, _ = mamba_train.apply_feature_mixup(
                tokens=token_batch,
                raw_features=None,
                targets=y_batch,
                sample_weights=sample_weights,
                mixup_prob=float(args.mixup_prob),
                mixup_alpha=float(args.mixup_alpha),
            )
            token_batch = mamba_train.apply_token_masking(
                tokens=token_batch,
                token_mask_prob=float(args.token_mask_prob),
                token_mask_max_frac=float(args.token_mask_max_frac),
            )
            logits = model(token_batch, None)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, fitted_mask],
                y_batch[:, fitted_mask],
                pos_weight=pos_weight[fitted_mask],
                reduction="mean",
            )
            if args.head_variant == "prototype_pooling" and args.prototype_orth_weight > 0:
                loss = loss + float(args.prototype_orth_weight) * mamba_train.prototype_orthogonality_loss(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if val_tokens_t is not None and val_targets_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_tokens_t, None)
                monitor_loss = float(
                    nn.functional.binary_cross_entropy_with_logits(
                        val_logits[:, fitted_mask],
                        val_targets_t[:, fitted_mask],
                        pos_weight=pos_weight[fitted_mask],
                        reduction="mean",
                    )
                    .detach()
                    .cpu()
                    .item()
                )
        else:
            monitor_loss = float(np.mean(losses))
        if monitor_loss < best_loss - 1e-5:
            best_loss = monitor_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    return best_state, {"best_epoch": float(best_epoch), "best_loss": float(best_loss)}


def predict_model(model: nn.Module, tokens: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), int(batch_size)):
            token_batch = torch.from_numpy(tokens[start:start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            pred = torch.sigmoid(model(token_batch, None)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    spatial_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    fitted_classes: int
    token_dim: int
    stage1_best_epoch: int
    stage1_best_loss: float
    stage2_best_epoch: int
    stage2_best_loss: float


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    class_names = load_class_names(Path(args.sample_submission_path))

    meta_df, scores_full_raw, emb_full = load_base_cache(
        cache_dir=Path(args.base_cache_dir),
        meta_path_arg=args.base_meta_path,
        arrays_path_arg=args.base_arrays_path,
    )
    y_true = build_aligned_labels(
        labels_path=Path(args.labels_path),
        class_names=class_names,
        meta_df=meta_df,
    )
    meta_df, y_true, scores_full_raw, emb_full = limit_by_files(
        meta_df=meta_df,
        y_true=y_true,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
        limit_files=args.limit_files,
    )
    spatial_meta, spatial_tokens, _, spatial_tokens_64 = mamba_train.load_spatial_cache_from_paths(
        cache_dir_arg=args.spatial_cache_dir,
        meta_path_arg=args.spatial_meta_path,
        arrays_path_arg=args.spatial_arrays_path,
    )
    if args.freq_pool == "flat64":
        if spatial_tokens_64 is None:
            raise ValueError("freq_pool=flat64 requires spatial_tokens_64 in the soundscape spatial cache.")
        spatial_tokens = spatial_tokens_64
    spatial_tokens = mamba_train.align_spatial_to_base(
        base_meta=meta_df,
        spatial_meta=spatial_meta,
        spatial_tokens=spatial_tokens,
    )
    audio_meta, audio_tokens, audio_y = load_audio_cache(
        cache_dir_arg=args.audio_cache_dir,
        meta_path_arg=args.audio_meta_path,
        arrays_path_arg=args.audio_arrays_path,
        freq_pool=args.freq_pool,
    )
    if audio_y.shape[1] != len(class_names):
        raise ValueError(f"Audio target class count mismatch: {audio_y.shape[1]} vs {len(class_names)}")

    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32)
    raw_sigmoid_auc = macro_auc_skip_empty(y_true, raw_sigmoid)
    splits = mamba_train.build_splits(meta_df, args=args)
    groups = meta_df["filename"].to_numpy()
    oof_pred = raw_sigmoid.copy()
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []

    print("[INFO] Train Perch Mamba audio-pretrain + soundscape finetune")
    print(f"[INFO] soundscape rows: {len(meta_df)} files={meta_df['filename'].nunique()}")
    print(f"[INFO] audio rows: {len(audio_meta)} files={audio_meta['filename'].nunique() if 'filename' in audio_meta else len(audio_meta)}")
    print(f"[INFO] spatial_tokens: {spatial_tokens.shape} audio_tokens: {audio_tokens.shape}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs}")
    print(
        "[INFO] train_aug: "
        f"token_mask_prob={args.token_mask_prob} token_mask_max_frac={args.token_mask_max_frac} "
        f"mixup_prob={args.mixup_prob} mixup_alpha={args.mixup_alpha}"
    )
    if args.fold_assignment_path:
        print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")

    for display_fold, train_idx, valid_idx in splits:
        token_projector = mamba_train.fit_tokens_projector(
            tokens_train=spatial_tokens[train_idx],
            token_pca_dim=args.token_pca_dim,
            seed=args.seed + int(display_fold),
        )
        tokens_train = mamba_train.transform_tokens_with_projector(spatial_tokens[train_idx], token_projector)
        tokens_valid = mamba_train.transform_tokens_with_projector(spatial_tokens[valid_idx], token_projector)
        audio_tokens_proj = mamba_train.transform_tokens_with_projector(audio_tokens, token_projector)

        real_pos = y_true[train_idx].sum(axis=0)
        real_neg = len(train_idx) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)
        inner_train_idx, inner_val_idx = make_inner_split(
            train_idx=np.arange(len(train_idx), dtype=np.int64),
            groups=groups[train_idx],
            inner_val_files=args.inner_val_files,
            seed=args.seed + int(display_fold) * 1000,
        )

        model = mamba_train.PerchSpatialMambaHead(
            token_dim=tokens_train.shape[-1],
            num_classes=len(class_names),
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            raw_dim=0,
            freq_pool=args.freq_pool,
            use_pos_embed=False,
            head_variant=args.head_variant,
            prototype_per_class=args.prototype_per_class,
            prototype_temperature=args.prototype_temperature,
        ).to(device)
        stage1_state, stage1_stats = train_epochs(
            model=model,
            tokens=audio_tokens_proj,
            targets=audio_y,
            fitted_indices=fitted_class_indices,
            args=args,
            lr=args.lr_stage1,
            epochs=args.stage1_epochs,
            seed=args.seed + int(display_fold) * 1000 + 13,
            device=device,
            val_tokens=None,
            val_targets=None,
        )
        model.load_state_dict(stage1_state)
        val_tokens = tokens_train[inner_val_idx] if len(inner_val_idx) else None
        val_targets = y_true[train_idx][inner_val_idx] if len(inner_val_idx) else None
        stage2_state, stage2_stats = train_epochs(
            model=model,
            tokens=tokens_train[inner_train_idx],
            targets=y_true[train_idx][inner_train_idx],
            fitted_indices=fitted_class_indices,
            args=args,
            lr=args.lr_stage2,
            epochs=args.stage2_epochs,
            seed=args.seed + int(display_fold) * 1000 + 29,
            device=device,
            val_tokens=val_tokens,
            val_targets=val_targets,
        )
        model.load_state_dict(stage2_state)
        spatial_all = predict_model(model=model, tokens=tokens_valid, device=device, batch_size=args.batch_size)
        fold_pred = raw_sigmoid[valid_idx].copy()
        fold_pred[:, fitted_class_indices] = spatial_all[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], raw_sigmoid[valid_idx])
        spatial_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        model_artifact = {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "token_dim": int(tokens_train.shape[-1]),
            "raw_dim": 0,
            "output_dim": len(class_names),
            "num_blocks": int(args.num_blocks),
            "kernel_size": int(args.kernel_size),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "token_mask_prob": float(args.token_mask_prob),
            "token_mask_max_frac": float(args.token_mask_max_frac),
            "mixup_prob": float(args.mixup_prob),
            "mixup_alpha": float(args.mixup_alpha),
            "freq_pool": str(args.freq_pool),
            "use_pos_embed": False,
            "head_variant": str(args.head_variant),
            "prototype_per_class": int(args.prototype_per_class),
            "prototype_temperature": float(args.prototype_temperature),
            "prototype_orth_weight": float(args.prototype_orth_weight),
            "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
            "stage1_best_epoch": int(stage1_stats["best_epoch"]),
            "stage1_best_loss": float(stage1_stats["best_loss"]),
            "stage2_best_epoch": int(stage2_stats["best_epoch"]),
            "stage2_best_loss": float(stage2_stats["best_loss"]),
        }
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "token_projector": mamba_train.tokens_projector_to_artifact(token_projector),
                "raw_projector": None,
                "model": model_artifact,
            }
        )
        result = FoldResult(
            fold=int(display_fold),
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            spatial_valid_auc=float(spatial_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            fitted_classes=int(len(fitted_class_indices)),
            token_dim=int(tokens_train.shape[-1]),
            stage1_best_epoch=int(stage1_stats["best_epoch"]),
            stage1_best_loss=float(stage1_stats["best_loss"]),
            stage2_best_epoch=int(stage2_stats["best_epoch"]),
            stage2_best_loss=float(stage2_stats["best_loss"]),
        )
        fold_results.append(result)
        print(
            f"[FOLD {display_fold}] raw_sigmoid_auc={raw_valid_auc:.6f} "
            f"spatial_auc={spatial_valid_auc:.6f} token_dim={result.token_dim} "
            f"fitted_classes={result.fitted_classes} stage1_epoch={result.stage1_best_epoch} "
            f"stage2_epoch={result.stage2_best_epoch}",
            flush=True,
        )

    spatial_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_spatial_auc = float(np.mean([item.spatial_valid_auc for item in fold_results]))
    artifact = {
        "artifact_version": 1,
        "model_type": "perch_spatial_mamba",
        "class_names": class_names,
        "config": {
            "audio_cache_dir": str(args.audio_cache_dir),
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "token_pca_dim": int(args.token_pca_dim),
            "freq_pool": str(args.freq_pool),
            "include_raw_scores": False,
            "head_variant": str(args.head_variant),
            "prototype_per_class": int(args.prototype_per_class),
            "prototype_temperature": float(args.prototype_temperature),
            "prototype_orth_weight": float(args.prototype_orth_weight),
            "num_blocks": int(args.num_blocks),
            "kernel_size": int(args.kernel_size),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "token_mask_prob": float(args.token_mask_prob),
            "token_mask_max_frac": float(args.token_mask_max_frac),
            "mixup_prob": float(args.mixup_prob),
            "mixup_alpha": float(args.mixup_alpha),
            "mlp_min_pos": int(args.mlp_min_pos),
            "stage1_epochs": int(args.stage1_epochs),
            "stage2_epochs": int(args.stage2_epochs),
            "batch_size": int(args.batch_size),
            "lr_stage1": float(args.lr_stage1),
            "lr_stage2": float(args.lr_stage2),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_spatial_mamba_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    pd.DataFrame([item.__dict__ for item in fold_results]).to_csv(output_dir / "fold_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=raw_sigmoid.astype(np.float32, copy=False),
        oof_pred=oof_pred.astype(np.float32, copy=False),
        row_id=meta_df["row_id"].astype(str).to_numpy(dtype=object),
        filename=meta_df["filename"].astype(str).to_numpy(dtype=object),
    )
    summary = {
        "rows": int(len(meta_df)),
        "files": int(meta_df["filename"].nunique()),
        "audio_rows": int(len(audio_meta)),
        "audio_classes": int(audio_meta["primary_label"].nunique()) if "primary_label" in audio_meta else -1,
        "classes": int(len(class_names)),
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "spatial_oof_auc": float(spatial_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_spatial_auc": float(mean_fold_spatial_auc),
        "fold_gap": float(mean_fold_spatial_auc - spatial_oof_auc),
        "audio_cache_dir": str(args.audio_cache_dir),
        "token_pca_dim": int(args.token_pca_dim),
        "head_variant": str(args.head_variant),
        "prototype_per_class": int(args.prototype_per_class),
        "prototype_temperature": float(args.prototype_temperature),
        "prototype_orth_weight": float(args.prototype_orth_weight),
        "token_mask_prob": float(args.token_mask_prob),
        "token_mask_max_frac": float(args.token_mask_max_frac),
        "mixup_prob": float(args.mixup_prob),
        "mixup_alpha": float(args.mixup_alpha),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "artifact_path": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] spatial_oof_auc: {spatial_oof_auc:.6f}")
    print(f"[INFO] artifact: {artifact_path}")


if __name__ == "__main__":
    main()
