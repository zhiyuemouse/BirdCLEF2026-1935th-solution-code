#!/usr/bin/env python3
"""Train a fold-safe MLP head on Perch global embeddings.

Input features are Perch ``embedding`` vectors with shape ``[B, 1536]``.  The
script supports:

1. labeled-only training on manually labeled soundscape windows
2. optional train_audio stage1 pretraining followed by fold-safe soundscape
   finetuning

The OOF convention matches our Perch head experiments: classes with enough
positives in a training fold use the learned head probability, while other
classes fall back to sigmoid(raw Perch logits).
"""

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
    parser = argparse.ArgumentParser(description="Train Perch embedding MLP folds.")
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--base-meta-path", type=str, default="")
    parser.add_argument("--base-arrays-path", type=str, default="")
    parser.add_argument("--audio-cache-dir", type=str, default="")
    parser.add_argument("--audio-meta-path", type=str, default="")
    parser.add_argument("--audio-arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_embedding_mlp_labeled_cnn195634folds_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--fold-assignment-path", type=str, default="")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--use-audio-pretrain", action="store_true")
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--hidden-dims", type=str, default="768,384")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--protoclr-weight-stage1", type=float, default=0.0)
    parser.add_argument("--protoclr-weight-stage2", type=float, default=0.0)
    parser.add_argument("--protoclr-temperature", type=float, default=0.12)
    parser.add_argument("--protoclr-min-classes", type=int, default=2)
    parser.add_argument("--protoclr-min-pos-per-class", type=int, default=1)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=220)
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


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer.")
    return dims


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_audio_cache_paths(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates = [Path(meta_path_arg)] if meta_path_arg else [
        cache_dir / "perch_audio_embedding_meta.parquet",
        cache_dir / "perch_audio_embedding_meta.csv",
    ]
    arrays_candidates = [Path(arrays_path_arg)] if arrays_path_arg else [
        cache_dir / "perch_audio_embedding_arrays.npz"
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find audio embedding meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find audio embedding arrays under {cache_dir}")
    return meta_path, arrays_path


def load_audio_embedding_cache(
    cache_dir_arg: str,
    meta_path_arg: str,
    arrays_path_arg: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path, arrays_path = resolve_audio_cache_paths(cache_dir_arg, meta_path_arg, arrays_path_arg)
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    if "embeddings" not in arrays or "y" not in arrays:
        raise KeyError(f"{arrays_path} must contain embeddings and y. Available keys: {arrays.files}")
    return (
        meta_df,
        arrays["embeddings"].astype(np.float32, copy=False),
        arrays["y"].astype(np.float32, copy=False),
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


def fit_standardizer(x_train: np.ndarray) -> Dict[str, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def transform_standardizer(x: np.ndarray, standardizer: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - standardizer["mean"]) / standardizer["std"]).astype(np.float32, copy=False)


class PerchEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def multilabel_protoclr_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    fitted_mask: torch.Tensor,
    temperature: float,
    min_classes: int,
    min_pos_per_class: int,
) -> torch.Tensor:
    fitted_targets = targets[:, fitted_mask]
    class_pos = fitted_targets.sum(dim=0)
    class_keep_local = torch.where(class_pos >= int(min_pos_per_class))[0]
    if class_keep_local.numel() < int(min_classes):
        return embeddings.sum() * 0.0

    z = nn.functional.normalize(embeddings, dim=1)
    kept_targets = fitted_targets[:, class_keep_local]
    prototypes: List[torch.Tensor] = []
    proto_target_columns: List[int] = []
    for target_col in range(kept_targets.shape[1]):
        pos_mask = kept_targets[:, target_col] > 0.5
        if not bool(pos_mask.any()):
            continue
        prototype = z[pos_mask].mean(dim=0)
        prototypes.append(prototype)
        proto_target_columns.append(int(target_col))
    if len(prototypes) < int(min_classes):
        return embeddings.sum() * 0.0

    proto_t = nn.functional.normalize(torch.stack(prototypes, dim=0), dim=1)
    sample_embeddings: List[torch.Tensor] = []
    sample_labels: List[torch.Tensor] = []
    for proto_idx, target_col in enumerate(proto_target_columns):
        pos_mask = kept_targets[:, target_col] > 0.5
        row_idx = torch.where(pos_mask)[0]
        if row_idx.numel() == 0:
            continue
        sample_embeddings.append(z[row_idx])
        sample_labels.append(torch.full((row_idx.numel(),), proto_idx, dtype=torch.long, device=z.device))
    if not sample_embeddings:
        return embeddings.sum() * 0.0

    sample_z = torch.cat(sample_embeddings, dim=0)
    sample_y = torch.cat(sample_labels, dim=0)
    logits = sample_z @ proto_t.T / max(float(temperature), 1e-6)
    return nn.functional.cross_entropy(logits, sample_y)


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.float32, copy=False)),
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


def forward_logits_and_embeddings(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(model, "encode") and hasattr(model, "head"):
        embeddings = model.encode(x)
        logits = model.head(embeddings)
        return logits, embeddings
    return model(x), None


def train_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    fitted_indices: np.ndarray,
    args: argparse.Namespace,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    protoclr_weight: float = 0.0,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    fitted_mask_np = np.zeros(y.shape[1], dtype=bool)
    fitted_mask_np[fitted_indices] = True
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight_np = compute_pos_weight(
        y,
        fitted_indices=fitted_indices,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )
    pos_weight = torch.from_numpy(pos_weight_np).to(device)
    loader = build_loader(
        x=x,
        y=y,
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
    if val_x is not None and val_y is not None and len(val_x) > 0:
        val_x_t = torch.from_numpy(val_x.astype(np.float32, copy=False)).to(device)
        val_y_t = torch.from_numpy(val_y.astype(np.float32, copy=False)).to(device)
    else:
        val_x_t = None
        val_y_t = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: List[float] = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = forward_logits_and_embeddings(model, x_batch)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, fitted_mask],
                y_batch[:, fitted_mask],
                pos_weight=pos_weight[fitted_mask],
                reduction="mean",
            )
            if protoclr_weight > 0 and embeddings is not None:
                loss = loss + float(protoclr_weight) * multilabel_protoclr_loss(
                    embeddings=embeddings,
                    targets=y_batch,
                    fitted_mask=fitted_mask,
                    temperature=args.protoclr_temperature,
                    min_classes=args.protoclr_min_classes,
                    min_pos_per_class=args.protoclr_min_pos_per_class,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if val_x_t is not None and val_y_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits, _ = forward_logits_and_embeddings(model, val_x_t)
                monitor_loss = float(
                    nn.functional.binary_cross_entropy_with_logits(
                        val_logits[:, fitted_mask],
                        val_y_t[:, fitted_mask],
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


def predict_model(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            batch = torch.from_numpy(x[start:start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    embedding_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    fitted_classes: int
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
    hidden_dims = parse_hidden_dims(args.hidden_dims)
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
    ).astype(np.float32, copy=False)
    meta_df, y_true, scores_full_raw, emb_full = limit_by_files(
        meta_df=meta_df,
        y_true=y_true,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
        limit_files=args.limit_files,
    )

    audio_meta = None
    audio_x = None
    audio_y = None
    if args.use_audio_pretrain:
        if not args.audio_cache_dir:
            raise ValueError("--audio-cache-dir is required when --use-audio-pretrain is set.")
        audio_meta, audio_x, audio_y = load_audio_embedding_cache(
            cache_dir_arg=args.audio_cache_dir,
            meta_path_arg=args.audio_meta_path,
            arrays_path_arg=args.audio_arrays_path,
        )
        if audio_y.shape[1] != len(class_names):
            raise ValueError(f"Audio target class count mismatch: {audio_y.shape[1]} vs {len(class_names)}")

    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32, copy=False)
    raw_sigmoid_auc = macro_auc_skip_empty(y_true, raw_sigmoid)
    splits = mamba_train.build_splits(meta_df, args=args)
    groups = meta_df["filename"].to_numpy()
    oof_pred = raw_sigmoid.copy()
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []

    print("[INFO] Train Perch embedding MLP")
    print(f"[INFO] soundscape rows: {len(meta_df)} files={meta_df['filename'].nunique()}")
    if audio_meta is not None:
        print(f"[INFO] audio rows: {len(audio_meta)} files={audio_meta['filename'].nunique() if 'filename' in audio_meta else len(audio_meta)}")
    print(f"[INFO] embeddings: {emb_full.shape}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] use_audio_pretrain={args.use_audio_pretrain}")
    print(f"[INFO] stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs}")
    print(f"[INFO] hidden_dims={hidden_dims} dropout={args.dropout}")
    print(
        f"[INFO] protoCLR stage1/stage2 weights="
        f"{args.protoclr_weight_stage1}/{args.protoclr_weight_stage2} "
        f"tau={args.protoclr_temperature}"
    )
    if args.fold_assignment_path:
        print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")

    for display_fold, train_idx, valid_idx in splits:
        standardizer = fit_standardizer(emb_full[train_idx])
        x_train = transform_standardizer(emb_full[train_idx], standardizer)
        x_valid = transform_standardizer(emb_full[valid_idx], standardizer)
        if audio_x is not None:
            x_audio = transform_standardizer(audio_x, standardizer)
        else:
            x_audio = None

        real_pos = y_true[train_idx].sum(axis=0)
        real_neg = len(train_idx) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)
        inner_train_idx, inner_val_idx = make_inner_split(
            train_idx=np.arange(len(train_idx), dtype=np.int64),
            groups=groups[train_idx],
            inner_val_files=args.inner_val_files,
            seed=args.seed + int(display_fold) * 1000,
        )

        model = PerchEmbeddingMLP(
            input_dim=x_train.shape[1],
            hidden_dims=hidden_dims,
            output_dim=len(class_names),
            dropout=args.dropout,
        ).to(device)
        stage1_stats = {"best_epoch": 0.0, "best_loss": float("nan")}
        if x_audio is not None and audio_y is not None and args.stage1_epochs > 0:
            stage1_state, stage1_stats = train_epochs(
                model=model,
                x=x_audio,
                y=audio_y,
                fitted_indices=fitted_class_indices,
                args=args,
                lr=args.lr_stage1,
                epochs=args.stage1_epochs,
                seed=args.seed + int(display_fold) * 1000 + 13,
                device=device,
                protoclr_weight=args.protoclr_weight_stage1,
                val_x=None,
                val_y=None,
            )
            model.load_state_dict(stage1_state)

        val_x = x_train[inner_val_idx] if len(inner_val_idx) else None
        val_y = y_true[train_idx][inner_val_idx] if len(inner_val_idx) else None
        stage2_state, stage2_stats = train_epochs(
            model=model,
            x=x_train[inner_train_idx],
            y=y_true[train_idx][inner_train_idx],
            fitted_indices=fitted_class_indices,
            args=args,
            lr=args.lr_stage2,
            epochs=args.stage2_epochs,
            seed=args.seed + int(display_fold) * 1000 + 29,
            device=device,
            protoclr_weight=args.protoclr_weight_stage2,
            val_x=val_x,
            val_y=val_y,
        )
        model.load_state_dict(stage2_state)
        mlp_valid = predict_model(model=model, x=x_valid, device=device, batch_size=args.batch_size)
        fold_pred = raw_sigmoid[valid_idx].copy()
        fold_pred[:, fitted_class_indices] = mlp_valid[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], raw_sigmoid[valid_idx])
        embedding_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        model_artifact = {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "input_dim": int(x_train.shape[1]),
            "output_dim": len(class_names),
            "hidden_dims": [int(dim) for dim in hidden_dims],
            "dropout": float(args.dropout),
            "protoclr_weight_stage1": float(args.protoclr_weight_stage1),
            "protoclr_weight_stage2": float(args.protoclr_weight_stage2),
            "protoclr_temperature": float(args.protoclr_temperature),
            "protoclr_min_classes": int(args.protoclr_min_classes),
            "protoclr_min_pos_per_class": int(args.protoclr_min_pos_per_class),
            "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
            "stage1_best_epoch": int(stage1_stats["best_epoch"]),
            "stage1_best_loss": float(stage1_stats["best_loss"]),
            "stage2_best_epoch": int(stage2_stats["best_epoch"]),
            "stage2_best_loss": float(stage2_stats["best_loss"]),
        }
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "embedding_standardizer": standardizer,
                "model": model_artifact,
            }
        )
        result = FoldResult(
            fold=int(display_fold),
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            embedding_valid_auc=float(embedding_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            fitted_classes=int(len(fitted_class_indices)),
            stage1_best_epoch=int(stage1_stats["best_epoch"]),
            stage1_best_loss=float(stage1_stats["best_loss"]),
            stage2_best_epoch=int(stage2_stats["best_epoch"]),
            stage2_best_loss=float(stage2_stats["best_loss"]),
        )
        fold_results.append(result)
        print(
            f"[FOLD {display_fold}] raw_sigmoid_auc={raw_valid_auc:.6f} "
            f"embedding_auc={embedding_valid_auc:.6f} fitted_classes={result.fitted_classes} "
            f"stage1_epoch={result.stage1_best_epoch} stage2_epoch={result.stage2_best_epoch}",
            flush=True,
        )

    embedding_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_embedding_auc = float(np.mean([item.embedding_valid_auc for item in fold_results]))
    artifact = {
        "artifact_version": 1,
        "model_type": "perch_embedding_mlp",
        "class_names": class_names,
        "config": {
            "audio_cache_dir": str(args.audio_cache_dir),
            "use_audio_pretrain": bool(args.use_audio_pretrain),
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "hidden_dims": [int(dim) for dim in hidden_dims],
            "dropout": float(args.dropout),
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
    artifact_path = output_dir / "perch_embedding_mlp_artifacts.joblib"
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
        "audio_rows": int(len(audio_meta)) if audio_meta is not None else 0,
        "audio_classes": int(audio_meta["primary_label"].nunique()) if audio_meta is not None and "primary_label" in audio_meta else -1,
        "classes": int(len(class_names)),
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "embedding_oof_auc": float(embedding_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_embedding_auc": float(mean_fold_embedding_auc),
        "fold_gap": float(mean_fold_embedding_auc - embedding_oof_auc),
        "audio_cache_dir": str(args.audio_cache_dir),
        "use_audio_pretrain": bool(args.use_audio_pretrain),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "dropout": float(args.dropout),
        "protoclr_weight_stage1": float(args.protoclr_weight_stage1),
        "protoclr_weight_stage2": float(args.protoclr_weight_stage2),
        "protoclr_temperature": float(args.protoclr_temperature),
        "protoclr_min_classes": int(args.protoclr_min_classes),
        "protoclr_min_pos_per_class": int(args.protoclr_min_pos_per_class),
        "mlp_min_pos": int(args.mlp_min_pos),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "artifact_path": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] embedding_oof_auc: {embedding_oof_auc:.6f}")
    print(f"[INFO] artifact: {artifact_path}")


if __name__ == "__main__":
    main()
