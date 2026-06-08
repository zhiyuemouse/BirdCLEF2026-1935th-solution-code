#!/usr/bin/env python3
"""Train fold-safe heads on frozen AudioMAE time tokens.

Input tokens are cached as ``[rows, 64, 768]`` from AudioMAE patch features.
The first experiments intentionally keep the head small because the labeled
soundscape set is tiny.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from MambaHead import LocalMambaBlock
from birdclef2026_perch_context_train import (
    build_aligned_labels,
    load_class_names,
    load_meta,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AudioMAE token heads.")
    parser.add_argument("--cache-dir", type=str, default="audiomae_soundscape_token_cache_cnn195634folds_v1")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/audiomae_token_mamba_labeled_cnn195634folds_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
    )
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--head-variant", type=str, choices=["mamba", "attention", "meanmax"], default="mamba")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mlp-min-pos", type=int, default=1)
    parser.add_argument("--fallback-prob", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        cache_dir / "audiomae_soundscape_token_meta.csv",
        cache_dir / "audiomae_soundscape_token_meta.parquet",
    ]
    arrays_candidates = [Path(arrays_path_arg)] if arrays_path_arg else [
        cache_dir / "audiomae_soundscape_tokens.npz"
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find AudioMAE token meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find AudioMAE token arrays under {cache_dir}")
    return meta_path, arrays_path


def load_token_cache(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[pd.DataFrame, np.ndarray]:
    meta_path, arrays_path = resolve_cache_paths(cache_dir_arg, meta_path_arg, arrays_path_arg)
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    if "tokens" not in arrays:
        raise KeyError(f"{arrays_path} must contain tokens. Available keys: {arrays.files}")
    tokens = arrays["tokens"].astype(np.float32, copy=False)
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [rows,tokens,dim], got {tokens.shape}")
    if len(meta_df) != len(tokens):
        raise ValueError(f"Meta rows and token rows mismatch: {len(meta_df)} vs {len(tokens)}")
    return meta_df, tokens


def load_fold_assignments(fold_assignment_path: Path, meta_df: pd.DataFrame) -> np.ndarray:
    fold_df = pd.read_csv(fold_assignment_path)
    required = {"row_id", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise KeyError(f"Fold assignment file is missing columns: {sorted(missing)}")
    fold_map = fold_df.drop_duplicates(subset=["row_id"]).set_index("row_id")["fold"]
    folds = meta_df["row_id"].map(fold_map)
    if folds.isna().any():
        examples = meta_df.loc[folds.isna(), "row_id"].astype(str).head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {examples}")
    return folds.astype(int).to_numpy()


def build_splits(meta_df: pd.DataFrame, args: argparse.Namespace) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    if args.fold_assignment_path:
        row_folds = load_fold_assignments(Path(args.fold_assignment_path), meta_df=meta_df)
        return [
            (
                int(fold_value),
                np.where(row_folds != int(fold_value))[0].astype(np.int64),
                np.where(row_folds == int(fold_value))[0].astype(np.int64),
            )
            for fold_value in sorted(pd.Index(row_folds).unique().tolist())
        ]
    groups = meta_df["filename"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(args.n_folds))
    return [
        (fold_idx, np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(meta_df)), groups=groups))
    ]


def limit_by_files(meta_df: pd.DataFrame, y_true: np.ndarray, tokens: np.ndarray, limit_files: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if limit_files <= 0:
        return meta_df, y_true, tokens
    keep_files = meta_df["filename"].drop_duplicates().iloc[: int(limit_files)].tolist()
    keep_mask = meta_df["filename"].isin(keep_files).to_numpy()
    return meta_df.loc[keep_mask].reset_index(drop=True), y_true[keep_mask], tokens[keep_mask]


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


def fit_standardizer(x_train: np.ndarray) -> Dict[str, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def transform_standardizer(x: np.ndarray, standardizer: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - standardizer["mean"][None, :, :]) / standardizer["std"][None, :, :]).astype(np.float32, copy=False)


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


class AudioMAETokenMambaHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            *[LocalMambaBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(int(num_blocks))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.fusion(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return self.head(pooled)


class AudioMAETokenAttentionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        weights = torch.softmax(self.attn(x), dim=1)
        attn_pool = (x * weights).sum(dim=1)
        mean_pool = x.mean(dim=1)
        return self.head(torch.cat([attn_pool, mean_pool], dim=-1))


class AudioMAETokenMeanMaxHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Linear(in_features * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1))


def build_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> nn.Module:
    if args.head_variant == "mamba":
        return AudioMAETokenMambaHead(
            in_features=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if args.head_variant == "attention":
        return AudioMAETokenAttentionHead(
            in_features=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    return AudioMAETokenMeanMaxHead(
        in_features=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    )


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, num_workers: int, seed: int, shuffle: bool) -> DataLoader:
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


def train_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    fitted_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    fitted_mask_np = np.zeros(y.shape[1], dtype=bool)
    fitted_mask_np[fitted_indices] = True
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight = torch.from_numpy(
        compute_pos_weight(y, fitted_indices=fitted_indices, power=args.pos_weight_power, max_value=args.pos_weight_max)
    ).to(device)
    loader = build_loader(x=x, y=y, batch_size=args.batch_size, num_workers=args.num_workers, seed=seed, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    val_x_t = torch.from_numpy(val_x.astype(np.float32, copy=False)).to(device) if val_x is not None and len(val_x) else None
    val_y_t = torch.from_numpy(val_y.astype(np.float32, copy=False)).to(device) if val_y is not None and len(val_y) else None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, fitted_mask],
                y_batch[:, fitted_mask],
                pos_weight=pos_weight[fitted_mask],
                reduction="mean",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if val_x_t is not None and val_y_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x_t)
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
    fallback_valid_auc: float
    token_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    fitted_classes: int
    best_epoch: int
    best_loss: float


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    class_names = load_class_names(Path(args.sample_submission_path))
    meta_df, tokens = load_token_cache(args.cache_dir, args.meta_path, args.arrays_path)
    y_true = build_aligned_labels(Path(args.labels_path), class_names=class_names, meta_df=meta_df).astype(np.float32, copy=False)
    meta_df, y_true, tokens = limit_by_files(meta_df, y_true, tokens, limit_files=args.limit_files)
    splits = build_splits(meta_df, args=args)
    groups = meta_df["filename"].astype(str).to_numpy()
    fallback_oof = np.full_like(y_true, fill_value=float(args.fallback_prob), dtype=np.float32)
    oof_pred = fallback_oof.copy()
    fold_results: List[FoldResult] = []
    fold_artifacts: List[Dict[str, object]] = []

    print("[INFO] Train AudioMAE token head")
    print(f"[INFO] rows: {len(meta_df)} files={meta_df['filename'].nunique()}")
    print(f"[INFO] tokens: {tokens.shape}")
    print(f"[INFO] head_variant={args.head_variant} hidden_dim={args.hidden_dim}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] mlp_min_pos={args.mlp_min_pos} fallback_prob={args.fallback_prob}")
    print(f"[INFO] epochs={args.epochs} patience={args.patience}")

    for display_fold, train_idx, valid_idx in splits:
        standardizer = fit_standardizer(tokens[train_idx])
        x_train_all = transform_standardizer(tokens[train_idx], standardizer)
        x_valid = transform_standardizer(tokens[valid_idx], standardizer)
        y_train_all = y_true[train_idx]

        real_pos = y_train_all.sum(axis=0)
        real_neg = len(train_idx) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)
        inner_train_rel, inner_val_rel = make_inner_split(
            train_idx=np.arange(len(train_idx), dtype=np.int64),
            groups=groups[train_idx],
            inner_val_files=args.inner_val_files,
            seed=args.seed + int(display_fold) * 1000,
        )

        model = build_model(args, input_dim=tokens.shape[-1], num_classes=len(class_names)).to(device)
        val_x = x_train_all[inner_val_rel] if len(inner_val_rel) else None
        val_y = y_train_all[inner_val_rel] if len(inner_val_rel) else None
        best_state, stats = train_epochs(
            model=model,
            x=x_train_all[inner_train_rel],
            y=y_train_all[inner_train_rel],
            fitted_indices=fitted_class_indices,
            args=args,
            seed=args.seed + int(display_fold) * 1000 + 29,
            device=device,
            val_x=val_x,
            val_y=val_y,
        )
        model.load_state_dict(best_state)
        fold_pred = np.full((len(valid_idx), y_true.shape[1]), fill_value=float(args.fallback_prob), dtype=np.float32)
        mlp_valid = predict_model(model=model, x=x_valid, device=device, batch_size=args.batch_size)
        fold_pred[:, fitted_class_indices] = mlp_valid[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        fallback_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fallback_oof[valid_idx])
        token_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        result = FoldResult(
            fold=int(display_fold),
            fallback_valid_auc=float(fallback_valid_auc),
            token_valid_auc=float(token_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            fitted_classes=int(len(fitted_class_indices)),
            best_epoch=int(stats["best_epoch"]),
            best_loss=float(stats["best_loss"]),
        )
        fold_results.append(result)
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "token_standardizer": standardizer,
                "model": {
                    "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                    "input_dim": int(tokens.shape[-1]),
                    "output_dim": len(class_names),
                    "head_variant": str(args.head_variant),
                    "hidden_dim": int(args.hidden_dim),
                    "num_blocks": int(args.num_blocks),
                    "kernel_size": int(args.kernel_size),
                    "dropout": float(args.dropout),
                    "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
                    "best_epoch": int(stats["best_epoch"]),
                    "best_loss": float(stats["best_loss"]),
                },
            }
        )
        print(
            f"[FOLD {display_fold}] fallback_auc={fallback_valid_auc:.6f} "
            f"token_auc={token_valid_auc:.6f} fitted_classes={result.fitted_classes} "
            f"best_epoch={result.best_epoch}",
            flush=True,
        )

    fallback_oof_auc = macro_auc_skip_empty(y_true, fallback_oof)
    token_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    artifact_path = output_dir / "audiomae_token_head_artifacts.joblib"
    artifact = {
        "artifact_version": 1,
        "model_type": "audiomae_token_head",
        "class_names": class_names,
        "config": {
            "cache_dir": str(args.cache_dir),
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "head_variant": str(args.head_variant),
            "hidden_dim": int(args.hidden_dim),
            "num_blocks": int(args.num_blocks),
            "kernel_size": int(args.kernel_size),
            "dropout": float(args.dropout),
            "mlp_min_pos": int(args.mlp_min_pos),
            "fallback_prob": float(args.fallback_prob),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "folds": fold_artifacts,
    }
    joblib.dump(artifact, artifact_path, compress=3)
    pd.DataFrame([item.__dict__ for item in fold_results]).to_csv(output_dir / "fold_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=fallback_oof.astype(np.float32, copy=False),
        oof_pred=oof_pred.astype(np.float32, copy=False),
        row_id=meta_df["row_id"].astype(str).to_numpy(dtype=object),
        filename=meta_df["filename"].astype(str).to_numpy(dtype=object),
    )
    summary = {
        "rows": int(len(meta_df)),
        "files": int(meta_df["filename"].nunique()),
        "classes": int(len(class_names)),
        "tokens_shape": list(tokens.shape),
        "fallback_oof_auc": float(fallback_oof_auc),
        "token_oof_auc": float(token_oof_auc),
        "mean_fold_token_auc": float(np.mean([item.token_valid_auc for item in fold_results])),
        "fold_gap": float(np.mean([item.token_valid_auc for item in fold_results]) - token_oof_auc),
        "head_variant": str(args.head_variant),
        "hidden_dim": int(args.hidden_dim),
        "num_blocks": int(args.num_blocks),
        "kernel_size": int(args.kernel_size),
        "dropout": float(args.dropout),
        "mlp_min_pos": int(args.mlp_min_pos),
        "fallback_prob": float(args.fallback_prob),
        "epochs": int(args.epochs),
        "artifact_path": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] fallback_oof_auc: {fallback_oof_auc:.6f}")
    print(f"[INFO] token_oof_auc: {token_oof_auc:.6f}")
    print(f"[INFO] artifact: {artifact_path}")


if __name__ == "__main__":
    main()
