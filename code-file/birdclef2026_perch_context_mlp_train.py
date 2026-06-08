#!/usr/bin/env python3
"""Train Perch + context MLP folds with leakage-aware OOF evaluation.

This is a Perch-only single-model experiment. Perch remains frozen and the MLP
only learns a lightweight multi-label head on top of Perch-derived features.

The OOF prediction convention matches deployment:

- classes with enough positives in a fold use the MLP probability
- other classes fall back to `sigmoid(raw Perch logits)`

Feature scaling, embedding PCA and early stopping are fitted inside each outer
training fold only. The outer validation fold is not used for early stopping.
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026_perch_context_train import (
    build_aligned_labels,
    build_context_tensor,
    build_metadata_features,
    build_position_features,
    fit_embedding_projector,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    parse_end_seconds,
    seed_everything,
    sigmoid_np,
    transform_embedding_projector,
)


CONTEXT_CORE = [
    "prev1",
    "next1",
    "file_mean",
    "file_max",
    "file_std",
    "neighbor_mean",
    "neighbor_max",
    "centered",
    "delta_prev1",
    "delta_next1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Perch context MLP folds.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_context_mlp_labeled_all_v1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--embedding-pca-dim", type=int, default=128)
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["raw", "base", "raw_context", "full_context"],
        default="base",
        help=(
            "`raw`: Perch logits + position. "
            "`base`: embedding PCA + Perch logits + position. "
            "`raw_context`: Perch logits + flattened context + position. "
            "`full_context`: embedding PCA + Perch logits + flattened context + position."
        ),
    )
    parser.add_argument("--context-mode", type=str, choices=["core", "all"], default="core")
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--hidden-dims", type=str, default="256")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=12.0)
    parser.add_argument("--inner-val-files", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--include-hour-features", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer.")
    return dims


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_context_indices(context_feature_names: Sequence[str], mode: str) -> List[int]:
    if mode == "all":
        return list(range(len(context_feature_names)))
    wanted = set(CONTEXT_CORE)
    return [idx for idx, name in enumerate(context_feature_names) if name in wanted]


def build_feature_matrix(
    emb_proj: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    context_indices: Sequence[int],
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    if feature_set in {"base", "full_context"}:
        parts.append(emb_proj.astype(np.float32, copy=False))
    parts.append(raw_scores.astype(np.float32, copy=False))
    if feature_set in {"raw_context", "full_context"}:
        ctx = context[:, :, list(context_indices)].reshape(len(raw_scores), -1)
        parts.append(ctx.astype(np.float32, copy=False))
    parts.append(position_features.astype(np.float32, copy=False))
    if metadata_features.shape[1] > 0:
        parts.append(metadata_features.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


class PerchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fitted_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits[:, fitted_mask]
    targets = targets[:, fitted_mask]
    pos_weight = pos_weight[fitted_mask]
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    return loss.mean()


def predict_model(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def train_fold_model(
    x_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    groups_train_outer: np.ndarray,
    hidden_dims: Sequence[int],
    fitted_class_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    inner_all = np.arange(len(x_train_outer), dtype=np.int64)
    inner_train_idx, inner_val_idx = make_inner_split(
        train_idx=inner_all,
        groups=groups_train_outer,
        inner_val_files=args.inner_val_files,
        seed=seed,
    )

    model = PerchMLP(
        input_dim=x_train_outer.shape[1],
        hidden_dims=hidden_dims,
        output_dim=y_train_outer.shape[1],
        dropout=args.dropout,
    ).to(device)

    pos = y_train_outer[inner_train_idx].sum(axis=0).astype(np.float32)
    neg = len(inner_train_idx) - pos
    pos_weight = np.ones(y_train_outer.shape[1], dtype=np.float32)
    valid_pos = pos > 0
    pos_weight[valid_pos] = np.power(neg[valid_pos] / np.maximum(pos[valid_pos], 1.0), args.pos_weight_power)
    pos_weight = np.clip(pos_weight, 1.0, float(args.pos_weight_max)).astype(np.float32)

    fitted_mask_np = np.zeros(y_train_outer.shape[1], dtype=bool)
    fitted_mask_np[fitted_class_indices] = True
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(
        torch.from_numpy(x_train_outer[inner_train_idx]),
        torch.from_numpy(y_train_outer[inner_train_idx].astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )

    if len(inner_val_idx) > 0:
        x_val_t = torch.from_numpy(x_train_outer[inner_val_idx]).to(device)
        y_val_t = torch.from_numpy(y_train_outer[inner_val_idx].astype(np.float32)).to(device)
    else:
        x_val_t = None
        y_val_t = None

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_bce_with_logits(logits, yb, fitted_mask=fitted_mask, pos_weight=pos_weight_t)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        if x_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_t)
                monitor_loss = float(
                    masked_bce_with_logits(val_logits, y_val_t, fitted_mask=fitted_mask, pos_weight=pos_weight_t)
                    .detach()
                    .cpu()
                    .item()
                )
        else:
            monitor_loss = float(np.mean(train_losses))

        if monitor_loss < best_loss - 1e-5:
            best_loss = monitor_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    artifact = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "input_dim": int(x_train_outer.shape[1]),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "output_dim": int(y_train_outer.shape[1]),
        "dropout": float(args.dropout),
        "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
    }
    stats = {
        "best_epoch": float(best_epoch),
        "best_loss": float(best_loss),
        "inner_train_rows": float(len(inner_train_idx)),
        "inner_val_rows": float(len(inner_val_idx)),
    }
    return artifact, stats


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    mlp_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    fitted_classes: int
    input_dim: int
    best_epoch: int
    best_loss: float
    inner_train_rows: int
    inner_val_rows: int


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

    meta_df, scores_full_raw, emb_full = load_cache(
        cache_dir=Path(args.cache_dir),
        meta_path_arg=args.meta_path,
        arrays_path_arg=args.arrays_path,
    )
    class_names = load_class_names(Path(args.sample_submission_path))
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

    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    metadata_features, metadata_feature_names = build_metadata_features(
        meta_df=meta_df,
        include_hour_features=args.include_hour_features,
    )
    context_tensor, context_feature_names = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)
    context_indices = select_context_indices(context_feature_names=context_feature_names, mode=args.context_mode)

    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need {args.n_folds}.")

    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32)
    raw_sigmoid_auc = macro_auc_skip_empty(y_true, raw_sigmoid)
    oof_pred = raw_sigmoid.copy()
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    gkf = GroupKFold(n_splits=args.n_folds)

    print("[INFO] Train deployable Perch + context MLP")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] feature_set: {args.feature_set}")
    print(f"[INFO] context_mode: {args.context_mode} ({len(context_indices)} features)")
    print(f"[INFO] embedding_pca_dim: {args.embedding_pca_dim}")
    print(f"[INFO] hidden_dims: {hidden_dims}")
    print(f"[INFO] mlp_min_pos: {args.mlp_min_pos}")

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(meta_df, groups=groups), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)

        if args.feature_set in {"base", "full_context"}:
            emb_train_proj, projector = fit_embedding_projector(
                emb_train=emb_full[train_idx],
                pca_dim=args.embedding_pca_dim,
                seed=args.seed + fold,
            )
            fold_projector = projector
            emb_valid_proj = transform_embedding_projector(
                emb_full[valid_idx],
                fold_artifact={
                    "embedding_scaler": projector["embedding_scaler"],
                    "embedding_pca": projector["embedding_pca"],
                },
            )
        else:
            emb_train_proj = np.zeros((len(train_idx), 0), dtype=np.float32)
            emb_valid_proj = np.zeros((len(valid_idx), 0), dtype=np.float32)
            fold_projector = {
                "embedding_scaler": None,
                "embedding_pca": None,
                "actual_embedding_dim": 0,
            }

        x_train = build_feature_matrix(
            emb_proj=emb_train_proj,
            raw_scores=scores_full_raw[train_idx],
            context=context_tensor[train_idx],
            context_indices=context_indices,
            position_features=position_features[train_idx],
            metadata_features=metadata_features[train_idx],
            feature_set=args.feature_set,
        )
        x_valid = build_feature_matrix(
            emb_proj=emb_valid_proj,
            raw_scores=scores_full_raw[valid_idx],
            context=context_tensor[valid_idx],
            context_indices=context_indices,
            position_features=position_features[valid_idx],
            metadata_features=metadata_features[valid_idx],
            feature_set=args.feature_set,
        )

        feature_scaler = StandardScaler()
        x_train_scaled = feature_scaler.fit_transform(x_train).astype(np.float32)
        x_valid_scaled = feature_scaler.transform(x_valid).astype(np.float32)

        y_train = y_true[train_idx]
        pos = y_train.sum(axis=0)
        neg = len(y_train) - pos
        fitted_class_indices = np.where((pos >= args.mlp_min_pos) & (neg > 0))[0].astype(np.int32)

        model_artifact, train_stats = train_fold_model(
            x_train_outer=x_train_scaled,
            y_train_outer=y_train,
            groups_train_outer=groups[train_idx],
            hidden_dims=hidden_dims,
            fitted_class_indices=fitted_class_indices,
            args=args,
            seed=args.seed + fold * 1000,
            device=device,
        )

        model = PerchMLP(
            input_dim=model_artifact["input_dim"],
            hidden_dims=model_artifact["hidden_dims"],
            output_dim=model_artifact["output_dim"],
            dropout=model_artifact["dropout"],
        ).to(device)
        model.load_state_dict(model_artifact["model_state"])
        mlp_all = predict_model(model=model, x=x_valid_scaled, device=device, batch_size=args.batch_size)
        fold_pred = raw_sigmoid[valid_idx].copy()
        fold_pred[:, fitted_class_indices] = mlp_all[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        fold_artifact = {
            "fold_name": f"fold_{fold}",
            "embedding_scaler": fold_projector["embedding_scaler"],
            "embedding_pca": fold_projector["embedding_pca"],
            "actual_embedding_dim": int(fold_projector["actual_embedding_dim"]),
            "feature_scaler": feature_scaler,
            "model": model_artifact,
        }
        fold_artifacts.append(fold_artifact)

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], raw_sigmoid[valid_idx])
        mlp_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        fold_result = FoldResult(
            fold=fold,
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            mlp_valid_auc=float(mlp_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            fitted_classes=int(len(fitted_class_indices)),
            input_dim=int(x_train_scaled.shape[1]),
            best_epoch=int(train_stats["best_epoch"]),
            best_loss=float(train_stats["best_loss"]),
            inner_train_rows=int(train_stats["inner_train_rows"]),
            inner_val_rows=int(train_stats["inner_val_rows"]),
        )
        fold_results.append(fold_result)
        print(
            f"[FOLD {fold}] raw_sigmoid_auc={raw_valid_auc:.6f} mlp_auc={mlp_valid_auc:.6f} "
            f"input_dim={fold_result.input_dim} fitted_classes={fold_result.fitted_classes} "
            f"best_epoch={fold_result.best_epoch} train_files={fold_result.n_train_files} "
            f"valid_files={fold_result.n_valid_files}"
        )

    mlp_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_mlp_auc = float(np.mean([item.mlp_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_mlp_auc - mlp_oof_auc)

    artifact = {
        "artifact_version": 1,
        "model_type": "perch_context_mlp",
        "class_names": class_names,
        "config": {
            "n_folds": int(args.n_folds),
            "embedding_pca_dim": int(args.embedding_pca_dim),
            "feature_set": str(args.feature_set),
            "context_mode": str(args.context_mode),
            "context_indices": [int(idx) for idx in context_indices],
            "mlp_min_pos": int(args.mlp_min_pos),
            "hidden_dims": [int(dim) for dim in hidden_dims],
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight_power": float(args.pos_weight_power),
            "pos_weight_max": float(args.pos_weight_max),
            "inner_val_files": int(args.inner_val_files),
            "patience": int(args.patience),
            "include_hour_features": bool(args.include_hour_features),
            "seed": int(args.seed),
            "context_feature_names": context_feature_names,
            "metadata_feature_names": metadata_feature_names,
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_context_mlp_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    fold_metrics_df = pd.DataFrame([item.__dict__ for item in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    oof_pred_path = output_dir / "oof_predictions.npz"
    np.savez_compressed(
        oof_pred_path,
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=scores_full_raw.astype(np.float32, copy=False),
        oof_pred=oof_pred.astype(np.float32, copy=False),
        row_id=meta_df["row_id"].to_numpy(),
        filename=meta_df["filename"].to_numpy(),
    )

    summary = {
        "rows": int(len(meta_df)),
        "files": int(len(unique_files)),
        "classes": int(len(class_names)),
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "mlp_oof_auc": float(mlp_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_mlp_auc": float(mean_fold_mlp_auc),
        "fold_gap": float(fold_gap),
        "embedding_pca_dim": int(args.embedding_pca_dim),
        "feature_set": str(args.feature_set),
        "context_mode": str(args.context_mode),
        "context_feature_count": int(len(context_indices)),
        "mlp_min_pos": int(args.mlp_min_pos),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "pos_weight_power": float(args.pos_weight_power),
        "pos_weight_max": float(args.pos_weight_max),
        "inner_val_files": int(args.inner_val_files),
        "patience": int(args.patience),
        "include_hour_features": bool(args.include_hour_features),
        "artifact_path": str(artifact_path),
        "context_feature_names": context_feature_names,
        "metadata_feature_names": metadata_feature_names,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] mlp_oof_auc: {mlp_oof_auc:.6f}")
    print(f"[INFO] mean_fold_mlp_auc: {mean_fold_mlp_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")
    print(f"[INFO] Saved artifact to: {artifact_path}")
    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF predictions to: {oof_pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
