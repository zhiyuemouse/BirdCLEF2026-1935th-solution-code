#!/usr/bin/env python3
"""Train a fold-safe MLP head on frozen AudioMAE embeddings.

This is a pure AudioMAE branch: the input is a cached 768-d AudioMAE embedding
per 5s soundscape window, and no Perch logits are used as fallback features.

Leakage policy:

- Outer folds are loaded from the same ``row_id -> fold`` CSV used by the CNN
  mainline, or built by grouped soundscape filename if no fold file is given.
- Embedding standardization is fitted on each outer train fold only.
- Early stopping uses a small inner split sampled from outer train files only.
- Classes without enough positives in the outer train fold fall back to the
  outer-train class prior for that fold.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026_perch_context_train import (
    build_aligned_labels,
    load_class_names,
    load_meta,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AudioMAE embedding MLP folds.")
    parser.add_argument("--cache-dir", type=str, default="audiomae_soundscape_cache_cnn195634folds_v1")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/audiomae_mlp_labeled_cnn195634folds_h512_256_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
    )
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument(
        "--fallback-prob",
        type=float,
        default=0.5,
        help="Probability used for classes not fitted in a fold. 0.5 is neutral for logit ensembling.",
    )
    parser.add_argument("--hidden-dims", type=str, default="512,256")
    parser.add_argument("--dropout", type=float, default=0.35)
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


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer.")
    return dims


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_cache_paths(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates = [Path(meta_path_arg)] if meta_path_arg else [
        cache_dir / "audiomae_soundscape_meta.csv",
        cache_dir / "audiomae_soundscape_meta.parquet",
    ]
    arrays_candidates = [Path(arrays_path_arg)] if arrays_path_arg else [
        cache_dir / "audiomae_soundscape_embeddings.npz"
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find AudioMAE meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find AudioMAE arrays under {cache_dir}")
    return meta_path, arrays_path


def load_audiomae_cache(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[pd.DataFrame, np.ndarray]:
    meta_path, arrays_path = resolve_cache_paths(cache_dir_arg, meta_path_arg, arrays_path_arg)
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    if "embeddings" not in arrays:
        raise KeyError(f"{arrays_path} must contain embeddings. Available keys: {arrays.files}")
    embeddings = arrays["embeddings"].astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [rows,dim], got {embeddings.shape}")
    if len(meta_df) != len(embeddings):
        raise ValueError(f"Meta rows and embeddings mismatch: {len(meta_df)} vs {len(embeddings)}")
    return meta_df, embeddings


def limit_by_files(meta_df: pd.DataFrame, y_true: np.ndarray, embeddings: np.ndarray, limit_files: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if limit_files <= 0:
        return meta_df, y_true, embeddings
    keep_files = meta_df["filename"].drop_duplicates().iloc[: int(limit_files)].tolist()
    keep_mask = meta_df["filename"].isin(keep_files).to_numpy()
    return meta_df.loc[keep_mask].reset_index(drop=True), y_true[keep_mask], embeddings[keep_mask]


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
        splits = []
        for fold_value in sorted(pd.Index(row_folds).unique().tolist()):
            valid_idx = np.where(row_folds == int(fold_value))[0].astype(np.int64)
            train_idx = np.where(row_folds != int(fold_value))[0].astype(np.int64)
            splits.append((int(fold_value), train_idx, valid_idx))
        return splits

    groups = meta_df["filename"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(args.n_folds))
    return [
        (fold_idx, np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(meta_df)), groups=groups))
    ]


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
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def transform_standardizer(x: np.ndarray, standardizer: Dict[str, np.ndarray]) -> np.ndarray:
    return ((x - standardizer["mean"]) / standardizer["std"]).astype(np.float32, copy=False)


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


class AudioMAEMLP(nn.Module):
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
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    if not fitted_mask_np.any():
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, {
            "best_epoch": 0.0,
            "best_loss": float("nan"),
        }
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight = torch.from_numpy(
        compute_pos_weight(
            y,
            fitted_indices=fitted_indices,
            power=args.pos_weight_power,
            max_value=args.pos_weight_max,
        )
    ).to(device)
    loader = build_loader(
        x=x,
        y=y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        shuffle=True,
    )
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


def smooth_prior(y_train: np.ndarray) -> np.ndarray:
    return ((y_train.sum(axis=0) + 0.5) / (len(y_train) + 1.0)).astype(np.float32)


@dataclass
class FoldResult:
    fold: int
    prior_valid_auc: float
    audiomae_valid_auc: float
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
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    class_names = load_class_names(Path(args.sample_submission_path))
    meta_df, embeddings = load_audiomae_cache(
        cache_dir_arg=args.cache_dir,
        meta_path_arg=args.meta_path,
        arrays_path_arg=args.arrays_path,
    )
    y_true = build_aligned_labels(
        labels_path=Path(args.labels_path),
        class_names=class_names,
        meta_df=meta_df,
    ).astype(np.float32, copy=False)
    meta_df, y_true, embeddings = limit_by_files(
        meta_df=meta_df,
        y_true=y_true,
        embeddings=embeddings,
        limit_files=args.limit_files,
    )
    splits = build_splits(meta_df, args=args)
    groups = meta_df["filename"].astype(str).to_numpy()
    oof_pred = np.zeros_like(y_true, dtype=np.float32)
    fallback_oof = np.full_like(y_true, fill_value=float(args.fallback_prob), dtype=np.float32)
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []

    print("[INFO] Train AudioMAE embedding MLP")
    print(f"[INFO] rows: {len(meta_df)} files={meta_df['filename'].nunique()}")
    print(f"[INFO] embeddings: {embeddings.shape}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] hidden_dims={hidden_dims} dropout={args.dropout}")
    print(f"[INFO] epochs={args.epochs} patience={args.patience}")
    print(f"[INFO] mlp_min_pos={args.mlp_min_pos}")
    if args.fold_assignment_path:
        print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")

    for display_fold, train_idx, valid_idx in splits:
        standardizer = fit_standardizer(embeddings[train_idx])
        x_train_all = transform_standardizer(embeddings[train_idx], standardizer)
        x_valid = transform_standardizer(embeddings[valid_idx], standardizer)
        y_train_all = y_true[train_idx]

        train_prior = smooth_prior(y_train_all)
        fold_pred = np.full(
            (len(valid_idx), y_true.shape[1]),
            fill_value=float(args.fallback_prob),
            dtype=np.float32,
        )

        real_pos = y_train_all.sum(axis=0)
        real_neg = len(train_idx) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)
        inner_train_rel, inner_val_rel = make_inner_split(
            train_idx=np.arange(len(train_idx), dtype=np.int64),
            groups=groups[train_idx],
            inner_val_files=args.inner_val_files,
            seed=args.seed + int(display_fold) * 1000,
        )

        model = AudioMAEMLP(
            input_dim=x_train_all.shape[1],
            hidden_dims=hidden_dims,
            output_dim=len(class_names),
            dropout=args.dropout,
        ).to(device)
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
        if len(fitted_class_indices) > 0:
            mlp_valid = predict_model(model=model, x=x_valid, device=device, batch_size=args.batch_size)
            fold_pred[:, fitted_class_indices] = mlp_valid[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        prior_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fallback_oof[valid_idx])
        audiomae_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "embedding_standardizer": standardizer,
                "class_prior": train_prior.astype(np.float32, copy=False),
                "model": {
                    "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                    "input_dim": int(x_train_all.shape[1]),
                    "output_dim": len(class_names),
                    "hidden_dims": [int(dim) for dim in hidden_dims],
                    "dropout": float(args.dropout),
                    "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
                    "best_epoch": int(stats["best_epoch"]),
                    "best_loss": float(stats["best_loss"]),
                },
            }
        )
        result = FoldResult(
            fold=int(display_fold),
            prior_valid_auc=float(prior_valid_auc),
            audiomae_valid_auc=float(audiomae_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            fitted_classes=int(len(fitted_class_indices)),
            best_epoch=int(stats["best_epoch"]),
            best_loss=float(stats["best_loss"]),
        )
        fold_results.append(result)
        print(
            f"[FOLD {display_fold}] prior_auc={prior_valid_auc:.6f} "
            f"audiomae_auc={audiomae_valid_auc:.6f} fitted_classes={result.fitted_classes} "
            f"best_epoch={result.best_epoch}",
            flush=True,
        )

    fallback_oof_auc = macro_auc_skip_empty(y_true, fallback_oof)
    audiomae_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    artifact_path = output_dir / "audiomae_mlp_artifacts.joblib"
    artifact = {
        "artifact_version": 1,
        "model_type": "audiomae_embedding_mlp",
        "class_names": class_names,
        "config": {
            "cache_dir": str(args.cache_dir),
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "hidden_dims": [int(dim) for dim in hidden_dims],
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
        "embedding_dim": int(embeddings.shape[1]),
        "fallback_oof_auc": float(fallback_oof_auc),
        "audiomae_oof_auc": float(audiomae_oof_auc),
        "mean_fold_prior_auc": float(np.mean([item.prior_valid_auc for item in fold_results])),
        "mean_fold_audiomae_auc": float(np.mean([item.audiomae_valid_auc for item in fold_results])),
        "fold_gap": float(np.mean([item.audiomae_valid_auc for item in fold_results]) - audiomae_oof_auc),
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "dropout": float(args.dropout),
        "mlp_min_pos": int(args.mlp_min_pos),
        "fallback_prob": float(args.fallback_prob),
        "epochs": int(args.epochs),
        "artifact_path": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] fallback_oof_auc: {fallback_oof_auc:.6f}")
    print(f"[INFO] audiomae_oof_auc: {audiomae_oof_auc:.6f}")
    print(f"[INFO] artifact: {artifact_path}")


if __name__ == "__main__":
    main()
