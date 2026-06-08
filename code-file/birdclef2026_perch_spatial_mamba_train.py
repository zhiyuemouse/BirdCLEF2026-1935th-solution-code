#!/usr/bin/env python3
"""Train a fold-safe Perch spatial-token Mamba-style head.

This experiment uses frozen Perch v2 ``spatial_embedding`` tokens:

``[row, 16, 1536]``

They are frequency-averaged tokens from the ONNX output
``[row, 16, 4, 1536]``.  The model follows ``Mamba.py``: gated depthwise 1D
convolution over the 16 local tokens, then pooling and a multilabel head.

Leakage policy:

- Labels are aligned by ``row_id`` only.
- Outer CV is grouped by full soundscape filename.
- Token normalization and optional PCA are fitted on the outer train fold only.
- Early stopping uses an inner split sampled from outer train files only.
- The outer validation fold is never used to choose epochs, scalers, PCA, or
  class masks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser = argparse.ArgumentParser(description="Train Perch spatial Mamba folds.")
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--base-meta-path", type=str, default="")
    parser.add_argument("--base-arrays-path", type=str, default="")
    parser.add_argument("--spatial-cache-dir", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--spatial-meta-path", type=str, default="")
    parser.add_argument("--spatial-arrays-path", type=str, default="")
    parser.add_argument("--pseudo-root", type=str, default="")
    parser.add_argument("--pseudo-spatial-cache-dir", type=str, default="")
    parser.add_argument("--pseudo-spatial-meta-path", type=str, default="")
    parser.add_argument("--pseudo-spatial-arrays-path", type=str, default="")
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--min-pseudo-max-prob", type=float, default=0.0)
    parser.add_argument("--max-pseudo-rows", type=int, default=-1)
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_spatial_mamba_labeled_all_v1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="",
        help="Optional CSV with row_id and fold columns. Use this to align folds with another run.",
    )
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--token-pca-dim", type=int, default=256)
    parser.add_argument("--freq-pool", type=str, choices=["mean", "meanmax", "flat64"], default="mean")
    parser.add_argument("--use-pos-embed", action="store_true")
    parser.add_argument("--include-raw-scores", action="store_true")
    parser.add_argument("--raw-proj-dim", type=int, default=128)
    parser.add_argument(
        "--head-variant",
        type=str,
        choices=[
            "generic",
            "perch_mamba_v1",
            "attention_pooling",
            "multihead_attention_pooling",
            "perch_transformer",
            "small_transformer",
            "mil",
            "logsumexp_mil",
            "attention_mil",
            "perch_fusion",
            "perch_strong",
            "prototype_pooling",
        ],
        default="generic",
    )
    parser.add_argument("--prototype-per-class", type=int, default=5)
    parser.add_argument("--prototype-temperature", type=float, default=12.0)
    parser.add_argument("--prototype-orth-weight", type=float, default=0.01)
    parser.add_argument(
        "--prototype-init-source",
        type=str,
        choices=["random", "audio_token", "soundscape_token"],
        default="random",
        help="Initialize prototype_pooling prototypes randomly, from train_audio tokens, or from outer-train soundscape tokens.",
    )
    parser.add_argument("--prototype-init-audio-cache-dir", type=str, default="")
    parser.add_argument("--prototype-init-audio-meta-path", type=str, default="")
    parser.add_argument("--prototype-init-audio-arrays-path", type=str, default="")
    parser.add_argument("--prototype-init-max-rows-per-class", type=int, default=80)
    parser.add_argument("--prototype-init-candidate-tokens", type=int, default=2048)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--token-mask-prob", type=float, default=0.0)
    parser.add_argument("--token-mask-max-frac", type=float, default=0.15)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
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


def resolve_spatial_paths_from_args(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
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


def resolve_spatial_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    return resolve_spatial_paths_from_args(
        cache_dir_arg=args.spatial_cache_dir,
        meta_path_arg=args.spatial_meta_path,
        arrays_path_arg=args.spatial_arrays_path,
    )


def load_spatial_cache_from_paths(
    cache_dir_arg: str,
    meta_path_arg: str = "",
    arrays_path_arg: str = "",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    meta_path, arrays_path = resolve_spatial_paths_from_args(
        cache_dir_arg=cache_dir_arg,
        meta_path_arg=meta_path_arg,
        arrays_path_arg=arrays_path_arg,
    )
    meta_df = load_meta(meta_path)
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


def load_spatial_cache(args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    return load_spatial_cache_from_paths(
        cache_dir_arg=args.spatial_cache_dir,
        meta_path_arg=args.spatial_meta_path,
        arrays_path_arg=args.spatial_arrays_path,
    )


def align_spatial_to_base(
    base_meta: pd.DataFrame,
    spatial_meta: pd.DataFrame,
    spatial_tokens: np.ndarray,
) -> np.ndarray:
    spatial_pos = pd.Series(np.arange(len(spatial_meta), dtype=np.int64), index=spatial_meta["row_id"])
    indices = base_meta["row_id"].map(spatial_pos)
    if indices.isna().any():
        missing = base_meta.loc[indices.isna(), "row_id"].astype(str).head(5).tolist()
        raise ValueError(f"Spatial cache missing {indices.isna().sum()} rows. Examples: {missing}")
    aligned = spatial_meta.iloc[indices.to_numpy(dtype=np.int64)].reset_index(drop=True)
    if not np.all(aligned["filename"].to_numpy() == base_meta["filename"].to_numpy()):
        raise AssertionError("Base and spatial filename order mismatch after row_id alignment.")
    return spatial_tokens[indices.to_numpy(dtype=np.int64)]


def load_fold_assignments(fold_assignment_path: Path, meta_df: pd.DataFrame) -> np.ndarray:
    fold_df = pd.read_csv(fold_assignment_path)
    required = {"row_id", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise KeyError(f"Fold assignment file is missing columns: {sorted(missing)}")
    fold_map = fold_df.drop_duplicates(subset=["row_id"]).set_index("row_id")["fold"]
    folds = meta_df["row_id"].map(fold_map)
    if folds.isna().any():
        missing_rows = meta_df.loc[folds.isna(), "row_id"].astype(str).head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {missing_rows}")
    return folds.astype(int).to_numpy()


def limit_by_files_spatial(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    emb_full: np.ndarray,
    spatial_tokens: np.ndarray,
    limit_files: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if limit_files <= 0:
        return meta_df, y_true, raw_scores, emb_full, spatial_tokens
    keep_files = meta_df["filename"].drop_duplicates().iloc[:limit_files].tolist()
    keep_mask = meta_df["filename"].isin(keep_files).to_numpy()
    return (
        meta_df.loc[keep_mask].reset_index(drop=True),
        y_true[keep_mask],
        raw_scores[keep_mask],
        emb_full[keep_mask],
        spatial_tokens[keep_mask],
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


class TokenProjector:
    def __init__(self, token_mean: np.ndarray, token_std: np.ndarray, pca: PCA | None, output_dim: int) -> None:
        self.token_mean = token_mean.astype(np.float32, copy=False)
        self.token_std = token_std.astype(np.float32, copy=False)
        self.pca = pca
        self.output_dim = int(output_dim)

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        n_rows, n_tokens, n_dim = tokens.shape
        flat = tokens.reshape(-1, n_dim).astype(np.float32, copy=False)
        scaled = ((flat - self.token_mean) / self.token_std).astype(np.float32, copy=False)
        if self.pca is not None:
            projected = self.pca.transform(scaled).astype(np.float32)
        else:
            projected = scaled
        return projected.reshape(n_rows, n_tokens, self.output_dim).astype(np.float32, copy=False)


def fit_token_projector(tokens_train: np.ndarray, token_pca_dim: int, seed: int) -> TokenProjector:
    _, _, n_dim = tokens_train.shape
    flat = tokens_train.reshape(-1, n_dim).astype(np.float32, copy=False)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    scaled = ((flat - mean) / std).astype(np.float32, copy=False)
    max_dim = min(int(token_pca_dim), scaled.shape[0] - 1, scaled.shape[1])
    if max_dim > 0 and max_dim < scaled.shape[1]:
        pca = PCA(n_components=max_dim, random_state=seed)
        pca.fit(scaled)
        output_dim = max_dim
    else:
        pca = None
        output_dim = scaled.shape[1]
    return TokenProjector(token_mean=mean, token_std=std, pca=pca, output_dim=output_dim)


def transform_tokens_with_projector(tokens: np.ndarray, projector: TokenProjector | Sequence[TokenProjector]) -> np.ndarray:
    if isinstance(projector, (list, tuple)):
        parts = [item.transform(tokens[:, part_idx, :]) for part_idx, item in enumerate(projector)]
        return np.stack(parts, axis=1).astype(np.float32, copy=False)
    return projector.transform(tokens)


def fit_tokens_projector(tokens_train: np.ndarray, token_pca_dim: int, seed: int) -> TokenProjector | List[TokenProjector]:
    if tokens_train.ndim == 4:
        return [
            fit_token_projector(tokens_train=tokens_train[:, part_idx, :, :], token_pca_dim=token_pca_dim, seed=seed + part_idx)
            for part_idx in range(tokens_train.shape[1])
        ]
    return fit_token_projector(tokens_train=tokens_train, token_pca_dim=token_pca_dim, seed=seed)


def resolve_audio_spatial_paths_from_args(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates: List[Path] = []
    arrays_candidates: List[Path] = []
    if meta_path_arg:
        meta_candidates.append(Path(meta_path_arg))
    else:
        meta_candidates.extend([cache_dir / "perch_audio_spatial_meta.parquet", cache_dir / "perch_audio_spatial_meta.csv"])
    if arrays_path_arg:
        arrays_candidates.append(Path(arrays_path_arg))
    else:
        arrays_candidates.append(cache_dir / "perch_audio_spatial_arrays.npz")
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find audio spatial meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find audio spatial arrays under {cache_dir}")
    return meta_path, arrays_path


def load_audio_prototype_init_cache(
    cache_dir_arg: str,
    meta_path_arg: str,
    arrays_path_arg: str,
    freq_pool: str,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path, arrays_path = resolve_audio_spatial_paths_from_args(
        cache_dir_arg=cache_dir_arg,
        meta_path_arg=meta_path_arg,
        arrays_path_arg=arrays_path_arg,
    )
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    token_key = "spatial_tokens_64" if freq_pool == "flat64" else "spatial_tokens"
    if token_key not in arrays or "y" not in arrays:
        raise KeyError(f"{arrays_path} must contain {token_key} and y. Available keys: {arrays.files}")
    tokens = arrays[token_key].astype(np.float32, copy=False)
    y = arrays["y"].astype(np.float32, copy=False)
    if y.shape[1] != len(class_names):
        raise ValueError(f"Audio prototype init target class count mismatch: {y.shape[1]} vs {len(class_names)}")
    return meta_df, tokens, y


def build_token_prototypes(
    source_tokens: np.ndarray,
    source_y: np.ndarray,
    fitted_class_indices: np.ndarray,
    num_classes: int,
    prototype_per_class: int,
    max_rows_per_class: int,
    candidate_tokens: int,
    seed: int,
) -> np.ndarray:
    """Choose class-specific local tokens as prototype initializers.

    The method is deliberately simple: for each class, sample positive rows,
    normalize their local tokens, compute a class centroid, then keep tokens
    nearest to that centroid.
    """
    if source_tokens.ndim != 3:
        raise ValueError(f"Expected source_tokens [rows,tokens,dim], got {source_tokens.shape}")
    rng = np.random.default_rng(seed)
    n_classes = int(num_classes)
    n_proto = int(max(1, prototype_per_class))
    token_dim = int(source_tokens.shape[-1])
    proto = rng.normal(0.0, 0.02, size=(n_classes, n_proto, token_dim)).astype(np.float32)
    fitted_set = set(np.asarray(fitted_class_indices, dtype=np.int64).tolist())
    row_limit = int(max_rows_per_class)
    token_limit = int(candidate_tokens)
    for class_idx in range(n_classes):
        positive_rows = np.where(source_y[:, class_idx] > 0.5)[0]
        if len(positive_rows) == 0:
            continue
        if row_limit > 0 and len(positive_rows) > row_limit:
            positive_rows = rng.choice(positive_rows, size=row_limit, replace=False)
        class_tokens = source_tokens[positive_rows].reshape(-1, token_dim).astype(np.float32, copy=False)
        if len(class_tokens) == 0:
            continue
        if token_limit > 0 and len(class_tokens) > token_limit:
            sampled = rng.choice(len(class_tokens), size=token_limit, replace=False)
            class_tokens = class_tokens[sampled]
        norms = np.linalg.norm(class_tokens, axis=1, keepdims=True)
        class_tokens_norm = class_tokens / np.maximum(norms, 1e-6)
        centroid = class_tokens_norm.mean(axis=0, keepdims=True)
        centroid = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-6)
        sim = (class_tokens_norm @ centroid.T).reshape(-1)
        if len(sim) >= n_proto:
            selected = np.argsort(-sim)[:n_proto]
        else:
            selected = np.resize(np.argsort(-sim), n_proto)
        proto[class_idx] = class_tokens[selected]
        if class_idx not in fitted_set:
            proto[class_idx] += rng.normal(0.0, 0.002, size=proto[class_idx].shape).astype(np.float32)
    return proto.astype(np.float32, copy=False)


def build_audio_token_prototypes(
    audio_tokens: np.ndarray,
    audio_y: np.ndarray,
    fitted_class_indices: np.ndarray,
    num_classes: int,
    prototype_per_class: int,
    max_rows_per_class: int,
    candidate_tokens: int,
    seed: int,
) -> np.ndarray:
    """Choose class-specific train_audio tokens as prototype initializers."""
    return build_token_prototypes(
        source_tokens=audio_tokens,
        source_y=audio_y,
        fitted_class_indices=fitted_class_indices,
        num_classes=num_classes,
        prototype_per_class=prototype_per_class,
        max_rows_per_class=max_rows_per_class,
        candidate_tokens=candidate_tokens,
        seed=seed,
    )


class RawScoreProjector:
    def __init__(self, mean: np.ndarray, std: np.ndarray, pca: PCA | None, output_dim: int) -> None:
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)
        self.pca = pca
        self.output_dim = int(output_dim)

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        scaled = ((raw_scores - self.mean) / self.std).astype(np.float32, copy=False)
        if self.pca is not None:
            return self.pca.transform(scaled).astype(np.float32)
        return scaled.astype(np.float32, copy=False)


def fit_raw_projector(raw_train: np.ndarray, raw_proj_dim: int, seed: int) -> RawScoreProjector:
    mean = raw_train.mean(axis=0, keepdims=True)
    std = raw_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    scaled = ((raw_train - mean) / std).astype(np.float32, copy=False)
    max_dim = min(int(raw_proj_dim), scaled.shape[0] - 1, scaled.shape[1])
    if max_dim > 0 and max_dim < scaled.shape[1]:
        pca = PCA(n_components=max_dim, random_state=seed)
        pca.fit(scaled)
        output_dim = max_dim
    else:
        pca = None
        output_dim = scaled.shape[1]
    return RawScoreProjector(mean=mean, std=std, pca=pca, output_dim=output_dim)


def token_projector_to_artifact(projector: TokenProjector) -> Dict[str, object]:
    return {
        "token_mean": projector.token_mean,
        "token_std": projector.token_std,
        "pca": projector.pca,
        "output_dim": int(projector.output_dim),
    }


def tokens_projector_to_artifact(projector: TokenProjector | Sequence[TokenProjector]) -> Dict[str, object] | List[Dict[str, object]]:
    if isinstance(projector, (list, tuple)):
        return [token_projector_to_artifact(item) for item in projector]
    return token_projector_to_artifact(projector)


def raw_projector_to_artifact(projector: RawScoreProjector | None) -> Dict[str, object] | None:
    if projector is None:
        return None
    return {
        "mean": projector.mean,
        "std": projector.std,
        "pca": projector.pca,
        "output_dim": int(projector.output_dim),
    }


def locate_pseudo_fold_dir(pseudo_root: Path, fold: int) -> Path:
    fold_dir = pseudo_root / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Could not find fold-specific pseudo dir for fold {fold}: {fold_dir}")
    return fold_dir


def load_pseudo_package(
    pseudo_root: Path,
    fold: int,
    class_names: Sequence[str],
    min_pseudo_max_prob: float,
    max_pseudo_rows: int,
) -> Tuple[pd.DataFrame, np.ndarray, Path]:
    fold_dir = locate_pseudo_fold_dir(pseudo_root=pseudo_root, fold=fold)
    pseudo_df = pd.read_csv(fold_dir / "pseudo_segments.csv")
    pseudo_probs = np.load(fold_dir / "pseudo_probs.npy").astype(np.float32, copy=False)
    if len(pseudo_df) != len(pseudo_probs):
        raise ValueError(f"Pseudo rows/probs length mismatch in {fold_dir}: {len(pseudo_df)} vs {len(pseudo_probs)}")
    if pseudo_probs.shape[1] != len(class_names):
        raise ValueError(
            f"Pseudo probs class count mismatch in {fold_dir}: {pseudo_probs.shape[1]} vs {len(class_names)}"
        )
    keep_mask = pseudo_df["max_prob"].to_numpy(dtype=np.float32) >= float(min_pseudo_max_prob)
    if "positive_count" in pseudo_df.columns:
        keep_mask &= pseudo_df["positive_count"].to_numpy(dtype=np.int64) > 0
    pseudo_df = pseudo_df.loc[keep_mask].reset_index(drop=True)
    pseudo_probs = pseudo_probs[keep_mask]
    if max_pseudo_rows > 0 and len(pseudo_df) > max_pseudo_rows:
        order = np.argsort(-pseudo_df["max_prob"].to_numpy(dtype=np.float32))[:max_pseudo_rows]
        pseudo_df = pseudo_df.iloc[order].reset_index(drop=True)
        pseudo_probs = pseudo_probs[order]
    return pseudo_df, pseudo_probs.astype(np.float32, copy=False), fold_dir


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


class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.proj(x))
        return shortcut + x


class PerchSpatialMambaHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        num_blocks: int,
        kernel_size: int,
        hidden_dim: int,
        dropout: float,
        raw_dim: int = 0,
        freq_pool: str = "mean",
        use_pos_embed: bool = False,
        head_variant: str = "generic",
        prototype_per_class: int = 5,
        prototype_temperature: float = 12.0,
    ) -> None:
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.freq_pool = str(freq_pool)
        self.use_pos_embed = bool(use_pos_embed)
        self.head_variant = str(head_variant)
        self.prototype_per_class = int(max(1, prototype_per_class))
        self.prototype_temperature = float(prototype_temperature)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, int(token_dim))) if self.use_pos_embed else None
        if self.head_variant == "perch_mamba_v1":
            if self.raw_dim != 0:
                raise ValueError("head_variant=perch_mamba_v1 does not support raw score features")
            if self.freq_pool != "mean":
                raise ValueError("head_variant=perch_mamba_v1 requires freq_pool=mean")
            if self.use_pos_embed:
                raise ValueError("head_variant=perch_mamba_v1 does not use flat64 positional embeddings")
        if self.head_variant == "perch_fusion":
            if self.raw_dim <= int(token_dim):
                raise ValueError("head_variant=perch_fusion requires raw_features=[embedding, selected_label_logits]")
            if self.freq_pool != "flat64":
                raise ValueError("head_variant=perch_fusion requires freq_pool=flat64")
            if self.use_pos_embed:
                raise ValueError("head_variant=perch_fusion does not use positional embeddings")
        if self.head_variant in {
            "attention_pooling",
            "multihead_attention_pooling",
            "perch_transformer",
            "small_transformer",
            "mil",
            "logsumexp_mil",
            "attention_mil",
            "perch_strong",
            "prototype_pooling",
        }:
            if self.raw_dim != 0:
                raise ValueError(f"head_variant={self.head_variant} does not support raw score features")
            if self.freq_pool != "flat64":
                raise ValueError(f"head_variant={self.head_variant} requires freq_pool=flat64")
            if self.use_pos_embed:
                raise ValueError(f"head_variant={self.head_variant} does not use positional embeddings")
        if self.freq_pool == "meanmax":
            self.freq_proj = nn.Sequential(
                nn.Linear(int(token_dim) * 2, int(token_dim)),
                nn.LayerNorm(int(token_dim)),
                nn.LeakyReLU(),
                nn.Dropout(float(dropout) * 0.4),
            )
        else:
            self.freq_proj = None
        if self.head_variant == "perch_mamba_v1":
            self.blocks = nn.Sequential(
                *[LocalMambaBlock(token_dim, kernel_size=5, dropout=0.1) for _ in range(2)]
            )
            head_hidden_dim = max(1, int(token_dim) // 2)
            self.head = nn.Sequential(
                nn.Linear(int(token_dim), head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(head_hidden_dim, int(num_classes)),
            )
        elif self.head_variant == "attention_pooling":
            self.blocks = nn.Identity()
            self.attn = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 768),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(768, int(num_classes)),
            )
        elif self.head_variant == "multihead_attention_pooling":
            num_heads = 4
            self.blocks = nn.Identity()
            self.attn = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 256),
                nn.Tanh(),
                nn.Linear(256, num_heads),
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(int(token_dim) * num_heads),
                nn.Linear(int(token_dim) * num_heads, int(token_dim)),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(int(token_dim), int(num_classes))
        elif self.head_variant == "perch_transformer":
            num_heads = 4
            if int(token_dim) % num_heads != 0:
                raise ValueError(f"head_variant=perch_transformer requires token_dim divisible by {num_heads}")
            self.blocks = nn.Identity()
            self.transformer_pos_embed = nn.Parameter(torch.zeros(1, 64, int(token_dim)))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(token_dim),
                nhead=num_heads,
                dim_feedforward=2048,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Sequential(
                nn.LayerNorm(int(token_dim) * 2),
                nn.Linear(int(token_dim) * 2, 768),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(768, int(num_classes)),
            )
        elif self.head_variant == "small_transformer":
            small_hidden = 512
            self.blocks = nn.Identity()
            self.input_proj = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), small_hidden),
                nn.GELU(),
            )
            self.transformer_pos_embed = nn.Parameter(torch.zeros(1, 64, small_hidden))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=small_hidden,
                nhead=8,
                dim_feedforward=small_hidden * 4,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Sequential(
                nn.LayerNorm(small_hidden * 2),
                nn.Linear(small_hidden * 2, small_hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(small_hidden, int(num_classes)),
            )
        elif self.head_variant == "mil":
            self.blocks = nn.Identity()
            self.token_classifier = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 768),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(768, int(num_classes)),
            )
        elif self.head_variant == "logsumexp_mil":
            self.blocks = nn.Identity()
            self.token_classifier = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 768),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(768, int(num_classes)),
            )
        elif self.head_variant == "attention_mil":
            self.blocks = nn.Identity()
            self.feat = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), 768),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.classifier = nn.Linear(768, int(num_classes))
            self.attention = nn.Linear(768, int(num_classes))
        elif self.head_variant == "perch_fusion":
            selected_label_dim = int(self.raw_dim) - int(token_dim)
            input_dim = int(token_dim) * 3 + selected_label_dim
            self.blocks = nn.Identity()
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 1024),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, int(num_classes)),
            )
        elif self.head_variant == "perch_strong":
            hidden = 768
            self.blocks = nn.Identity()
            self.token_feat = nn.Sequential(
                nn.LayerNorm(int(token_dim)),
                nn.Linear(int(token_dim), hidden),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.token_classifier = nn.Linear(hidden, int(num_classes))
            self.token_attention = nn.Linear(hidden, int(num_classes))
            self.global_head = nn.Sequential(
                nn.LayerNorm(int(token_dim) * 2),
                nn.Linear(int(token_dim) * 2, hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, int(num_classes)),
            )
            self.final_scale = nn.Parameter(torch.tensor(0.5))
        elif self.head_variant == "prototype_pooling":
            self.blocks = nn.Identity()
            self.num_classes = int(num_classes)
            self.prototypes = nn.Parameter(
                torch.randn(int(num_classes), self.prototype_per_class, int(token_dim)) * 0.02
            )
            self.class_proto_weight = nn.Parameter(torch.full((int(num_classes), self.prototype_per_class), -2.0))
            self.class_bias = nn.Parameter(torch.zeros(int(num_classes)))
        else:
            self.blocks = nn.Sequential(
                *[LocalMambaBlock(token_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(num_blocks)]
            )
            head_in = int(token_dim) + self.raw_dim
            self.head = nn.Sequential(
                nn.Linear(head_in, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(num_classes)),
            )

    def forward(self, tokens: torch.Tensor, raw_features: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim == 4:
            if self.freq_proj is None:
                raise ValueError("4D tokens require freq_pool=meanmax")
            mean_tokens = tokens[:, 0, :, :]
            max_tokens = tokens[:, 1, :, :]
            tokens = self.freq_proj(torch.cat([mean_tokens, max_tokens], dim=-1))
        if self.pos_embed is not None:
            if tokens.ndim != 3 or tokens.shape[1] != self.pos_embed.shape[1]:
                raise ValueError("use_pos_embed requires flat64 tokens with shape [batch,64,dim]")
            tokens = tokens + self.pos_embed
        if self.head_variant == "attention_pooling":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=attention_pooling expects flat64 tokens with shape [batch,64,dim]")
            score = self.attn(tokens)
            weight = torch.softmax(score, dim=1)
            pooled = (tokens * weight).sum(dim=1)
            return self.head(pooled)
        if self.head_variant == "multihead_attention_pooling":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError(
                    "head_variant=multihead_attention_pooling expects flat64 tokens with shape [batch,64,dim]"
                )
            score = self.attn(tokens)
            weight = torch.softmax(score, dim=1)
            pooled = (tokens.unsqueeze(2) * weight.unsqueeze(-1)).sum(dim=1)
            pooled = pooled.reshape(tokens.shape[0], -1)
            feat = self.proj(pooled)
            return self.classifier(feat)
        if self.head_variant == "perch_transformer":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=perch_transformer expects flat64 tokens with shape [batch,64,dim]")
            x = self.encoder(tokens + self.transformer_pos_embed)
            x_mean = x.mean(dim=1)
            x_max = x.amax(dim=1)
            return self.head(torch.cat([x_mean, x_max], dim=-1))
        if self.head_variant == "small_transformer":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=small_transformer expects flat64 tokens with shape [batch,64,dim]")
            x = self.input_proj(tokens)
            x = self.encoder(x + self.transformer_pos_embed)
            x_mean = x.mean(dim=1)
            x_max = x.amax(dim=1)
            return self.head(torch.cat([x_mean, x_max], dim=-1))
        if self.head_variant == "mil":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=mil expects flat64 tokens with shape [batch,64,dim]")
            token_logits = self.token_classifier(tokens)
            return token_logits.amax(dim=1)
        if self.head_variant == "logsumexp_mil":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=logsumexp_mil expects flat64 tokens with shape [batch,64,dim]")
            token_logits = self.token_classifier(tokens)
            return torch.logsumexp(token_logits, dim=1) - math.log(tokens.shape[1])
        if self.head_variant == "attention_mil":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=attention_mil expects flat64 tokens with shape [batch,64,dim]")
            hidden = self.feat(tokens)
            token_logits = self.classifier(hidden)
            attn_score = self.attention(hidden)
            attn_weight = torch.softmax(attn_score, dim=1)
            return (token_logits * attn_weight).sum(dim=1)
        if self.head_variant == "perch_fusion":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=perch_fusion expects flat64 tokens with shape [batch,64,dim]")
            if raw_features is None:
                raise ValueError("head_variant=perch_fusion requires raw_features=[embedding, selected_label_logits]")
            embedding = raw_features[:, :tokens.shape[-1]]
            selected_label_logits = raw_features[:, tokens.shape[-1]:]
            spatial_mean = tokens.mean(dim=1)
            spatial_max = tokens.amax(dim=1)
            return self.head(torch.cat([embedding, spatial_mean, spatial_max, selected_label_logits], dim=-1))
        if self.head_variant == "perch_strong":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=perch_strong expects flat64 tokens with shape [batch,64,dim]")
            hidden = self.token_feat(tokens)
            token_logits = self.token_classifier(hidden)
            attn_score = self.token_attention(hidden)
            attn_weight = torch.softmax(attn_score, dim=1)
            mil_logits = (token_logits * attn_weight).sum(dim=1)
            spatial_mean = tokens.mean(dim=1)
            spatial_max = tokens.amax(dim=1)
            global_logits = self.global_head(torch.cat([spatial_mean, spatial_max], dim=-1))
            return mil_logits + self.final_scale * global_logits
        if self.head_variant == "prototype_pooling":
            if tokens.ndim != 3 or tokens.shape[1] != 64:
                raise ValueError("head_variant=prototype_pooling expects flat64 tokens with shape [batch,64,dim]")
            tokens_norm = nn.functional.normalize(tokens, dim=-1)
            proto_norm = nn.functional.normalize(self.prototypes, dim=-1)
            sim = torch.einsum("bnd,cpd->bcnp", tokens_norm, proto_norm)
            token_sim = sim.max(dim=2).values.clamp_min(0.0) * float(self.prototype_temperature)  # [B, C, P]
            weighted = token_sim * nn.functional.softplus(self.class_proto_weight)[None, :, :]
            return weighted.sum(dim=-1) + self.class_bias[None, :]
        x = self.blocks(tokens)
        pooled = x.mean(dim=1)
        if self.raw_dim > 0:
            if raw_features is None:
                raise ValueError("raw_features are required when raw_dim > 0")
            pooled = torch.cat([pooled, raw_features], dim=1)
        return self.head(pooled)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fitted_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits[:, fitted_mask]
    targets = targets[:, fitted_mask]
    pos_weight = pos_weight[fitted_mask]
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="mean",
    )


def prototype_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    prototypes = getattr(model, "prototypes", None)
    if prototypes is None or prototypes.ndim != 3 or prototypes.shape[1] <= 1:
        first_param = next(model.parameters())
        return first_param.sum() * 0.0
    proto_norm = nn.functional.normalize(prototypes, dim=-1)
    gram = torch.matmul(proto_norm, proto_norm.transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None, :, :]
    return ((gram - eye) ** 2).mean()


def apply_token_masking(tokens: torch.Tensor, token_mask_prob: float, token_mask_max_frac: float) -> torch.Tensor:
    if token_mask_prob <= 0.0 or token_mask_max_frac <= 0.0 or tokens.ndim != 3:
        return tokens
    batch_size, n_tokens, _ = tokens.shape
    max_mask = min(n_tokens - 1, max(1, int(math.ceil(n_tokens * float(token_mask_max_frac)))))
    if batch_size <= 0 or max_mask <= 0:
        return tokens
    sample_mask = torch.rand(batch_size, device=tokens.device) < float(token_mask_prob)
    sample_indices = torch.nonzero(sample_mask, as_tuple=False).flatten()
    if sample_indices.numel() == 0:
        return tokens
    tokens = tokens.clone()
    for sample_idx in sample_indices.tolist():
        n_mask = int(torch.randint(1, max_mask + 1, (1,), device=tokens.device).item())
        token_idx = torch.randperm(n_tokens, device=tokens.device)[:n_mask]
        tokens[sample_idx, token_idx] = 0.0
    return tokens


def apply_feature_mixup(
    tokens: torch.Tensor,
    raw_features: torch.Tensor | None,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    mixup_prob: float,
    mixup_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if mixup_prob <= 0.0 or mixup_alpha <= 0.0 or tokens.shape[0] < 2:
        return tokens, raw_features, targets, sample_weights
    if float(torch.rand((), device=tokens.device).item()) >= float(mixup_prob):
        return tokens, raw_features, targets, sample_weights
    batch_size = tokens.shape[0]
    perm = torch.randperm(batch_size, device=tokens.device)
    beta = torch.distributions.Beta(float(mixup_alpha), float(mixup_alpha))
    lam_value = float(beta.sample().item())
    lam = torch.tensor(lam_value, device=tokens.device, dtype=tokens.dtype)
    tokens = tokens * lam + tokens[perm] * (1.0 - lam)
    targets = targets * lam + targets[perm] * (1.0 - lam)
    sample_weights = sample_weights * lam + sample_weights[perm] * (1.0 - lam)
    if raw_features is not None:
        raw_features = raw_features * lam + raw_features[perm] * (1.0 - lam)
    return tokens, raw_features, targets, sample_weights


def predict_model(
    model: nn.Module,
    tokens: np.ndarray,
    raw_features: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            token_batch = torch.from_numpy(tokens[start:start + batch_size]).to(device)
            raw_batch = None
            if raw_features is not None:
                raw_batch = torch.from_numpy(raw_features[start:start + batch_size]).to(device)
            pred = torch.sigmoid(model(token_batch, raw_batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def build_loader(
    tokens: np.ndarray,
    raw_features: np.ndarray | None,
    targets: np.ndarray,
    sample_weights: np.ndarray | None,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    token_t = torch.from_numpy(tokens)
    target_t = torch.from_numpy(targets.astype(np.float32, copy=False))
    if sample_weights is None:
        weight_t = torch.ones((len(tokens),), dtype=torch.float32)
    else:
        weight_t = torch.from_numpy(sample_weights.astype(np.float32, copy=False))
    if raw_features is None:
        raw_t = torch.zeros((len(tokens), 0), dtype=torch.float32)
    else:
        raw_t = torch.from_numpy(raw_features)
    dataset = TensorDataset(token_t, raw_t, target_t, weight_t)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )


def train_fold_model(
    tokens_train_outer: np.ndarray,
    raw_train_outer: np.ndarray | None,
    y_train_outer: np.ndarray,
    groups_train_outer: np.ndarray,
    sample_weights_outer: np.ndarray | None,
    real_row_mask_outer: np.ndarray | None,
    fitted_class_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    prototype_init: np.ndarray | None = None,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    inner_all = np.arange(len(tokens_train_outer), dtype=np.int64)
    real_row_mask = np.ones(len(tokens_train_outer), dtype=bool) if real_row_mask_outer is None else real_row_mask_outer
    real_indices = inner_all[real_row_mask]
    if len(real_indices) == 0:
        raise ValueError("At least one real labeled row is required for inner validation.")
    real_inner_train_idx, inner_val_idx = make_inner_split(
        train_idx=real_indices,
        groups=groups_train_outer,
        inner_val_files=args.inner_val_files,
        seed=seed,
    )
    pseudo_indices = inner_all[~real_row_mask]
    inner_train_idx = np.concatenate([real_inner_train_idx, pseudo_indices]).astype(np.int64, copy=False)
    if sample_weights_outer is None:
        sample_weights_outer = np.ones(len(tokens_train_outer), dtype=np.float32)

    raw_dim = 0 if raw_train_outer is None else int(raw_train_outer.shape[1])
    model = PerchSpatialMambaHead(
        token_dim=tokens_train_outer.shape[-1],
        num_classes=y_train_outer.shape[1],
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        raw_dim=raw_dim,
        freq_pool=args.freq_pool,
        use_pos_embed=args.use_pos_embed,
        head_variant=args.head_variant,
        prototype_per_class=args.prototype_per_class,
        prototype_temperature=args.prototype_temperature,
    ).to(device)
    if prototype_init is not None:
        if args.head_variant != "prototype_pooling":
            raise ValueError("prototype_init is only valid with head_variant=prototype_pooling")
        expected_shape = (
            int(y_train_outer.shape[1]),
            int(args.prototype_per_class),
            int(tokens_train_outer.shape[-1]),
        )
        if tuple(prototype_init.shape) != expected_shape:
            raise ValueError(f"prototype_init shape mismatch: {prototype_init.shape} vs {expected_shape}")
        with torch.no_grad():
            model.prototypes.copy_(torch.from_numpy(prototype_init.astype(np.float32, copy=False)).to(device))

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

    train_loader = build_loader(
        tokens=tokens_train_outer[inner_train_idx],
        raw_features=None if raw_train_outer is None else raw_train_outer[inner_train_idx],
        targets=y_train_outer[inner_train_idx],
        sample_weights=sample_weights_outer[inner_train_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )

    if len(inner_val_idx) > 0:
        val_tokens_t = torch.from_numpy(tokens_train_outer[inner_val_idx]).to(device)
        val_raw_t = None
        if raw_train_outer is not None:
            val_raw_t = torch.from_numpy(raw_train_outer[inner_val_idx]).to(device)
        val_y_t = torch.from_numpy(y_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device)
    else:
        val_tokens_t = None
        val_raw_t = None
        val_y_t = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for token_batch, raw_batch, y_batch, weight_batch in train_loader:
            token_batch = token_batch.to(device)
            y_batch = y_batch.to(device)
            weight_batch = weight_batch.to(device)
            raw_batch_t = raw_batch.to(device) if raw_dim > 0 else None
            optimizer.zero_grad(set_to_none=True)
            token_batch, raw_batch_t, y_batch, weight_batch = apply_feature_mixup(
                tokens=token_batch,
                raw_features=raw_batch_t,
                targets=y_batch,
                sample_weights=weight_batch,
                mixup_prob=float(args.mixup_prob),
                mixup_alpha=float(args.mixup_alpha),
            )
            token_batch = apply_token_masking(
                tokens=token_batch,
                token_mask_prob=float(args.token_mask_prob),
                token_mask_max_frac=float(args.token_mask_max_frac),
            )
            logits = model(token_batch, raw_batch_t)
            per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, fitted_mask],
                y_batch[:, fitted_mask],
                pos_weight=pos_weight_t[fitted_mask],
                reduction="none",
            ).mean(dim=1)
            loss = (per_sample_loss * weight_batch).sum() / weight_batch.sum().clamp_min(1e-6)
            if args.head_variant == "prototype_pooling" and args.prototype_orth_weight > 0:
                loss = loss + float(args.prototype_orth_weight) * prototype_orthogonality_loss(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        if val_tokens_t is not None and val_y_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_tokens_t, val_raw_t)
                monitor_loss = float(
                    masked_bce_with_logits(val_logits, val_y_t, fitted_mask=fitted_mask, pos_weight=pos_weight_t)
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
        "token_dim": int(tokens_train_outer.shape[-1]),
        "raw_dim": int(raw_dim),
        "output_dim": int(y_train_outer.shape[1]),
        "num_blocks": int(args.num_blocks),
        "kernel_size": int(args.kernel_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "freq_pool": str(args.freq_pool),
        "use_pos_embed": bool(args.use_pos_embed),
        "head_variant": str(args.head_variant),
        "prototype_per_class": int(args.prototype_per_class),
        "prototype_temperature": float(args.prototype_temperature),
        "prototype_orth_weight": float(args.prototype_orth_weight),
        "prototype_init_source": str(args.prototype_init_source),
        "prototype_init_audio_cache_dir": str(args.prototype_init_audio_cache_dir),
        "prototype_init_max_rows_per_class": int(args.prototype_init_max_rows_per_class),
        "prototype_init_candidate_tokens": int(args.prototype_init_candidate_tokens),
        "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
        "token_mask_prob": float(args.token_mask_prob),
        "token_mask_max_frac": float(args.token_mask_max_frac),
        "mixup_prob": float(args.mixup_prob),
        "mixup_alpha": float(args.mixup_alpha),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
    }
    stats = {
        "best_epoch": float(best_epoch),
        "best_loss": float(best_loss),
        "inner_train_rows": float(len(inner_train_idx)),
        "inner_val_rows": float(len(inner_val_idx)),
        "inner_real_train_rows": float(len(real_inner_train_idx)),
        "inner_pseudo_train_rows": float(len(pseudo_indices)),
    }
    return artifact, stats


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    spatial_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_pseudo_rows: int
    n_train_files: int
    n_valid_files: int
    n_pseudo_files: int
    fitted_classes: int
    token_dim: int
    raw_dim: int
    best_epoch: int
    best_loss: float
    inner_train_rows: int
    inner_val_rows: int
    inner_real_train_rows: int
    inner_pseudo_train_rows: int


def build_splits(meta_df: pd.DataFrame, args: argparse.Namespace) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    groups = meta_df["filename"].to_numpy()
    if args.fold_assignment_path:
        folds = load_fold_assignments(Path(args.fold_assignment_path), meta_df=meta_df)
        split_items: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for fold in sorted(np.unique(folds).tolist()):
            valid_idx = np.where(folds == fold)[0].astype(np.int64)
            train_idx = np.where(folds != fold)[0].astype(np.int64)
            split_items.append((int(fold), train_idx, valid_idx))
        return split_items

    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need {args.n_folds}.")
    gkf = GroupKFold(n_splits=args.n_folds)
    return [
        (fold, np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
        for fold, (train_idx, valid_idx) in enumerate(gkf.split(meta_df, groups=groups), start=1)
    ]


def main() -> None:
    args = parse_args()
    if args.use_pos_embed and args.freq_pool != "flat64":
        raise ValueError("--use-pos-embed is only valid with --freq-pool flat64")
    if args.head_variant == "perch_mamba_v1":
        if args.freq_pool != "mean":
            raise ValueError("--head-variant perch_mamba_v1 requires --freq-pool mean")
        if args.include_raw_scores:
            raise ValueError("--head-variant perch_mamba_v1 does not support --include-raw-scores")
    if args.head_variant in {
        "attention_pooling",
        "multihead_attention_pooling",
        "perch_transformer",
        "small_transformer",
        "mil",
        "logsumexp_mil",
        "attention_mil",
        "perch_strong",
        "prototype_pooling",
    }:
        if args.freq_pool != "flat64":
            raise ValueError(f"--head-variant {args.head_variant} requires --freq-pool flat64")
        if args.include_raw_scores:
            raise ValueError(f"--head-variant {args.head_variant} does not support --include-raw-scores")
        if args.use_pos_embed:
            raise ValueError(f"--head-variant {args.head_variant} does not support --use-pos-embed")
    if args.head_variant == "perch_fusion":
        if args.freq_pool != "flat64":
            raise ValueError("--head-variant perch_fusion requires --freq-pool flat64")
        if args.use_pos_embed:
            raise ValueError("--head-variant perch_fusion does not support --use-pos-embed")
        if args.include_raw_scores:
            raise ValueError("--head-variant perch_fusion builds embedding+raw score features automatically")
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
    spatial_meta, spatial_tokens, spatial_tokens_max, spatial_tokens_64 = load_spatial_cache(args)
    spatial_tokens = align_spatial_to_base(
        base_meta=meta_df,
        spatial_meta=spatial_meta,
        spatial_tokens=spatial_tokens,
    )
    if spatial_tokens_max is not None:
        spatial_tokens_max = align_spatial_to_base(
            base_meta=meta_df,
            spatial_meta=spatial_meta,
            spatial_tokens=spatial_tokens_max,
        )
    if spatial_tokens_64 is not None:
        spatial_tokens_64 = align_spatial_to_base(
            base_meta=meta_df,
            spatial_meta=spatial_meta,
            spatial_tokens=spatial_tokens_64,
        )

    audio_init_meta = None
    audio_init_tokens = None
    audio_init_y = None
    if args.prototype_init_source == "audio_token":
        if args.head_variant != "prototype_pooling":
            raise ValueError("--prototype-init-source audio_token requires --head-variant prototype_pooling")
        if not args.prototype_init_audio_cache_dir:
            raise ValueError("--prototype-init-audio-cache-dir is required when using audio_token init")
        audio_init_meta, audio_init_tokens, audio_init_y = load_audio_prototype_init_cache(
            cache_dir_arg=args.prototype_init_audio_cache_dir,
            meta_path_arg=args.prototype_init_audio_meta_path,
            arrays_path_arg=args.prototype_init_audio_arrays_path,
            freq_pool=args.freq_pool,
            class_names=class_names,
        )

    pseudo_root = Path(args.pseudo_root) if args.pseudo_root else None
    pseudo_spatial_meta = None
    pseudo_spatial_tokens = None
    pseudo_spatial_tokens_max = None
    pseudo_spatial_tokens_64 = None
    if pseudo_root is not None:
        if not args.pseudo_spatial_cache_dir and not args.pseudo_spatial_meta_path:
            raise ValueError("--pseudo-spatial-cache-dir or --pseudo-spatial-meta-path is required with --pseudo-root")
        pseudo_spatial_meta, pseudo_spatial_tokens, pseudo_spatial_tokens_max, pseudo_spatial_tokens_64 = (
            load_spatial_cache_from_paths(
                cache_dir_arg=args.pseudo_spatial_cache_dir,
                meta_path_arg=args.pseudo_spatial_meta_path,
                arrays_path_arg=args.pseudo_spatial_arrays_path,
            )
        )

    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32)
    raw_sigmoid_auc = macro_auc_skip_empty(y_true, raw_sigmoid)
    oof_pred = raw_sigmoid.copy()
    splits = build_splits(meta_df, args=args)

    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    source_summaries: List[pd.DataFrame] = []

    print("[INFO] Train Perch spatial Mamba head")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] spatial_tokens: {spatial_tokens.shape}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] freq_pool: {args.freq_pool}")
    print(f"[INFO] use_pos_embed: {args.use_pos_embed}")
    print(f"[INFO] token_pca_dim: {args.token_pca_dim}")
    print(f"[INFO] include_raw_scores: {args.include_raw_scores} raw_proj_dim={args.raw_proj_dim}")
    print(f"[INFO] head_variant: {args.head_variant}")
    print(f"[INFO] num_blocks={args.num_blocks} kernel_size={args.kernel_size} hidden_dim={args.hidden_dim}")
    print(
        "[INFO] train_aug: "
        f"token_mask_prob={args.token_mask_prob} token_mask_max_frac={args.token_mask_max_frac} "
        f"mixup_prob={args.mixup_prob} mixup_alpha={args.mixup_alpha}"
    )
    print(f"[INFO] mlp_min_pos: {args.mlp_min_pos}")
    if args.fold_assignment_path:
        print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")
    if pseudo_root is not None:
        print(f"[INFO] pseudo_root: {pseudo_root}")
        print(f"[INFO] pseudo_spatial_rows: {len(pseudo_spatial_meta)}")
        print(f"[INFO] pseudo_loss_weight: {args.pseudo_loss_weight}")
        print(f"[INFO] min_pseudo_max_prob: {args.min_pseudo_max_prob}")
    if args.prototype_init_source == "audio_token":
        print("[INFO] prototype_init_source: audio_token")
        print(f"[INFO] prototype_init_audio_rows: {len(audio_init_meta) if audio_init_meta is not None else 0}")
        print(f"[INFO] prototype_init_max_rows_per_class: {args.prototype_init_max_rows_per_class}")
        print(f"[INFO] prototype_init_candidate_tokens: {args.prototype_init_candidate_tokens}")
    if args.prototype_init_source == "soundscape_token":
        print("[INFO] prototype_init_source: soundscape_token")
        print(f"[INFO] prototype_init_max_rows_per_class: {args.prototype_init_max_rows_per_class}")
        print(f"[INFO] prototype_init_candidate_tokens: {args.prototype_init_candidate_tokens}")

    for display_fold, train_idx, valid_idx in splits:
        token_source_train = spatial_tokens[train_idx]
        token_source_valid = spatial_tokens[valid_idx]
        pseudo_df = None
        pseudo_probs = None
        pseudo_token_source = None
        pseudo_fold_dir = None
        pseudo_files = 0
        if pseudo_root is not None:
            pseudo_df, pseudo_probs, pseudo_fold_dir = load_pseudo_package(
                pseudo_root=pseudo_root,
                fold=int(display_fold),
                class_names=class_names,
                min_pseudo_max_prob=args.min_pseudo_max_prob,
                max_pseudo_rows=args.max_pseudo_rows,
            )
            keep_pseudo = ~pseudo_df["filename"].astype(str).isin(set(meta_df["filename"].astype(str)))
            if not keep_pseudo.all():
                pseudo_df = pseudo_df.loc[keep_pseudo].reset_index(drop=True)
                pseudo_probs = pseudo_probs[keep_pseudo.to_numpy()]
            pseudo_files = int(pseudo_df["filename"].nunique()) if len(pseudo_df) else 0
            pseudo_token_source = align_array_by_row_id(
                source_meta=pseudo_spatial_meta,
                source_array=pseudo_spatial_tokens,
                target_row_ids=pseudo_df["row_id"].astype(str).tolist(),
                name="pseudo spatial_tokens",
            )
        if args.freq_pool == "meanmax":
            if spatial_tokens_max is None:
                raise ValueError("freq_pool=meanmax requires spatial_tokens_max in the spatial cache.")
            token_source_train = np.stack([spatial_tokens[train_idx], spatial_tokens_max[train_idx]], axis=1)
            token_source_valid = np.stack([spatial_tokens[valid_idx], spatial_tokens_max[valid_idx]], axis=1)
            if pseudo_df is not None:
                if pseudo_spatial_tokens_max is None:
                    raise ValueError("freq_pool=meanmax requires pseudo spatial_tokens_max in the pseudo spatial cache.")
                pseudo_token_mean = align_array_by_row_id(
                    source_meta=pseudo_spatial_meta,
                    source_array=pseudo_spatial_tokens,
                    target_row_ids=pseudo_df["row_id"].astype(str).tolist(),
                    name="pseudo spatial_tokens",
                )
                pseudo_token_max = align_array_by_row_id(
                    source_meta=pseudo_spatial_meta,
                    source_array=pseudo_spatial_tokens_max,
                    target_row_ids=pseudo_df["row_id"].astype(str).tolist(),
                    name="pseudo spatial_tokens_max",
                )
                pseudo_token_source = np.stack([pseudo_token_mean, pseudo_token_max], axis=1)
        elif args.freq_pool == "flat64":
            if spatial_tokens_64 is None:
                raise ValueError("freq_pool=flat64 requires spatial_tokens_64 in the spatial cache.")
            token_source_train = spatial_tokens_64[train_idx]
            token_source_valid = spatial_tokens_64[valid_idx]
            if pseudo_df is not None:
                if pseudo_spatial_tokens_64 is None:
                    raise ValueError("freq_pool=flat64 requires pseudo spatial_tokens_64 in the pseudo spatial cache.")
                pseudo_token_source = align_array_by_row_id(
                    source_meta=pseudo_spatial_meta,
                    source_array=pseudo_spatial_tokens_64,
                    target_row_ids=pseudo_df["row_id"].astype(str).tolist(),
                    name="pseudo spatial_tokens_64",
                )

        projector_source_train = token_source_train
        if pseudo_token_source is not None and len(pseudo_token_source) > 0:
            projector_source_train = np.concatenate([token_source_train, pseudo_token_source], axis=0)

        token_projector = fit_tokens_projector(
            tokens_train=projector_source_train,
            token_pca_dim=args.token_pca_dim,
            seed=args.seed + int(display_fold),
        )
        tokens_train = transform_tokens_with_projector(token_source_train, token_projector)
        tokens_valid = transform_tokens_with_projector(token_source_valid, token_projector)
        pseudo_tokens = None
        if pseudo_token_source is not None and len(pseudo_token_source) > 0:
            pseudo_tokens = transform_tokens_with_projector(pseudo_token_source, token_projector)
        y_train = y_true[train_idx]
        real_pos = y_true[train_idx].sum(axis=0)
        real_neg = len(train_idx) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)

        prototype_init = None
        if args.head_variant == "prototype_pooling" and args.prototype_init_source in {"audio_token", "soundscape_token"}:
            if audio_init_meta is None or audio_init_tokens is None or audio_init_y is None:
                if args.prototype_init_source == "audio_token":
                    raise ValueError("audio_token prototype init requested but audio init cache is missing")
            if args.prototype_init_source == "audio_token":
                source_tokens_projected = transform_tokens_with_projector(audio_init_tokens, token_projector)
                source_y = audio_init_y
            else:
                source_tokens_projected = tokens_train
                source_y = y_train
            prototype_init = build_token_prototypes(
                source_tokens=source_tokens_projected,
                source_y=source_y,
                fitted_class_indices=fitted_class_indices,
                num_classes=y_true.shape[1],
                prototype_per_class=args.prototype_per_class,
                max_rows_per_class=args.prototype_init_max_rows_per_class,
                candidate_tokens=args.prototype_init_candidate_tokens,
                seed=args.seed + int(display_fold) * 37,
            )

        raw_projector = None
        raw_train_features = None
        raw_valid_features = None
        if args.head_variant == "perch_fusion":
            fusion_train_raw = np.concatenate([emb_full[train_idx], scores_full_raw[train_idx]], axis=1).astype(
                np.float32,
                copy=False,
            )
            fusion_valid_raw = np.concatenate([emb_full[valid_idx], scores_full_raw[valid_idx]], axis=1).astype(
                np.float32,
                copy=False,
            )
            fusion_mean = fusion_train_raw.mean(axis=0, keepdims=True)
            fusion_std = fusion_train_raw.std(axis=0, keepdims=True)
            fusion_std = np.where(fusion_std < 1e-6, 1.0, fusion_std).astype(np.float32)
            raw_projector = RawScoreProjector(
                mean=fusion_mean,
                std=fusion_std,
                pca=None,
                output_dim=fusion_train_raw.shape[1],
            )
            raw_train_features = raw_projector.transform(fusion_train_raw)
            raw_valid_features = raw_projector.transform(fusion_valid_raw)
        elif args.include_raw_scores:
            raw_projector = fit_raw_projector(
                raw_train=scores_full_raw[train_idx],
                raw_proj_dim=args.raw_proj_dim,
                seed=args.seed + int(display_fold) * 17,
            )
            raw_train_features = raw_projector.transform(scores_full_raw[train_idx])
            raw_valid_features = raw_projector.transform(scores_full_raw[valid_idx])

        train_groups_outer = groups[train_idx]
        sample_weights_outer = np.ones(len(y_train), dtype=np.float32)
        real_row_mask_outer = np.ones(len(y_train), dtype=bool)
        if pseudo_tokens is not None and pseudo_probs is not None and len(pseudo_tokens) > 0:
            tokens_train = np.concatenate([tokens_train, pseudo_tokens], axis=0).astype(np.float32, copy=False)
            y_train = np.concatenate([y_train.astype(np.float32, copy=False), pseudo_probs], axis=0).astype(
                np.float32,
                copy=False,
            )
            train_groups_outer = np.concatenate(
                [train_groups_outer, pseudo_df["filename"].astype(str).to_numpy()],
                axis=0,
            )
            pseudo_weights = (
                np.full(len(pseudo_probs), float(args.pseudo_loss_weight), dtype=np.float32)
                * pseudo_df["max_prob"].to_numpy(dtype=np.float32)
            )
            sample_weights_outer = np.concatenate([sample_weights_outer, pseudo_weights], axis=0)
            real_row_mask_outer = np.concatenate([real_row_mask_outer, np.zeros(len(pseudo_probs), dtype=bool)])
            if raw_train_features is not None:
                raise ValueError("Pseudo training with raw score features is not supported yet.")
        model_artifact, train_stats = train_fold_model(
            tokens_train_outer=tokens_train,
            raw_train_outer=raw_train_features,
            y_train_outer=y_train,
            groups_train_outer=train_groups_outer,
            sample_weights_outer=sample_weights_outer,
            real_row_mask_outer=real_row_mask_outer,
            fitted_class_indices=fitted_class_indices,
            args=args,
            seed=args.seed + int(display_fold) * 1000,
            device=device,
            prototype_init=prototype_init,
        )

        model = PerchSpatialMambaHead(
            token_dim=model_artifact["token_dim"],
            num_classes=model_artifact["output_dim"],
            num_blocks=model_artifact["num_blocks"],
            kernel_size=model_artifact["kernel_size"],
            hidden_dim=model_artifact["hidden_dim"],
            dropout=model_artifact["dropout"],
            raw_dim=model_artifact["raw_dim"],
            freq_pool=model_artifact.get("freq_pool", args.freq_pool),
            use_pos_embed=bool(model_artifact.get("use_pos_embed", False)),
            head_variant=str(model_artifact.get("head_variant", "generic")),
            prototype_per_class=int(model_artifact.get("prototype_per_class", args.prototype_per_class)),
            prototype_temperature=float(model_artifact.get("prototype_temperature", args.prototype_temperature)),
        ).to(device)
        model.load_state_dict(model_artifact["model_state"])
        spatial_all = predict_model(
            model=model,
            tokens=tokens_valid,
            raw_features=raw_valid_features,
            device=device,
            batch_size=args.batch_size,
        )
        fold_pred = raw_sigmoid[valid_idx].copy()
        fold_pred[:, fitted_class_indices] = spatial_all[:, fitted_class_indices]
        fold_pred = np.clip(fold_pred, 0.0, 1.0).astype(np.float32, copy=False)
        oof_pred[valid_idx] = fold_pred

        fold_artifact = {
            "fold_name": f"fold_{display_fold}",
            "token_projector": tokens_projector_to_artifact(token_projector),
            "raw_projector": raw_projector_to_artifact(raw_projector),
            "model": model_artifact,
            "pseudo_fold_dir": "" if pseudo_fold_dir is None else str(pseudo_fold_dir),
        }
        fold_artifacts.append(fold_artifact)
        source_rows = [
            {
                "source": "real",
                "rows": int(len(train_idx)),
                "fold": int(display_fold),
                "pseudo_dir": "" if pseudo_fold_dir is None else str(pseudo_fold_dir),
            }
        ]
        if pseudo_df is not None:
            source_rows.append(
                {
                    "source": "pseudo",
                    "rows": int(len(pseudo_df)),
                    "fold": int(display_fold),
                    "pseudo_dir": "" if pseudo_fold_dir is None else str(pseudo_fold_dir),
                }
            )
        source_summaries.append(pd.DataFrame(source_rows))

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], raw_sigmoid[valid_idx])
        spatial_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        fold_result = FoldResult(
            fold=int(display_fold),
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            spatial_valid_auc=float(spatial_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_pseudo_rows=0 if pseudo_df is None else int(len(pseudo_df)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            n_pseudo_files=int(pseudo_files),
            fitted_classes=int(len(fitted_class_indices)),
            token_dim=int(tokens_train.shape[-1]),
            raw_dim=0 if raw_train_features is None else int(raw_train_features.shape[1]),
            best_epoch=int(train_stats["best_epoch"]),
            best_loss=float(train_stats["best_loss"]),
            inner_train_rows=int(train_stats["inner_train_rows"]),
            inner_val_rows=int(train_stats["inner_val_rows"]),
            inner_real_train_rows=int(train_stats["inner_real_train_rows"]),
            inner_pseudo_train_rows=int(train_stats["inner_pseudo_train_rows"]),
        )
        fold_results.append(fold_result)
        print(
            f"[FOLD {display_fold}] raw_sigmoid_auc={raw_valid_auc:.6f} "
            f"spatial_auc={spatial_valid_auc:.6f} token_dim={fold_result.token_dim} "
            f"raw_dim={fold_result.raw_dim} fitted_classes={fold_result.fitted_classes} "
            f"pseudo_rows={fold_result.n_pseudo_rows} "
            f"best_epoch={fold_result.best_epoch} train_files={fold_result.n_train_files} "
            f"valid_files={fold_result.n_valid_files}",
            flush=True,
        )

    spatial_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_spatial_auc = float(np.mean([item.spatial_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_spatial_auc - spatial_oof_auc)

    artifact = {
        "artifact_version": 1,
        "model_type": "perch_spatial_mamba",
        "class_names": class_names,
        "config": {
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "pseudo_root": str(args.pseudo_root),
            "pseudo_spatial_cache_dir": str(args.pseudo_spatial_cache_dir),
            "pseudo_loss_weight": float(args.pseudo_loss_weight),
            "min_pseudo_max_prob": float(args.min_pseudo_max_prob),
            "max_pseudo_rows": int(args.max_pseudo_rows),
            "token_pca_dim": int(args.token_pca_dim),
            "freq_pool": str(args.freq_pool),
            "use_pos_embed": bool(args.use_pos_embed),
            "include_raw_scores": bool(args.include_raw_scores),
            "raw_proj_dim": int(args.raw_proj_dim),
            "head_variant": str(args.head_variant),
            "num_blocks": int(args.num_blocks),
            "kernel_size": int(args.kernel_size),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "token_mask_prob": float(args.token_mask_prob),
            "token_mask_max_frac": float(args.token_mask_max_frac),
            "mixup_prob": float(args.mixup_prob),
            "mixup_alpha": float(args.mixup_alpha),
            "mlp_min_pos": int(args.mlp_min_pos),
            "prototype_per_class": int(args.prototype_per_class),
            "prototype_temperature": float(args.prototype_temperature),
            "prototype_orth_weight": float(args.prototype_orth_weight),
            "prototype_init_source": str(args.prototype_init_source),
            "prototype_init_audio_cache_dir": str(args.prototype_init_audio_cache_dir),
            "prototype_init_max_rows_per_class": int(args.prototype_init_max_rows_per_class),
            "prototype_init_candidate_tokens": int(args.prototype_init_candidate_tokens),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight_power": float(args.pos_weight_power),
            "pos_weight_max": float(args.pos_weight_max),
            "inner_val_files": int(args.inner_val_files),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_spatial_mamba_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    fold_metrics_df = pd.DataFrame([item.__dict__ for item in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    source_summary_path = output_dir / "train_source_summary.csv"
    pd.concat(source_summaries, axis=0, ignore_index=True).to_csv(source_summary_path, index=False)

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
        "spatial_oof_auc": float(spatial_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_spatial_auc": float(mean_fold_spatial_auc),
        "fold_gap": float(fold_gap),
        "pseudo_root": str(args.pseudo_root),
        "pseudo_spatial_cache_dir": str(args.pseudo_spatial_cache_dir),
        "pseudo_loss_weight": float(args.pseudo_loss_weight),
        "min_pseudo_max_prob": float(args.min_pseudo_max_prob),
        "max_pseudo_rows": int(args.max_pseudo_rows),
        "token_pca_dim": int(args.token_pca_dim),
        "freq_pool": str(args.freq_pool),
        "use_pos_embed": bool(args.use_pos_embed),
        "include_raw_scores": bool(args.include_raw_scores),
        "raw_proj_dim": int(args.raw_proj_dim),
        "head_variant": str(args.head_variant),
        "num_blocks": int(args.num_blocks),
        "kernel_size": int(args.kernel_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "token_mask_prob": float(args.token_mask_prob),
        "token_mask_max_frac": float(args.token_mask_max_frac),
        "mixup_prob": float(args.mixup_prob),
        "mixup_alpha": float(args.mixup_alpha),
        "mlp_min_pos": int(args.mlp_min_pos),
        "prototype_per_class": int(args.prototype_per_class),
        "prototype_temperature": float(args.prototype_temperature),
        "prototype_orth_weight": float(args.prototype_orth_weight),
        "prototype_init_source": str(args.prototype_init_source),
        "prototype_init_audio_cache_dir": str(args.prototype_init_audio_cache_dir),
        "prototype_init_max_rows_per_class": int(args.prototype_init_max_rows_per_class),
        "prototype_init_candidate_tokens": int(args.prototype_init_candidate_tokens),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "pos_weight_power": float(args.pos_weight_power),
        "pos_weight_max": float(args.pos_weight_max),
        "inner_val_files": int(args.inner_val_files),
        "patience": int(args.patience),
        "artifact_path": str(artifact_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] spatial_oof_auc: {spatial_oof_auc:.6f}")
    print(f"[INFO] mean_fold_spatial_auc: {mean_fold_spatial_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")
    print(f"[INFO] Saved artifact to: {artifact_path}")
    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved train source summary to: {source_summary_path}")
    print(f"[INFO] Saved OOF predictions to: {oof_pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
