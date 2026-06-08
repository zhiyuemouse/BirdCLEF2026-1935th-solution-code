#!/usr/bin/env python3
"""Train a fold-safe 60s temporal head on frozen Perch spatial tokens.

Unlike the earlier Perch spatial heads, this model sees a full soundscape at
once:

    [file, 12 windows, 16 Perch tokens, 1536 dim] -> [file, 12, classes]

The loss is only applied to windows that are present in
``train_soundscapes_labels.csv``.  Unlabeled windows inside partial files are
allowed to provide temporal context, but they are never treated as negative
labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026_perch_context_train import (
    load_cache as load_base_cache,
    load_class_names,
    load_meta,
    macro_auc_skip_empty,
    seed_everything,
    sigmoid_np,
)


N_WINDOWS = 12
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Perch 60s temporal head folds.")
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--base-meta-path", type=str, default="")
    parser.add_argument("--base-arrays-path", type=str, default="")
    parser.add_argument("--spatial-cache-dir", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--spatial-meta-path", type=str, default="")
    parser.add_argument("--spatial-arrays-path", type=str, default="")
    parser.add_argument("--spatial-token-key", type=str, default="spatial_tokens")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--fold-assignment-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_temporal_head_labeled_all_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--local-blocks", type=int, default=0)
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--local-on-raw-tokens", action="store_true")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=12.0)
    parser.add_argument("--inner-val-files", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--teacher-target-path", type=str, default="")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-use-unlabeled-windows", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_end_seconds(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(str(row_id).rsplit("_", 1)[-1]) for row_id in row_ids], dtype=np.int16)


def load_spatial_cache(
    cache_dir_arg: str,
    meta_path_arg: str,
    arrays_path_arg: str,
    token_key: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
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

    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    if token_key not in arrays:
        raise KeyError(f"{arrays_path} must contain {token_key!r}. Available keys: {arrays.files}")
    tokens = arrays[token_key].astype(np.float32, copy=False)
    if tokens.ndim != 3:
        raise ValueError(f"Expected spatial_tokens [rows,16,dim], got {tokens.shape}")
    return meta_df, tokens


def parse_label_cell(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def union_labels(series: Sequence[object]) -> List[str]:
    labels = set()
    for value in series:
        labels.update(parse_label_cell(value))
    return sorted(labels)


def build_labels_and_mask(
    labels_path: Path,
    class_names: Sequence[str],
    row_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(labels_path)
    sc_clean = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    row_to_labels = sc_clean.set_index("row_id")["label_list"].to_dict()
    y = np.zeros((len(row_ids), len(class_names)), dtype=np.uint8)
    mask = np.zeros((len(row_ids),), dtype=bool)
    for i, row_id in enumerate(row_ids):
        labels = row_to_labels.get(str(row_id))
        if labels is None:
            continue
        mask[i] = True
        idxs = [label_to_idx[label] for label in labels if label in label_to_idx]
        if idxs:
            y[i, idxs] = 1
    return y, mask


@dataclass
class FileTensorPack:
    filenames: np.ndarray
    row_ids: np.ndarray
    tokens: np.ndarray
    y: np.ndarray
    label_mask: np.ndarray


@dataclass
class TeacherTargetPack:
    row_id: np.ndarray
    fold_values: np.ndarray
    teacher_by_fold: np.ndarray


def build_file_tensors(meta_df: pd.DataFrame, spatial_tokens: np.ndarray, y: np.ndarray, label_mask: np.ndarray) -> FileTensorPack:
    meta = meta_df.copy().reset_index(drop=True)
    meta["_end_sec"] = parse_end_seconds(meta["row_id"].astype(str).tolist())
    meta["_pos"] = np.arange(len(meta), dtype=np.int64)
    meta = meta.sort_values(["filename", "_end_sec"]).reset_index(drop=True)

    filenames: List[str] = []
    row_ids: List[np.ndarray] = []
    token_files: List[np.ndarray] = []
    y_files: List[np.ndarray] = []
    mask_files: List[np.ndarray] = []
    for filename, group in meta.groupby("filename", sort=False):
        if len(group) != N_WINDOWS:
            raise ValueError(f"Expected {N_WINDOWS} windows for {filename}, got {len(group)}")
        expected = np.arange(5, 65, 5)
        got = group["_end_sec"].to_numpy(dtype=np.int16)
        if not np.array_equal(got, expected):
            raise ValueError(f"Unexpected end seconds for {filename}: {got.tolist()}")
        idx = group["_pos"].to_numpy(dtype=np.int64)
        filenames.append(str(filename))
        row_ids.append(group["row_id"].astype(str).to_numpy())
        token_files.append(spatial_tokens[idx])
        y_files.append(y[idx])
        mask_files.append(label_mask[idx])

    return FileTensorPack(
        filenames=np.asarray(filenames, dtype=object),
        row_ids=np.stack(row_ids, axis=0),
        tokens=np.stack(token_files, axis=0).astype(np.float32, copy=False),
        y=np.stack(y_files, axis=0).astype(np.uint8, copy=False),
        label_mask=np.stack(mask_files, axis=0).astype(bool, copy=False),
    )


def align_base_raw_to_spatial(
    base_meta: pd.DataFrame,
    scores_full_raw: np.ndarray,
    target_row_ids: np.ndarray,
) -> np.ndarray:
    source_pos = pd.Series(np.arange(len(base_meta), dtype=np.int64), index=base_meta["row_id"].astype(str))
    flat_ids = target_row_ids.reshape(-1)
    indices = pd.Series(flat_ids, dtype=str).map(source_pos)
    raw = np.full((len(flat_ids), scores_full_raw.shape[1]), np.nan, dtype=np.float32)
    known = ~indices.isna().to_numpy()
    raw[known] = scores_full_raw[indices[known].to_numpy(dtype=np.int64)]
    return raw.reshape(target_row_ids.shape[0], target_row_ids.shape[1], scores_full_raw.shape[1])


def load_teacher_targets(path_arg: str, target_row_ids: np.ndarray) -> TeacherTargetPack | None:
    if not path_arg:
        return None
    package = np.load(path_arg, allow_pickle=True)
    required = {"row_id", "fold_values", "teacher_by_fold"}
    missing = required - set(package.files)
    if missing:
        raise KeyError(f"Teacher target package is missing keys: {sorted(missing)}")
    source_row_id = package["row_id"].astype(str)
    teacher_by_fold = package["teacher_by_fold"].astype(np.float32, copy=False)
    source_pos = pd.Series(np.arange(len(source_row_id), dtype=np.int64), index=source_row_id)
    flat_ids = target_row_ids.reshape(-1)
    indices = pd.Series(flat_ids, dtype=str).map(source_pos)
    missing_mask = indices.isna().to_numpy()
    aligned_flat = np.full(
        (teacher_by_fold.shape[0], len(flat_ids), teacher_by_fold.shape[-1]),
        np.nan,
        dtype=np.float32,
    )
    if (~missing_mask).any():
        idx = indices[~missing_mask].to_numpy(dtype=np.int64)
        aligned_flat[:, ~missing_mask, :] = teacher_by_fold[:, idx, :]
    if missing_mask.any():
        missing_rows = pd.Series(flat_ids, dtype=str).loc[missing_mask].head(5).tolist()
        print(
            "[WARN] Teacher target package misses "
            f"{int(missing_mask.sum())} rows; these windows will not receive teacher loss. "
            f"Examples: {missing_rows}",
            flush=True,
        )
    aligned = aligned_flat.reshape(
        teacher_by_fold.shape[0],
        target_row_ids.shape[0],
        target_row_ids.shape[1],
        teacher_by_fold.shape[-1],
    )
    return TeacherTargetPack(
        row_id=target_row_ids.copy(),
        fold_values=package["fold_values"].astype(np.int16, copy=False),
        teacher_by_fold=aligned.astype(np.float32, copy=False),
    )


class TokenStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        return ((tokens - self.mean) / self.std).astype(np.float32, copy=False)


def fit_standardizer(tokens_train: np.ndarray) -> TokenStandardizer:
    flat = tokens_train.reshape(-1, tokens_train.shape[-1]).astype(np.float32, copy=False)
    mean = flat.mean(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    std = flat.std(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return TokenStandardizer(mean=mean, std=std)


def standardizer_to_artifact(standardizer: TokenStandardizer) -> Dict[str, np.ndarray]:
    return {"mean": standardizer.mean, "std": standardizer.std}


def build_file_folds(pack: FileTensorPack, fold_assignment_path: str, n_folds: int) -> np.ndarray:
    if fold_assignment_path:
        fold_df = pd.read_csv(fold_assignment_path)
        required = {"filename", "fold"}
        missing = required - set(fold_df.columns)
        if missing:
            raise KeyError(f"Fold assignment file is missing columns: {sorted(missing)}")
        file_folds = fold_df.drop_duplicates(subset=["filename", "fold"]).groupby("filename")["fold"].nunique()
        bad = file_folds[file_folds != 1]
        if len(bad) > 0:
            raise ValueError(f"Some filenames have multiple folds: {bad.head().to_dict()}")
        fold_map = fold_df.drop_duplicates(subset=["filename"]).set_index("filename")["fold"]
        folds = pd.Series(pack.filenames.astype(str)).map(fold_map)
        if folds.isna().any():
            missing_files = pd.Series(pack.filenames.astype(str)).loc[folds.isna()].head(5).tolist()
            raise ValueError(f"Fold assignment misses files: {missing_files}")
        return folds.astype(int).to_numpy()

    groups = pack.filenames.astype(str)
    gkf = GroupKFold(n_splits=n_folds)
    folds = np.full(len(groups), -1, dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups), start=1):
        folds[valid_idx] = fold
    return folds


def make_inner_split(train_files_idx: np.ndarray, pack: FileTensorPack, inner_val_files: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if inner_val_files <= 0 or len(train_files_idx) <= 2:
        return train_files_idx, np.asarray([], dtype=np.int64)
    eligible = train_files_idx[pack.label_mask[train_files_idx].any(axis=1)]
    if len(eligible) <= 2:
        return train_files_idx, np.asarray([], dtype=np.int64)
    n_val = min(int(inner_val_files), max(1, len(eligible) // 5))
    rng = np.random.default_rng(seed)
    val_idx = np.sort(rng.choice(eligible, size=n_val, replace=False)).astype(np.int64)
    val_set = set(val_idx.tolist())
    train_idx = np.asarray([idx for idx in train_files_idx if int(idx) not in val_set], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        return train_files_idx, np.asarray([], dtype=np.int64)
    return train_idx, val_idx


class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.dwconv = nn.Conv1d(
            int(dim),
            int(dim),
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=int(dim),
        )
        self.gate = nn.Linear(int(dim), int(dim))
        self.proj = nn.Linear(int(dim), int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.proj(x))
        return shortcut + x


class PerchTemporalHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_classes: int,
        local_blocks: int,
        local_kernel_size: int,
        local_on_raw_tokens: bool,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local_on_raw_tokens = bool(local_on_raw_tokens)
        self.raw_local_blocks = None
        if self.local_on_raw_tokens:
            self.raw_local_blocks = nn.Sequential(
                *[
                    LocalMambaBlock(
                        dim=int(token_dim),
                        kernel_size=int(local_kernel_size),
                        dropout=float(dropout) * 0.4,
                    )
                    for _ in range(int(local_blocks))
                ]
            )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        hidden_local_blocks = 0 if self.local_on_raw_tokens else int(local_blocks)
        self.local_blocks = nn.Sequential(
            *[
                LocalMambaBlock(
                    dim=int(hidden_dim),
                    kernel_size=int(local_kernel_size),
                    dropout=float(dropout) * 0.4,
                )
                for _ in range(hidden_local_blocks)
            ]
        )
        self.local_attn = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), max(64, int(hidden_dim) // 4)),
            nn.Tanh(),
            nn.Linear(max(64, int(hidden_dim) // 4), 1),
        )
        self.window_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        self.time_pos = nn.Parameter(torch.zeros(1, N_WINDOWS, int(hidden_dim)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 12, 16, D]
        bsz, n_win, n_tok, _ = tokens.shape
        if self.raw_local_blocks is not None:
            tokens = self.raw_local_blocks(tokens.reshape(bsz * n_win, n_tok, -1)).reshape(bsz, n_win, n_tok, -1)
        x = self.input_proj(tokens)
        x = self.local_blocks(x.reshape(bsz * n_win, n_tok, -1)).reshape(bsz, n_win, n_tok, -1)
        attn = torch.softmax(self.local_attn(x), dim=2)
        local_attn_pool = (x * attn).sum(dim=2)
        local_max_pool = x.amax(dim=2)
        window_feat = self.window_proj(torch.cat([local_attn_pool, local_max_pool], dim=-1))
        context = self.encoder(window_feat + self.time_pos[:, :n_win, :])
        logits = self.head(torch.cat([window_feat, context], dim=-1))
        return logits


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    window_mask: torch.Tensor,
    fitted_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    valid_logits = logits[window_mask][:, fitted_mask]
    valid_targets = targets[window_mask][:, fitted_mask]
    if valid_logits.numel() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=pos_weight[fitted_mask],
        reduction="mean",
    )


def build_loader(
    tokens: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    teacher_targets: np.ndarray | None,
    teacher_mask: np.ndarray | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    if teacher_targets is None:
        teacher_targets = np.zeros_like(targets, dtype=np.float32)
    if teacher_mask is None:
        teacher_mask = np.zeros_like(label_mask, dtype=bool)
    dataset = TensorDataset(
        torch.from_numpy(tokens.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
        torch.from_numpy(label_mask.astype(bool, copy=False)),
        torch.from_numpy(teacher_targets.astype(np.float32, copy=False)),
        torch.from_numpy(teacher_mask.astype(bool, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def predict_model(model: nn.Module, tokens: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.from_numpy(tokens[start:start + batch_size].astype(np.float32, copy=False)).to(device)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def train_fold_model(
    tokens_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    mask_train_outer: np.ndarray,
    teacher_train_outer: np.ndarray | None,
    teacher_mask_train_outer: np.ndarray | None,
    fitted_class_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    all_idx = np.arange(len(tokens_train_outer), dtype=np.int64)
    inner_train_idx, inner_val_idx = make_inner_split(
        train_files_idx=all_idx,
        pack=FileTensorPack(
            filenames=np.asarray([str(i) for i in all_idx], dtype=object),
            row_ids=np.empty((len(all_idx), N_WINDOWS), dtype=object),
            tokens=tokens_train_outer,
            y=y_train_outer,
            label_mask=mask_train_outer,
        ),
        inner_val_files=args.inner_val_files,
        seed=seed,
    )

    pos = y_train_outer[inner_train_idx][mask_train_outer[inner_train_idx]].sum(axis=0).astype(np.float32)
    n_labeled = int(mask_train_outer[inner_train_idx].sum())
    neg = n_labeled - pos
    pos_weight = np.ones(y_train_outer.shape[-1], dtype=np.float32)
    valid_pos = pos > 0
    pos_weight[valid_pos] = np.power(neg[valid_pos] / np.maximum(pos[valid_pos], 1.0), args.pos_weight_power)
    pos_weight = np.clip(pos_weight, 1.0, float(args.pos_weight_max)).astype(np.float32)

    fitted_mask_np = np.zeros(y_train_outer.shape[-1], dtype=bool)
    fitted_mask_np[fitted_class_indices] = True
    fitted_mask = torch.from_numpy(fitted_mask_np).to(device)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    model = PerchTemporalHead(
        token_dim=tokens_train_outer.shape[-1],
        hidden_dim=args.hidden_dim,
        num_classes=y_train_outer.shape[-1],
        local_blocks=args.local_blocks,
        local_kernel_size=args.local_kernel_size,
        local_on_raw_tokens=args.local_on_raw_tokens,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    train_loader = build_loader(
        tokens=tokens_train_outer[inner_train_idx],
        targets=y_train_outer[inner_train_idx],
        label_mask=mask_train_outer[inner_train_idx],
        teacher_targets=None if teacher_train_outer is None else teacher_train_outer[inner_train_idx],
        teacher_mask=None if teacher_mask_train_outer is None else teacher_mask_train_outer[inner_train_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        shuffle=True,
    )

    if len(inner_val_idx) > 0:
        val_tokens_t = torch.from_numpy(tokens_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device)
        val_y_t = torch.from_numpy(y_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device)
        val_mask_t = torch.from_numpy(mask_train_outer[inner_val_idx].astype(bool, copy=False)).to(device)
    else:
        val_tokens_t = None
        val_y_t = None
        val_mask_t = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for token_batch, y_batch, mask_batch, teacher_batch, teacher_mask_batch in train_loader:
            token_batch = token_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            teacher_batch = teacher_batch.to(device)
            teacher_mask_batch = teacher_mask_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_batch)
            loss = masked_bce_with_logits(
                logits=logits,
                targets=y_batch,
                window_mask=mask_batch,
                fitted_mask=fitted_mask,
                pos_weight=pos_weight_t,
            )
            if float(args.teacher_loss_weight) > 0.0 and teacher_mask_batch.any():
                teacher_logits = logits[teacher_mask_batch][:, fitted_mask]
                teacher_targets = teacher_batch[teacher_mask_batch][:, fitted_mask].clamp(EPS, 1.0 - EPS)
                teacher_loss = nn.functional.binary_cross_entropy_with_logits(
                    teacher_logits,
                    teacher_targets,
                    reduction="mean",
                )
                loss = loss + float(args.teacher_loss_weight) * teacher_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if val_tokens_t is not None and val_y_t is not None and val_mask_t is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_tokens_t)
                monitor_loss = float(
                    masked_bce_with_logits(
                        logits=val_logits,
                        targets=val_y_t,
                        window_mask=val_mask_t,
                        fitted_mask=fitted_mask,
                        pos_weight=pos_weight_t,
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
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    artifact = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "token_dim": int(tokens_train_outer.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "local_blocks": int(args.local_blocks),
        "local_kernel_size": int(args.local_kernel_size),
        "local_on_raw_tokens": bool(args.local_on_raw_tokens),
        "output_dim": int(y_train_outer.shape[-1]),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "fitted_class_indices": fitted_class_indices.astype(np.int32, copy=False),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
    }
    stats = {
        "best_epoch": float(best_epoch),
        "best_loss": float(best_loss),
        "inner_train_files": float(len(inner_train_idx)),
        "inner_val_files": float(len(inner_val_idx)),
        "inner_train_labeled_windows": float(mask_train_outer[inner_train_idx].sum()),
        "inner_val_labeled_windows": float(mask_train_outer[inner_val_idx].sum()) if len(inner_val_idx) else 0.0,
    }
    return artifact, stats


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    temporal_valid_auc: float
    n_train_files: int
    n_valid_files: int
    n_train_labeled_windows: int
    n_valid_labeled_windows: int
    fitted_classes: int
    best_epoch: int
    best_loss: float
    inner_train_files: int
    inner_val_files: int
    inner_train_labeled_windows: int
    inner_val_labeled_windows: int


def main() -> None:
    args = parse_args()
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --num-heads")
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    class_names = load_class_names(Path(args.sample_submission_path))

    spatial_meta, spatial_tokens = load_spatial_cache(
        cache_dir_arg=args.spatial_cache_dir,
        meta_path_arg=args.spatial_meta_path,
        arrays_path_arg=args.spatial_arrays_path,
        token_key=args.spatial_token_key,
    )
    y_spatial, label_mask_spatial = build_labels_and_mask(
        labels_path=Path(args.labels_path),
        class_names=class_names,
        row_ids=spatial_meta["row_id"].astype(str).tolist(),
    )
    pack = build_file_tensors(
        meta_df=spatial_meta,
        spatial_tokens=spatial_tokens,
        y=y_spatial,
        label_mask=label_mask_spatial,
    )

    if args.limit_files > 0:
        keep = np.arange(min(args.limit_files, len(pack.filenames)), dtype=np.int64)
        pack = FileTensorPack(
            filenames=pack.filenames[keep],
            row_ids=pack.row_ids[keep],
            tokens=pack.tokens[keep],
            y=pack.y[keep],
            label_mask=pack.label_mask[keep],
        )

    base_meta, scores_full_raw, _ = load_base_cache(
        cache_dir=Path(args.base_cache_dir),
        meta_path_arg=args.base_meta_path,
        arrays_path_arg=args.base_arrays_path,
    )
    raw_file_scores = align_base_raw_to_spatial(base_meta=base_meta, scores_full_raw=scores_full_raw, target_row_ids=pack.row_ids)
    raw_sigmoid_files = sigmoid_np(np.nan_to_num(raw_file_scores, nan=0.0)).astype(np.float32)

    file_folds = build_file_folds(pack, fold_assignment_path=args.fold_assignment_path, n_folds=args.n_folds)
    fold_values = sorted(pd.Index(file_folds).unique().tolist())
    teacher_pack = load_teacher_targets(args.teacher_target_path, target_row_ids=pack.row_ids)
    teacher_fold_to_pos: Dict[int, int] = {}
    if teacher_pack is not None:
        if teacher_pack.teacher_by_fold.shape[1:] != (len(pack.filenames), N_WINDOWS, len(class_names)):
            raise ValueError(
                "Teacher target shape mismatch: "
                f"{teacher_pack.teacher_by_fold.shape} vs expected "
                f"({len(teacher_pack.fold_values)}, {len(pack.filenames)}, {N_WINDOWS}, {len(class_names)})"
            )
        teacher_fold_to_pos = {int(fold): idx for idx, fold in enumerate(teacher_pack.fold_values.tolist())}
        missing_teacher_folds = sorted(set(int(fold) for fold in fold_values) - set(teacher_fold_to_pos))
        if missing_teacher_folds:
            raise ValueError(f"Teacher target package misses folds: {missing_teacher_folds}")

    labeled_flat_mask = pack.label_mask.reshape(-1)
    y_labeled = pack.y.reshape(-1, len(class_names))[labeled_flat_mask]
    raw_labeled = raw_sigmoid_files.reshape(-1, len(class_names))[labeled_flat_mask]
    row_id_labeled = pack.row_ids.reshape(-1)[labeled_flat_mask]
    filename_labeled = np.repeat(pack.filenames, N_WINDOWS)[labeled_flat_mask]
    raw_sigmoid_auc = macro_auc_skip_empty(y_labeled, raw_labeled)
    oof_pred_labeled = raw_labeled.copy()

    print("[INFO] Train Perch 60s temporal head")
    print(f"[INFO] files: {len(pack.filenames)}")
    print(f"[INFO] windows: {pack.tokens.shape[0] * pack.tokens.shape[1]}")
    print(f"[INFO] labeled_windows: {int(pack.label_mask.sum())}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] tokens: {pack.tokens.shape}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] folds: {fold_values}")
    print(f"[INFO] device: {device}")
    print(
        "[INFO] model: "
        f"hidden_dim={args.hidden_dim} local_blocks={args.local_blocks} "
        f"local_kernel={args.local_kernel_size} local_on_raw={args.local_on_raw_tokens} layers={args.num_layers} "
        f"heads={args.num_heads} dropout={args.dropout}"
    )
    if args.fold_assignment_path:
        print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")
    if teacher_pack is not None:
        print(
            f"[INFO] teacher_target_path: {args.teacher_target_path} | "
            f"teacher_loss_weight={args.teacher_loss_weight} | "
            f"teacher_use_unlabeled={args.teacher_use_unlabeled_windows}"
        )

    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []

    for display_fold in fold_values:
        valid_files_idx = np.where(file_folds == int(display_fold))[0].astype(np.int64)
        train_files_idx = np.where(file_folds != int(display_fold))[0].astype(np.int64)
        standardizer = fit_standardizer(pack.tokens[train_files_idx])
        tokens_train = standardizer.transform(pack.tokens[train_files_idx])
        tokens_valid = standardizer.transform(pack.tokens[valid_files_idx])
        y_train = pack.y[train_files_idx]
        mask_train = pack.label_mask[train_files_idx]
        teacher_train = None
        teacher_mask_train = None
        if teacher_pack is not None and float(args.teacher_loss_weight) > 0.0:
            teacher_pos = teacher_fold_to_pos[int(display_fold)]
            teacher_train = teacher_pack.teacher_by_fold[teacher_pos, train_files_idx]
            if args.teacher_use_unlabeled_windows:
                teacher_mask_train = np.isfinite(teacher_train).all(axis=-1)
            else:
                teacher_mask_train = mask_train & np.isfinite(teacher_train).all(axis=-1)
            teacher_train = np.nan_to_num(teacher_train, nan=0.5).astype(np.float32, copy=False)

        train_labeled = y_train[mask_train]
        real_pos = train_labeled.sum(axis=0).astype(np.float32)
        real_neg = len(train_labeled) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)

        model_artifact, train_stats = train_fold_model(
            tokens_train_outer=tokens_train,
            y_train_outer=y_train,
            mask_train_outer=mask_train,
            teacher_train_outer=teacher_train,
            teacher_mask_train_outer=teacher_mask_train,
            fitted_class_indices=fitted_class_indices,
            args=args,
            seed=args.seed + int(display_fold) * 1000,
            device=device,
        )

        model = PerchTemporalHead(
            token_dim=model_artifact["token_dim"],
            hidden_dim=model_artifact["hidden_dim"],
            num_classes=model_artifact["output_dim"],
            local_blocks=model_artifact.get("local_blocks", args.local_blocks),
            local_kernel_size=model_artifact.get("local_kernel_size", args.local_kernel_size),
            local_on_raw_tokens=model_artifact.get("local_on_raw_tokens", args.local_on_raw_tokens),
            num_layers=model_artifact["num_layers"],
            num_heads=model_artifact["num_heads"],
            dropout=model_artifact["dropout"],
        ).to(device)
        model.load_state_dict(model_artifact["model_state"])
        pred_valid_files = predict_model(model=model, tokens=tokens_valid, device=device, batch_size=args.batch_size)

        valid_label_mask = pack.label_mask[valid_files_idx]
        valid_y = pack.y[valid_files_idx][valid_label_mask]
        valid_raw = raw_sigmoid_files[valid_files_idx][valid_label_mask]
        valid_pred = valid_raw.copy()
        valid_pred[:, fitted_class_indices] = pred_valid_files[valid_label_mask][:, fitted_class_indices]
        valid_pred = np.clip(valid_pred, 0.0, 1.0).astype(np.float32, copy=False)

        valid_flat_global = np.isin(row_id_labeled, pack.row_ids[valid_files_idx].reshape(-1))
        oof_pred_labeled[valid_flat_global] = valid_pred

        raw_valid_auc = macro_auc_skip_empty(valid_y, valid_raw)
        temporal_valid_auc = macro_auc_skip_empty(valid_y, valid_pred)
        fold_result = FoldResult(
            fold=int(display_fold),
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            temporal_valid_auc=float(temporal_valid_auc),
            n_train_files=int(len(train_files_idx)),
            n_valid_files=int(len(valid_files_idx)),
            n_train_labeled_windows=int(mask_train.sum()),
            n_valid_labeled_windows=int(valid_label_mask.sum()),
            fitted_classes=int(len(fitted_class_indices)),
            best_epoch=int(train_stats["best_epoch"]),
            best_loss=float(train_stats["best_loss"]),
            inner_train_files=int(train_stats["inner_train_files"]),
            inner_val_files=int(train_stats["inner_val_files"]),
            inner_train_labeled_windows=int(train_stats["inner_train_labeled_windows"]),
            inner_val_labeled_windows=int(train_stats["inner_val_labeled_windows"]),
        )
        fold_results.append(fold_result)
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "token_standardizer": standardizer_to_artifact(standardizer),
                "model": model_artifact,
            }
        )
        print(
            f"[FOLD {display_fold}] raw_sigmoid_auc={raw_valid_auc:.6f} "
            f"temporal_auc={temporal_valid_auc:.6f} fitted_classes={fold_result.fitted_classes} "
            f"best_epoch={fold_result.best_epoch} train_files={fold_result.n_train_files} "
            f"valid_files={fold_result.n_valid_files} train_labeled={fold_result.n_train_labeled_windows} "
            f"valid_labeled={fold_result.n_valid_labeled_windows}",
            flush=True,
        )

    temporal_oof_auc = macro_auc_skip_empty(y_labeled, oof_pred_labeled)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_temporal_auc = float(np.mean([item.temporal_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_temporal_auc - temporal_oof_auc)

    artifact = {
        "artifact_version": 1,
        "model_type": "perch_temporal_head",
        "class_names": class_names,
        "config": {
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "spatial_cache_dir": str(args.spatial_cache_dir),
            "spatial_token_key": str(args.spatial_token_key),
            "hidden_dim": int(args.hidden_dim),
            "local_blocks": int(args.local_blocks),
            "local_kernel_size": int(args.local_kernel_size),
            "local_on_raw_tokens": bool(args.local_on_raw_tokens),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "mlp_min_pos": int(args.mlp_min_pos),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight_power": float(args.pos_weight_power),
            "pos_weight_max": float(args.pos_weight_max),
            "inner_val_files": int(args.inner_val_files),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "teacher_target_path": str(args.teacher_target_path),
            "teacher_loss_weight": float(args.teacher_loss_weight),
            "teacher_use_unlabeled_windows": bool(args.teacher_use_unlabeled_windows),
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_temporal_head_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    fold_metrics_df = pd.DataFrame([item.__dict__ for item in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y_true=y_labeled.astype(np.uint8, copy=False),
        raw_scores=raw_labeled.astype(np.float32, copy=False),
        oof_pred=oof_pred_labeled.astype(np.float32, copy=False),
        row_id=row_id_labeled.astype(object),
        filename=filename_labeled.astype(object),
    )

    summary = {
        "files": int(len(pack.filenames)),
        "windows": int(pack.tokens.shape[0] * pack.tokens.shape[1]),
        "labeled_windows": int(pack.label_mask.sum()),
        "classes": int(len(class_names)),
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "temporal_oof_auc": float(temporal_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_temporal_auc": float(mean_fold_temporal_auc),
        "fold_gap": float(fold_gap),
        "spatial_cache_dir": str(args.spatial_cache_dir),
        "spatial_token_key": str(args.spatial_token_key),
        "hidden_dim": int(args.hidden_dim),
        "local_blocks": int(args.local_blocks),
        "local_kernel_size": int(args.local_kernel_size),
        "local_on_raw_tokens": bool(args.local_on_raw_tokens),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "mlp_min_pos": int(args.mlp_min_pos),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "pos_weight_power": float(args.pos_weight_power),
        "pos_weight_max": float(args.pos_weight_max),
        "inner_val_files": int(args.inner_val_files),
        "patience": int(args.patience),
        "artifact_path": str(artifact_path),
        "teacher_target_path": str(args.teacher_target_path),
        "teacher_loss_weight": float(args.teacher_loss_weight),
        "teacher_use_unlabeled_windows": bool(args.teacher_use_unlabeled_windows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] temporal_oof_auc: {temporal_oof_auc:.6f}")
    print(f"[INFO] mean_fold_temporal_auc: {mean_fold_temporal_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")
    print(f"[INFO] Saved artifact to: {artifact_path}")
    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF predictions to: {output_dir / 'oof_predictions.npz'}")
    print(f"[INFO] Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
