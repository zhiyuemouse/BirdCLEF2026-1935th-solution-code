#!/usr/bin/env python3
"""Honest local CV for Perch + temporal residual modeling.

This script is intentionally conservative about evaluation:

- uses cached local Perch outputs from `perch_cache/`
- aligns labels strictly by `row_id`
- rebuilds each soundscape as a 12-step sequence
- splits folds by `filename`
- reports row-level macro ROC-AUC over the 234 competition classes

Important:
- This is a lightweight PyTorch re-implementation in the *style* of
  "Perch + ProtoSSM", not an exact copy of any public Kaggle notebook.
- The model is residual: it starts from a first-pass base predictor and learns
  a temporal correction on top of it.
- For the strongest default setting, the first-pass base is a non-site
  Perch + context + per-class LogReg model trained inside each outer fold.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdclef2026_perch_context_logreg import (
    build_context_tensor,
    build_metadata_features,
    build_position_features,
)
from birdclef2026_perch_context_train import (
    fit_context_artifact as fit_deploy_context_artifact,
    predict_context_artifact as predict_deploy_context_artifact,
)
from birdclef2026_perch_probe_cv import (
    build_aligned_labels,
    build_features,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Honest local CV for Perch + temporal residual modeling.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_protossm_cv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--expected-seq-len", type=int, default=12)
    parser.add_argument(
        "--base-score-mode",
        type=str,
        choices=["raw_perch", "context_logreg"],
        default="context_logreg",
    )
    parser.add_argument("--base-inner-folds", type=int, default=4)
    parser.add_argument("--base-embedding-pca-dim", type=int, default=64)
    parser.add_argument("--base-logreg-c", type=float, default=0.125)
    parser.add_argument("--base-logreg-max-iter", type=int, default=1000)
    parser.add_argument("--base-logreg-min-pos", type=int, default=8)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["embedding", "embedding_plus_scores"],
        default="embedding",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["protossm_lite", "bigru"],
        default="protossm_lite",
    )
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--proto-temperature", type=float, default=8.0)
    parser.add_argument("--proto-weight-init", type=float, default=0.0)
    parser.add_argument("--temporal-blend", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--inner-val-files", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_end_seconds(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(str(row_id).rsplit("_", 1)[-1]) for row_id in row_ids], dtype=np.int64)


@dataclass
class SequencePack:
    features_seq: np.ndarray
    base_scores_seq: np.ndarray
    y_seq: np.ndarray
    row_index_seq: np.ndarray
    row_id_seq: np.ndarray
    filename_seq: np.ndarray
    end_seconds_seq: np.ndarray


def group_rows_into_sequences(
    meta_df: pd.DataFrame,
    x_rows: np.ndarray,
    base_scores_rows: np.ndarray,
    y_rows: np.ndarray,
    expected_seq_len: int,
    source_row_indices: np.ndarray | None = None,
) -> SequencePack:
    file_groups = meta_df.groupby("filename", sort=False).indices

    features_seq: List[np.ndarray] = []
    base_scores_seq: List[np.ndarray] = []
    y_seq: List[np.ndarray] = []
    row_index_seq: List[np.ndarray] = []
    row_id_seq: List[np.ndarray] = []
    filename_seq: List[str] = []
    end_seconds_seq: List[np.ndarray] = []

    for filename, row_indices in file_groups.items():
        idx = np.asarray(row_indices, dtype=np.int64)
        row_ids = meta_df.iloc[idx]["row_id"].tolist()
        end_seconds = parse_end_seconds(row_ids)
        order = np.argsort(end_seconds, kind="stable")
        idx_sorted = idx[order]
        end_seconds_sorted = end_seconds[order]
        row_ids_sorted = meta_df.iloc[idx_sorted]["row_id"].to_numpy(dtype=object)

        if len(idx_sorted) != expected_seq_len:
            raise ValueError(
                f"File {filename} has {len(idx_sorted)} rows, expected {expected_seq_len}. "
                "This script assumes the `full_files` Perch cache format."
            )

        features_seq.append(x_rows[idx_sorted].astype(np.float32, copy=False))
        base_scores_seq.append(base_scores_rows[idx_sorted].astype(np.float32, copy=False))
        y_seq.append(y_rows[idx_sorted].astype(np.uint8, copy=False))
        if source_row_indices is None:
            row_index_seq.append(idx_sorted.astype(np.int64, copy=False))
        else:
            row_index_seq.append(np.asarray(source_row_indices[idx_sorted], dtype=np.int64))
        row_id_seq.append(row_ids_sorted)
        filename_seq.append(str(filename))
        end_seconds_seq.append(end_seconds_sorted.astype(np.int64, copy=False))

    return SequencePack(
        features_seq=np.stack(features_seq).astype(np.float32, copy=False),
        base_scores_seq=np.stack(base_scores_seq).astype(np.float32, copy=False),
        y_seq=np.stack(y_seq).astype(np.uint8, copy=False),
        row_index_seq=np.stack(row_index_seq).astype(np.int64, copy=False),
        row_id_seq=np.stack(row_id_seq),
        filename_seq=np.asarray(filename_seq, dtype=object),
        end_seconds_seq=np.stack(end_seconds_seq).astype(np.int64, copy=False),
    )


def preprocess_feature_sequences(
    x_train_seq: np.ndarray,
    x_valid_seq: np.ndarray,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n_train, seq_len, input_dim = x_train_seq.shape
    n_valid = x_valid_seq.shape[0]

    train_flat = x_train_seq.reshape(n_train * seq_len, input_dim)
    valid_flat = x_valid_seq.reshape(n_valid * seq_len, input_dim)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).astype(np.float32)
    valid_scaled = scaler.transform(valid_flat).astype(np.float32)

    actual_dim = train_scaled.shape[1]
    if pca_dim > 0:
        max_dim = min(pca_dim, train_scaled.shape[0] - 1, train_scaled.shape[1])
        if max_dim >= 1 and max_dim < train_scaled.shape[1]:
            pca = PCA(n_components=max_dim, random_state=seed)
            train_scaled = pca.fit_transform(train_scaled).astype(np.float32)
            valid_scaled = pca.transform(valid_scaled).astype(np.float32)
            actual_dim = int(max_dim)

    train_seq = train_scaled.reshape(n_train, seq_len, actual_dim).astype(np.float32, copy=False)
    valid_seq = valid_scaled.reshape(n_valid, seq_len, actual_dim).astype(np.float32, copy=False)
    return train_seq, valid_seq, int(actual_dim)


def scores_to_logits(scores: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    scores = scores.clamp(min=eps, max=1.0 - eps)
    return torch.log(scores / (1.0 - scores))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x_seq: np.ndarray, base_scores_seq: np.ndarray, y_seq: np.ndarray):
        self.x_seq = torch.from_numpy(x_seq.astype(np.float32, copy=False))
        self.base_scores_seq = torch.from_numpy(base_scores_seq.astype(np.float32, copy=False))
        self.y_seq = torch.from_numpy(y_seq.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int):
        return self.x_seq[idx], self.base_scores_seq[idx], self.y_seq[idx]


class SelectiveSSM1D(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 4)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, forget_gate, input_gate, output_gate = self.in_proj(x).chunk(4, dim=-1)
        candidate = torch.tanh(candidate)
        forget_gate = torch.sigmoid(forget_gate + 1.0)
        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)

        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(x.shape[1]):
            state = forget_gate[:, step] * state + input_gate[:, step] * candidate[:, step]
            outputs.append(output_gate[:, step] * torch.tanh(state))

        y = torch.stack(outputs, dim=1)
        return self.out_proj(self.dropout(y))


class ProtoSSMBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm_fwd = SelectiveSSM1D(d_model=d_model, dropout=dropout)
        self.ssm_bwd = SelectiveSSM1D(d_model=d_model, dropout=dropout)
        self.mix_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        y_fwd = self.ssm_fwd(z)
        y_bwd = torch.flip(self.ssm_bwd(torch.flip(z, dims=[1])), dims=[1])
        y = self.mix_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        x = x + self.dropout(y)
        x = x + self.ffn(self.norm2(x))
        return x


class ResidualSequenceHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        proto_temperature: float,
        proto_weight_init: float,
    ):
        super().__init__()
        self.out_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Linear(d_model, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.proto_scale = nn.Parameter(torch.tensor(float(proto_weight_init)))
        self.proto_temperature = float(proto_temperature)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, hidden: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        hidden = self.out_norm(hidden)
        base_logits = scores_to_logits(base_scores)
        delta_logits = self.delta_head(hidden)

        hidden_norm = F.normalize(hidden, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        proto_logits = torch.einsum("btd,cd->btc", hidden_norm, proto_norm) * self.proto_temperature

        return base_logits + delta_logits + (self.proto_scale * proto_logits)


class ProtoSSMLite(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        proto_temperature: float,
        proto_weight_init: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([ProtoSSMBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)])
        self.head = ResidualSequenceHead(
            d_model=d_model,
            num_classes=num_classes,
            proto_temperature=proto_temperature,
            proto_weight_init=proto_weight_init,
        )

    def forward(self, x_seq: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x_seq)
        hidden = self.input_norm(hidden + self.pos_embed[:, : x_seq.shape[1]])
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden, base_scores=base_scores)


class BiGRUResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        proto_temperature: float,
        proto_weight_init: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = ResidualSequenceHead(
            d_model=d_model,
            num_classes=num_classes,
            proto_temperature=proto_temperature,
            proto_weight_init=proto_weight_init,
        )

    def forward(self, x_seq: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x_seq)
        hidden = self.input_norm(hidden + self.pos_embed[:, : x_seq.shape[1]])
        hidden, _ = self.gru(hidden)
        hidden = self.output_proj(self.dropout(hidden))
        return self.head(hidden, base_scores=base_scores)


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    num_classes: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    proto_temperature: float,
    proto_weight_init: float,
) -> nn.Module:
    if model_type == "protossm_lite":
        return ProtoSSMLite(
            input_dim=input_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            proto_temperature=proto_temperature,
            proto_weight_init=proto_weight_init,
        )
    if model_type == "bigru":
        return BiGRUResidual(
            input_dim=input_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            proto_temperature=proto_temperature,
            proto_weight_init=proto_weight_init,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def split_inner_sequence_indices(
    n_train_files: int,
    inner_val_files: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(n_train_files, dtype=np.int64)
    if inner_val_files <= 0 or n_train_files <= 2:
        return all_idx, np.asarray([], dtype=np.int64)
    n_val = min(int(inner_val_files), max(1, n_train_files // 5))
    rng = np.random.default_rng(seed)
    valid_idx = np.sort(rng.choice(all_idx, size=n_val, replace=False)).astype(np.int64)
    valid_mask = np.zeros(n_train_files, dtype=bool)
    valid_mask[valid_idx] = True
    inner_train_idx = all_idx[~valid_mask]
    if len(inner_train_idx) == 0 or len(valid_idx) == 0:
        return all_idx, np.asarray([], dtype=np.int64)
    return inner_train_idx, valid_idx


def train_temporal_model_for_epochs(
    args: argparse.Namespace,
    x_train_seq: np.ndarray,
    base_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    actual_input_dim: int,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    train_dataset = SequenceDataset(
        x_seq=x_train_seq,
        base_scores_seq=base_train_seq,
        y_seq=y_train_seq,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    num_classes = y_train_seq.shape[2]
    seq_len = y_train_seq.shape[1]
    model = build_model(
        model_type=args.model_type,
        input_dim=actual_input_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        proto_temperature=args.proto_temperature,
        proto_weight_init=args.proto_weight_init,
    ).to(device)

    y_train_flat = torch.from_numpy(y_train_seq.reshape(-1, num_classes).astype(np.float32, copy=False))
    pos = y_train_flat.sum(dim=0)
    neg = y_train_flat.shape[0] - pos
    pos_weight = (neg / (pos + 1.0)).clamp(max=50.0).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(1, max(int(epochs), 1) + 1):
        model.train()
        for batch_x, batch_base, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_base = batch_base.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, base_scores=batch_base)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
    return model


def predict_temporal_model(
    model: nn.Module,
    x_seq: np.ndarray,
    base_seq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_seq.astype(np.float32, copy=False)).to(device)
        base_t = torch.from_numpy(base_seq.astype(np.float32, copy=False)).to(device)
        pred = torch.sigmoid(model(x_t, base_scores=base_t)).cpu().numpy().astype(np.float32, copy=False)
    return pred


@dataclass
class FoldResult:
    fold: int
    n_train_files: int
    n_valid_files: int
    n_train_rows: int
    n_valid_rows: int
    raw_valid_auc: float
    base_valid_auc: float
    temporal_valid_auc: float
    best_epoch: int
    actual_input_dim: int
    base_actual_embedding_dim: int
    base_fitted_classes: int


def train_context_base_split(
    args: argparse.Namespace,
    emb_full: np.ndarray,
    raw_scores_full: np.ndarray,
    y_true: np.ndarray,
    context_tensor: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed_offset: int,
) -> Tuple[np.ndarray, int, int]:
    fold_artifact = fit_deploy_context_artifact(
        emb_train=emb_full[train_idx],
        raw_scores_train=raw_scores_full[train_idx],
        context_train=context_tensor[train_idx],
        position_train=position_features[train_idx],
        metadata_train=metadata_features[train_idx],
        y_train=y_true[train_idx],
        c_value=args.base_logreg_c,
        max_iter=args.base_logreg_max_iter,
        min_pos=args.base_logreg_min_pos,
        embedding_pca_dim=args.base_embedding_pca_dim,
        seed=args.seed + seed_offset,
        fold_name=f"protossm_base_{seed_offset}",
    )
    pred = predict_deploy_context_artifact(
        fold_artifact=fold_artifact,
        emb=emb_full[valid_idx],
        raw_scores=raw_scores_full[valid_idx],
        context=context_tensor[valid_idx],
        position_features=position_features[valid_idx],
        metadata_features=metadata_features[valid_idx],
        sigmoid_fallback=True,
    )
    actual_embedding_dim = int(fold_artifact["actual_embedding_dim"])
    fitted_classes = int(len(fold_artifact["fitted_class_indices"]))
    return pred, actual_embedding_dim, fitted_classes


def build_base_predictions_for_outer_fold(
    args: argparse.Namespace,
    meta_df: pd.DataFrame,
    emb_full: np.ndarray,
    raw_scores_full: np.ndarray,
    y_true: np.ndarray,
    context_tensor: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_valid_idx: np.ndarray,
    outer_fold: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if args.base_score_mode == "raw_perch":
        return (
            sigmoid_np(raw_scores_full[outer_train_idx]).astype(np.float32, copy=False),
            sigmoid_np(raw_scores_full[outer_valid_idx]).astype(np.float32, copy=False),
            0,
            0,
        )

    outer_valid_pred, base_actual_embedding_dim, base_fitted_classes = train_context_base_split(
        args=args,
        emb_full=emb_full,
        raw_scores_full=raw_scores_full,
        y_true=y_true,
        context_tensor=context_tensor,
        position_features=position_features,
        metadata_features=metadata_features,
        train_idx=outer_train_idx,
        valid_idx=outer_valid_idx,
        seed_offset=1000 + outer_fold,
    )

    outer_train_groups = meta_df.iloc[outer_train_idx]["filename"].to_numpy()
    inner_unique_files = pd.Index(outer_train_groups).unique()
    inner_n_splits = min(args.base_inner_folds, len(inner_unique_files))
    if inner_n_splits < 2:
        raise ValueError(
            "base_inner_folds requires at least 2 unique files in the outer train split. "
            f"Got {len(inner_unique_files)}."
        )

    inner_oof = np.zeros((len(outer_train_idx), raw_scores_full.shape[1]), dtype=np.float32)
    dummy_inner = np.zeros((len(outer_train_idx), 1), dtype=np.float32)
    inner_gkf = GroupKFold(n_splits=inner_n_splits)

    for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(
        inner_gkf.split(dummy_inner, groups=outer_train_groups),
        start=1,
    ):
        inner_train_abs = outer_train_idx[np.asarray(inner_train_rel, dtype=np.int64)]
        inner_valid_abs = outer_train_idx[np.asarray(inner_valid_rel, dtype=np.int64)]
        inner_valid_pred, _, _ = train_context_base_split(
            args=args,
            emb_full=emb_full,
            raw_scores_full=raw_scores_full,
            y_true=y_true,
            context_tensor=context_tensor,
            position_features=position_features,
            metadata_features=metadata_features,
            train_idx=inner_train_abs,
            valid_idx=inner_valid_abs,
            seed_offset=2000 + (outer_fold * 100) + inner_fold,
        )
        inner_oof[np.asarray(inner_valid_rel, dtype=np.int64)] = inner_valid_pred

    return inner_oof, outer_valid_pred, base_actual_embedding_dim, base_fitted_classes


def train_one_fold(
    args: argparse.Namespace,
    x_train_seq: np.ndarray,
    x_valid_seq: np.ndarray,
    base_train_seq: np.ndarray,
    base_valid_seq: np.ndarray,
    y_train_seq: np.ndarray,
    y_valid_seq: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, int, int]:
    x_train_seq, x_valid_seq, actual_input_dim = preprocess_feature_sequences(
        x_train_seq=x_train_seq,
        x_valid_seq=x_valid_seq,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )

    num_classes = y_train_seq.shape[2]
    inner_train_idx, inner_valid_idx = split_inner_sequence_indices(
        n_train_files=len(x_train_seq),
        inner_val_files=args.inner_val_files,
        seed=args.seed,
    )

    best_epoch = 0
    best_auc = -math.inf
    wait = 0

    if len(inner_valid_idx) == 0:
        best_epoch = max(1, min(args.epochs, args.patience))
    else:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(
            model_type=args.model_type,
            input_dim=actual_input_dim,
            seq_len=y_train_seq.shape[1],
            num_classes=num_classes,
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            proto_temperature=args.proto_temperature,
            proto_weight_init=args.proto_weight_init,
        ).to(device)

        inner_dataset = SequenceDataset(
            x_seq=x_train_seq[inner_train_idx],
            base_scores_seq=base_train_seq[inner_train_idx],
            y_seq=y_train_seq[inner_train_idx],
        )
        inner_loader = torch.utils.data.DataLoader(
            inner_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )

        y_inner_flat = torch.from_numpy(
            y_train_seq[inner_train_idx].reshape(-1, num_classes).astype(np.float32, copy=False)
        )
        pos = y_inner_flat.sum(dim=0)
        neg = y_inner_flat.shape[0] - pos
        pos_weight = (neg / (pos + 1.0)).clamp(max=50.0).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        y_inner_valid_flat = y_train_seq[inner_valid_idx].reshape(-1, num_classes)

        x_inner_valid = x_train_seq[inner_valid_idx]
        base_inner_valid = base_train_seq[inner_valid_idx]

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch_x, batch_base, batch_y in inner_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_base = batch_base.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x, base_scores=batch_base)
                loss = criterion(logits, batch_y)
                loss.backward()
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                optimizer.step()

                batch_size = batch_x.shape[0]
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            inner_valid_pred = predict_temporal_model(
                model=model,
                x_seq=x_inner_valid,
                base_seq=base_inner_valid,
                device=device,
            )
            inner_valid_auc = macro_auc_skip_empty(
                y_inner_valid_flat,
                inner_valid_pred.reshape(-1, num_classes),
            )

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                mean_train_loss = train_loss_sum / max(train_count, 1)
                print(
                    f"[MODEL] epoch={epoch:04d} train_loss={mean_train_loss:.6f} "
                    f"inner_auc={inner_valid_auc:.6f}"
                )

            if inner_valid_auc > best_auc:
                best_auc = inner_valid_auc
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

    # Refit on the full outer-train fold for the selected epoch count. The
    # outer-valid labels are deliberately not used for early stopping.
    torch.manual_seed(args.seed + 17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 17)
    model = train_temporal_model_for_epochs(
        args=args,
        x_train_seq=x_train_seq,
        base_train_seq=base_train_seq,
        y_train_seq=y_train_seq,
        actual_input_dim=actual_input_dim,
        device=device,
        epochs=best_epoch,
    )
    final_pred = predict_temporal_model(
        model=model,
        x_seq=x_valid_seq,
        base_seq=base_valid_seq,
        device=device,
    )
    raw_final_pred = final_pred.astype(np.float32, copy=False)
    if args.temporal_blend < 1.0:
        final_pred = (
            (1.0 - float(args.temporal_blend)) * base_valid_seq
            + float(args.temporal_blend) * final_pred
        ).astype(np.float32, copy=False)
    return final_pred, raw_final_pred, best_epoch, actual_input_dim


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    labels_path = Path(args.labels_path)
    sample_submission_path = Path(args.sample_submission_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df, scores_full_raw, emb_full = load_cache(
        cache_dir=cache_dir,
        meta_path_arg=args.meta_path,
        arrays_path_arg=args.arrays_path,
    )
    class_names = load_class_names(sample_submission_path)
    y_true = build_aligned_labels(
        labels_path=labels_path,
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

    x_rows = build_features(
        emb_full=emb_full,
        scores_full_raw=scores_full_raw,
        feature_mode=args.feature_mode,
    )
    seq_pack = group_rows_into_sequences(
        meta_df=meta_df,
        x_rows=x_rows,
        base_scores_rows=scores_full_raw,
        y_rows=y_true,
        expected_seq_len=args.expected_seq_len,
        source_row_indices=np.arange(len(meta_df), dtype=np.int64),
    )

    n_files = len(seq_pack.filename_seq)
    if n_files < args.n_folds:
        raise ValueError(f"Not enough files for GroupKFold: have {n_files}, need at least {args.n_folds}.")

    device = resolve_device(args.device)
    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    raw_temporal_oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_folds = np.full(len(meta_df), -1, dtype=np.int64)
    fold_results: List[FoldResult] = []

    gkf = GroupKFold(n_splits=args.n_folds)
    dummy_x = np.zeros((n_files, 1), dtype=np.float32)

    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    metadata_features, _ = build_metadata_features(
        meta_df=meta_df,
        include_site_onehot=False,
        include_hour_features=False,
    )
    context_tensor, context_feature_names = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    print("[INFO] Perch + temporal residual local CV")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {n_files}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] base_score_mode: {args.base_score_mode}")
    print(f"[INFO] base_inner_folds: {args.base_inner_folds}")
    print(f"[INFO] base_embedding_pca_dim: {args.base_embedding_pca_dim}")
    print(f"[INFO] base_logreg_c: {args.base_logreg_c}")
    print(f"[INFO] base_logreg_min_pos: {args.base_logreg_min_pos}")
    print(f"[INFO] feature_mode: {args.feature_mode}")
    print(f"[INFO] model_type: {args.model_type}")
    print(f"[INFO] input_dim: {x_rows.shape[1]}")
    print(f"[INFO] pca_dim: {args.pca_dim}")
    print(f"[INFO] d_model: {args.d_model}")
    print(f"[INFO] num_layers: {args.num_layers}")
    print(f"[INFO] temporal_blend: {args.temporal_blend}")
    print(f"[INFO] epochs: {args.epochs}")
    print(f"[INFO] inner_val_files: {args.inner_val_files}")
    print(f"[INFO] patience: {args.patience}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] context_features: {', '.join(context_feature_names)}")
    print(f"[INFO] device: {device}")

    base_oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(
        gkf.split(dummy_x, groups=seq_pack.filename_seq),
        start=1,
    ):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        train_files = set(seq_pack.filename_seq[train_idx].tolist())
        valid_files = set(seq_pack.filename_seq[valid_idx].tolist())
        overlap = train_files & valid_files
        if overlap:
            raise RuntimeError(
                f"Leakage guard failed: fold {fold} train/valid filename overlap: {sorted(overlap)[:5]}"
            )

        train_row_idx = seq_pack.row_index_seq[train_idx].reshape(-1)
        valid_row_idx = seq_pack.row_index_seq[valid_idx].reshape(-1)
        row_train_files = set(meta_df.iloc[train_row_idx]["filename"].tolist())
        row_valid_files = set(meta_df.iloc[valid_row_idx]["filename"].tolist())
        row_overlap = row_train_files & row_valid_files
        if row_overlap:
            raise RuntimeError(
                f"Leakage guard failed: fold {fold} row-level filename overlap: {sorted(row_overlap)[:5]}"
            )
        base_train_rows, base_valid_rows, base_actual_embedding_dim, base_fitted_classes = build_base_predictions_for_outer_fold(
            args=args,
            meta_df=meta_df,
            emb_full=emb_full,
            raw_scores_full=scores_full_raw,
            y_true=y_true,
            context_tensor=context_tensor,
            position_features=position_features,
            metadata_features=metadata_features,
            outer_train_idx=train_row_idx,
            outer_valid_idx=valid_row_idx,
            outer_fold=fold,
        )

        base_oof_pred[valid_row_idx] = base_valid_rows

        train_base_seq_pack = group_rows_into_sequences(
            meta_df=meta_df.iloc[train_row_idx].reset_index(drop=True),
            x_rows=x_rows[train_row_idx],
            base_scores_rows=base_train_rows,
            y_rows=y_true[train_row_idx],
            expected_seq_len=args.expected_seq_len,
            source_row_indices=train_row_idx,
        )
        valid_base_seq_pack = group_rows_into_sequences(
            meta_df=meta_df.iloc[valid_row_idx].reset_index(drop=True),
            x_rows=x_rows[valid_row_idx],
            base_scores_rows=base_valid_rows,
            y_rows=y_true[valid_row_idx],
            expected_seq_len=args.expected_seq_len,
            source_row_indices=valid_row_idx,
        )

        x_train_seq = train_base_seq_pack.features_seq
        x_valid_seq = valid_base_seq_pack.features_seq
        base_train_seq = train_base_seq_pack.base_scores_seq
        base_valid_seq = valid_base_seq_pack.base_scores_seq
        y_train_seq = train_base_seq_pack.y_seq
        y_valid_seq = valid_base_seq_pack.y_seq

        raw_valid_auc = macro_auc_skip_empty(
            y_valid_seq.reshape(-1, y_valid_seq.shape[-1]),
            scores_full_raw[valid_row_idx],
        )
        base_valid_auc = macro_auc_skip_empty(
            y_valid_seq.reshape(-1, y_valid_seq.shape[-1]),
            base_valid_seq.reshape(-1, base_valid_seq.shape[-1]),
        )

        print(
            f"[INFO] Fold {fold} | "
            f"train_files={len(train_idx)} valid_files={len(valid_idx)} "
            f"train_rows={len(train_idx) * args.expected_seq_len} valid_rows={len(valid_idx) * args.expected_seq_len}"
        )
        fold_pred_seq, raw_fold_pred_seq, best_epoch, actual_input_dim = train_one_fold(
            args=args,
            x_train_seq=x_train_seq,
            x_valid_seq=x_valid_seq,
            base_train_seq=base_train_seq,
            base_valid_seq=base_valid_seq,
            y_train_seq=y_train_seq,
            y_valid_seq=y_valid_seq,
            device=device,
        )

        temporal_valid_auc = macro_auc_skip_empty(
            y_valid_seq.reshape(-1, y_valid_seq.shape[-1]),
            fold_pred_seq.reshape(-1, fold_pred_seq.shape[-1]),
        )

        valid_row_indices = valid_base_seq_pack.row_index_seq.reshape(-1)
        flat_fold_pred = fold_pred_seq.reshape(-1, fold_pred_seq.shape[-1])
        if flat_fold_pred.shape[0] != len(valid_row_indices):
            raise AssertionError(
                f"Temporal fold prediction row count mismatch: pred_rows={flat_fold_pred.shape[0]} "
                f"vs valid_row_indices={len(valid_row_indices)}"
            )
        oof_pred[valid_row_indices] = flat_fold_pred
        raw_temporal_oof_pred[valid_row_indices] = raw_fold_pred_seq.reshape(-1, raw_fold_pred_seq.shape[-1])
        oof_folds[valid_row_indices] = fold

        fold_result = FoldResult(
            fold=fold,
            n_train_files=len(train_idx),
            n_valid_files=len(valid_idx),
            n_train_rows=len(train_idx) * args.expected_seq_len,
            n_valid_rows=len(valid_idx) * args.expected_seq_len,
            raw_valid_auc=raw_valid_auc,
            base_valid_auc=base_valid_auc,
            temporal_valid_auc=temporal_valid_auc,
            best_epoch=best_epoch,
            actual_input_dim=actual_input_dim,
            base_actual_embedding_dim=base_actual_embedding_dim,
            base_fitted_classes=base_fitted_classes,
        )
        fold_results.append(fold_result)

        print(
            f"[FOLD {fold}] raw_auc={raw_valid_auc:.6f} "
            f"base_auc={base_valid_auc:.6f} "
            f"temporal_auc={temporal_valid_auc:.6f} "
            f"best_epoch={best_epoch} "
            f"actual_input_dim={actual_input_dim} "
            f"base_embed_dim={base_actual_embedding_dim} "
            f"base_fitted_classes={base_fitted_classes}"
        )

    base_oof_auc = macro_auc_skip_empty(y_true, base_oof_pred)
    temporal_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([result.raw_valid_auc for result in fold_results]))
    mean_fold_base_auc = float(np.mean([result.base_valid_auc for result in fold_results]))
    mean_fold_temporal_auc = float(np.mean([result.temporal_valid_auc for result in fold_results]))

    print(f"[INFO] base_oof_auc: {base_oof_auc:.6f}")
    print(f"[INFO] temporal_oof_auc: {temporal_oof_auc:.6f}")
    print(f"[INFO] mean_fold_raw_auc: {mean_fold_raw_auc:.6f}")
    print(f"[INFO] mean_fold_base_auc: {mean_fold_base_auc:.6f}")
    print(f"[INFO] mean_fold_temporal_auc: {mean_fold_temporal_auc:.6f}")

    fold_metrics_df = pd.DataFrame([result.__dict__ for result in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    oof_meta_df = meta_df.copy()
    oof_meta_df["fold"] = oof_folds
    oof_meta_path = output_dir / "oof_meta.csv"
    oof_meta_df.to_csv(oof_meta_path, index=False)

    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=scores_full_raw.astype(np.float32, copy=False),
        base_oof_scores=base_oof_pred.astype(np.float32, copy=False),
        temporal_oof_scores=oof_pred.astype(np.float32, copy=False),
        temporal_raw_oof_scores=raw_temporal_oof_pred.astype(np.float32, copy=False),
    )

    summary = {
        "rows": int(len(meta_df)),
        "files": int(n_files),
        "classes": int(len(class_names)),
        "n_folds": int(args.n_folds),
        "expected_seq_len": int(args.expected_seq_len),
        "base_score_mode": args.base_score_mode,
        "base_inner_folds": int(args.base_inner_folds),
        "base_embedding_pca_dim": int(args.base_embedding_pca_dim),
        "base_logreg_c": float(args.base_logreg_c),
        "base_logreg_max_iter": int(args.base_logreg_max_iter),
        "base_logreg_min_pos": int(args.base_logreg_min_pos),
        "feature_mode": args.feature_mode,
        "model_type": args.model_type,
        "input_dim": int(x_rows.shape[1]),
        "pca_dim": int(args.pca_dim),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "proto_temperature": float(args.proto_temperature),
        "proto_weight_init": float(args.proto_weight_init),
        "temporal_blend": float(args.temporal_blend),
        "epochs": int(args.epochs),
        "inner_val_files": int(args.inner_val_files),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad_norm": float(args.clip_grad_norm),
        "device": str(device),
        "raw_perch_auc": float(raw_perch_auc),
        "base_oof_auc": float(base_oof_auc),
        "temporal_oof_auc": float(temporal_oof_auc),
        "mean_fold_raw_auc": mean_fold_raw_auc,
        "mean_fold_base_auc": mean_fold_base_auc,
        "mean_fold_temporal_auc": mean_fold_temporal_auc,
        "mean_best_epoch": float(np.mean([result.best_epoch for result in fold_results])),
        "mean_actual_input_dim": float(np.mean([result.actual_input_dim for result in fold_results])),
        "mean_base_actual_embedding_dim": float(np.mean([result.base_actual_embedding_dim for result in fold_results])),
        "mean_base_fitted_classes": float(np.mean([result.base_fitted_classes for result in fold_results])),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF metadata to: {oof_meta_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
