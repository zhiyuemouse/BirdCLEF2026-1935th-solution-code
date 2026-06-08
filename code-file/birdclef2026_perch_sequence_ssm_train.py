#!/usr/bin/env python3
"""Train fold-safe ProtoSSM-style temporal models on Perch sequences.

This adapts the public high-score idea in a leak-safe way:

    Perch 5s embedding/logits -> [file, 12, feature] -> bidirectional SSM

Perch itself still sees independent 5s windows.  Only this downstream model
reads the 60s context.  Loss is applied only to labeled windows; unlabeled
windows in partial soundscapes are context only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026_perch_context_train import (
    load_class_names,
    load_meta,
    macro_auc_skip_empty,
    seed_everything,
    sigmoid_np,
)
from birdclef2026_perch_temporal_head_train import (
    N_WINDOWS,
    FileTensorPack,
    build_file_folds,
    build_labels_and_mask,
    make_inner_split,
    parse_end_seconds,
)


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fold-safe Perch sequence ProtoSSM.")
    parser.add_argument("--sequence-cache-dir", type=str, default="perch_sequence_cache_labeled_all_full")
    parser.add_argument("--sequence-meta-path", type=str, default="")
    parser.add_argument("--sequence-arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--fold-assignment-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_sequence_ssm_labeled_all_cnn195634folds_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--n-ssm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--n-sites", type=int, default=20)
    parser.add_argument("--meta-dim", type=int, default=16)
    parser.add_argument("--use-cross-attn", action="store_true")
    parser.add_argument("--cross-attn-heads", type=int, default=4)
    parser.add_argument("--mlp-min-pos", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=12.0)
    parser.add_argument("--protoclr-weight", type=float, default=0.0)
    parser.add_argument("--protoclr-temperature", type=float, default=0.12)
    parser.add_argument("--protoclr-min-classes", type=int, default=2)
    parser.add_argument("--protoclr-min-pos-per-class", type=int, default=1)
    parser.add_argument("--teacher-oof-path", type=str, default="")
    parser.add_argument("--teacher-key", type=str, default="best")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--inner-val-files", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_sequence_cache(cache_dir_arg: str, meta_path_arg: str, arrays_path_arg: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir_arg)
    meta_candidates = [Path(meta_path_arg)] if meta_path_arg else [
        cache_dir / "perch_sequence_meta.parquet",
        cache_dir / "perch_sequence_meta.csv",
        cache_dir / "perch_meta.parquet",
        cache_dir / "full_perch_meta.parquet",
    ]
    arrays_candidates = [Path(arrays_path_arg)] if arrays_path_arg else [
        cache_dir / "perch_sequence_arrays.npz",
        cache_dir / "perch_arrays.npz",
        cache_dir / "full_perch_arrays.npz",
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find sequence meta under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find sequence arrays under {cache_dir}")
    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    for key in ["scores_full_raw", "emb_full"]:
        if key not in arrays:
            raise KeyError(f"{arrays_path} must contain {key!r}. Available keys: {arrays.files}")
    scores = arrays["scores_full_raw"].astype(np.float32, copy=False)
    emb = arrays["emb_full"].astype(np.float32, copy=False)
    return meta_df, scores, emb


def build_site_ids(filenames: np.ndarray, n_sites: int) -> Tuple[np.ndarray, Dict[str, int]]:
    sites = []
    for name in filenames.astype(str):
        parts = Path(name).stem.split("_")
        site = next((part for part in parts if part.startswith("S") and part[1:].isdigit()), "unknown")
        sites.append(site)
    unique_sites = sorted(set(sites))
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites[: max(1, n_sites - 1)])}
    unk = max(0, n_sites - 1)
    site_ids = np.asarray([site_to_idx.get(site, unk) for site in sites], dtype=np.int64)
    return site_ids, site_to_idx


@dataclass
class SequencePack:
    filenames: np.ndarray
    row_ids: np.ndarray
    emb: np.ndarray
    raw_logits: np.ndarray
    y: np.ndarray
    label_mask: np.ndarray
    teacher_probs: Optional[np.ndarray]
    site_ids: np.ndarray
    hours: np.ndarray


def build_sequence_pack(
    meta_df: pd.DataFrame,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
    y: np.ndarray,
    label_mask: np.ndarray,
    n_sites: int,
) -> Tuple[SequencePack, Dict[str, int]]:
    meta = meta_df.copy().reset_index(drop=True)
    meta["_end_sec"] = parse_end_seconds(meta["row_id"].astype(str).tolist())
    meta["_pos"] = np.arange(len(meta), dtype=np.int64)
    meta = meta.sort_values(["filename", "_end_sec"]).reset_index(drop=True)

    filenames: List[str] = []
    row_ids: List[np.ndarray] = []
    emb_files: List[np.ndarray] = []
    raw_files: List[np.ndarray] = []
    y_files: List[np.ndarray] = []
    mask_files: List[np.ndarray] = []
    hour_files: List[int] = []
    expected = np.arange(5, 65, 5, dtype=np.int16)

    for filename, group in meta.groupby("filename", sort=False):
        if len(group) != N_WINDOWS:
            raise ValueError(f"Expected {N_WINDOWS} windows for {filename}, got {len(group)}")
        got = group["_end_sec"].to_numpy(dtype=np.int16)
        if not np.array_equal(got, expected):
            raise ValueError(f"Unexpected end seconds for {filename}: {got.tolist()}")
        idx = group["_pos"].to_numpy(dtype=np.int64)
        filenames.append(str(filename))
        row_ids.append(group["row_id"].astype(str).to_numpy())
        emb_files.append(emb_full[idx])
        raw_files.append(scores_full_raw[idx])
        y_files.append(y[idx])
        mask_files.append(label_mask[idx])
        hour_files.append(int(group["hour_utc"].iloc[0]) if "hour_utc" in group else 0)

    filename_arr = np.asarray(filenames, dtype=object)
    site_ids, site_to_idx = build_site_ids(filename_arr, n_sites=n_sites)
    pack = SequencePack(
        filenames=filename_arr,
        row_ids=np.stack(row_ids, axis=0),
        emb=np.stack(emb_files, axis=0).astype(np.float32, copy=False),
        raw_logits=np.stack(raw_files, axis=0).astype(np.float32, copy=False),
        y=np.stack(y_files, axis=0).astype(np.uint8, copy=False),
        label_mask=np.stack(mask_files, axis=0).astype(bool, copy=False),
        teacher_probs=None,
        site_ids=site_ids,
        hours=np.asarray(hour_files, dtype=np.int64).clip(0, 23),
    )
    return pack, site_to_idx


def load_teacher_probs_for_pack(
    teacher_path: Path,
    teacher_key: str,
    class_names: Sequence[str],
    pack: SequencePack,
) -> np.ndarray:
    """Load row-level OOF teacher probabilities and align to [file, window, class]."""

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher OOF path not found: {teacher_path}")
    flat_row_ids = pack.row_ids.reshape(-1).astype(str)
    flat_mask = pack.label_mask.reshape(-1)
    teacher_flat = np.zeros((len(flat_row_ids), len(class_names)), dtype=np.float32)

    if teacher_path.suffix.lower() == ".npz":
        npz = np.load(teacher_path, allow_pickle=True)
        if "row_id" not in npz.files:
            raise KeyError(f"{teacher_path} must contain 'row_id'. Available: {npz.files}")
        if teacher_key not in npz.files:
            raise KeyError(f"{teacher_path} missing teacher key {teacher_key!r}. Available: {npz.files}")
        source_row_ids = npz["row_id"].astype(str)
        source_pred = npz[teacher_key].astype(np.float32, copy=False)
        if source_pred.shape[1] != len(class_names):
            raise ValueError(
                f"Teacher key {teacher_key!r} has {source_pred.shape[1]} classes, expected {len(class_names)}"
            )
        if "y_true" in npz.files:
            source_y = npz["y_true"].astype(np.float32, copy=False)
            if source_y.shape == source_pred.shape:
                expected_y = pd.DataFrame({"row_id": flat_row_ids[flat_mask]}).merge(
                    pd.DataFrame({"row_id": source_row_ids}).assign(_pos=np.arange(len(source_row_ids))),
                    on="row_id",
                    how="left",
                    validate="one_to_one",
                )
                if expected_y["_pos"].isna().any():
                    missing = expected_y.loc[expected_y["_pos"].isna(), "row_id"].head(5).tolist()
                    raise ValueError(f"Teacher y_true alignment missing rows: {missing}")
                aligned_y = source_y[expected_y["_pos"].to_numpy(dtype=np.int64)]
                pack_y = pack.y.reshape(-1, len(class_names))[flat_mask].astype(np.float32, copy=False)
                if not np.array_equal(aligned_y, pack_y):
                    raise ValueError("Teacher y_true does not match aligned hard labels.")
    else:
        teacher_df = pd.read_csv(teacher_path)
        if "row_id" not in teacher_df.columns:
            raise KeyError(f"{teacher_path} must contain row_id")
        missing_cols = [name for name in class_names if name not in teacher_df.columns]
        if missing_cols:
            raise KeyError(f"{teacher_path} missing class columns, examples: {missing_cols[:5]}")
        source_row_ids = teacher_df["row_id"].astype(str).to_numpy()
        source_pred = teacher_df[list(class_names)].to_numpy(dtype=np.float32)

    source = pd.DataFrame({"row_id": source_row_ids, "_pos": np.arange(len(source_row_ids), dtype=np.int64)})
    aligned = pd.DataFrame({"row_id": flat_row_ids[flat_mask]}).merge(
        source,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["_pos"].isna().any():
        missing = aligned.loc[aligned["_pos"].isna(), "row_id"].head(5).tolist()
        raise ValueError(f"Teacher predictions missing aligned rows. Examples: {missing}")
    teacher_flat[flat_mask] = np.clip(source_pred[aligned["_pos"].to_numpy(dtype=np.int64)], 0.0, 1.0)
    return teacher_flat.reshape(pack.row_ids.shape[0], pack.row_ids.shape[1], len(class_names)).astype(np.float32)


class FeatureStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)


def fit_standardizer(x_train: np.ndarray) -> FeatureStandardizer:
    flat = x_train.reshape(-1, x_train.shape[-1]).astype(np.float32, copy=False)
    mean = flat.mean(axis=0, keepdims=True).reshape(1, 1, -1)
    std = flat.std(axis=0, keepdims=True).reshape(1, 1, -1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureStandardizer(mean=mean, std=std)


def standardizer_to_artifact(standardizer: FeatureStandardizer) -> Dict[str, np.ndarray]:
    return {"mean": standardizer.mean, "std": standardizer.std}


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.conv1d = nn.Conv1d(self.d_model, self.d_model, int(d_conv), padding=int(d_conv) - 1, groups=self.d_model)
        self.dt_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_model, -1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_model))
        self.B_proj = nn.Linear(self.d_model, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_model, self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        xz = self.in_proj(x)
        x_ssm, _ = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :steps].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        a = -torch.exp(self.A_log)
        b = self.B_proj(x_conv)
        c = self.C_proj(x_conv)
        h = torch.zeros(bsz, dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(steps):
            dt_t = dt[:, t, :]
            d_a = torch.exp(a[None, :, :] * dt_t[:, :, None])
            d_b = dt_t[:, :, None] * b[:, t, None, :]
            h = h * d_a + x[:, t, :, None] * d_b
            ys.append((h * c[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return self.out_proj(y + x * self.D[None, None, :])


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(int(d_model), int(n_heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.Dropout(float(dropout)),
        )
        self.norm2 = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        return residual + self.ffn(x)


class ProtoSSMHead(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_state: int,
        n_ssm_layers: int,
        n_classes: int,
        n_windows: int,
        dropout: float,
        n_sites: int,
        meta_dim: int,
        use_cross_attn: bool,
        cross_attn_heads: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(d_input), int(d_model)),
            nn.LayerNorm(int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, int(n_windows), int(d_model)) * 0.02)
        self.site_emb = nn.Embedding(int(n_sites), int(meta_dim))
        self.hour_emb = nn.Embedding(24, int(meta_dim))
        self.meta_proj = nn.Linear(2 * int(meta_dim), int(d_model))
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(int(n_ssm_layers)):
            self.ssm_fwd.append(SelectiveSSM(int(d_model), int(d_state)))
            self.ssm_bwd.append(SelectiveSSM(int(d_model), int(d_state)))
            self.ssm_merge.append(nn.Linear(2 * int(d_model), int(d_model)))
            self.ssm_norm.append(nn.LayerNorm(int(d_model)))
        self.ssm_drop = nn.Dropout(float(dropout))
        self.use_cross_attn = bool(use_cross_attn)
        if self.use_cross_attn:
            self.cross_attn = TemporalCrossAttention(int(d_model), int(cross_attn_heads), float(dropout))
        self.prototypes = nn.Parameter(torch.randn(int(n_classes), int(d_model)) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(int(n_classes)))
        self.fusion_alpha = nn.Parameter(torch.zeros(int(n_classes)))

    def encode(
        self,
        emb: torch.Tensor,
        site_ids: torch.Tensor,
        hours: torch.Tensor,
    ) -> torch.Tensor:
        steps = emb.shape[1]
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :steps, :]
        meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
        h = h + meta[:, None, :]
        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)
        if self.use_cross_attn:
            h = self.cross_attn(h)
        return h

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor,
        site_ids: torch.Tensor,
        hours: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(emb, site_ids, hours)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(h_norm, p_norm.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]
        alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
        logits = alpha * sim + (1.0 - alpha) * perch_logits
        if return_features:
            return logits, h
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


def masked_teacher_bce_with_logits(
    logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    teacher_mask: torch.Tensor,
    fitted_mask: torch.Tensor,
) -> torch.Tensor:
    valid_logits = logits[teacher_mask][:, fitted_mask]
    valid_targets = teacher_targets[teacher_mask][:, fitted_mask]
    if valid_logits.numel() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        reduction="mean",
    )


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

    z = F.normalize(embeddings, dim=1)
    kept_targets = fitted_targets[:, class_keep_local]
    prototypes: List[torch.Tensor] = []
    proto_target_columns: List[int] = []
    for target_col in range(kept_targets.shape[1]):
        pos_mask = kept_targets[:, target_col] > 0.5
        if not bool(pos_mask.any()):
            continue
        prototypes.append(z[pos_mask].mean(dim=0))
        proto_target_columns.append(int(target_col))
    if len(prototypes) < int(min_classes):
        return embeddings.sum() * 0.0

    proto_t = F.normalize(torch.stack(prototypes, dim=0), dim=1)
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
    return F.cross_entropy(logits, sample_y)


def build_loader(
    emb: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    teacher_targets: Optional[np.ndarray],
    site_ids: np.ndarray,
    hours: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    if teacher_targets is None:
        teacher_targets = np.zeros_like(targets, dtype=np.float32)
        teacher_mask = np.zeros(label_mask.shape, dtype=bool)
    else:
        teacher_targets = teacher_targets.astype(np.float32, copy=False)
        teacher_mask = label_mask.astype(bool, copy=False)
    dataset = TensorDataset(
        torch.from_numpy(emb.astype(np.float32, copy=False)),
        torch.from_numpy(logits.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
        torch.from_numpy(label_mask.astype(bool, copy=False)),
        torch.from_numpy(teacher_targets.astype(np.float32, copy=False)),
        torch.from_numpy(teacher_mask.astype(bool, copy=False)),
        torch.from_numpy(site_ids.astype(np.int64, copy=False)),
        torch.from_numpy(hours.astype(np.int64, copy=False)),
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


def predict_model(
    model: nn.Module,
    emb: np.ndarray,
    logits: np.ndarray,
    site_ids: np.ndarray,
    hours: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(emb), int(batch_size)):
            emb_t = torch.from_numpy(emb[start:start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            logits_t = torch.from_numpy(logits[start:start + int(batch_size)].astype(np.float32, copy=False)).to(device)
            site_t = torch.from_numpy(site_ids[start:start + int(batch_size)].astype(np.int64, copy=False)).to(device)
            hour_t = torch.from_numpy(hours[start:start + int(batch_size)].astype(np.int64, copy=False)).to(device)
            pred = torch.sigmoid(model(emb_t, logits_t, site_t, hour_t)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def train_fold_model(
    emb_train_outer: np.ndarray,
    logits_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    mask_train_outer: np.ndarray,
    teacher_train_outer: Optional[np.ndarray],
    site_train_outer: np.ndarray,
    hour_train_outer: np.ndarray,
    fitted_class_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    all_idx = np.arange(len(emb_train_outer), dtype=np.int64)
    inner_train_idx, inner_val_idx = make_inner_split(
        train_files_idx=all_idx,
        pack=FileTensorPack(
            filenames=np.asarray([str(i) for i in all_idx], dtype=object),
            row_ids=np.empty((len(all_idx), N_WINDOWS), dtype=object),
            tokens=emb_train_outer,
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

    model = ProtoSSMHead(
        d_input=emb_train_outer.shape[-1],
        d_model=args.d_model,
        d_state=args.d_state,
        n_ssm_layers=args.n_ssm_layers,
        n_classes=y_train_outer.shape[-1],
        n_windows=N_WINDOWS,
        dropout=args.dropout,
        n_sites=args.n_sites,
        meta_dim=args.meta_dim,
        use_cross_attn=args.use_cross_attn,
        cross_attn_heads=args.cross_attn_heads,
    ).to(device)
    train_loader = build_loader(
        emb=emb_train_outer[inner_train_idx],
        logits=logits_train_outer[inner_train_idx],
        targets=y_train_outer[inner_train_idx],
        label_mask=mask_train_outer[inner_train_idx],
        teacher_targets=None if teacher_train_outer is None else teacher_train_outer[inner_train_idx],
        site_ids=site_train_outer[inner_train_idx],
        hours=hour_train_outer[inner_train_idx],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        shuffle=True,
    )
    if len(inner_val_idx) > 0:
        val = (
            torch.from_numpy(emb_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device),
            torch.from_numpy(logits_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device),
            torch.from_numpy(y_train_outer[inner_val_idx].astype(np.float32, copy=False)).to(device),
            torch.from_numpy(mask_train_outer[inner_val_idx].astype(bool, copy=False)).to(device),
            torch.from_numpy(site_train_outer[inner_val_idx].astype(np.int64, copy=False)).to(device),
            torch.from_numpy(hour_train_outer[inner_val_idx].astype(np.int64, copy=False)).to(device),
        )
    else:
        val = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for emb_b, logits_b, y_b, mask_b, teacher_b, teacher_mask_b, site_b, hour_b in train_loader:
            emb_b = emb_b.to(device)
            logits_b = logits_b.to(device)
            y_b = y_b.to(device)
            mask_b = mask_b.to(device)
            teacher_b = teacher_b.to(device)
            teacher_mask_b = teacher_mask_b.to(device)
            site_b = site_b.to(device)
            hour_b = hour_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            out, features = model(emb_b, logits_b, site_b, hour_b, return_features=True)
            loss = masked_bce_with_logits(out, y_b, mask_b, fitted_mask, pos_weight_t)
            if float(args.teacher_loss_weight) > 0.0 and bool(teacher_mask_b.any().item()):
                loss = loss + float(args.teacher_loss_weight) * masked_teacher_bce_with_logits(
                    out,
                    teacher_targets=teacher_b,
                    teacher_mask=teacher_mask_b,
                    fitted_mask=fitted_mask,
                )
            if float(args.protoclr_weight) > 0.0:
                valid_features = features[mask_b]
                valid_targets = y_b[mask_b]
                if valid_features.numel() > 0:
                    loss = loss + float(args.protoclr_weight) * multilabel_protoclr_loss(
                        embeddings=valid_features,
                        targets=valid_targets,
                        fitted_mask=fitted_mask,
                        temperature=args.protoclr_temperature,
                        min_classes=args.protoclr_min_classes,
                        min_pos_per_class=args.protoclr_min_pos_per_class,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if val is not None:
            model.eval()
            with torch.inference_mode():
                out = model(val[0], val[1], val[4], val[5])
                monitor_loss = float(masked_bce_with_logits(out, val[2], val[3], fitted_mask, pos_weight_t).detach().cpu().item())
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
    model.load_state_dict(best_state)
    artifact = {
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "d_input": int(emb_train_outer.shape[-1]),
        "d_model": int(args.d_model),
        "d_state": int(args.d_state),
        "n_ssm_layers": int(args.n_ssm_layers),
        "output_dim": int(y_train_outer.shape[-1]),
        "dropout": float(args.dropout),
        "n_sites": int(args.n_sites),
        "meta_dim": int(args.meta_dim),
        "use_cross_attn": bool(args.use_cross_attn),
        "cross_attn_heads": int(args.cross_attn_heads),
        "protoclr_weight": float(args.protoclr_weight),
        "protoclr_temperature": float(args.protoclr_temperature),
        "protoclr_min_classes": int(args.protoclr_min_classes),
        "protoclr_min_pos_per_class": int(args.protoclr_min_pos_per_class),
        "teacher_loss_weight": float(args.teacher_loss_weight),
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
    ssm_valid_auc: float
    n_train_files: int
    n_valid_files: int
    n_train_labeled_windows: int
    n_valid_labeled_windows: int
    fitted_classes: int
    best_epoch: int
    best_loss: float


def main() -> None:
    args = parse_args()
    if args.d_model % args.cross_attn_heads != 0:
        raise ValueError("--d-model must be divisible by --cross-attn-heads")
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    class_names = load_class_names(Path(args.sample_submission_path))
    meta_df, raw_logits, emb_full = load_sequence_cache(
        cache_dir_arg=args.sequence_cache_dir,
        meta_path_arg=args.sequence_meta_path,
        arrays_path_arg=args.sequence_arrays_path,
    )
    y, label_mask = build_labels_and_mask(
        labels_path=Path(args.labels_path),
        class_names=class_names,
        row_ids=meta_df["row_id"].astype(str).tolist(),
    )
    pack, site_to_idx = build_sequence_pack(
        meta_df=meta_df,
        scores_full_raw=raw_logits,
        emb_full=emb_full,
        y=y,
        label_mask=label_mask,
        n_sites=args.n_sites,
    )
    teacher_probs = None
    if args.teacher_oof_path and float(args.teacher_loss_weight) > 0.0:
        teacher_probs = load_teacher_probs_for_pack(
            teacher_path=Path(args.teacher_oof_path),
            teacher_key=args.teacher_key,
            class_names=class_names,
            pack=pack,
        )
        pack.teacher_probs = teacher_probs
    if args.limit_files > 0:
        keep = np.arange(min(args.limit_files, len(pack.filenames)), dtype=np.int64)
        pack = SequencePack(
            filenames=pack.filenames[keep],
            row_ids=pack.row_ids[keep],
            emb=pack.emb[keep],
            raw_logits=pack.raw_logits[keep],
            y=pack.y[keep],
            label_mask=pack.label_mask[keep],
            teacher_probs=None if pack.teacher_probs is None else pack.teacher_probs[keep],
            site_ids=pack.site_ids[keep],
            hours=pack.hours[keep],
        )
    file_folds = build_file_folds(
        FileTensorPack(
            filenames=pack.filenames,
            row_ids=pack.row_ids,
            tokens=pack.emb,
            y=pack.y,
            label_mask=pack.label_mask,
        ),
        fold_assignment_path=args.fold_assignment_path,
        n_folds=args.n_folds,
    )
    fold_values = sorted(pd.Index(file_folds).unique().tolist())
    raw_sigmoid_files = sigmoid_np(pack.raw_logits).astype(np.float32)
    labeled_flat_mask = pack.label_mask.reshape(-1)
    y_labeled = pack.y.reshape(-1, len(class_names))[labeled_flat_mask]
    raw_labeled = raw_sigmoid_files.reshape(-1, len(class_names))[labeled_flat_mask]
    row_id_labeled = pack.row_ids.reshape(-1)[labeled_flat_mask]
    filename_labeled = np.repeat(pack.filenames, N_WINDOWS)[labeled_flat_mask]
    raw_sigmoid_auc = macro_auc_skip_empty(y_labeled, raw_labeled)
    oof_pred_labeled = raw_labeled.copy()

    print("[INFO] Train Perch sequence ProtoSSM")
    print(f"[INFO] files: {len(pack.filenames)}")
    print(f"[INFO] windows: {pack.emb.shape[0] * pack.emb.shape[1]}")
    print(f"[INFO] labeled_windows: {int(pack.label_mask.sum())}")
    print(f"[INFO] emb: {pack.emb.shape} raw_logits: {pack.raw_logits.shape}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] folds: {fold_values}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] fold_assignment_path: {args.fold_assignment_path}")
    print(
        "[INFO] model: "
        f"d_model={args.d_model} d_state={args.d_state} layers={args.n_ssm_layers} "
        f"dropout={args.dropout} cross_attn={args.use_cross_attn}"
    )
    print(
        "[INFO] ProtoCLR: "
        f"weight={args.protoclr_weight} tau={args.protoclr_temperature} "
        f"min_classes={args.protoclr_min_classes} min_pos_per_class={args.protoclr_min_pos_per_class}"
    )
    print(
        "[INFO] Teacher distill: "
        f"path={args.teacher_oof_path or '<disabled>'} key={args.teacher_key} "
        f"weight={args.teacher_loss_weight}"
    )

    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    for display_fold in fold_values:
        valid_idx = np.where(file_folds == int(display_fold))[0].astype(np.int64)
        train_idx = np.where(file_folds != int(display_fold))[0].astype(np.int64)
        standardizer = fit_standardizer(pack.emb[train_idx])
        emb_train = standardizer.transform(pack.emb[train_idx])
        emb_valid = standardizer.transform(pack.emb[valid_idx])
        y_train = pack.y[train_idx]
        mask_train = pack.label_mask[train_idx]
        train_labeled = y_train[mask_train]
        real_pos = train_labeled.sum(axis=0).astype(np.float32)
        real_neg = len(train_labeled) - real_pos
        fitted_class_indices = np.where((real_pos >= args.mlp_min_pos) & (real_neg > 0))[0].astype(np.int32)
        model_artifact, train_stats = train_fold_model(
            emb_train_outer=emb_train,
            logits_train_outer=pack.raw_logits[train_idx],
            y_train_outer=y_train,
            mask_train_outer=mask_train,
            teacher_train_outer=None if pack.teacher_probs is None else pack.teacher_probs[train_idx],
            site_train_outer=pack.site_ids[train_idx],
            hour_train_outer=pack.hours[train_idx],
            fitted_class_indices=fitted_class_indices,
            args=args,
            seed=args.seed + int(display_fold) * 1000,
            device=device,
        )
        model = ProtoSSMHead(
            d_input=model_artifact["d_input"],
            d_model=model_artifact["d_model"],
            d_state=model_artifact["d_state"],
            n_ssm_layers=model_artifact["n_ssm_layers"],
            n_classes=model_artifact["output_dim"],
            n_windows=N_WINDOWS,
            dropout=model_artifact["dropout"],
            n_sites=model_artifact["n_sites"],
            meta_dim=model_artifact["meta_dim"],
            use_cross_attn=model_artifact["use_cross_attn"],
            cross_attn_heads=model_artifact["cross_attn_heads"],
        ).to(device)
        model.load_state_dict(model_artifact["model_state"])
        pred_valid_files = predict_model(
            model=model,
            emb=emb_valid,
            logits=pack.raw_logits[valid_idx],
            site_ids=pack.site_ids[valid_idx],
            hours=pack.hours[valid_idx],
            device=device,
            batch_size=args.batch_size,
        )
        valid_label_mask = pack.label_mask[valid_idx]
        valid_y = pack.y[valid_idx][valid_label_mask]
        valid_raw = raw_sigmoid_files[valid_idx][valid_label_mask]
        valid_pred = valid_raw.copy()
        valid_pred[:, fitted_class_indices] = pred_valid_files[valid_label_mask][:, fitted_class_indices]
        valid_pred = np.clip(valid_pred, 0.0, 1.0).astype(np.float32, copy=False)
        valid_flat_global = np.isin(row_id_labeled, pack.row_ids[valid_idx].reshape(-1))
        oof_pred_labeled[valid_flat_global] = valid_pred
        raw_valid_auc = macro_auc_skip_empty(valid_y, valid_raw)
        ssm_valid_auc = macro_auc_skip_empty(valid_y, valid_pred)
        result = FoldResult(
            fold=int(display_fold),
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            ssm_valid_auc=float(ssm_valid_auc),
            n_train_files=int(len(train_idx)),
            n_valid_files=int(len(valid_idx)),
            n_train_labeled_windows=int(mask_train.sum()),
            n_valid_labeled_windows=int(valid_label_mask.sum()),
            fitted_classes=int(len(fitted_class_indices)),
            best_epoch=int(train_stats["best_epoch"]),
            best_loss=float(train_stats["best_loss"]),
        )
        fold_results.append(result)
        fold_artifacts.append(
            {
                "fold_name": f"fold_{display_fold}",
                "embedding_standardizer": standardizer_to_artifact(standardizer),
                "model": model_artifact,
            }
        )
        print(
            f"[FOLD {display_fold}] raw_sigmoid_auc={raw_valid_auc:.6f} "
            f"ssm_auc={ssm_valid_auc:.6f} fitted_classes={result.fitted_classes} "
            f"best_epoch={result.best_epoch} train_files={result.n_train_files} valid_files={result.n_valid_files}",
            flush=True,
        )

    ssm_oof_auc = macro_auc_skip_empty(y_labeled, oof_pred_labeled)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_ssm_auc = float(np.mean([item.ssm_valid_auc for item in fold_results]))
    artifact = {
        "artifact_version": 1,
        "model_type": "perch_sequence_ssm",
        "class_names": class_names,
        "site_to_idx": site_to_idx,
        "config": {
            "n_folds": int(args.n_folds),
            "fold_assignment_path": str(args.fold_assignment_path),
            "sequence_cache_dir": str(args.sequence_cache_dir),
            "d_model": int(args.d_model),
            "d_state": int(args.d_state),
            "n_ssm_layers": int(args.n_ssm_layers),
            "dropout": float(args.dropout),
            "n_sites": int(args.n_sites),
            "meta_dim": int(args.meta_dim),
            "use_cross_attn": bool(args.use_cross_attn),
            "cross_attn_heads": int(args.cross_attn_heads),
            "mlp_min_pos": int(args.mlp_min_pos),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight_power": float(args.pos_weight_power),
            "pos_weight_max": float(args.pos_weight_max),
            "protoclr_weight": float(args.protoclr_weight),
            "protoclr_temperature": float(args.protoclr_temperature),
            "protoclr_min_classes": int(args.protoclr_min_classes),
            "protoclr_min_pos_per_class": int(args.protoclr_min_pos_per_class),
            "teacher_oof_path": str(args.teacher_oof_path),
            "teacher_key": str(args.teacher_key),
            "teacher_loss_weight": float(args.teacher_loss_weight),
            "inner_val_files": int(args.inner_val_files),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_sequence_ssm_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    pd.DataFrame([item.__dict__ for item in fold_results]).to_csv(output_dir / "fold_metrics.csv", index=False)
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
        "windows": int(pack.emb.shape[0] * pack.emb.shape[1]),
        "labeled_windows": int(pack.label_mask.sum()),
        "classes": int(len(class_names)),
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "ssm_oof_auc": float(ssm_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_ssm_auc": float(mean_fold_ssm_auc),
        "fold_gap": float(mean_fold_ssm_auc - ssm_oof_auc),
        "sequence_cache_dir": str(args.sequence_cache_dir),
        "d_model": int(args.d_model),
        "d_state": int(args.d_state),
        "n_ssm_layers": int(args.n_ssm_layers),
        "dropout": float(args.dropout),
        "use_cross_attn": bool(args.use_cross_attn),
        "mlp_min_pos": int(args.mlp_min_pos),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "protoclr_weight": float(args.protoclr_weight),
        "protoclr_temperature": float(args.protoclr_temperature),
        "protoclr_min_classes": int(args.protoclr_min_classes),
        "protoclr_min_pos_per_class": int(args.protoclr_min_pos_per_class),
        "teacher_oof_path": str(args.teacher_oof_path),
        "teacher_key": str(args.teacher_key),
        "teacher_loss_weight": float(args.teacher_loss_weight),
        "artifact_path": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] ssm_oof_auc: {ssm_oof_auc:.6f}")
    print(f"[INFO] artifact: {artifact_path}")


if __name__ == "__main__":
    main()
