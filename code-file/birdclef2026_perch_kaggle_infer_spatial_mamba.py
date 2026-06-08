#!/usr/bin/env python3
"""Kaggle inference for Perch spatial-token Mamba artifacts.

This script is the deployment companion for
``birdclef2026_perch_spatial_mamba_train.py``.  It does not need a precomputed
spatial cache on Kaggle:

1. Read each 60s soundscape and split it into 12 x 5s windows.
2. Run frozen Perch v2 ONNX.
3. Use ``spatial_embedding [B,16,4,1536]`` -> mean over axis 2 -> ``[B,16,1536]``.
4. Apply each fold's train-only token projector.
5. Run the fold's Mamba-style head.
6. Average fold predictions and save ``submission.csv``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def resolve_user_path(path_str: str, competition_root: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    competition_candidate = competition_root / candidate
    if competition_candidate.exists():
        return competition_candidate
    return cwd_candidate


def discover_existing_path(explicit_path: str, candidates: Sequence[str], competition_root: Path) -> Path | None:
    if explicit_path:
        path = resolve_user_path(explicit_path, competition_root=competition_root)
        if not path.exists():
            raise FileNotFoundError(f"Explicit runtime model path does not exist: {path}")
        return path
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def discover_model_path(model_path_arg: str) -> Path:
    if model_path_arg:
        model_path = Path(model_path_arg)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Explicit model artifact does not exist: {model_path}")
        return model_path

    search_roots = [Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working")]
    candidates: List[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("perch_spatial_mamba_artifacts.joblib"))
    if not candidates:
        raise FileNotFoundError("No perch_spatial_mamba_artifacts.joblib found. Pass --model-path explicitly.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"[INFO] Auto-discovered model artifact: {candidates[0]}")
    return candidates[0]


def list_soundscape_files(soundscapes_dir: Path, debug: bool, debug_limit: int) -> List[Path]:
    files = sorted(path for path in soundscapes_dir.iterdir() if path.suffix == ".ogg")
    if debug:
        files = files[:debug_limit]
    return files


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def load_perch_label_table(perch_dir: Path, onnx_path: Path) -> pd.DataFrame:
    candidates = [
        perch_dir / "assets" / "labels.csv",
        perch_dir / "labels.csv",
        onnx_path.parent / "labels.csv",
    ]
    labels_path = next((path for path in candidates if path.exists()), None)
    if labels_path is None:
        raise FileNotFoundError(
            "Could not find Perch labels.csv. Checked:\n" + "\n".join(str(path) for path in candidates)
        )
    bc_labels = (
        pd.read_csv(labels_path)
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    if "scientific_name" not in bc_labels.columns:
        raise KeyError(f"Perch labels.csv must contain `inat2024_fsd50k`: {labels_path}")
    return bc_labels


def build_competition_mapping(
    primary_labels: List[str],
    taxonomy_path: Path,
    bc_labels: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    taxonomy = pd.read_csv(taxonomy_path).copy()
    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]
    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = taxonomy.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    no_label_index = len(bc_labels)
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)

    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    bc_indices = np.array([int(label_to_bc_index.get(label, no_label_index)) for label in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label_index
    mapped_bc_indices = bc_indices[mapped_mask].astype(np.int32)
    return bc_indices, mapped_bc_indices, mapping


def build_selected_proxy_targets(
    primary_labels: List[str],
    mapping: pd.DataFrame,
    bc_labels: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    class_name_map = mapping.set_index("primary_label")["class_name"].to_dict()
    label_to_idx = {label: i for i, label in enumerate(primary_labels)}
    no_label_index = len(bc_labels)
    unmapped_df = mapping[mapping["bc_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    proxy_pos_to_bc: Dict[int, np.ndarray] = {}
    scientific_name_series = bc_labels["scientific_name"].astype(str)
    for _, row in unmapped_non_sonotype.iterrows():
        target = str(row["primary_label"])
        if target not in label_to_idx:
            continue
        if class_name_map.get(target) not in PROXY_TAXA:
            continue
        genus = str(row["scientific_name"]).split()[0]
        hits = bc_labels[scientific_name_series.str.match(rf"^{re.escape(genus)}\s", na=False)].copy()
        if len(hits) == 0:
            continue
        proxy_pos_to_bc[label_to_idx[target]] = hits["bc_index"].to_numpy(dtype=np.int32)
    return proxy_pos_to_bc


def parse_soundscape_filename(name: str) -> Dict[str, object]:
    stem = Path(name).stem
    match = re.match(r"^.+?_(?:Train|Test)_.+?_(S\d+)_(\d{8})_(\d{6})$", stem)
    if not match:
        return {"site": "unknown", "hour_utc": -1, "month": -1}
    site, yyyymmdd, hhmmss = match.groups()
    return {"site": site, "hour_utc": int(hhmmss[:2]), "month": int(yyyymmdd[4:6])}


def read_soundscape_60s(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Perch spatial Mamba Kaggle inference.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--taxonomy-path", type=str, default="")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--perch-onnx-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--features-npz", type=str, default="", help="Optional debug-only cached features NPZ.")
    parser.add_argument("--features-meta-path", type=str, default="", help="Optional debug-only cached feature meta.")
    parser.add_argument("--batch-files", type=int, default=32)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--proxy-reduce", type=str, choices=["max", "mean"], default="max")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


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
        if self.head_variant == "perch_fusion":
            if self.raw_dim <= int(token_dim):
                raise ValueError("head_variant=perch_fusion requires raw_features=[embedding, selected_label_logits]")
            if self.freq_pool != "flat64":
                raise ValueError("head_variant=perch_fusion requires freq_pool=flat64")
            if self.use_pos_embed:
                raise ValueError("head_variant=perch_fusion does not use positional embeddings")
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
            tokens = self.freq_proj(torch.cat([tokens[:, 0, :, :], tokens[:, 1, :, :]], dim=-1))
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
            token_sim = sim.max(dim=2).values.clamp_min(0.0) * float(self.prototype_temperature)
            weighted = token_sim * nn.functional.softplus(self.class_proto_weight)[None, :, :]
            return weighted.sum(dim=-1) + self.class_bias[None, :]
        x = self.blocks(tokens)
        pooled = x.mean(dim=1)
        if self.raw_dim > 0:
            if raw_features is None:
                raise ValueError("raw_features are required when raw_dim > 0")
            pooled = torch.cat([pooled, raw_features], dim=1)
        return self.head(pooled)


def transform_one_token_projector(tokens: np.ndarray, projector: Dict[str, object]) -> np.ndarray:
    n_rows, n_tokens, n_dim = tokens.shape
    flat = tokens.reshape(-1, n_dim).astype(np.float32, copy=False)
    mean = np.asarray(projector["token_mean"], dtype=np.float32)
    std = np.asarray(projector["token_std"], dtype=np.float32)
    scaled = ((flat - mean) / std).astype(np.float32, copy=False)
    pca = projector.get("pca")
    if pca is not None:
        transformed = pca.transform(scaled).astype(np.float32)
    else:
        transformed = scaled
    output_dim = int(projector["output_dim"])
    return transformed.reshape(n_rows, n_tokens, output_dim).astype(np.float32, copy=False)


def transform_token_projector(tokens: np.ndarray, projector: Dict[str, object] | Sequence[Dict[str, object]]) -> np.ndarray:
    if isinstance(projector, (list, tuple)):
        if tokens.ndim != 4:
            raise ValueError(f"List token projector expects [rows,parts,tokens,dim], got {tokens.shape}")
        parts = [
            transform_one_token_projector(tokens[:, part_idx, :, :], part_projector)
            for part_idx, part_projector in enumerate(projector)
        ]
        return np.stack(parts, axis=1).astype(np.float32, copy=False)
    if tokens.ndim != 3:
        raise ValueError(f"Dict token projector expects [rows,tokens,dim], got {tokens.shape}")
    return transform_one_token_projector(tokens, projector)


def transform_raw_projector(raw_scores: np.ndarray, projector: Dict[str, object] | None) -> np.ndarray | None:
    if projector is None:
        return None
    mean = np.asarray(projector["mean"], dtype=np.float32)
    std = np.asarray(projector["std"], dtype=np.float32)
    scaled = ((raw_scores - mean) / std).astype(np.float32, copy=False)
    pca = projector.get("pca")
    if pca is not None:
        return pca.transform(scaled).astype(np.float32)
    return scaled.astype(np.float32, copy=False)


def map_logits_to_competition(
    logits: np.ndarray,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
) -> np.ndarray:
    scores = np.zeros((len(logits), n_classes), dtype=np.float32)
    scores[:, mapped_pos] = logits[:, mapped_bc_indices]
    for pos, bc_idx_arr in proxy_pos_to_bc.items():
        sub = logits[:, bc_idx_arr]
        proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
        scores[:, pos] = proxy_score.astype(np.float32, copy=False)
    return scores


def build_meta_for_paths(paths: Sequence[Path]) -> pd.DataFrame:
    n_rows = len(paths) * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    for file_idx, path in enumerate(paths):
        row_start = file_idx * N_WINDOWS
        row_end = row_start + N_WINDOWS
        stem = path.stem
        meta = parse_soundscape_filename(path.name)
        row_ids[row_start:row_end] = [f"{stem}_{t}" for t in range(5, 65, 5)]
        filenames[row_start:row_end] = path.name
        sites[row_start:row_end] = meta["site"]
        hours[row_start:row_end] = int(meta["hour_utc"])

    return pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours})


def infer_perch_onnx_spatial(
    paths: Sequence[Path],
    onnx_path: Path,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
    num_threads: int,
    batch_files: int,
    freq_pool: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    import onnxruntime as ort

    start_time = time.time()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    required = {"embedding", "spatial_embedding", "label"}
    missing = required - set(output_names)
    if missing:
        raise RuntimeError(f"Perch ONNX missing outputs {sorted(missing)}. Outputs: {output_names}")

    paths = [Path(path) for path in paths]
    all_embedding: List[np.ndarray] = []
    all_spatial: List[np.ndarray] = []
    all_raw_scores: List[np.ndarray] = []
    batch_files = max(1, int(batch_files))
    print(f"[INFO] Using Perch ONNX spatial: {onnx_path}")

    for start in range(0, len(paths), batch_files):
        batch_paths = paths[start:start + batch_files]
        x = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            row_start = batch_idx * N_WINDOWS
            row_end = row_start + N_WINDOWS
            x[row_start:row_end] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

        embedding, spatial, logits = session.run(["embedding", "spatial_embedding", "label"], {input_name: x})
        embedding = embedding.astype(np.float32, copy=False)
        spatial = spatial.astype(np.float32, copy=False)
        if embedding.shape[1:] != (1536,):
            raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")
        if spatial.shape[1:] != (16, 4, 1536):
            raise RuntimeError(f"Unexpected spatial_embedding shape: {spatial.shape}")
        if freq_pool == "flat64":
            spatial_tokens = spatial.reshape(len(batch_paths) * N_WINDOWS, 64, 1536).astype(np.float32, copy=False)
        elif freq_pool == "meanmax":
            spatial_tokens = np.stack(
                [
                    spatial.mean(axis=2).astype(np.float32, copy=False),
                    spatial.max(axis=2).astype(np.float32, copy=False),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
        else:
            spatial_tokens = spatial.mean(axis=2).astype(np.float32, copy=False)
        raw_scores = map_logits_to_competition(
            logits=logits.astype(np.float32, copy=False),
            n_classes=n_classes,
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=proxy_reduce,
        )
        all_embedding.append(embedding)
        all_spatial.append(spatial_tokens)
        all_raw_scores.append(raw_scores)

        done = start + len(batch_paths)
        if done == len(paths) or done % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] ONNX spatial Perch {done}/{len(paths)} files | elapsed={elapsed:.1f}s")

    meta_df = build_meta_for_paths(paths)
    embedding_full = np.concatenate(all_embedding, axis=0).astype(np.float32, copy=False)
    spatial_tokens_full = np.concatenate(all_spatial, axis=0).astype(np.float32, copy=False)
    raw_scores_full = np.concatenate(all_raw_scores, axis=0).astype(np.float32, copy=False)
    print(f"[INFO] ONNX spatial Perch done: {len(paths)} files in {time.time() - start_time:.1f}s")
    return meta_df, spatial_tokens_full, raw_scores_full, embedding_full


def load_meta(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() == ".parquet":
        return pd.read_parquet(meta_path)
    if meta_path.suffix.lower() == ".csv":
        return pd.read_csv(meta_path)
    raise ValueError(f"Unsupported meta suffix: {meta_path.suffix}")


def load_debug_features(
    features_npz: Path,
    features_meta_path: Path,
    freq_pool: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    arrays = np.load(features_npz, allow_pickle=True)
    if "spatial_tokens" not in arrays:
        raise KeyError(f"{features_npz} must contain spatial_tokens")
    if "raw_scores" not in arrays:
        raise KeyError(f"{features_npz} must contain raw_scores")
    if features_meta_path:
        meta_df = load_meta(features_meta_path)
    elif "row_id" in arrays and "filename" in arrays:
        row_ids = arrays["row_id"]
        filenames = arrays["filename"]
        meta_df = pd.DataFrame(
            {
                "row_id": row_ids,
                "filename": filenames,
                "site": np.asarray(["unknown"] * len(row_ids), dtype=object),
                "hour_utc": np.asarray([-1] * len(row_ids), dtype=np.int16),
            }
        )
    else:
        raise ValueError("--features-meta-path is required unless NPZ contains row_id and filename")
    if freq_pool == "flat64" and "spatial_tokens_64" in arrays:
        spatial_tokens = arrays["spatial_tokens_64"].astype(np.float32, copy=False)
    elif freq_pool == "meanmax" and "spatial_tokens_max" in arrays:
        spatial_tokens = np.stack(
            [
                arrays["spatial_tokens"].astype(np.float32, copy=False),
                arrays["spatial_tokens_max"].astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    else:
        spatial_tokens = arrays["spatial_tokens"].astype(np.float32, copy=False)
    embedding = arrays["embedding"].astype(np.float32, copy=False) if "embedding" in arrays else None
    return meta_df, spatial_tokens, arrays["raw_scores"].astype(np.float32, copy=False), embedding


def predict_fold(
    fold_artifact: Dict[str, object],
    spatial_tokens: np.ndarray,
    raw_scores: np.ndarray,
    embedding: np.ndarray | None,
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    model_artifact = fold_artifact["model"]
    tokens = transform_token_projector(
        tokens=spatial_tokens,
        projector=fold_artifact["token_projector"],
    )
    raw_input = raw_scores
    if str(model_artifact.get("head_variant", "generic")) == "perch_fusion":
        if embedding is None:
            raise ValueError("head_variant=perch_fusion requires embedding features.")
        raw_input = np.concatenate([embedding, raw_scores], axis=1).astype(np.float32, copy=False)
    raw_features = transform_raw_projector(raw_scores=raw_input, projector=fold_artifact.get("raw_projector"))

    model = PerchSpatialMambaHead(
        token_dim=int(model_artifact["token_dim"]),
        num_classes=int(model_artifact["output_dim"]),
        num_blocks=int(model_artifact["num_blocks"]),
        kernel_size=int(model_artifact["kernel_size"]),
        hidden_dim=int(model_artifact["hidden_dim"]),
        dropout=float(model_artifact["dropout"]),
        raw_dim=int(model_artifact["raw_dim"]),
        freq_pool=str(model_artifact.get("freq_pool", "mean")),
        use_pos_embed=bool(model_artifact.get("use_pos_embed", False)),
        head_variant=str(model_artifact.get("head_variant", "generic")),
        prototype_per_class=int(model_artifact.get("prototype_per_class", 5)),
        prototype_temperature=float(model_artifact.get("prototype_temperature", 12.0)),
    )
    model.load_state_dict(model_artifact["model_state"])
    model.eval()

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            token_batch = torch.from_numpy(tokens[start:start + batch_size])
            raw_batch = None
            if raw_features is not None:
                raw_batch = torch.from_numpy(raw_features[start:start + batch_size])
            pred = torch.sigmoid(model(token_batch, raw_batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    spatial_all = np.concatenate(preds, axis=0)

    pred = sigmoid_np(raw_scores).astype(np.float32)
    fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int32)
    pred[:, fitted] = spatial_all[:, fitted]
    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0), int(len(fitted))


def predict_ensemble(
    artifact: Dict[str, object],
    spatial_tokens: np.ndarray,
    raw_scores: np.ndarray,
    embedding: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    if artifact.get("model_type") != "perch_spatial_mamba":
        raise ValueError(f"Expected model_type=perch_spatial_mamba, got {artifact.get('model_type')}")

    fold_preds = []
    for fold_artifact in artifact["folds"]:
        fold_pred, fitted_count = predict_fold(
            fold_artifact=fold_artifact,
            spatial_tokens=spatial_tokens,
            raw_scores=raw_scores,
            embedding=embedding,
            batch_size=batch_size,
        )
        fold_preds.append(fold_pred)
        print(
            f"[INFO] Applied {fold_artifact.get('fold_name', 'fold')} "
            f"fitted_classes={fitted_count}"
        )

    pred = np.mean(fold_preds, axis=0).astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, int(args.runtime_num_threads)))

    competition_root = Path(args.competition_root)
    sample_submission_path = (
        resolve_user_path(args.sample_submission_path, competition_root)
        if args.sample_submission_path
        else competition_root / "sample_submission.csv"
    )
    taxonomy_path = (
        resolve_user_path(args.taxonomy_path, competition_root)
        if args.taxonomy_path
        else competition_root / "taxonomy.csv"
    )
    soundscapes_dir = (
        resolve_user_path(args.soundscapes_dir, competition_root)
        if args.soundscapes_dir
        else competition_root / ("train_soundscapes" if args.debug else "test_soundscapes")
    )
    perch_dir = resolve_user_path(args.perch_dir, competition_root)
    model_path = discover_model_path(args.model_path)
    output_path = Path(args.output_path)

    artifact = joblib.load(model_path)
    class_names = load_class_names(sample_submission_path)
    if list(artifact["class_names"]) != list(class_names):
        raise ValueError("Artifact class_names do not match sample_submission columns.")

    onnx_path = discover_existing_path(
        args.perch_onnx_path,
        candidates=[
            "PerchV2Onnx/perch_v2.onnx",
            "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
            "/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx",
        ],
        competition_root=competition_root,
    )
    if onnx_path is None and not args.features_npz:
        raise FileNotFoundError("No Perch ONNX model found. Pass --perch-onnx-path explicitly.")

    soundscape_files: List[Path] = []
    if not args.features_npz:
        soundscape_files = list_soundscape_files(soundscapes_dir, debug=args.debug, debug_limit=args.debug_limit)
        if not soundscape_files:
            raise FileNotFoundError(f"No .ogg files found under {soundscapes_dir}")

    print("[INFO] Perch spatial Mamba inference")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] files: {len(soundscape_files) if soundscape_files else 'features-npz'}")
    print(f"[INFO] perch_dir: {perch_dir}")
    print(f"[INFO] perch_onnx_path: {onnx_path}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] output_path: {output_path}")
    print(f"[INFO] seed: {args.seed}")
    print(f"[INFO] artifact_config: {artifact.get('config', {})}")
    freq_pool = str(artifact.get("config", {}).get("freq_pool", "mean"))

    if args.features_npz:
        meta_df, spatial_tokens, raw_scores, embedding = load_debug_features(
            features_npz=resolve_user_path(args.features_npz, competition_root=competition_root),
            features_meta_path=resolve_user_path(args.features_meta_path, competition_root=competition_root)
            if args.features_meta_path
            else Path(""),
            freq_pool=freq_pool,
        )
    else:
        if onnx_path is None:
            raise FileNotFoundError("No Perch ONNX model found. Pass --perch-onnx-path explicitly.")
        bc_labels = load_perch_label_table(perch_dir=perch_dir, onnx_path=onnx_path)
        bc_indices, mapped_bc_indices, mapping = build_competition_mapping(
            primary_labels=class_names,
            taxonomy_path=taxonomy_path,
            bc_labels=bc_labels,
        )
        mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
        proxy_pos_to_bc = build_selected_proxy_targets(primary_labels=class_names, mapping=mapping, bc_labels=bc_labels)

        meta_df, spatial_tokens, raw_scores, embedding = infer_perch_onnx_spatial(
            paths=soundscape_files,
            onnx_path=onnx_path,
            n_classes=len(class_names),
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=args.proxy_reduce,
            num_threads=args.runtime_num_threads,
            batch_files=args.batch_files,
            freq_pool=freq_pool,
        )
    pred = predict_ensemble(
        artifact=artifact,
        spatial_tokens=spatial_tokens,
        raw_scores=raw_scores,
        embedding=embedding,
        batch_size=args.batch_files * N_WINDOWS,
    )

    submission = pd.concat(
        [pd.DataFrame({"row_id": meta_df["row_id"].to_numpy()}), pd.DataFrame(pred, columns=class_names)],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved submission to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
