#!/usr/bin/env python3
"""Unified Kaggle inference for Perch LR + spatial heads + Stage3 CNN.

The important deployment detail is that frozen Perch v2 ONNX is executed only
once.  Its shared outputs are then routed to:

- Perch context LogReg: raw class logits + embedding
- Perch Mamba head: spatial_embedding pooled to [B, 16, 1536]
- Perch Attention head: spatial_embedding flattened to [B, 64, 1536]
- Perch 60s temporal head: full-file flat64 tokens [file, 12, 64, 1536]

Stage3 CNN and optional raw waveform models are inferred separately, then all
enabled branches are logit-blended and can be post-processed with file-level
scaling and temporal smoothing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def register_numpy_core_compat() -> None:
    try:
        import numpy._core as numpy_core_mod  # type: ignore[attr-defined]
        has_numpy_core = True
    except Exception:
        import numpy.core as numpy_core_mod  # type: ignore[no-redef]
        has_numpy_core = False

    sys.modules.setdefault("numpy._core", numpy_core_mod)
    if not hasattr(np, "_core"):
        np._core = numpy_core_mod  # type: ignore[attr-defined]

    if has_numpy_core:
        return

    alias_targets = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.arrayprint": "numpy.core.arrayprint",
        "numpy._core.records": "numpy.core.records",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core._dtype_ctypes": "numpy.core._dtype_ctypes",
        "numpy._core._methods": "numpy.core._methods",
    }
    for alias, target in alias_targets.items():
        try:
            sys.modules.setdefault(alias, importlib.import_module(target))
        except Exception:
            pass


register_numpy_core_compat()

import birdclef2026_blend_submissions_postprocess as blend_postprocess
import birdclef2026_gm_kaggle_infer as cnn_base
import birdclef2026_gm_kaggle_infer_stage3 as stage3_infer
import birdclef2026_perch_kaggle_infer_spatial_mamba as spatial_infer
from waveform_model import RawWaveTransformerMixerModel, RawWaveTransformerModel


EPS = 1e-6
DEFAULT_WEIGHTS_5WAY = {
    "perch_lr": 0.2275,
    "mamba": 0.273,
    "stage3": 0.1365,
    "attention": 0.273,
    "raw_wave": 0.09,
    "base_cnn": 0.0,
}
DEFAULT_WEIGHTS_SSM_MIXED = {
    "perch_lr": 0.22625,
    "mamba": 0.1465,
    "stage3": 0.13575,
    "attention": 0.1465,
    "raw_wave": 0.095,
    "ssm": 0.25,
    "temporal": 0.0,
    "audiomae_token": 0.0,
    "base_cnn": 0.0,
}
DEFAULT_WEIGHTS_6WAY = {
    "perch_lr": 0.2275,
    "mamba": 0.273,
    "stage3": 0.1365,
    "attention": 0.193,
    "raw_wave": 0.09,
    "temporal": 0.08,
    "base_cnn": 0.0,
}
DEFAULT_WEIGHTS_AUDIOMAE_TOKEN = {
    "perch_lr": 0.0,
    "mamba": 0.275,
    "stage3": 0.075,
    "attention": 0.075,
    "raw_wave": 0.15,
    "temporal": 0.0,
    "audiomae_token": 0.425,
    "base_cnn": 0.0,
}


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified BirdCLEF 2026 Perch+Stage3 ensemble inference.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--taxonomy-path", type=str, default="")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--perch-onnx-path", type=str, default="")
    parser.add_argument("--perch-lr-model-path", type=str, default="")
    parser.add_argument("--mamba-model-path", type=str, default="")
    parser.add_argument("--attention-model-path", type=str, default="")
    parser.add_argument("--temporal-model-path", type=str, default="")
    parser.add_argument("--ssm-model-path", type=str, default="")
    parser.add_argument("--stage3-model-root", type=str, default="")
    parser.add_argument("--base-cnn-model-root", type=str, default="")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--stage3-backend", type=str, choices=["torch", "openvino"], default="torch")
    parser.add_argument("--stage3-segment-batch-size", type=int, default=12)
    parser.add_argument("--base-cnn-segment-batch-size", type=int, default=12)
    parser.add_argument("--proxy-reduce", type=str, choices=["max", "mean"], default="max")
    parser.add_argument("--perch-lr-weight", type=float, default=None)
    parser.add_argument("--mamba-weight", type=float, default=None)
    parser.add_argument(
        "--mamba-tta-offsets",
        type=str,
        default="",
        help="Comma-separated shifted 5s offsets in seconds for the Mamba branch, e.g. '-1,1'. Empty disables TTA.",
    )
    parser.add_argument("--stage3-weight", type=float, default=None)
    parser.add_argument(
        "--stage3-tta-offsets",
        type=str,
        default="",
        help="Comma-separated shifted 5s offsets in seconds for the Stage3 CNN branch, e.g. '-1,1'. Empty disables TTA.",
    )
    parser.add_argument("--attention-weight", type=float, default=None)
    parser.add_argument("--base-cnn-weight", type=float, default=None)
    parser.add_argument(
        "--base-cnn-tta-offsets",
        type=str,
        default="",
        help="Comma-separated shifted 5s offsets in seconds for the optional base CNN branch.",
    )
    parser.add_argument("--raw-wave-weight", type=float, default=None)
    parser.add_argument("--temporal-weight", type=float, default=None)
    parser.add_argument("--ssm-weight", type=float, default=None)
    parser.add_argument("--audiomae-token-weight", type=float, default=None)
    parser.add_argument("--audiomae-token-model-path", type=str, default="")
    parser.add_argument("--audiomae-ckpt-dir", type=str, default="")
    parser.add_argument("--audiomae-token-batch-size", type=int, default=32)
    parser.add_argument("--audiomae-token-device", type=str, default="auto")
    parser.add_argument("--raw-wave-model-root", type=str, default="")
    parser.add_argument("--raw-wave-backend", type=str, choices=["torch", "openvino"], default="torch")
    parser.add_argument("--raw-wave-segment-batch-size", type=int, default=12)
    parser.add_argument("--file-scale-mode", type=str, choices=["none", "topk_mean", "max_power"], default="topk_mean")
    parser.add_argument("--file-scale-value", type=float, default=None)
    parser.add_argument("--file-scale-topk", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--disable-file-scale", action="store_true")
    parser.add_argument(
        "--file-gate-mode",
        type=str,
        choices=["none", "topk_mean", "topk_max", "topk_geom", "mean", "mean_max", "topk_mean_max", "noisy_or"],
        default="none",
        help="Alternative file-level presence gate selected from OOF experiments. Overrides file-scale when enabled.",
    )
    parser.add_argument("--file-gate-topk", type=int, default=2)
    parser.add_argument("--file-gate-gamma", type=float, default=1.25)
    parser.add_argument("--file-gate-strength", type=float, default=1.0)
    parser.add_argument("--smooth-mode", type=str, choices=["none", "plain", "adaptive"], default="none")
    parser.add_argument("--smooth-alpha", type=float, default=0.10)
    parser.add_argument(
        "--blend-mode",
        type=str,
        choices=["logit", "mixed_rank", "family3"],
        default="logit",
        help=(
            "Final blend mode. family3 uses rank blend for 47158son* classes. "
        ),
    )
    parser.add_argument("--rank-blend-alpha-logit", type=float, default=0.70)
    parser.add_argument("--save-branch-submissions", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def parse_float_list(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Blend weights must sum to a positive value, got {weights}")
    return {key: float(value) / total for key, value in weights.items()}


def resolve_optional_path(path_str: str, competition_root: Path) -> Path | None:
    if not path_str:
        return None
    path = spatial_infer.resolve_user_path(path_str, competition_root=competition_root)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def discover_unique_file(explicit_path: str, candidates: Sequence[str], filename: str, label: str, competition_root: Path) -> Path:
    explicit = resolve_optional_path(explicit_path, competition_root=competition_root)
    if explicit is not None:
        return explicit

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    hits: List[Path] = []
    for root in [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]:
        if root.exists():
            hits.extend(root.rglob(filename))
    hits = sorted(set(hits))
    if len(hits) == 1:
        print(f"[INFO] Auto-discovered {label}: {hits[0]}")
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"Could not find {label}. Pass the explicit path.")
    raise FileExistsError(
        f"Found multiple candidates for {label}; pass the explicit path:\n"
        + "\n".join(str(path) for path in hits[:20])
    )


def discover_onnx_path(explicit_path: str, competition_root: Path) -> Path:
    onnx_path = spatial_infer.discover_existing_path(
        explicit_path,
        candidates=[
            "PerchV2Onnx/perch_v2.onnx",
            "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
            "/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx",
        ],
        competition_root=competition_root,
    )
    if onnx_path is None:
        raise FileNotFoundError("No Perch ONNX model found. Pass --perch-onnx-path explicitly.")
    return onnx_path


def load_artifact(path: Path, class_names: Sequence[str], label: str) -> Dict[str, object]:
    artifact = joblib.load(path)
    if list(artifact["class_names"]) != list(class_names):
        raise ValueError(f"{label} artifact class_names do not match sample_submission columns.")
    return artifact


def resolve_default_weights(
    args: argparse.Namespace,
    temporal_enabled: bool,
    ssm_enabled: bool,
    raw_wave_enabled: bool,
    audiomae_token_enabled: bool = False,
    base_cnn_enabled: bool = False,
) -> Dict[str, float]:
    defaults = DEFAULT_WEIGHTS_SSM_MIXED if ssm_enabled else DEFAULT_WEIGHTS_AUDIOMAE_TOKEN if audiomae_token_enabled else (
        DEFAULT_WEIGHTS_6WAY if temporal_enabled else DEFAULT_WEIGHTS_5WAY
    )
    raw = {
        "perch_lr": defaults.get("perch_lr", 0.0) if args.perch_lr_weight is None else args.perch_lr_weight,
        "mamba": defaults.get("mamba", 0.0) if args.mamba_weight is None else args.mamba_weight,
        "stage3": defaults.get("stage3", 0.0) if args.stage3_weight is None else args.stage3_weight,
        "attention": defaults.get("attention", 0.0) if args.attention_weight is None else args.attention_weight,
        "raw_wave": defaults.get("raw_wave", 0.0) if args.raw_wave_weight is None else args.raw_wave_weight,
        "temporal": defaults.get("temporal", 0.0) if args.temporal_weight is None else args.temporal_weight,
        "ssm": defaults.get("ssm", 0.0) if args.ssm_weight is None else args.ssm_weight,
        "audiomae_token": defaults.get("audiomae_token", 0.0) if args.audiomae_token_weight is None else args.audiomae_token_weight,
        "base_cnn": defaults.get("base_cnn", 0.0) if args.base_cnn_weight is None else args.base_cnn_weight,
    }
    if not raw_wave_enabled:
        raw["raw_wave"] = 0.0
    if not temporal_enabled:
        raw["temporal"] = 0.0
    if not ssm_enabled:
        raw["ssm"] = 0.0
    if not audiomae_token_enabled:
        raw["audiomae_token"] = 0.0
    if not base_cnn_enabled:
        raw["base_cnn"] = 0.0
    return normalize_weights(raw)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def discover_raw_wave_model_root(explicit_path: str, competition_root: Path) -> Path | None:
    if not explicit_path:
        return None
    path = spatial_infer.resolve_user_path(explicit_path, competition_root=competition_root)
    if not path.exists():
        raise FileNotFoundError(f"Raw waveform model root does not exist: {path}")
    return path


def load_raw_wave_models(model_root: Path, num_classes: int, device: torch.device):
    config_path = model_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Raw waveform model root is missing config.json: {model_root}")
    cfg = load_json(config_path)
    if str(cfg.get("tokenizer_type", "conv_stack")) != "conv_stack":
        raise ValueError("Unified raw-wave inference currently supports tokenizer_type=conv_stack only.")
    variant = str(cfg.get("waveform_model_variant", "base"))
    model_cls = RawWaveTransformerMixerModel if variant == "mixer" else RawWaveTransformerModel
    kwargs = {
        "num_classes": int(num_classes),
        "embed_dim": int(cfg.get("d_model", 768)),
        "depth": int(cfg.get("transformer_layers", 4)),
        "num_heads": int(cfg.get("transformer_heads", 8)),
        "mlp_ratio": int(cfg.get("transformer_ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.2)),
        "num_tokens": int(cfg.get("num_tokens", 32)),
    }
    sample_rate = int(cfg.get("sample_rate", 32000))
    clip_seconds = float(cfg.get("clip_seconds", 5.0))
    fold_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No raw waveform fold checkpoints found under {model_root}")

    models = []
    for fold_path in fold_paths:
        model = model_cls(**kwargs)
        checkpoint = torch.load(fold_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded raw waveform checkpoint: {fold_path}")
    print(
        "[INFO] Raw waveform config: "
        f"variant={variant} sample_rate={sample_rate} clip_seconds={clip_seconds} "
        f"tokens={kwargs['num_tokens']} d_model={kwargs['embed_dim']} depth={kwargs['depth']}"
    )
    return models, sample_rate, clip_seconds


def load_raw_wave_openvino_models(model_root: Path, num_classes: int) -> Tuple[List[object], int, float]:
    import openvino as ov

    _patch_torch_numpy_for_openvino()
    config_path = model_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Raw waveform model root is missing config.json: {model_root}")
    cfg = load_json(config_path)
    if str(cfg.get("tokenizer_type", "conv_stack")) != "conv_stack":
        raise ValueError("OpenVINO raw-wave inference currently supports tokenizer_type=conv_stack only.")
    variant = str(cfg.get("waveform_model_variant", "base"))
    model_cls = RawWaveTransformerMixerModel if variant == "mixer" else RawWaveTransformerModel
    kwargs = {
        "num_classes": int(num_classes),
        "embed_dim": int(cfg.get("d_model", 768)),
        "depth": int(cfg.get("transformer_layers", 4)),
        "num_heads": int(cfg.get("transformer_heads", 8)),
        "mlp_ratio": int(cfg.get("transformer_ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.2)),
        "num_tokens": int(cfg.get("num_tokens", 32)),
    }
    sample_rate = int(cfg.get("sample_rate", 32000))
    clip_seconds = float(cfg.get("clip_seconds", 5.0))
    fold_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No raw waveform fold checkpoints found under {model_root}")

    core = ov.Core()
    example = torch.randn(1, int(round(sample_rate * clip_seconds)), dtype=torch.float32)
    compiled_models: List[object] = []
    for fold_path in fold_paths:
        model = model_cls(**kwargs)
        checkpoint = torch.load(fold_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        ov_model = convert_torch_model_to_openvino(
            model,
            example=example,
            label=f"raw waveform {fold_path.name}",
            force_trace=env_flag("OPENVINO_FORCE_TRACE_RAW_WAVE"),
        )
        ov_model.reshape({0: [-1, int(round(sample_rate * clip_seconds))]})
        compiled_models.append(core.compile_model(ov_model, "CPU"))
        print(f"[INFO] Loaded OpenVINO raw waveform checkpoint: {fold_path}")
        del model, checkpoint, state_dict, ov_model
    print(
        "[INFO] OpenVINO raw waveform config: "
        f"variant={variant} sample_rate={sample_rate} clip_seconds={clip_seconds} "
        f"tokens={kwargs['num_tokens']} d_model={kwargs['embed_dim']} depth={kwargs['depth']}"
    )
    return compiled_models, sample_rate, clip_seconds


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_model,
            self.d_model,
            int(d_conv),
            padding=int(d_conv) - 1,
            groups=self.d_model,
        )
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
    """Inference-only copy of the trained SSM head.

    Keeping this class local avoids importing the training module on Kaggle,
    which pulls sklearn/scipy just to reconstruct a torch module.
    """

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

    def encode(self, emb: torch.Tensor, site_ids: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
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


def parse_end_seconds(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(str(row_id).rsplit("_", 1)[-1]) for row_id in row_ids], dtype=np.int64)


def build_sequence_file_tensor(
    meta_df: pd.DataFrame,
    embedding: np.ndarray,
    raw_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_windows = spatial_infer.N_WINDOWS
    expected_end = np.arange(5, 65, 5, dtype=np.int64)
    row_end = parse_end_seconds(meta_df["row_id"].tolist())
    file_emb: List[np.ndarray] = []
    file_raw: List[np.ndarray] = []
    file_indices: List[np.ndarray] = []
    file_sites: List[str] = []
    file_hours: List[int] = []
    for filename, indices in meta_df.groupby("filename", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(row_end[idx], kind="stable")
        idx_sorted = idx[order]
        ends = row_end[idx_sorted]
        if len(idx_sorted) != n_windows or not np.array_equal(ends, expected_end):
            raise ValueError(f"SSM got incomplete or misordered windows for {filename}: {ends.tolist()}")
        file_emb.append(embedding[idx_sorted])
        file_raw.append(raw_scores[idx_sorted])
        file_indices.append(idx_sorted)
        file_sites.append(str(meta_df.iloc[idx_sorted[0]].get("site", "unknown")))
        hour_value = int(meta_df.iloc[idx_sorted[0]].get("hour_utc", 0))
        file_hours.append(max(0, min(23, hour_value)))
    return (
        np.stack(file_emb, axis=0).astype(np.float32, copy=False),
        np.stack(file_raw, axis=0).astype(np.float32, copy=False),
        np.stack(file_indices, axis=0).astype(np.int64, copy=False),
        np.asarray(file_sites, dtype=object),
        np.asarray(file_hours, dtype=np.int64),
    )


def predict_ssm_fold(
    fold_artifact: Dict[str, object],
    file_embedding: np.ndarray,
    file_raw_scores: np.ndarray,
    row_indices_by_file: np.ndarray,
    file_sites: np.ndarray,
    file_hours: np.ndarray,
    site_to_idx: Dict[str, int],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    model_artifact = fold_artifact["model"]
    standardizer = fold_artifact["embedding_standardizer"]
    emb = ((file_embedding - standardizer["mean"]) / standardizer["std"]).astype(np.float32, copy=False)
    n_sites = int(model_artifact["n_sites"])
    unk_site = max(0, n_sites - 1)
    site_ids = np.asarray([site_to_idx.get(str(site), unk_site) for site in file_sites], dtype=np.int64)
    site_ids = np.clip(site_ids, 0, n_sites - 1)

    model = ProtoSSMHead(
        d_input=int(model_artifact["d_input"]),
        d_model=int(model_artifact["d_model"]),
        d_state=int(model_artifact["d_state"]),
        n_ssm_layers=int(model_artifact["n_ssm_layers"]),
        n_classes=int(model_artifact["output_dim"]),
        n_windows=spatial_infer.N_WINDOWS,
        dropout=float(model_artifact["dropout"]),
        n_sites=n_sites,
        meta_dim=int(model_artifact["meta_dim"]),
        use_cross_attn=bool(model_artifact["use_cross_attn"]),
        cross_attn_heads=int(model_artifact["cross_attn_heads"]),
    )
    model.load_state_dict(model_artifact["model_state"], strict=True)
    model.to(device)
    model.eval()

    file_preds: List[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(emb), batch_size):
            emb_t = torch.from_numpy(emb[start:start + batch_size]).to(device)
            raw_t = torch.from_numpy(file_raw_scores[start:start + batch_size].astype(np.float32, copy=False)).to(device)
            site_t = torch.from_numpy(site_ids[start:start + batch_size]).to(device)
            hour_t = torch.from_numpy(file_hours[start:start + batch_size].astype(np.int64, copy=False)).to(device)
            pred = torch.sigmoid(model(emb_t, raw_t, site_t, hour_t)).detach().cpu().numpy().astype(np.float32)
            file_preds.append(pred)
            del emb_t, raw_t, site_t, hour_t
    file_pred = np.concatenate(file_preds, axis=0)
    flat_raw = file_raw_scores.reshape(-1, file_raw_scores.shape[-1])
    flat_pred = file_pred.reshape(-1, file_pred.shape[-1])
    flat_indices = row_indices_by_file.reshape(-1)
    fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int32)
    out_sorted = sigmoid_np(flat_raw).astype(np.float32)
    out_sorted[:, fitted] = flat_pred[:, fitted]
    out = np.zeros_like(out_sorted)
    out[flat_indices] = out_sorted
    return np.clip(out.astype(np.float32, copy=False), 0.0, 1.0), int(len(fitted))


def predict_ssm_ensemble(
    artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    embedding: np.ndarray,
    raw_scores: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if artifact.get("model_type") != "perch_sequence_ssm":
        raise ValueError(f"Expected model_type=perch_sequence_ssm, got {artifact.get('model_type')}")
    file_embedding, file_raw_scores, row_indices_by_file, file_sites, file_hours = build_sequence_file_tensor(
        meta_df=meta_df,
        embedding=embedding,
        raw_scores=raw_scores,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] SSM head device: {device}")
    site_to_idx = {str(key): int(value) for key, value in dict(artifact.get("site_to_idx", {})).items()}
    fold_preds = []
    for fold_artifact in artifact["folds"]:
        fold_pred, fitted_count = predict_ssm_fold(
            fold_artifact=fold_artifact,
            file_embedding=file_embedding,
            file_raw_scores=file_raw_scores,
            row_indices_by_file=row_indices_by_file,
            file_sites=file_sites,
            file_hours=file_hours,
            site_to_idx=site_to_idx,
            batch_size=batch_size,
            device=device,
        )
        fold_preds.append(fold_pred)
        print(
            f"[INFO] Applied SSM {fold_artifact.get('fold_name', 'fold')} "
            f"fitted_classes={fitted_count}"
        )
    return np.clip(np.mean(fold_preds, axis=0).astype(np.float32), 0.0, 1.0)


def build_position_features(end_seconds: np.ndarray) -> np.ndarray:
    pos = end_seconds.astype(np.float32) / 60.0
    angle = 2.0 * np.pi * pos
    return np.stack(
        [
            pos,
            np.sin(angle).astype(np.float32, copy=False),
            np.cos(angle).astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def build_metadata_features(meta_df: pd.DataFrame, include_hour_features: bool) -> np.ndarray:
    if not include_hour_features:
        return np.zeros((len(meta_df), 0), dtype=np.float32)
    hour = meta_df["hour_utc"].to_numpy(dtype=np.float32, copy=False)
    hour_phase = 2.0 * np.pi * (hour / 24.0)
    return np.stack(
        [
            hour / 24.0,
            np.sin(hour_phase).astype(np.float32, copy=False),
            np.cos(hour_phase).astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def previous_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.empty_like(values)
    out[:steps] = values[:1]
    out[steps:] = values[:-steps]
    return out


def next_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.empty_like(values)
    out[:-steps] = values[steps:]
    out[-steps:] = values[-1:]
    return out


def build_context_tensor(meta_df: pd.DataFrame, scores_full_raw: np.ndarray) -> np.ndarray:
    end_seconds = parse_end_seconds(meta_df["row_id"].tolist())
    n_rows, n_classes = scores_full_raw.shape
    context = np.zeros((n_rows, n_classes, 13), dtype=np.float32)
    eps = 1e-6

    for _, row_indices in meta_df.groupby("filename", sort=False).indices.items():
        idx = np.asarray(row_indices, dtype=np.int64)
        order = np.argsort(end_seconds[idx], kind="stable")
        idx_sorted = idx[order]
        scores = scores_full_raw[idx_sorted].astype(np.float32, copy=False)

        prev1 = previous_with_edge(scores, steps=1)
        next1 = next_with_edge(scores, steps=1)
        prev2 = previous_with_edge(scores, steps=2)
        next2 = next_with_edge(scores, steps=2)
        file_mean = np.repeat(scores.mean(axis=0, keepdims=True), len(idx_sorted), axis=0)
        file_max = np.repeat(scores.max(axis=0, keepdims=True), len(idx_sorted), axis=0)
        file_std = np.repeat(scores.std(axis=0, keepdims=True), len(idx_sorted), axis=0)
        neighbor_mean = 0.5 * (prev1 + next1)
        neighbor_max = np.maximum(prev1, next1)
        centered = scores - file_mean
        delta_prev1 = scores - prev1
        delta_next1 = scores - next1
        relative_to_file_max = scores / (file_max + eps)

        context[idx_sorted] = np.stack(
            [
                prev1,
                next1,
                prev2,
                next2,
                file_mean,
                file_max,
                file_std,
                neighbor_mean,
                neighbor_max,
                centered,
                delta_prev1,
                delta_next1,
                relative_to_file_max,
            ],
            axis=2,
        ).astype(np.float32, copy=False)

    return context


def build_base_features(
    emb_part: np.ndarray,
    raw_scores: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    parts = [emb_part, raw_scores.astype(np.float32, copy=False), position_features]
    if metadata_features.shape[1] > 0:
        parts.append(metadata_features)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def predict_binary_logreg_proba(model, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)
    if coef.shape[0] != 1:
        raise ValueError(f"Expected binary LogisticRegression coef shape (1, n_features), got {coef.shape}")
    logits = x @ coef[0] + float(intercept[0])
    proba = sigmoid_np(logits).astype(np.float32, copy=False)
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2 and int(classes[1]) != 1:
        proba = 1.0 - proba
    return proba


def transform_embedding_projector(emb: np.ndarray, fold_artifact: Dict[str, object]) -> np.ndarray:
    embedding_scaler = fold_artifact["embedding_scaler"]
    embedding_pca = fold_artifact["embedding_pca"]
    if embedding_scaler is None:
        return emb.astype(np.float32, copy=False)
    emb_scaled = embedding_scaler.transform(emb).astype(np.float32)
    if embedding_pca is None:
        return emb_scaled
    return embedding_pca.transform(emb_scaled).astype(np.float32)


def predict_context_logreg_fold(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    emb_proj = transform_embedding_projector(emb, fold_artifact=fold_artifact)
    base = build_base_features(
        emb_part=emb_proj,
        raw_scores=raw_scores,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    base_scaled = fold_artifact["base_scaler"].transform(base).astype(np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32)

    class_models = fold_artifact["class_models"]
    context_mean = fold_artifact["context_mean"]
    context_std = fold_artifact["context_std"]
    for class_idx in fold_artifact["fitted_class_indices"]:
        class_idx = int(class_idx)
        model = class_models[class_idx]
        ctx = context[:, class_idx, :].astype(np.float32, copy=False)
        ctx_scaled = ((ctx - context_mean[class_idx]) / context_std[class_idx]).astype(np.float32, copy=False)
        x = np.concatenate([base_scaled, ctx_scaled], axis=1).astype(np.float32, copy=False)
        pred[:, class_idx] = predict_binary_logreg_proba(model, x)
    return pred.astype(np.float32, copy=False)


def predict_context_logreg_ensemble(
    artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
) -> np.ndarray:
    model_type = str(artifact.get("model_type", "perch_context_logreg"))
    if model_type != "perch_context_logreg":
        raise ValueError(f"Unified script expects model_type=perch_context_logreg, got {model_type}")

    config = artifact["config"]
    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    metadata_features = build_metadata_features(
        meta_df=meta_df,
        include_hour_features=bool(config.get("include_hour_features", False)),
    )
    context = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    fold_preds = []
    for fold_artifact in artifact["folds"]:
        fold_pred = predict_context_logreg_fold(
            fold_artifact=fold_artifact,
            emb=emb_full,
            raw_scores=scores_full_raw,
            context=context,
            position_features=position_features,
            metadata_features=metadata_features,
        )
        fold_preds.append(fold_pred)
        print(
            f"[INFO] Applied {fold_artifact.get('fold_name', 'fold')} "
            f"fitted_classes={len(fold_artifact['fitted_class_indices'])}"
        )
    pred = np.mean(fold_preds, axis=0).astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def required_freq_pools(*artifacts: Dict[str, object]) -> List[str]:
    pools = []
    for artifact in artifacts:
        pool = str(artifact.get("config", {}).get("freq_pool", "mean"))
        if pool not in pools:
            pools.append(pool)
    return pools


def pool_spatial_tokens(spatial: np.ndarray, freq_pool: str) -> np.ndarray:
    if spatial.shape[1:] != (16, 4, 1536):
        raise RuntimeError(f"Unexpected spatial_embedding shape: {spatial.shape}")
    if freq_pool == "flat64":
        return spatial.reshape(len(spatial), 64, 1536).astype(np.float32, copy=False)
    if freq_pool == "meanmax":
        return np.stack(
            [
                spatial.mean(axis=2).astype(np.float32, copy=False),
                spatial.max(axis=2).astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    return spatial.mean(axis=2).astype(np.float32, copy=False)


class TemporalLocalMambaBlock(nn.Module):
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
                    TemporalLocalMambaBlock(
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
                TemporalLocalMambaBlock(
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
        self.time_pos = nn.Parameter(torch.zeros(1, spatial_infer.N_WINDOWS, int(hidden_dim)))
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
        # tokens: [B, 12, T, D], where T is usually 64 for flat spatial tokens.
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


def build_shifted_windows(audio: np.ndarray, offset_samples: int) -> np.ndarray:
    windows = np.zeros((spatial_infer.N_WINDOWS, spatial_infer.WINDOW_SAMPLES), dtype=np.float32)
    for window_idx in range(spatial_infer.N_WINDOWS):
        base_start = window_idx * spatial_infer.WINDOW_SAMPLES
        src_start = base_start + int(offset_samples)
        src_end = src_start + spatial_infer.WINDOW_SAMPLES
        dst_start = max(0, -src_start)
        dst_end = spatial_infer.WINDOW_SAMPLES - max(0, src_end - len(audio))
        clipped_start = max(0, src_start)
        clipped_end = min(len(audio), src_end)
        if clipped_end > clipped_start and dst_end > dst_start:
            windows[window_idx, dst_start:dst_end] = audio[clipped_start:clipped_end]
    return windows


def infer_perch_shared_onnx(
    paths: Sequence[Path],
    onnx_path: Path,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
    num_threads: int,
    batch_files: int,
    freq_pools: Sequence[str],
    clip_offset_seconds: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
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
    batch_files = max(1, int(batch_files))
    freq_pools = list(freq_pools)
    pooled_parts: Dict[str, List[np.ndarray]] = {pool: [] for pool in freq_pools}
    all_embedding: List[np.ndarray] = []
    all_raw_scores: List[np.ndarray] = []
    print(f"[INFO] Using shared Perch ONNX: {onnx_path}")
    print(f"[INFO] Shared Perch freq_pools: {freq_pools}")
    print(f"[INFO] Shared Perch clip_offset_seconds: {clip_offset_seconds}")
    offset_samples = int(round(float(clip_offset_seconds) * (spatial_infer.WINDOW_SAMPLES // 5)))

    for start in range(0, len(paths), batch_files):
        batch_paths = paths[start : start + batch_files]
        x = np.empty((len(batch_paths) * spatial_infer.N_WINDOWS, spatial_infer.WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = spatial_infer.read_soundscape_60s(path)
            row_start = batch_idx * spatial_infer.N_WINDOWS
            row_end = row_start + spatial_infer.N_WINDOWS
            if offset_samples == 0:
                x[row_start:row_end] = y.reshape(spatial_infer.N_WINDOWS, spatial_infer.WINDOW_SAMPLES)
            else:
                x[row_start:row_end] = build_shifted_windows(y, offset_samples=offset_samples)

        embedding, spatial, logits = session.run(["embedding", "spatial_embedding", "label"], {input_name: x})
        embedding = embedding.astype(np.float32, copy=False)
        spatial = spatial.astype(np.float32, copy=False)
        logits = logits.astype(np.float32, copy=False)
        if embedding.shape[1:] != (1536,):
            raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")

        for pool in freq_pools:
            pooled_parts[pool].append(pool_spatial_tokens(spatial, freq_pool=pool))
        raw_scores = spatial_infer.map_logits_to_competition(
            logits=logits,
            n_classes=n_classes,
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=proxy_reduce,
        )
        all_embedding.append(embedding)
        all_raw_scores.append(raw_scores)

        done = start + len(batch_paths)
        if done == len(paths) or done % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Shared ONNX Perch {done}/{len(paths)} files | elapsed={elapsed:.1f}s")

    meta_df = spatial_infer.build_meta_for_paths(paths)
    pooled_full = {
        pool: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for pool, parts in pooled_parts.items()
    }
    embedding_full = np.concatenate(all_embedding, axis=0).astype(np.float32, copy=False)
    raw_scores_full = np.concatenate(all_raw_scores, axis=0).astype(np.float32, copy=False)
    print(f"[INFO] Shared ONNX Perch done: {len(paths)} files in {time.time() - start_time:.1f}s")
    return meta_df, pooled_full, raw_scores_full, embedding_full


def predict_stage3_cnn(
    model_root: Path,
    soundscape_files: Sequence[Path],
    class_names: Sequence[str],
    competition_root: Path,
    debug: bool,
    debug_limit: int,
    segment_batch_size: int,
    seed: int,
    tta_offsets: Sequence[float] | None = None,
    label: str = "Stage3 CNN",
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = stage3_infer.InferConfig(
        competition_root=str(competition_root),
        output_path="/tmp/unused_submission.csv",
        model_root=str(model_root),
        soundscapes_dir="",
        debug=debug,
        debug_limit=debug_limit,
        segment_batch_size=segment_batch_size,
        seed=seed,
    )
    cnn_base.seed_everything(cfg.seed)
    spec = stage3_infer.resolve_model_spec(model_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = cnn_base.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )
    models = stage3_infer.load_models(spec=spec, num_classes=len(class_names), device=device)
    offsets = list(tta_offsets) if tta_offsets else [0.0]
    print(f"[INFO] {label} device: {device}")
    print(f"[INFO] {label} model_root: {model_root}")
    print(f"[INFO] {label} config_source: {spec.config_source}")
    print(f"[INFO] {label} tta_offsets: {offsets}")

    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(
        soundscape_files,
        total=len(soundscape_files),
        desc=label,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for audio_path in progress:
        audio = cnn_base.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
        offset_preds: List[np.ndarray] = []
        row_ids: List[str] | None = None
        for offset in offsets:
            segments, current_row_ids = cnn_base.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=spec.sample_rate,
                clip_seconds=spec.clip_seconds,
                clip_offset_seconds=float(offset),
            )
            preds = cnn_base.predict_file_segments(
                segments=segments,
                models=models,
                renderer=renderer,
                device=device,
                segment_batch_size=segment_batch_size,
            )
            if row_ids is None:
                row_ids = current_row_ids
            elif row_ids != current_row_ids:
                raise RuntimeError(f"Stage3 TTA row_id mismatch for {audio_path}")
            offset_preds.append(preds.astype(np.float32, copy=False))
        preds = np.mean(np.stack(offset_preds, axis=0), axis=0).astype(np.float32, copy=False)
        if row_ids is None:
            raise RuntimeError(f"No Stage3 offsets were evaluated for {audio_path}")
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    return np.asarray(all_row_ids, dtype=object), np.concatenate(all_preds, axis=0).astype(np.float32, copy=False)


def _patch_torch_numpy_for_openvino() -> None:
    original_numpy = torch.Tensor.numpy
    if getattr(original_numpy, "_openvino_force_compat", False):
        return

    def numpy_compat(self, *args, **kwargs):
        kwargs.pop("force", None)
        tensor = self.detach().cpu() if self.requires_grad else self.cpu()
        return original_numpy(tensor, *args, **kwargs)

    numpy_compat._openvino_force_compat = True  # type: ignore[attr-defined]
    torch.Tensor.numpy = numpy_compat  # type: ignore[assignment]


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def convert_torch_model_to_openvino(
    model: nn.Module,
    example: torch.Tensor,
    label: str,
    force_trace: bool = False,
):
    import openvino as ov

    model.eval()
    example = example.detach().cpu()
    if not force_trace:
        try:
            with torch.inference_mode():
                return ov.convert_model(model, example_input=example)
        except Exception as exc:
            print(
                f"[WARN] OpenVINO direct conversion failed for {label}; "
                f"retrying with TorchScript trace check disabled. Error: {type(exc).__name__}: {exc}"
            )
    else:
        print(f"[INFO] OPENVINO_FORCE_TRACE enabled for {label}")

    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False, check_trace=False)
        traced.eval()
    try:
        return ov.convert_model(traced)
    except Exception as exc:
        print(
            f"[WARN] OpenVINO TorchScript conversion without example_input failed for {label}; "
            f"retrying with example_input. Error: {type(exc).__name__}: {exc}"
        )
        return ov.convert_model(traced, example_input=example)


def load_stage3_openvino_models(
    spec: stage3_infer.ResolvedModelSpec,
    num_classes: int,
) -> List[object]:
    import openvino as ov

    _patch_torch_numpy_for_openvino()
    if spec.run_kind == "stage3":
        fold_paths = sorted(spec.model_root.glob("fold_*/stage3_best.pth"))
    else:
        fold_paths = sorted(spec.model_root.glob("fold_*/stage2_fold*_best.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No {spec.run_kind} fold checkpoints found under {spec.model_root}")

    core = ov.Core()
    example = torch.randn(1, 3, spec.image_height, spec.image_width, dtype=torch.float32)
    compiled_models: List[object] = []
    for fold_path in fold_paths:
        model = cnn_base.BirdCLEFNet(
            model_name=spec.model_name,
            num_classes=num_classes,
            dropout=spec.dropout,
            drop_path=spec.drop_path,
            head_type=spec.head_type,
        )
        checkpoint_obj = torch.load(fold_path, map_location="cpu")
        model.load_state_dict(cnn_base.extract_state_dict(checkpoint_obj), strict=True)
        model.eval()
        ov_model = convert_torch_model_to_openvino(
            model,
            example=example,
            label=f"{spec.run_kind} {fold_path.name}",
            force_trace=env_flag("OPENVINO_FORCE_TRACE") or env_flag("OPENVINO_FORCE_TRACE_STAGE3"),
        )
        ov_model.reshape({0: [-1, 3, spec.image_height, spec.image_width]})
        compiled_models.append(core.compile_model(ov_model, "CPU"))
        print(f"[INFO] Loaded OpenVINO {spec.run_kind} checkpoint: {fold_path}")
        del model, checkpoint_obj, ov_model
    return compiled_models


def predict_file_segments_openvino(
    segments: np.ndarray,
    models: Sequence[object],
    renderer: cnn_base.SpectrogramRenderer,
    segment_batch_size: int,
) -> np.ndarray:
    all_preds = []
    for start in range(0, len(segments), int(segment_batch_size)):
        batch_segments = segments[start:start + int(segment_batch_size)]
        images = renderer(batch_segments).numpy().astype(np.float32, copy=False)
        ensemble = None
        for model in models:
            logits = next(iter(model(images).values()))
            probs = sigmoid_np(np.asarray(logits, dtype=np.float32))
            ensemble = probs if ensemble is None else ensemble + probs
        all_preds.append((ensemble / float(len(models))).astype(np.float32, copy=False))
        del images, ensemble
    return np.concatenate(all_preds, axis=0).astype(np.float32, copy=False)


def predict_stage3_cnn_openvino(
    model_root: Path,
    soundscape_files: Sequence[Path],
    class_names: Sequence[str],
    competition_root: Path,
    debug: bool,
    debug_limit: int,
    segment_batch_size: int,
    seed: int,
    tta_offsets: Sequence[float] | None = None,
    label: str = "Stage3 CNN",
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = stage3_infer.InferConfig(
        competition_root=str(competition_root),
        output_path="/tmp/unused_submission.csv",
        model_root=str(model_root),
        soundscapes_dir="",
        debug=debug,
        debug_limit=debug_limit,
        segment_batch_size=segment_batch_size,
        seed=seed,
    )
    cnn_base.seed_everything(cfg.seed)
    spec = stage3_infer.resolve_model_spec(model_root)
    renderer = cnn_base.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )
    models = load_stage3_openvino_models(spec=spec, num_classes=len(class_names))
    offsets = list(tta_offsets) if tta_offsets else [0.0]
    print(f"[INFO] {label} backend: openvino")
    print(f"[INFO] {label} model_root: {model_root}")
    print(f"[INFO] {label} config_source: {spec.config_source}")
    print(f"[INFO] {label} tta_offsets: {offsets}")

    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(
        soundscape_files,
        total=len(soundscape_files),
        desc=f"{label} OpenVINO",
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for audio_path in progress:
        audio = cnn_base.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
        offset_preds: List[np.ndarray] = []
        row_ids: List[str] | None = None
        for offset in offsets:
            segments, current_row_ids = cnn_base.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=spec.sample_rate,
                clip_seconds=spec.clip_seconds,
                clip_offset_seconds=float(offset),
            )
            preds = predict_file_segments_openvino(
                segments=segments,
                models=models,
                renderer=renderer,
                segment_batch_size=segment_batch_size,
            )
            if row_ids is None:
                row_ids = current_row_ids
            elif row_ids != current_row_ids:
                raise RuntimeError(f"Stage3 OpenVINO TTA row_id mismatch for {audio_path}")
            offset_preds.append(preds.astype(np.float32, copy=False))
        preds = np.mean(np.stack(offset_preds, axis=0), axis=0).astype(np.float32, copy=False)
        if row_ids is None:
            raise RuntimeError(f"No Stage3 OpenVINO offsets were evaluated for {audio_path}")
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    return np.asarray(all_row_ids, dtype=object), np.concatenate(all_preds, axis=0).astype(np.float32, copy=False)


def predict_raw_wave(
    model_root: Path,
    soundscape_files: Sequence[Path],
    class_names: Sequence[str],
    segment_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, sample_rate, clip_seconds = load_raw_wave_models(
        model_root=model_root,
        num_classes=len(class_names),
        device=device,
    )
    print(f"[INFO] Raw waveform device: {device}")
    print(f"[INFO] Raw waveform model_root: {model_root}")

    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(
        soundscape_files,
        total=len(soundscape_files),
        desc="RawWave",
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for audio_path in progress:
        audio = cnn_base.load_soundscape_audio(audio_path, sample_rate=sample_rate)
        segments, row_ids = cnn_base.build_segments_for_file(
            audio=audio,
            file_stem=audio_path.stem,
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
        )
        file_preds: List[np.ndarray] = []
        for start in range(0, len(segments), int(segment_batch_size)):
            batch = torch.from_numpy(segments[start:start + int(segment_batch_size)]).float().to(device)
            with torch.inference_mode():
                ensemble = None
                for model in models:
                    probs = torch.sigmoid(model(batch))
                    ensemble = probs if ensemble is None else ensemble + probs
                ensemble = ensemble / len(models)
            file_preds.append(ensemble.detach().cpu().numpy().astype(np.float32))
            del batch, ensemble
        all_row_ids.extend(row_ids)
        all_preds.append(np.concatenate(file_preds, axis=0).astype(np.float32, copy=False))

    return np.asarray(all_row_ids, dtype=object), np.concatenate(all_preds, axis=0).astype(np.float32, copy=False)


def predict_raw_wave_openvino(
    model_root: Path,
    soundscape_files: Sequence[Path],
    class_names: Sequence[str],
    segment_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        models, sample_rate, clip_seconds = load_raw_wave_openvino_models(
            model_root=model_root,
            num_classes=len(class_names),
        )
    except Exception as exc:
        if env_flag("RAW_WAVE_OPENVINO_STRICT"):
            raise
        print(
            "[WARN] RawWave OpenVINO conversion failed; falling back to PyTorch RawWave. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        return predict_raw_wave(
            model_root=model_root,
            soundscape_files=soundscape_files,
            class_names=class_names,
            segment_batch_size=segment_batch_size,
        )
    print("[INFO] Raw waveform backend: openvino")
    print(f"[INFO] Raw waveform model_root: {model_root}")

    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(
        soundscape_files,
        total=len(soundscape_files),
        desc="RawWave OpenVINO",
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for audio_path in progress:
        audio = cnn_base.load_soundscape_audio(audio_path, sample_rate=sample_rate)
        segments, row_ids = cnn_base.build_segments_for_file(
            audio=audio,
            file_stem=audio_path.stem,
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
        )
        file_preds: List[np.ndarray] = []
        for start in range(0, len(segments), int(segment_batch_size)):
            batch = segments[start:start + int(segment_batch_size)].astype(np.float32, copy=False)
            ensemble = None
            for model in models:
                logits = next(iter(model(batch).values()))
                probs = sigmoid_np(np.asarray(logits, dtype=np.float32))
                ensemble = probs if ensemble is None else ensemble + probs
            file_preds.append((ensemble / float(len(models))).astype(np.float32, copy=False))
            del batch, ensemble
        all_row_ids.extend(row_ids)
        all_preds.append(np.concatenate(file_preds, axis=0).astype(np.float32, copy=False))

    return np.asarray(all_row_ids, dtype=object), np.concatenate(all_preds, axis=0).astype(np.float32, copy=False)


def predict_audiomae_token(
    ckpt_dir: Path,
    model_path: Path,
    soundscape_files: Sequence[Path],
    class_names: Sequence[str],
    batch_size: int,
    device_arg: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import birdclef2026_kaggle_infer_audiomae_token as audiomae_token

    device = audiomae_token.resolve_device(device_arg)
    encoder = audiomae_token.load_audiomae_encoder(ckpt_dir, device=device)
    artifact, head_models = audiomae_token.load_head_ensemble(model_path, class_names=class_names, device=device)
    fallback_prob = float(artifact.get("config", {}).get("fallback_prob", 0.5))
    batch_size = max(1, int(batch_size))
    print(f"[INFO] AudioMAE token device: {device}")
    print(f"[INFO] AudioMAE token ckpt_dir: {ckpt_dir}")
    print(f"[INFO] AudioMAE token model_path: {model_path}")

    row_ids: List[str] = []
    filenames: List[str] = []
    token_batches: List[np.ndarray] = []
    current_features: List[torch.Tensor] = []
    current_row_ids: List[str] = []
    current_filenames: List[str] = []
    progress = tqdm(
        soundscape_files,
        total=len(soundscape_files),
        desc="AudioMAE",
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    with torch.inference_mode():
        for path in progress:
            file_features = audiomae_token.build_file_features(Path(path))
            for win_idx in range(audiomae_token.N_WINDOWS):
                current_features.append(file_features[win_idx])
                end_sec = (win_idx + 1) * 5
                current_row_ids.append(f"{Path(path).stem}_{end_sec}")
                current_filenames.append(Path(path).name)
            while len(current_features) >= batch_size:
                batch = torch.stack(current_features[:batch_size], dim=0).to(device)
                token_batches.append(
                    audiomae_token.features_to_time_tokens(encoder, batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                row_ids.extend(current_row_ids[:batch_size])
                filenames.extend(current_filenames[:batch_size])
                del current_features[:batch_size]
                del current_row_ids[:batch_size]
                del current_filenames[:batch_size]
                del batch
        if current_features:
            batch = torch.stack(current_features, dim=0).to(device)
            token_batches.append(
                audiomae_token.features_to_time_tokens(encoder, batch)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            row_ids.extend(current_row_ids)
            filenames.extend(current_filenames)
            del batch

    tokens = np.concatenate(token_batches, axis=0).astype(np.float32, copy=False)
    pred = audiomae_token.predict_heads(
        models=head_models,
        tokens=tokens,
        device=device,
        batch_size=batch_size,
        fallback_prob=fallback_prob,
    )
    return np.asarray(row_ids, dtype=object), pred.astype(np.float32, copy=False)


def build_temporal_file_tensor(meta_df: pd.DataFrame, spatial_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spatial_tokens.ndim != 3:
        raise ValueError(f"Temporal spatial_tokens must be [rows, tokens, dim], got {spatial_tokens.shape}")
    n_files = meta_df["filename"].nunique()
    n_windows = spatial_infer.N_WINDOWS
    expected_rows = n_files * n_windows
    if len(meta_df) != expected_rows:
        raise ValueError(
            f"Temporal head requires complete 12-window files. "
            f"Got rows={len(meta_df)} files={n_files}, expected rows={expected_rows}."
        )

    row_end = parse_end_seconds(meta_df["row_id"].tolist())
    file_names: List[str] = []
    file_row_indices: List[np.ndarray] = []
    file_tokens: List[np.ndarray] = []
    expected_end = np.arange(5, 65, 5, dtype=np.int64)

    for filename, indices in meta_df.groupby("filename", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(row_end[idx], kind="stable")
        idx_sorted = idx[order]
        ends = row_end[idx_sorted]
        if len(idx_sorted) != n_windows or not np.array_equal(ends, expected_end):
            raise ValueError(f"Temporal head got incomplete or misordered windows for {filename}: {ends.tolist()}")
        file_names.append(str(filename))
        file_row_indices.append(idx_sorted)
        file_tokens.append(spatial_tokens[idx_sorted])

    return (
        np.stack(file_tokens, axis=0).astype(np.float32, copy=False),
        np.asarray(file_names, dtype=object),
        np.stack(file_row_indices, axis=0).astype(np.int64, copy=False),
    )


def predict_temporal_fold(
    fold_artifact: Dict[str, object],
    file_tokens: np.ndarray,
    raw_scores: np.ndarray,
    row_indices_by_file: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    model_artifact = fold_artifact["model"]
    standardizer = fold_artifact["token_standardizer"]
    tokens = ((file_tokens - standardizer["mean"]) / standardizer["std"]).astype(np.float32, copy=False)
    model = PerchTemporalHead(
        token_dim=int(model_artifact["token_dim"]),
        hidden_dim=int(model_artifact["hidden_dim"]),
        num_classes=int(model_artifact["output_dim"]),
        local_blocks=int(model_artifact["local_blocks"]),
        local_kernel_size=int(model_artifact["local_kernel_size"]),
        local_on_raw_tokens=bool(model_artifact["local_on_raw_tokens"]),
        num_layers=int(model_artifact["num_layers"]),
        num_heads=int(model_artifact["num_heads"]),
        dropout=float(model_artifact["dropout"]),
    )
    model.load_state_dict(model_artifact["model_state"], strict=True)
    model.to(device)
    model.eval()

    file_preds: List[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch = torch.from_numpy(tokens[start:start + batch_size]).to(device)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32)
            file_preds.append(pred)
            del batch

    temporal_file_pred = np.concatenate(file_preds, axis=0)
    pred = sigmoid_np(raw_scores).astype(np.float32)
    fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int32)
    flat_indices = row_indices_by_file.reshape(-1)
    flat_temporal = temporal_file_pred.reshape(-1, temporal_file_pred.shape[-1])
    pred[flat_indices[:, None], fitted[None, :]] = flat_temporal[:, fitted]
    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0), int(len(fitted))


def predict_temporal_ensemble(
    artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    flat64_tokens: np.ndarray,
    raw_scores: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if artifact.get("model_type") != "perch_temporal_head":
        raise ValueError(f"Expected model_type=perch_temporal_head, got {artifact.get('model_type')}")
    file_tokens, _, row_indices_by_file = build_temporal_file_tensor(meta_df=meta_df, spatial_tokens=flat64_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Temporal head device: {device}")

    fold_preds = []
    for fold_artifact in artifact["folds"]:
        fold_pred, fitted_count = predict_temporal_fold(
            fold_artifact=fold_artifact,
            file_tokens=file_tokens,
            raw_scores=raw_scores,
            row_indices_by_file=row_indices_by_file,
            batch_size=batch_size,
            device=device,
        )
        fold_preds.append(fold_pred)
        print(
            f"[INFO] Applied temporal {fold_artifact.get('fold_name', 'fold')} "
            f"fitted_classes={fitted_count}"
        )

    pred = np.mean(fold_preds, axis=0).astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def align_prediction_by_row_id(
    source_row_ids: np.ndarray,
    source_pred: np.ndarray,
    target_row_ids: np.ndarray,
    class_names: Sequence[str],
    label: str,
) -> np.ndarray:
    pred_df = pd.concat(
        [pd.DataFrame({"row_id": source_row_ids}), pd.DataFrame(source_pred, columns=class_names)],
        axis=1,
    )
    aligned = pd.DataFrame({"row_id": target_row_ids}).merge(pred_df, on="row_id", how="left", validate="one_to_one")
    if aligned[list(class_names)].isna().any().any():
        raise ValueError(f"{label} predictions are missing rows after row_id alignment.")
    return aligned[list(class_names)].to_numpy(dtype=np.float32)


def topk_presence_stat(pred: np.ndarray, topk: int, method: str) -> np.ndarray:
    k = max(1, min(int(topk), len(pred)))
    if method == "topk_mean":
        return np.sort(pred, axis=0)[-k:].mean(axis=0)
    if method == "topk_max":
        return pred.max(axis=0)
    if method == "topk_geom":
        top = np.sort(np.clip(pred, EPS, 1.0), axis=0)[-k:]
        return np.exp(np.log(top).mean(axis=0))
    if method == "mean":
        return pred.mean(axis=0)
    if method == "mean_max":
        return 0.5 * pred.mean(axis=0) + 0.5 * pred.max(axis=0)
    if method == "topk_mean_max":
        return 0.5 * np.sort(pred, axis=0)[-k:].mean(axis=0) + 0.5 * pred.max(axis=0)
    if method == "noisy_or":
        return 1.0 - np.prod(1.0 - np.clip(pred, 0.0, 1.0), axis=0)
    raise ValueError(f"Unknown file gate mode: {method}")


def file_presence_gate(
    pred: np.ndarray,
    filename: np.ndarray,
    method: str,
    topk: int,
    gamma: float,
    strength: float,
) -> np.ndarray:
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        if len(idx) == 0:
            continue
        p = pred[idx].astype(np.float32, copy=False)
        presence = np.clip(topk_presence_stat(p, topk=topk, method=method), EPS, 1.0)
        gate = (1.0 - float(strength)) + float(strength) * np.power(presence, float(gamma))
        out[idx] = p * gate[None, :]
    return np.clip(out.astype(np.float32, copy=False), 0.0, 1.0)


def apply_final_postprocess(
    pred: np.ndarray,
    filename: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    scale_mode = "none" if args.disable_file_scale else args.file_scale_mode
    scale_value = args.file_scale_value
    if scale_value is None:
        scale_value = float(args.file_scale_topk)

    out = pred.astype(np.float32, copy=True)
    if args.file_gate_mode != "none":
        out = file_presence_gate(
            out,
            filename=filename,
            method=str(args.file_gate_mode),
            topk=int(args.file_gate_topk),
            gamma=float(args.file_gate_gamma),
            strength=float(args.file_gate_strength),
        )
    elif scale_mode != "none":
        out = blend_postprocess.file_level_scale(
            out,
            file_keys=filename,
            mode=scale_mode,
            value=float(scale_value),
        )
    out = blend_postprocess.temporal_smooth(
        out,
        file_keys=filename,
        mode=args.smooth_mode,
        alpha=float(args.smooth_alpha),
    )
    return np.clip(out.astype(np.float32, copy=False), 0.0, 1.0)


def rank01_matrix(pred: np.ndarray) -> np.ndarray:
    out = np.empty_like(pred, dtype=np.float32)
    n = pred.shape[0]
    if n <= 1:
        out.fill(0.0)
        return out
    for class_idx in range(pred.shape[1]):
        values = pred[:, class_idx].astype(np.float64, copy=False)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_values[end] == sorted_values[start]:
                end += 1
            avg_rank = (start + end - 1) / 2.0
            ranks[order[start:end]] = avg_rank
            start = end
        out[:, class_idx] = (ranks / float(n - 1)).astype(np.float32)
    return out


def weighted_logit_blend(branch_preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    fused_logit = None
    for name, weight in weights.items():
        if float(weight) <= 0.0:
            continue
        if name not in branch_preds:
            continue
        term = float(weight) * logit_np(branch_preds[name])
        fused_logit = term if fused_logit is None else fused_logit + term
    if fused_logit is None:
        raise ValueError(f"No positive-weight branches available for weights={weights}")
    return sigmoid_np(fused_logit).astype(np.float32, copy=False)


def weighted_rank_blend(branch_preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    fused = None
    for name, weight in weights.items():
        if float(weight) <= 0.0:
            continue
        if name not in branch_preds:
            continue
        term = float(weight) * rank01_matrix(branch_preds[name])
        fused = term if fused is None else fused + term
    if fused is None:
        raise ValueError(f"No positive-weight branches available for rank blend weights={weights}")
    return np.clip(fused.astype(np.float32, copy=False), 0.0, 1.0)


def apply_family_blend(
    logit_pred: np.ndarray,
    rank_pred: np.ndarray,
    class_names: Sequence[str],
    mode: str,
) -> np.ndarray:
    mode = str(mode)
    if mode == "logit":
        return logit_pred
    if mode == "mixed_rank":
        raise ValueError("apply_family_blend should not receive mixed_rank directly.")
    if mode != "family3":
        raise ValueError(f"Unknown blend mode: {mode}")
    out = logit_pred.astype(np.float32, copy=True)
    son_mask = np.asarray([str(name).startswith("47158son") for name in class_names], dtype=bool)
    out[:, son_mask] = rank_pred[:, son_mask]
    return np.clip(out, 0.0, 1.0)


def save_branch_submission(output_path: Path, row_ids: np.ndarray, pred: np.ndarray, class_names: Sequence[str]) -> None:
    submission = pd.concat(
        [pd.DataFrame({"row_id": row_ids}), pd.DataFrame(pred, columns=class_names)],
        axis=1,
    )
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved branch submission: {output_path}")


def main() -> None:
    total_start = time.perf_counter()
    timing: Dict[str, float] = {}
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, int(args.runtime_num_threads)))

    competition_root = Path(args.competition_root)
    sample_submission_path = (
        spatial_infer.resolve_user_path(args.sample_submission_path, competition_root)
        if args.sample_submission_path
        else competition_root / "sample_submission.csv"
    )
    taxonomy_path = (
        spatial_infer.resolve_user_path(args.taxonomy_path, competition_root)
        if args.taxonomy_path
        else competition_root / "taxonomy.csv"
    )
    soundscapes_dir = (
        spatial_infer.resolve_user_path(args.soundscapes_dir, competition_root)
        if args.soundscapes_dir
        else competition_root / ("train_soundscapes" if args.debug else "test_soundscapes")
    )
    perch_dir = spatial_infer.resolve_user_path(args.perch_dir, competition_root)
    output_path = Path(args.output_path)

    class_names = spatial_infer.load_class_names(sample_submission_path)
    onnx_path = discover_onnx_path(args.perch_onnx_path, competition_root=competition_root)
    perch_lr_path = discover_unique_file(
        args.perch_lr_model_path,
        candidates=[],
        filename="perch_context_logreg_artifacts.joblib",
        label="Perch context LogReg artifact",
        competition_root=competition_root,
    )
    mamba_path = discover_unique_file(
        args.mamba_model_path,
        candidates=["mamba/perch_spatial_mamba_artifacts.joblib"],
        filename="perch_spatial_mamba_artifacts.joblib",
        label="Perch Mamba artifact",
        competition_root=competition_root,
    )
    attention_path = discover_unique_file(
        args.attention_model_path,
        candidates=["attention/perch_spatial_mamba_artifacts.joblib"],
        filename="perch_spatial_mamba_artifacts.joblib",
        label="Perch Attention artifact",
        competition_root=competition_root,
    )
    temporal_path = resolve_optional_path(args.temporal_model_path, competition_root=competition_root)
    ssm_path = resolve_optional_path(args.ssm_model_path, competition_root=competition_root)
    stage3_model_root = stage3_infer.discover_model_root(args.stage3_model_root)
    base_cnn_model_root = stage3_infer.discover_model_root(args.base_cnn_model_root) if args.base_cnn_model_root else None
    raw_wave_model_root = discover_raw_wave_model_root(args.raw_wave_model_root, competition_root=competition_root)
    audiomae_token_model_path = resolve_optional_path(args.audiomae_token_model_path, competition_root=competition_root)
    audiomae_ckpt_dir = resolve_optional_path(args.audiomae_ckpt_dir, competition_root=competition_root) if audiomae_token_model_path is not None else None

    perch_lr_artifact = load_artifact(perch_lr_path, class_names=class_names, label="Perch LR")
    mamba_artifact = load_artifact(mamba_path, class_names=class_names, label="Mamba")
    attention_artifact = load_artifact(attention_path, class_names=class_names, label="Attention")
    temporal_artifact = (
        load_artifact(temporal_path, class_names=class_names, label="Temporal") if temporal_path is not None else None
    )
    ssm_artifact = load_artifact(ssm_path, class_names=class_names, label="SSM") if ssm_path is not None else None
    freq_pools = required_freq_pools(mamba_artifact, attention_artifact)
    if temporal_artifact is not None and "flat64" not in freq_pools:
        freq_pools.append("flat64")

    soundscape_files = spatial_infer.list_soundscape_files(soundscapes_dir, debug=args.debug, debug_limit=args.debug_limit)
    if not soundscape_files:
        raise FileNotFoundError(f"No .ogg files found under {soundscapes_dir}")

    weights = resolve_default_weights(
        args,
        temporal_enabled=temporal_artifact is not None,
        ssm_enabled=ssm_artifact is not None,
        raw_wave_enabled=raw_wave_model_root is not None,
        audiomae_token_enabled=audiomae_token_model_path is not None,
        base_cnn_enabled=base_cnn_model_root is not None,
    )
    mamba_tta_offsets = parse_float_list(args.mamba_tta_offsets)
    stage3_tta_offsets = parse_float_list(args.stage3_tta_offsets)
    base_cnn_tta_offsets = parse_float_list(args.base_cnn_tta_offsets)

    print("[INFO] Unified Perch+Stage3 ensemble inference")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] files: {len(soundscape_files)}")
    print(f"[INFO] perch_dir: {perch_dir}")
    print(f"[INFO] perch_onnx_path: {onnx_path}")
    print(f"[INFO] perch_lr_path: {perch_lr_path}")
    print(f"[INFO] mamba_path: {mamba_path}")
    print(f"[INFO] attention_path: {attention_path}")
    print(f"[INFO] temporal_path: {temporal_path}")
    print(f"[INFO] ssm_path: {ssm_path}")
    print(f"[INFO] stage3_model_root: {stage3_model_root}")
    print(f"[INFO] stage3_backend: {args.stage3_backend}")
    print(f"[INFO] base_cnn_model_root: {base_cnn_model_root}")
    print(f"[INFO] raw_wave_model_root: {raw_wave_model_root}")
    print(f"[INFO] raw_wave_backend: {args.raw_wave_backend}")
    print(f"[INFO] audiomae_token_model_path: {audiomae_token_model_path}")
    print(f"[INFO] audiomae_ckpt_dir: {audiomae_ckpt_dir}")
    print(f"[INFO] weights: {weights}")
    print(f"[INFO] mamba_tta_offsets: {mamba_tta_offsets}")
    print(f"[INFO] stage3_tta_offsets: {stage3_tta_offsets}")
    print(f"[INFO] base_cnn_tta_offsets: {base_cnn_tta_offsets}")
    print(
        "[INFO] postprocess: "
        f"file_scale_mode={('none' if args.disable_file_scale else args.file_scale_mode)} "
        f"file_scale_value={args.file_scale_value if args.file_scale_value is not None else args.file_scale_topk} "
        f"smooth_mode={args.smooth_mode} smooth_alpha={args.smooth_alpha} "
        f"blend_mode={args.blend_mode} rank_blend_alpha_logit={args.rank_blend_alpha_logit}"
    )
    print(f"[INFO] output_path: {output_path}")

    bc_labels = spatial_infer.load_perch_label_table(perch_dir=perch_dir, onnx_path=onnx_path)
    bc_indices, mapped_bc_indices, mapping = spatial_infer.build_competition_mapping(
        primary_labels=class_names,
        taxonomy_path=taxonomy_path,
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = spatial_infer.build_selected_proxy_targets(
        primary_labels=class_names,
        mapping=mapping,
        bc_labels=bc_labels,
    )

    step_start = time.perf_counter()
    meta_df, spatial_tokens_by_pool, raw_scores, embedding = infer_perch_shared_onnx(
        paths=soundscape_files,
        onnx_path=onnx_path,
        n_classes=len(class_names),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        proxy_reduce=args.proxy_reduce,
        num_threads=args.runtime_num_threads,
        batch_files=args.batch_files,
        freq_pools=freq_pools,
    )
    timing["perch_shared_onnx"] = time.perf_counter() - step_start

    row_ids = meta_df["row_id"].to_numpy()
    filenames = meta_df["filename"].to_numpy()
    step_start = time.perf_counter()
    perch_lr_pred = predict_context_logreg_ensemble(
        artifact=perch_lr_artifact,
        meta_df=meta_df,
        scores_full_raw=raw_scores,
        emb_full=embedding,
    )
    timing["perch_lr_head"] = time.perf_counter() - step_start

    mamba_pool = str(mamba_artifact.get("config", {}).get("freq_pool", "mean"))
    attention_pool = str(attention_artifact.get("config", {}).get("freq_pool", "mean"))
    step_start = time.perf_counter()
    mamba_pred = spatial_infer.predict_ensemble(
        artifact=mamba_artifact,
        spatial_tokens=spatial_tokens_by_pool[mamba_pool],
        raw_scores=raw_scores,
        embedding=embedding,
        batch_size=args.batch_files * spatial_infer.N_WINDOWS,
    )
    timing["mamba_head"] = time.perf_counter() - step_start
    if mamba_tta_offsets:
        step_start = time.perf_counter()
        mamba_tta_preds: List[np.ndarray] = []
        for offset in mamba_tta_offsets:
            if abs(float(offset)) < 1e-8:
                mamba_tta_preds.append(mamba_pred)
                continue
            _, shifted_tokens_by_pool, _, _ = infer_perch_shared_onnx(
                paths=soundscape_files,
                onnx_path=onnx_path,
                n_classes=len(class_names),
                mapped_pos=mapped_pos,
                mapped_bc_indices=mapped_bc_indices,
                proxy_pos_to_bc=proxy_pos_to_bc,
                proxy_reduce=args.proxy_reduce,
                num_threads=args.runtime_num_threads,
                batch_files=args.batch_files,
                freq_pools=[mamba_pool],
                clip_offset_seconds=float(offset),
            )
            shifted_pred = spatial_infer.predict_ensemble(
                artifact=mamba_artifact,
                spatial_tokens=shifted_tokens_by_pool[mamba_pool],
                raw_scores=raw_scores,
                embedding=embedding,
                batch_size=args.batch_files * spatial_infer.N_WINDOWS,
            )
            mamba_tta_preds.append(shifted_pred)
        mamba_pred = np.mean(np.stack(mamba_tta_preds, axis=0), axis=0).astype(np.float32, copy=False)
        timing["mamba_tta_extra"] = time.perf_counter() - step_start
    step_start = time.perf_counter()
    attention_pred = spatial_infer.predict_ensemble(
        artifact=attention_artifact,
        spatial_tokens=spatial_tokens_by_pool[attention_pool],
        raw_scores=raw_scores,
        embedding=embedding,
        batch_size=args.batch_files * spatial_infer.N_WINDOWS,
    )
    timing["attention_head"] = time.perf_counter() - step_start

    temporal_pred = None
    if temporal_artifact is not None:
        step_start = time.perf_counter()
        temporal_pred = predict_temporal_ensemble(
            artifact=temporal_artifact,
            meta_df=meta_df,
            flat64_tokens=spatial_tokens_by_pool["flat64"],
            raw_scores=raw_scores,
            batch_size=args.batch_files,
        )
        timing["temporal_head"] = time.perf_counter() - step_start

    ssm_pred = None
    if ssm_artifact is not None:
        step_start = time.perf_counter()
        ssm_pred = predict_ssm_ensemble(
            artifact=ssm_artifact,
            meta_df=meta_df,
            embedding=embedding,
            raw_scores=raw_scores,
            batch_size=args.batch_files,
        )
        timing["ssm_head"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    stage3_predict_fn = predict_stage3_cnn_openvino if args.stage3_backend == "openvino" else predict_stage3_cnn
    stage3_row_ids, stage3_pred_raw = stage3_predict_fn(
        model_root=stage3_model_root,
        soundscape_files=soundscape_files,
        class_names=class_names,
        competition_root=competition_root,
        debug=args.debug,
        debug_limit=args.debug_limit,
        segment_batch_size=args.stage3_segment_batch_size,
        seed=args.seed,
        tta_offsets=stage3_tta_offsets,
    )
    timing["stage3_cnn"] = time.perf_counter() - step_start
    stage3_pred = align_prediction_by_row_id(
        source_row_ids=stage3_row_ids,
        source_pred=stage3_pred_raw,
        target_row_ids=row_ids,
        class_names=class_names,
        label="Stage3 CNN",
    )
    base_cnn_pred = None
    if base_cnn_model_root is not None:
        step_start = time.perf_counter()
        base_cnn_row_ids, base_cnn_pred_raw = predict_stage3_cnn(
            model_root=base_cnn_model_root,
            soundscape_files=soundscape_files,
            class_names=class_names,
            competition_root=competition_root,
            debug=args.debug,
            debug_limit=args.debug_limit,
            segment_batch_size=args.base_cnn_segment_batch_size,
            seed=args.seed,
            tta_offsets=base_cnn_tta_offsets,
            label="Base CNN",
        )
        timing["base_cnn"] = time.perf_counter() - step_start
        base_cnn_pred = align_prediction_by_row_id(
            source_row_ids=base_cnn_row_ids,
            source_pred=base_cnn_pred_raw,
            target_row_ids=row_ids,
            class_names=class_names,
            label="Base CNN",
        )
    raw_wave_pred = None
    if raw_wave_model_root is not None:
        step_start = time.perf_counter()
        raw_wave_predict_fn = predict_raw_wave_openvino if args.raw_wave_backend == "openvino" else predict_raw_wave
        raw_wave_row_ids, raw_wave_pred_raw = raw_wave_predict_fn(
            model_root=raw_wave_model_root,
            soundscape_files=soundscape_files,
            class_names=class_names,
            segment_batch_size=args.raw_wave_segment_batch_size,
        )
        timing["raw_wave"] = time.perf_counter() - step_start
        raw_wave_pred = align_prediction_by_row_id(
            source_row_ids=raw_wave_row_ids,
            source_pred=raw_wave_pred_raw,
            target_row_ids=row_ids,
            class_names=class_names,
            label="Raw waveform",
        )

    audiomae_token_pred = None
    if audiomae_token_model_path is not None:
        if audiomae_ckpt_dir is None:
            raise ValueError("--audiomae-ckpt-dir is required when --audiomae-token-model-path is set")
        step_start = time.perf_counter()
        audiomae_row_ids, audiomae_pred_raw = predict_audiomae_token(
            ckpt_dir=audiomae_ckpt_dir,
            model_path=audiomae_token_model_path,
            soundscape_files=soundscape_files,
            class_names=class_names,
            batch_size=args.audiomae_token_batch_size,
            device_arg=args.audiomae_token_device,
        )
        timing["audiomae_token"] = time.perf_counter() - step_start
        audiomae_token_pred = align_prediction_by_row_id(
            source_row_ids=audiomae_row_ids,
            source_pred=audiomae_pred_raw,
            target_row_ids=row_ids,
            class_names=class_names,
            label="AudioMAE token",
        )

    step_start = time.perf_counter()
    branch_preds: Dict[str, np.ndarray] = {
        "perch_lr": perch_lr_pred,
        "mamba": mamba_pred,
        "stage3": stage3_pred,
        "attention": attention_pred,
    }
    if raw_wave_pred is not None:
        branch_preds["raw_wave"] = raw_wave_pred
    if temporal_pred is not None:
        branch_preds["temporal"] = temporal_pred
    if ssm_pred is not None:
        branch_preds["ssm"] = ssm_pred
    if audiomae_token_pred is not None:
        branch_preds["audiomae_token"] = audiomae_token_pred
    if base_cnn_pred is not None:
        branch_preds["base_cnn"] = base_cnn_pred

    logit_fused = weighted_logit_blend(branch_preds, weights)
    if args.blend_mode == "logit":
        fused = logit_fused
    else:
        rank_fused = weighted_rank_blend(branch_preds, weights)
        if args.blend_mode == "mixed_rank":
            alpha = float(args.rank_blend_alpha_logit)
            fused = np.clip(alpha * logit_fused + (1.0 - alpha) * rank_fused, 0.0, 1.0).astype(np.float32)
        else:
            fused = apply_family_blend(
                logit_pred=logit_fused,
                rank_pred=rank_fused,
                class_names=class_names,
                mode=args.blend_mode,
            )
    fused = apply_final_postprocess(fused, filename=filenames, args=args)
    timing["fusion_postprocess"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.concat(
        [pd.DataFrame({"row_id": row_ids}), pd.DataFrame(fused, columns=class_names)],
        axis=1,
    )
    submission.to_csv(output_path, index=False)
    timing["save_submission"] = time.perf_counter() - step_start
    timing["total"] = time.perf_counter() - total_start
    print("[INFO] Timing summary:")
    for key, value in timing.items():
        print(f"[INFO]   {key}: {value:.1f}s")
    if len(soundscape_files) > 0:
        print(f"[INFO]   seconds_per_file_total: {timing['total'] / len(soundscape_files):.2f}s")
    print(f"[INFO] Saved unified submission to {output_path}")
    print(submission.head())

    if args.save_branch_submissions:
        stem = output_path.with_suffix("")
        save_branch_submission(stem.with_name(stem.name + "_perch_lr.csv"), row_ids, perch_lr_pred, class_names)
        save_branch_submission(stem.with_name(stem.name + "_mamba.csv"), row_ids, mamba_pred, class_names)
        save_branch_submission(stem.with_name(stem.name + "_attention.csv"), row_ids, attention_pred, class_names)
        if temporal_pred is not None:
            save_branch_submission(stem.with_name(stem.name + "_temporal.csv"), row_ids, temporal_pred, class_names)
        if ssm_pred is not None:
            save_branch_submission(stem.with_name(stem.name + "_ssm.csv"), row_ids, ssm_pred, class_names)
        save_branch_submission(stem.with_name(stem.name + "_stage3.csv"), row_ids, stage3_pred, class_names)
        if base_cnn_pred is not None:
            save_branch_submission(stem.with_name(stem.name + "_base_cnn.csv"), row_ids, base_cnn_pred, class_names)
        if raw_wave_pred is not None:
            save_branch_submission(stem.with_name(stem.name + "_raw_wave.csv"), row_ids, raw_wave_pred, class_names)
        if audiomae_token_pred is not None:
            save_branch_submission(stem.with_name(stem.name + "_audiomae_token.csv"), row_ids, audiomae_token_pred, class_names)


if __name__ == "__main__":
    main()
