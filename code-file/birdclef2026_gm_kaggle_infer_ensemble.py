from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Failed to load image Python extension:.*")


@dataclass
class InferConfig:
    competition_root: str = "/kaggle/input/competitions/birdclef-2026"
    output_path: str = "/kaggle/working/submission.csv"
    model_roots: List[str] = None
    soundscapes_dir: str = ""
    debug: bool = False
    debug_limit: int = 4
    segment_batch_size: int = 12
    tta_offsets: List[float] = None
    smoothing_kernel: List[float] = None
    soundscape_top_k: int = 0
    seed: int = 2026


@dataclass
class ResolvedModelSpec:
    model_root: Path
    run_kind: str
    checkpoint_name: str
    model_name: str
    sample_rate: int
    clip_seconds: float
    image_height: int
    image_width: int
    dropout: float
    drop_path: float
    head_type: str
    config_source: str
    student_run_dir: Optional[Path]


DEFAULT_MODEL_CONFIG = {
    "sample_rate": 32000,
    "clip_seconds": 5.0,
    "image_height": 256,
    "image_width": 320,
    "dropout": 0.2,
    "drop_path": 0.1,
    "head_type": "linear",
}


def parse_multi_string_args(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    items = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                items.append(part)
    deduped = []
    seen = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def parse_float_list(text: str, default: List[float]) -> List[float]:
    if text is None:
        return list(default)
    text = str(text).strip()
    if not text:
        return list(default)
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values if values else list(default)


def parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Kaggle inference for birdclef2026_gm.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument(
        "--model-root",
        type=str,
        action="append",
        default=None,
        help="Model run directory. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument(
        "--tta-offsets",
        type=str,
        default="0",
        help="Comma-separated time shifts in seconds for row-level TTA, e.g. '0,-1.25,1.25'.",
    )
    parser.add_argument(
        "--smoothing-kernel",
        type=str,
        default="",
        help="Optional temporal smoothing kernel, e.g. '0.1,0.8,0.1'.",
    )
    parser.add_argument(
        "--soundscape-top-k",
        type=int,
        default=0,
        help="Optional soundscape-level scaling using the mean of top-k chunk probabilities per class.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return InferConfig(
        competition_root=args.competition_root,
        output_path=args.output_path,
        model_roots=parse_multi_string_args(args.model_root),
        soundscapes_dir=args.soundscapes_dir,
        debug=args.debug,
        debug_limit=args.debug_limit,
        segment_batch_size=args.segment_batch_size,
        tta_offsets=parse_float_list(args.tta_offsets, default=[0.0]),
        smoothing_kernel=parse_float_list(args.smoothing_kernel, default=[]),
        soundscape_top_k=args.soundscape_top_k,
        seed=args.seed,
    )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except TypeError:
            torch.use_deterministic_algorithms(True, warn_only=False)


def get_determinism_status() -> Dict[str, bool]:
    deterministic_algorithms = False
    cudnn_deterministic = False
    cudnn_benchmark = False
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    if hasattr(torch.backends, "cudnn"):
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    return {
        "deterministic_algorithms": deterministic_algorithms,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_search_roots(model_root: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []
    if model_root is not None:
        roots.append(model_root)
        roots.extend(model_root.parents)
    roots.extend(
        [
            Path.cwd(),
            Path("/kaggle/working"),
            Path("/kaggle/input/models"),
            Path("/kaggle/input"),
        ]
    )

    deduped = []
    seen = set()
    for root in roots:
        root = Path(root)
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def resolve_existing_path(path_str: str, model_root: Optional[Path] = None) -> Optional[Path]:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    for root in build_search_roots(model_root=model_root):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def resolve_model_root_path(path_str: str) -> Path:
    resolved = resolve_existing_path(path_str)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"Explicit model root does not exist: {path_str}")


def auto_discover_best_model_root() -> Path:
    candidates = []
    for root in build_search_roots():
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            candidate_root = config_path.parent
            stage3_paths = sorted(candidate_root.glob("fold_*/stage3_best.pth"))
            stage2_paths = sorted(candidate_root.glob("fold_*/stage2_fold*_best.pth"))

            if stage3_paths:
                candidates.append((candidate_root, 2, len(stage3_paths), config_path.stat().st_mtime))
            elif stage2_paths:
                candidates.append((candidate_root, 1, len(stage2_paths), config_path.stat().st_mtime))

    if not candidates:
        raise FileNotFoundError(
            "No candidate model directory found under /kaggle/input. "
            "Please upload the trained model folder or pass --model-root."
        )

    candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    chosen_root = candidates[0][0]
    print(f"[INFO] Auto-discovered model root: {chosen_root}")
    return chosen_root


def discover_model_roots(explicit_model_roots: List[str]) -> List[Path]:
    if explicit_model_roots:
        return [resolve_model_root_path(path_str) for path_str in explicit_model_roots]
    return [auto_discover_best_model_root()]


def detect_run_kind(model_root: Path) -> Tuple[str, str]:
    stage3_paths = sorted(model_root.glob("fold_*/stage3_best.pth"))
    if stage3_paths:
        return "stage3", "stage3_best.pth"

    stage2_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
    if stage2_paths:
        return "stage2", "stage2_fold*_best.pth"

    raise FileNotFoundError(f"No supported fold checkpoints found under {model_root}")


def infer_model_name_from_run_name(run_name: str) -> Optional[str]:
    patterns = [
        r"^\d{8}_\d{6}_(.+)_stage3_pseudo$",
        r"^\d{8}_\d{6}_(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, run_name)
        if match:
            return match.group(1)
    return None


def resolve_student_run_config(model_root: Path, run_cfg: dict) -> Tuple[Optional[Path], Optional[dict]]:
    candidate_paths = []
    for key in ["student_run_dir"]:
        value = run_cfg.get(key)
        if value:
            candidate_paths.append(str(value))

    metrics_path = model_root / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        value = metrics.get("student_run_dir")
        if value:
            candidate_paths.append(str(value))

    for candidate_path in candidate_paths:
        resolved = resolve_existing_path(candidate_path, model_root=model_root)
        if resolved is None or not resolved.is_dir():
            continue
        config_path = resolved / "config.json"
        if config_path.exists():
            return resolved, load_json(config_path)

    return None, None


def build_fallback_model_config(model_root: Path, run_cfg: dict) -> dict:
    model_name = str(run_cfg.get("model_name", "")).strip() or infer_model_name_from_run_name(model_root.name)
    if model_name is None:
        student_run_name = Path(str(run_cfg.get("student_run_dir", ""))).name
        model_name = infer_model_name_from_run_name(student_run_name)
    if model_name is None:
        raise ValueError(
            "Could not infer model_name for this stage3 run. "
            "Please upload the original stage2 run directory alongside the stage3 run, or pass a model root with config."
        )

    fallback = dict(DEFAULT_MODEL_CONFIG)
    fallback["model_name"] = model_name
    for key in ["sample_rate", "clip_seconds", "image_height", "image_width", "dropout", "drop_path", "head_type"]:
        if key in run_cfg:
            fallback[key] = run_cfg[key]
    return fallback


def resolve_model_spec(model_root: Path) -> ResolvedModelSpec:
    run_kind, checkpoint_name = detect_run_kind(model_root)
    run_cfg = load_json(model_root / "config.json")

    student_run_dir = None
    if run_kind == "stage2":
        model_cfg = run_cfg
        config_source = "stage2 config.json"
    else:
        student_run_dir, student_cfg = resolve_student_run_config(model_root, run_cfg)
        if student_cfg is not None:
            model_cfg = student_cfg
            config_source = f"student config from {student_run_dir}"
        else:
            model_cfg = build_fallback_model_config(model_root, run_cfg)
            config_source = "stage3 fallback defaults"
            print(
                "[WARN] Could not resolve student_run_dir for this stage3 run. "
                "Falling back to inferred model_name + default audio/image settings. "
                "For the safest Kaggle submission, upload the original stage2 run folder alongside the stage3 run."
            )

    required = ["model_name", "sample_rate", "clip_seconds", "image_height", "image_width", "dropout", "drop_path"]
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise KeyError(f"Missing required model config keys for inference: {missing}")

    return ResolvedModelSpec(
        model_root=model_root,
        run_kind=run_kind,
        checkpoint_name=checkpoint_name,
        model_name=str(model_cfg["model_name"]),
        sample_rate=int(model_cfg["sample_rate"]),
        clip_seconds=float(model_cfg["clip_seconds"]),
        image_height=int(model_cfg["image_height"]),
        image_width=int(model_cfg["image_width"]),
        dropout=float(model_cfg["dropout"]),
        drop_path=float(model_cfg["drop_path"]),
        head_type=str(model_cfg.get("head_type", "linear")),
        config_source=config_source,
        student_run_dir=student_run_dir,
    )


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


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def linear_resample(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    if len(audio) <= 1:
        return np.zeros(int(round(len(audio) * target_sr / max(original_sr, 1))), dtype=np.float32)

    duration = (len(audio) - 1) / float(original_sr)
    target_len = max(1, int(round(len(audio) * target_sr / float(original_sr))))
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=True, dtype=np.float64)
    new_times = np.linspace(0.0, duration, num=target_len, endpoint=True, dtype=np.float64)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq / 700.0))


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, num=(n_fft // 2) + 1, dtype=np.float32)
    mel_edges = np.linspace(
        hz_to_mel(np.array([f_min], dtype=np.float32))[0],
        hz_to_mel(np.array([f_max], dtype=np.float32))[0],
        num=n_mels + 2,
        dtype=np.float32,
    )
    hz_edges = mel_to_hz(mel_edges)

    filterbank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for mel_idx in range(n_mels):
        left = hz_edges[mel_idx]
        center = hz_edges[mel_idx + 1]
        right = hz_edges[mel_idx + 2]
        if center <= left or right <= center:
            continue

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        filterbank[mel_idx, left_mask] = (fft_freqs[left_mask] - left) / max(center - left, 1e-8)
        filterbank[mel_idx, right_mask] = (right - fft_freqs[right_mask]) / max(right - center, 1e-8)

    return torch.from_numpy(filterbank)


def power_to_db(spec: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    spec = torch.clamp(spec, min=1e-10)
    db = 10.0 * torch.log10(spec)
    max_db = db.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp(db, min=max_db - top_db)


class SpectrogramRenderer:
    def __init__(self, sample_rate: int, image_height: int, image_width: int):
        self.sample_rate = sample_rate
        self.image_height = image_height
        self.image_width = image_width
        self.top_db = 80.0
        self.specs = [
            self._build_spec(n_fft=1024, hop_length=256, n_mels=128, f_min=20.0, f_max=16000.0),
            self._build_spec(n_fft=2048, hop_length=512, n_mels=128, f_min=20.0, f_max=12000.0),
            self._build_spec(n_fft=4096, hop_length=1024, n_mels=128, f_min=20.0, f_max=8000.0),
        ]

    def _build_spec(self, n_fft: int, hop_length: int, n_mels: int, f_min: float, f_max: float) -> Dict[str, torch.Tensor]:
        return {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": torch.hann_window(n_fft),
            "mel_filter": build_mel_filterbank(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            ),
        }

    def _mel_spectrogram_batch(self, waveform_batch: torch.Tensor, spec_cfg: Dict[str, torch.Tensor]) -> torch.Tensor:
        stft = torch.stft(
            waveform_batch,
            n_fft=int(spec_cfg["n_fft"]),
            hop_length=int(spec_cfg["hop_length"]),
            win_length=int(spec_cfg["n_fft"]),
            window=spec_cfg["window"],
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power_spec = stft.abs().pow(2.0)
        mel_filter = spec_cfg["mel_filter"].to(power_spec.dtype)
        mel_spec = torch.einsum("mf,bft->bmt", mel_filter, power_spec)
        return power_to_db(mel_spec, top_db=self.top_db)

    def __call__(self, segments: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(segments).float()
        channels = []
        for spec_cfg in self.specs:
            mel = self._mel_spectrogram_batch(x, spec_cfg)
            mel = mel.unsqueeze(1)
            mel = F.interpolate(
                mel,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
            channels.append(mel)
        image = torch.cat(channels, dim=1)
        mins = image.amin(dim=(-2, -1), keepdim=True)
        maxs = image.amax(dim=(-2, -1), keepdim=True)
        image = (image - mins) / (maxs - mins + 1e-6)
        image = (image - 0.5) / 0.5
        return image


class LocalSequenceBlock(nn.Module):
    def __init__(self, in_features: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(in_features)
        self.pointwise_in = nn.Linear(in_features, in_features)
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_features,
            bias=True,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.pointwise_out = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pointwise_in(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pointwise_out(x)
        return residual + x


class CSIROHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        hidden_features = max(in_features // 2, 64)
        self.fusion = nn.Sequential(
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)
        x = self.pool(x.transpose(1, 2)).flatten(1)
        return self.out_head(x)


class BirdCLEFNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float, drop_path: float, head_type: str = "linear"):
        super().__init__()
        self.head_type = str(head_type)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="avg" if self.head_type == "linear" else "",
            drop_path_rate=drop_path,
        )
        self.dropout = nn.Dropout(dropout)
        if self.head_type == "linear":
            self.head = nn.Linear(self.backbone.num_features, num_classes)
        elif self.head_type == "csiro_conv_v1":
            self.head = CSIROHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def _forward_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if isinstance(features, dict):
            for key in ["x", "feat", "features", "last_hidden_state"]:
                if key in features:
                    features = features[key]
                    break
            else:
                raise TypeError(f"Unsupported backbone feature dict keys: {list(features.keys())}")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Unsupported backbone feature type: {type(features)}")
        return features

    def _flatten_feature_sequence(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3:
            if features.shape[-1] == self.backbone.num_features:
                return features
            if features.shape[1] == self.backbone.num_features:
                return features.transpose(1, 2)
            raise ValueError(f"Unsupported 3D backbone feature shape: {tuple(features.shape)}")
        if features.ndim == 2:
            return features.unsqueeze(1)
        raise ValueError(f"Unsupported backbone feature shape: {tuple(features.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "linear":
            features = self.backbone(x)
            return self.head(self.dropout(features))
        features = self._forward_backbone_features(x)
        feature_sequence = self._flatten_feature_sequence(features)
        return self.head(feature_sequence)


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
        return checkpoint_obj["model"]
    return checkpoint_obj


def load_models(spec: ResolvedModelSpec, num_classes: int, device: torch.device):
    if spec.run_kind == "stage3":
        fold_paths = sorted(spec.model_root.glob("fold_*/stage3_best.pth"))
    else:
        fold_paths = sorted(spec.model_root.glob("fold_*/stage2_fold*_best.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No {spec.run_kind} fold checkpoints found under {spec.model_root}")

    models = []
    for fold_path in fold_paths:
        model = BirdCLEFNet(
            model_name=spec.model_name,
            num_classes=num_classes,
            dropout=spec.dropout,
            drop_path=spec.drop_path,
            head_type=spec.head_type,
        )
        checkpoint_obj = torch.load(fold_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint_obj)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded {spec.run_kind} checkpoint: {fold_path}")
    return models


@dataclass
class ModelBundle:
    root: Path
    run_kind: str
    model_name: str
    sample_rate: int
    clip_seconds: float
    image_height: int
    image_width: int
    renderer: SpectrogramRenderer
    models: List[nn.Module]
    config_source: str
    student_run_dir: Optional[Path]


@lru_cache(maxsize=32)
def read_audio_file(path: str):
    return sf.read(path, dtype="float32")


@lru_cache(maxsize=256)
def load_soundscape_audio_cached(path_str: str, sample_rate: int) -> np.ndarray:
    audio, sr = read_audio_file(path_str)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = linear_resample(audio, sr, sample_rate)
    return audio


def load_soundscape_audio(path: Path, sample_rate: int) -> np.ndarray:
    return np.asarray(load_soundscape_audio_cached(str(path), sample_rate), dtype=np.float32)


def extract_centered_window(audio: np.ndarray, sample_rate: int, center_sec: float, clip_seconds: float) -> np.ndarray:
    clip_len = int(round(sample_rate * clip_seconds))
    start = int(round((center_sec - clip_seconds / 2.0) * sample_rate))
    end = start + clip_len
    pad_left = max(0, -start)
    pad_right = max(0, end - len(audio))
    if pad_left > 0 or pad_right > 0:
        clipped = audio[max(start, 0) : min(end, len(audio))]
        clipped = np.pad(clipped, (pad_left, pad_right), mode="constant")
    else:
        clipped = audio[start:end]
    if len(clipped) != clip_len:
        clipped = np.pad(clipped[:clip_len], (0, max(0, clip_len - len(clipped))), mode="constant")
    return clipped.astype(np.float32)


def build_segments_for_file(
    audio: np.ndarray,
    file_stem: str,
    sample_rate: int,
    clip_seconds: float,
    tta_offsets: List[float],
):
    segments = []
    row_ids = []
    row_indices = []
    for end_sec in range(5, 61, 5):
        row_center_sec = end_sec - 2.5
        row_ids.append(f"{file_stem}_{end_sec}")
        for offset_sec in tta_offsets:
            segment = extract_centered_window(
                audio=audio,
                sample_rate=sample_rate,
                center_sec=row_center_sec + offset_sec,
                clip_seconds=clip_seconds,
            )
            segments.append(segment)
            row_indices.append(len(row_ids) - 1)
    return np.stack(segments), row_ids, np.asarray(row_indices, dtype=np.int64)


def list_soundscape_files(test_dir: Path, debug: bool, debug_limit: int) -> List[Path]:
    files = sorted(path for path in test_dir.iterdir() if path.suffix == ".ogg")
    if debug:
        files = files[:debug_limit]
    return files


def predict_file_segments(
    segments: np.ndarray,
    models: List[nn.Module],
    renderer: SpectrogramRenderer,
    device: torch.device,
    segment_batch_size: int,
) -> np.ndarray:
    all_preds = []
    for start in range(0, len(segments), segment_batch_size):
        batch_segments = segments[start : start + segment_batch_size]
        images = renderer(batch_segments).to(device)
        with torch.inference_mode():
            ensemble = None
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                ensemble = probs if ensemble is None else ensemble + probs
            ensemble = ensemble / len(models)
        all_preds.append(ensemble.cpu().numpy())
        del images, ensemble
    return np.concatenate(all_preds, axis=0)


def aggregate_tta_predictions(window_preds: np.ndarray, row_indices: np.ndarray, n_rows: int) -> np.ndarray:
    prediction_sum = np.zeros((n_rows, window_preds.shape[1]), dtype=np.float32)
    counts = np.zeros((n_rows, 1), dtype=np.float32)
    for window_idx, row_idx in enumerate(row_indices):
        prediction_sum[row_idx] += window_preds[window_idx]
        counts[row_idx, 0] += 1.0
    return prediction_sum / np.clip(counts, 1.0, None)


def apply_temporal_smoothing(pred_matrix: np.ndarray, kernel: List[float]) -> np.ndarray:
    if not kernel or len(kernel) <= 1:
        return pred_matrix
    if len(kernel) % 2 == 0:
        raise ValueError("Smoothing kernel length must be odd.")
    kernel_array = np.asarray(kernel, dtype=np.float32)
    kernel_sum = float(kernel_array.sum())
    if abs(kernel_sum) > 1e-8:
        kernel_array = kernel_array / kernel_sum
    radius = len(kernel_array) // 2
    padded = np.pad(pred_matrix, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.zeros_like(pred_matrix)
    for row_idx in range(pred_matrix.shape[0]):
        smoothed[row_idx] = (padded[row_idx : row_idx + len(kernel_array)] * kernel_array[:, None]).sum(axis=0)
    return smoothed


def apply_soundscape_postprocess(pred_matrix: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return pred_matrix
    top_k = min(top_k, pred_matrix.shape[0])
    strength = np.sort(pred_matrix, axis=0)[-top_k:].mean(axis=0, keepdims=True)
    return pred_matrix * strength


def load_model_bundle(model_root: Path, class_names: List[str], device: torch.device) -> ModelBundle:
    spec = resolve_model_spec(model_root)
    renderer = SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )
    models = load_models(
        spec=spec,
        num_classes=len(class_names),
        device=device,
    )
    print(f"[INFO] Model root: {spec.model_root}")
    print(f"[INFO] Run kind: {spec.run_kind}")
    print(f"[INFO] Config source: {spec.config_source}")
    if spec.student_run_dir is not None:
        print(f"[INFO] Student run dir: {spec.student_run_dir}")
    return ModelBundle(
        root=spec.model_root,
        run_kind=spec.run_kind,
        model_name=spec.model_name,
        sample_rate=spec.sample_rate,
        clip_seconds=spec.clip_seconds,
        image_height=spec.image_height,
        image_width=spec.image_width,
        renderer=renderer,
        models=models,
        config_source=spec.config_source,
        student_run_dir=spec.student_run_dir,
    )


def run_inference(cfg: InferConfig):
    seed_everything(cfg.seed)
    model_roots = discover_model_roots(cfg.model_roots)

    competition_root = Path(cfg.competition_root)
    if cfg.soundscapes_dir:
        test_dir = resolve_user_path(cfg.soundscapes_dir, competition_root=competition_root)
    else:
        test_dir = competition_root / ("train_soundscapes" if cfg.debug else "test_soundscapes")
    sample_submission_path = competition_root / "sample_submission.csv"
    output_path = Path(cfg.output_path)

    class_names = load_class_names(sample_submission_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Using soundscapes dir: {test_dir}")
    determinism = get_determinism_status()
    print(
        f"[INFO] Seed={cfg.seed} | deterministic_algorithms={determinism['deterministic_algorithms']} | "
        f"cudnn_deterministic={determinism['cudnn_deterministic']} | "
        f"cudnn_benchmark={determinism['cudnn_benchmark']}"
    )
    print(f"[INFO] TTA offsets (sec): {cfg.tta_offsets}")
    print(f"[INFO] Smoothing kernel: {cfg.smoothing_kernel if cfg.smoothing_kernel else 'disabled'}")
    print(f"[INFO] Soundscape top-k postprocess: {cfg.soundscape_top_k}")

    bundles = [load_model_bundle(model_root=model_root, class_names=class_names, device=device) for model_root in model_roots]
    total_models = sum(len(bundle.models) for bundle in bundles)
    print(f"[INFO] Loaded {len(bundles)} model run(s), total fold models = {total_models}")

    soundscape_files = list_soundscape_files(test_dir, debug=cfg.debug, debug_limit=cfg.debug_limit)
    if not soundscape_files:
        raise FileNotFoundError(f"No .ogg files found under {test_dir}")

    all_row_ids = []
    all_preds = []
    progress = tqdm(soundscape_files, total=len(soundscape_files), desc="Infer soundscapes", dynamic_ncols=True)
    for audio_path in progress:
        row_ids = None
        bundle_preds = []
        for bundle in bundles:
            audio = load_soundscape_audio(audio_path, sample_rate=bundle.sample_rate)
            segments, bundle_row_ids, row_indices = build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=bundle.sample_rate,
                clip_seconds=bundle.clip_seconds,
                tta_offsets=cfg.tta_offsets,
            )
            window_preds = predict_file_segments(
                segments=segments,
                models=bundle.models,
                renderer=bundle.renderer,
                device=device,
                segment_batch_size=cfg.segment_batch_size,
            )
            pred_matrix = aggregate_tta_predictions(window_preds, row_indices=row_indices, n_rows=len(bundle_row_ids))
            bundle_preds.append(pred_matrix)
            if row_ids is None:
                row_ids = bundle_row_ids
            elif row_ids != bundle_row_ids:
                raise ValueError("Row ids mismatch across model bundles.")

        preds = np.mean(np.stack(bundle_preds, axis=0), axis=0)
        preds = apply_temporal_smoothing(preds, cfg.smoothing_kernel)
        preds = apply_soundscape_postprocess(preds, cfg.soundscape_top_k)
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    prediction_matrix = np.concatenate(all_preds, axis=0)
    prediction_df = pd.DataFrame(prediction_matrix, columns=class_names)
    submission = pd.concat([pd.DataFrame({"row_id": all_row_ids}), prediction_df], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved submission to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    run_inference(parse_args())
