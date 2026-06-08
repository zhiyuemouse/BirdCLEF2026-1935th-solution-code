import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

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
    model_root: str = ""
    soundscapes_dir: str = ""
    debug: bool = False
    debug_limit: int = 4
    segment_batch_size: int = 12
    seed: int = 2026


def parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Kaggle inference for birdclef2026_gm.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--model-root", type=str, default="")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return InferConfig(
        competition_root=args.competition_root,
        output_path=args.output_path,
        model_root=args.model_root,
        soundscapes_dir=args.soundscapes_dir,
        debug=args.debug,
        debug_limit=args.debug_limit,
        segment_batch_size=args.segment_batch_size,
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


def discover_model_root(explicit_model_root: str) -> Path:
    if explicit_model_root:
        model_root = Path(explicit_model_root)
        if not model_root.is_absolute():
            model_root = Path.cwd() / model_root
        if not model_root.exists():
            raise FileNotFoundError(f"Explicit model root does not exist: {model_root}")
        return model_root

    search_roots = [
        Path.cwd(),
        Path("/kaggle/working"),
        Path("/kaggle/input/models"),
        Path("/kaggle/input"),
    ]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            candidate_root = config_path.parent
            fold_paths = sorted(candidate_root.glob("fold_*/stage2_fold*_best.pth"))
            if fold_paths:
                candidates.append((candidate_root, len(fold_paths), config_path.stat().st_mtime))

    if not candidates:
        raise FileNotFoundError(
            "No candidate model directory found under /kaggle/input. "
            "Please upload the trained model folder or pass --model-root."
        )

    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    chosen_root = candidates[0][0]
    print(f"[INFO] Auto-discovered model root: {chosen_root}")
    return chosen_root


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
    norm: str = "",
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

    if str(norm).lower() == "slaney":
        enorm = 2.0 / np.maximum(hz_edges[2 : n_mels + 2] - hz_edges[:n_mels], 1e-8)
        filterbank *= enorm[:, np.newaxis].astype(np.float32)

    return torch.from_numpy(filterbank)


def power_to_db(spec: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    spec = torch.clamp(spec, min=1e-10)
    db = 10.0 * torch.log10(spec)
    max_db = db.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp(db, min=max_db - top_db)


def pcen_transform(
    spec: torch.Tensor,
    eps: float = 1e-6,
    alpha: float = 0.98,
    delta: float = 2.0,
    root: float = 0.5,
    smooth_kernel: int = 31,
) -> torch.Tensor:
    """Apply a lightweight PCEN-style dynamic range compression along time."""

    spec = torch.clamp(spec, min=0.0)
    time_steps = int(spec.shape[-1])
    kernel = min(int(smooth_kernel), time_steps if time_steps % 2 == 1 else max(time_steps - 1, 1))
    if kernel <= 1:
        smooth = spec
    else:
        pad = kernel // 2
        original_shape = spec.shape
        x = spec.reshape(-1, original_shape[-2], original_shape[-1])
        x = F.pad(x, (pad, pad), mode="replicate")
        smooth = F.avg_pool1d(x, kernel_size=kernel, stride=1)
        smooth = smooth.reshape(original_shape)
    return (spec / torch.pow(eps + smooth, alpha) + delta).pow(root) - (delta**root)


class SpectrogramRenderer:
    def __init__(
        self,
        sample_rate: int,
        image_height: int,
        image_width: int,
        spectrogram_variant: str = "logmel",
        input_channels: int = 3,
        image_normalize: str = "minus_one_one",
    ):
        self.sample_rate = sample_rate
        self.image_height = image_height
        self.image_width = image_width
        self.input_channels = int(input_channels)
        self.image_normalize = str(image_normalize)
        self.top_db = 80.0
        self.spectrogram_variant = str(spectrogram_variant).lower()
        if self.spectrogram_variant not in {"logmel", "pcen", "logmel_v8"}:
            raise ValueError(f"Unsupported spectrogram_variant: {spectrogram_variant}")
        if self.spectrogram_variant == "logmel_v8":
            self.specs = [
                self._build_spec(
                    n_fft=2048,
                    hop_length=313,
                    n_mels=256,
                    f_min=20.0,
                    f_max=16000.0,
                    win_length=626,
                    mel_norm="slaney",
                )
            ]
        else:
            self.specs = [
                self._build_spec(n_fft=1024, hop_length=256, n_mels=128, f_min=20.0, f_max=16000.0),
                self._build_spec(n_fft=2048, hop_length=512, n_mels=128, f_min=20.0, f_max=12000.0),
                self._build_spec(n_fft=4096, hop_length=1024, n_mels=128, f_min=20.0, f_max=8000.0),
            ]

    def _build_spec(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        win_length: Optional[int] = None,
        mel_norm: str = "",
    ) -> Dict[str, torch.Tensor]:
        win_length = int(win_length or n_fft)
        return {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": torch.hann_window(win_length),
            "mel_filter": build_mel_filterbank(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                norm=mel_norm,
            ),
        }

    def _mel_spectrogram_batch(self, waveform_batch: torch.Tensor, spec_cfg: Dict[str, torch.Tensor]) -> torch.Tensor:
        stft = torch.stft(
            waveform_batch,
            n_fft=int(spec_cfg["n_fft"]),
            hop_length=int(spec_cfg["hop_length"]),
            win_length=int(spec_cfg.get("win_length", spec_cfg["n_fft"])),
            window=spec_cfg["window"],
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power_spec = stft.abs().pow(2.0)
        mel_filter = spec_cfg["mel_filter"].to(power_spec.dtype)
        mel_spec = torch.einsum("mf,bft->bmt", mel_filter, power_spec)
        if self.spectrogram_variant == "pcen":
            return pcen_transform(mel_spec)
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
        if self.input_channels == 1 and image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)
        elif self.input_channels == 3 and image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        mins = image.amin(dim=(-2, -1), keepdim=True)
        maxs = image.amax(dim=(-2, -1), keepdim=True)
        image = (image - mins) / (maxs - mins + 1e-6)
        if self.image_normalize == "minus_one_one":
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


class LSEHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.logsumexp(x / self.temperature, dim=1) * self.temperature - math.log(x.shape[1])
        return self.classifier(self.dropout(pooled))


class SequencePooling(nn.Module):
    def __init__(self, pool_type: str = "avg", p: float = 3.0, lse_temperature: float = 1.0):
        super().__init__()
        self.pool_type = str(pool_type)
        self.p = float(p)
        self.lse_temperature = float(lse_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return x.mean(dim=1)
        if self.pool_type == "avg_max":
            return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
        if self.pool_type == "lse":
            return torch.logsumexp(x / self.lse_temperature, dim=1) * self.lse_temperature - math.log(x.shape[1])
        shifted = x - x.amin(dim=1, keepdim=True)
        pooled = shifted.clamp_min(1e-6).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return pooled + x.amin(dim=1)


class CSIROMultiContextHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float, num_slots: int, pool_type: str = "avg"):
        super().__init__()
        hidden_features = max(in_features // 2, 64)
        self.num_slots = int(num_slots)
        self.fusion = nn.Sequential(
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
            LocalSequenceBlock(in_features, kernel_size=5, dropout=0.1),
        )
        self.global_pool = SequencePooling(pool_type=pool_type)
        self.slot_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )
        self.global_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def _slot_pool(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.shape[1]
        pooled = []
        for slot in range(self.num_slots):
            start = int(math.floor(slot * n_tokens / self.num_slots))
            end = int(math.floor((slot + 1) * n_tokens / self.num_slots))
            end = max(end, start + 1)
            pooled.append(x[:, start:end].mean(dim=1))
        return torch.stack(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.fusion(x)
        slot_features = self._slot_pool(x)
        global_features = self.global_pool(x)
        return {
            "slot_logits": self.slot_head(slot_features),
            "global_logits": self.global_head(global_features),
        }


class BirdCLEFNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float,
        drop_path: float,
        head_type: str = "linear",
        head_pool_type: str = "avg",
        multi_context_num_slots: int = 3,
        input_channels: int = 3,
        lse_temperature: float = 1.0,
    ):
        super().__init__()
        self.head_type = str(head_type)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=int(input_channels),
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
        elif self.head_type == "lse_head_v1":
            self.head = LSEHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=dropout,
                temperature=lse_temperature,
            )
        elif self.head_type == "csiro_multicontext_v1":
            self.head = CSIROMultiContextHead(
                in_features=self.backbone.num_features,
                num_classes=num_classes,
                dropout=dropout,
                num_slots=multi_context_num_slots,
                pool_type=head_pool_type,
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


def load_models(
        model_root: Path,
        model_name: str,
        num_classes: int,
        dropout: float,
        drop_path: float,
        device: torch.device,
        head_type: str = "linear",
        head_pool_type: str = "avg",
        multi_context_num_slots: int = 3,
        input_channels: int = 3,
        lse_temperature: float = 1.0,
):
    fold_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
    if not fold_paths:
        raise FileNotFoundError(f"No stage2 fold checkpoints found under {model_root}")

    models = []
    for fold_path in fold_paths:
        model = BirdCLEFNet(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            drop_path=drop_path,
            head_type=head_type,
            head_pool_type=head_pool_type,
            multi_context_num_slots=multi_context_num_slots,
            input_channels=input_channels,
            lse_temperature=lse_temperature,
        )
        checkpoint_obj = torch.load(fold_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint_obj)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded fold checkpoint: {fold_path}")
    return models


@lru_cache(maxsize=32)
def read_audio_file(path: str):
    return sf.read(path, dtype="float32")


def load_soundscape_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = read_audio_file(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = linear_resample(audio, sr, sample_rate)
    return audio


def slice_audio_with_padding(audio: np.ndarray, start: int, clip_len: int) -> np.ndarray:
    end = start + clip_len
    src_start = max(0, start)
    src_end = min(len(audio), end)
    segment = np.zeros(clip_len, dtype=np.float32)
    if src_end > src_start:
        dst_start = max(0, -start)
        dst_end = dst_start + (src_end - src_start)
        segment[dst_start:dst_end] = audio[src_start:src_end]
    return segment


def build_segments_for_file(
    audio: np.ndarray,
    file_stem: str,
    sample_rate: int,
    clip_seconds: float,
    clip_offset_seconds: float = 0.0,
):
    clip_len = int(round(sample_rate * clip_seconds))
    segments = []
    row_ids = []
    for end_sec in range(5, 61, 5):
        # Match stage2 training semantics: each row is anchored at its labeled 5-second window start,
        # while clip_seconds controls how much right-context the model sees from that start.
        start_sec = (end_sec - 5) + float(clip_offset_seconds)
        start = int(round(start_sec * sample_rate))
        segment = slice_audio_with_padding(audio=audio, start=start, clip_len=clip_len)
        segments.append(segment)
        row_ids.append(f"{file_stem}_{end_sec}")
    return np.stack(segments), row_ids


def build_multicontext_segments_for_file(
    audio: np.ndarray,
    file_stem: str,
    sample_rate: int,
    clip_seconds: float,
    num_slots: int,
):
    clip_len = int(round(sample_rate * clip_seconds))
    segments = []
    window_slot_indices = []
    row_ids = [f"{file_stem}_{end_sec}" for end_sec in range(5, 61, 5)]
    for start_slot in range(0, 12 - int(num_slots) + 1):
        start = int(round(start_slot * 5 * sample_rate))
        end = start + clip_len
        segment = audio[start:end]
        if len(segment) < clip_len:
            segment = np.pad(segment, (0, clip_len - len(segment)), mode="constant")
        segments.append(segment.astype(np.float32))
        window_slot_indices.append([start_slot + slot for slot in range(int(num_slots))])
    return np.stack(segments), row_ids, np.asarray(window_slot_indices, dtype=np.int64)


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
                if isinstance(logits, dict):
                    logits = logits["slot_logits"]
                probs = torch.sigmoid(logits)
                ensemble = probs if ensemble is None else ensemble + probs
            ensemble = ensemble / len(models)
        all_preds.append(ensemble.cpu().numpy())
        del images, ensemble
    return np.concatenate(all_preds, axis=0)


def run_inference(cfg: InferConfig):
    seed_everything(cfg.seed)
    model_root = discover_model_root(cfg.model_root)
    model_cfg = load_json(model_root / "config.json")

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

    renderer = SpectrogramRenderer(
        sample_rate=int(model_cfg["sample_rate"]),
        image_height=int(model_cfg["image_height"]),
        image_width=int(model_cfg["image_width"]),
        spectrogram_variant=str(model_cfg.get("spectrogram_variant", "logmel")),
        input_channels=int(model_cfg.get("input_channels", 3)),
        image_normalize=str(model_cfg.get("image_normalize", "minus_one_one")),
    )
    print(
        f"[INFO] Spectrogram variant: {renderer.spectrogram_variant} | "
        f"input_channels={renderer.input_channels} | normalize={renderer.image_normalize}"
    )
    models = load_models(
        model_root=model_root,
        model_name=str(model_cfg["model_name"]),
        num_classes=len(class_names),
        dropout=float(model_cfg["dropout"]),
        drop_path=float(model_cfg["drop_path"]),
        device=device,
        head_type=str(model_cfg.get("head_type", "linear")),
        head_pool_type=str(model_cfg.get("head_pool_type", "avg")),
        multi_context_num_slots=int(model_cfg.get("multi_context_num_slots", 3)),
        input_channels=int(model_cfg.get("input_channels", 3)),
        lse_temperature=float(model_cfg.get("lse_temperature", 1.0)),
    )
    head_type = str(model_cfg.get("head_type", "linear"))
    multi_context_num_slots = int(model_cfg.get("multi_context_num_slots", 3))

    soundscape_files = list_soundscape_files(test_dir, debug=cfg.debug, debug_limit=cfg.debug_limit)
    if not soundscape_files:
        raise FileNotFoundError(f"No .ogg files found under {test_dir}")

    all_row_ids = []
    all_preds = []
    progress = tqdm(soundscape_files, total=len(soundscape_files), desc="Infer soundscapes", dynamic_ncols=True)
    for audio_path in progress:
        audio = load_soundscape_audio(audio_path, sample_rate=int(model_cfg["sample_rate"]))
        if head_type == "csiro_multicontext_v1":
            segments, row_ids, window_slot_indices = build_multicontext_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=int(model_cfg["sample_rate"]),
                clip_seconds=float(model_cfg["clip_seconds"]),
                num_slots=multi_context_num_slots,
            )
            slot_preds = predict_file_segments(
                segments=segments,
                models=models,
                renderer=renderer,
                device=device,
                segment_batch_size=cfg.segment_batch_size,
            )
            pred_sum = np.zeros((12, len(class_names)), dtype=np.float32)
            pred_count = np.zeros(12, dtype=np.float32)
            for window_idx, slot_indices in enumerate(window_slot_indices):
                for slot_idx, row_idx in enumerate(slot_indices):
                    pred_sum[row_idx] += slot_preds[window_idx, slot_idx]
                    pred_count[row_idx] += 1.0
            preds = pred_sum / np.maximum(pred_count[:, None], 1.0)
        else:
            segments, row_ids = build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=int(model_cfg["sample_rate"]),
                clip_seconds=float(model_cfg["clip_seconds"]),
            )
            preds = predict_file_segments(
                segments=segments,
                models=models,
                renderer=renderer,
                device=device,
                segment_batch_size=cfg.segment_batch_size,
            )
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
