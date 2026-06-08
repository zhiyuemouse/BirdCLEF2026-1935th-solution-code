#!/usr/bin/env python3
"""Kaggle inference for the AudioMAE token-attention branch.

This script runs the frozen AudioMAE encoder once per 5s window, converts
``forward_features`` to ``[B, 64, 768]`` time tokens, applies the trained
fold ensemble head, and writes a BirdCLEF-style submission.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torchaudio.compliance import kaldi
from tqdm.auto import tqdm

N_WINDOWS = 12
ORIG_SR = 32000
MODEL_SR = 16000
WINDOW_SEC = 5.0
MODEL_FRAMES = 1024
MODEL_MELS = 128
AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974
TIME_TOKENS = 64
FREQ_TOKENS = 8
FEATURE_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioMAE token branch inference for BirdCLEF 2026.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--audiomae-ckpt-dir", type=str, default="ckpt/AudioMAE-HF")
    parser.add_argument("--model-path", type=str, default="audiomae_token_head_artifacts.joblib")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--file-scale-topk", type=int, default=2)
    parser.add_argument("--disable-file-scale", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_user_path(path_str: str, competition_root: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = competition_root / path
        if candidate.exists():
            return candidate
    return path


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample = pd.read_csv(sample_submission_path, nrows=0)
    return [col for col in sample.columns if col != "row_id"]


def load_audiomae_encoder(ckpt_dir: Path, device: torch.device) -> nn.Module:
    import timm

    ckpt_path = ckpt_dir / "model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find model.safetensors or pytorch_model.bin under {ckpt_dir}")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
        in_chans=1,
        img_size=(MODEL_FRAMES, MODEL_MELS),
        pretrained_cfg_overlay={"file": str(ckpt_path)},
    )
    model.eval()
    model.to(device)
    return model


def list_soundscape_paths(soundscapes_dir: Path, debug: bool, debug_limit: int) -> List[Path]:
    paths = sorted(soundscapes_dir.glob("*.ogg"))
    if not paths:
        paths = sorted(soundscapes_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No audio files found under {soundscapes_dir}")
    if debug:
        paths = paths[: int(debug_limit)]
    return paths


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def crop_or_pad(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len].astype(np.float32, copy=False)
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def waveform_to_fbank(window: np.ndarray, sr: int) -> torch.Tensor:
    wav = torch.from_numpy(window.astype(np.float32, copy=False))
    if sr != MODEL_SR:
        wav = AF.resample(wav, orig_freq=int(sr), new_freq=MODEL_SR)
    wav = wav.unsqueeze(0)
    fbank = kaldi.fbank(
        wav,
        htk_compat=True,
        window_type="hanning",
        num_mel_bins=MODEL_MELS,
        sample_frequency=MODEL_SR,
    )
    if fbank.shape[0] < MODEL_FRAMES:
        fbank = F.pad(fbank, (0, 0, 0, MODEL_FRAMES - fbank.shape[0]))
    else:
        fbank = fbank[:MODEL_FRAMES]
    fbank = (fbank - AUDIOMAE_MEAN) / (AUDIOMAE_STD * 2.0)
    return fbank.unsqueeze(0)


def build_file_features(path: Path) -> torch.Tensor:
    audio, sr = read_audio_mono(path)
    file_len = int(round(60.0 * sr))
    audio = crop_or_pad(audio, file_len)
    feats: List[torch.Tensor] = []
    window_len = int(round(WINDOW_SEC * sr))
    for win_idx in range(N_WINDOWS):
        start = win_idx * window_len
        end = start + window_len
        feats.append(waveform_to_fbank(audio[start:end], sr))
    return torch.stack(feats, dim=0)


def features_to_time_tokens(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = encoder.forward_features(x)
    expected = (1 + TIME_TOKENS * FREQ_TOKENS, FEATURE_DIM)
    if tuple(features.shape[1:]) != expected:
        raise RuntimeError(f"Unexpected AudioMAE feature shape: {tuple(features.shape)}")
    patch = features[:, 1:, :].reshape(len(x), TIME_TOKENS, FREQ_TOKENS, FEATURE_DIM)
    return patch.mean(dim=2)


class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0) -> None:
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


class AudioMAETokenMambaHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            *[LocalMambaBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(int(num_blocks))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.fusion(x)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        return self.head(pooled)


class AudioMAETokenAttentionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(64, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(64, hidden_dim // 2), 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        weights = torch.softmax(self.attn(x), dim=1)
        attn_pool = (x * weights).sum(dim=1)
        mean_pool = x.mean(dim=1)
        return self.head(torch.cat([attn_pool, mean_pool], dim=-1))


class AudioMAETokenMeanMaxHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Linear(in_features * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1))


def build_head(model_info: Dict[str, object], num_classes: int) -> nn.Module:
    variant = str(model_info.get("head_variant", "attention"))
    input_dim = int(model_info.get("input_dim", FEATURE_DIM))
    hidden_dim = int(model_info.get("hidden_dim", 384))
    dropout = float(model_info.get("dropout", 0.3))
    if variant == "mamba":
        return AudioMAETokenMambaHead(
            in_features=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_blocks=int(model_info.get("num_blocks", 2)),
            kernel_size=int(model_info.get("kernel_size", 9)),
            dropout=dropout,
        )
    if variant == "meanmax":
        return AudioMAETokenMeanMaxHead(
            in_features=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    return AudioMAETokenAttentionHead(
        in_features=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def transform_tokens(tokens: np.ndarray, standardizer: Dict[str, np.ndarray]) -> np.ndarray:
    mean = standardizer["mean"]
    std = standardizer["std"]
    return ((tokens - mean[None, :, :]) / std[None, :, :]).astype(np.float32, copy=False)


def load_head_ensemble(artifact_path: Path, class_names: Sequence[str], device: torch.device):
    artifact = joblib.load(artifact_path)
    if list(artifact["class_names"]) != list(class_names):
        raise ValueError("AudioMAE token artifact class_names do not match sample_submission columns.")
    models = []
    for fold_artifact in artifact["folds"]:
        model_info = fold_artifact["model"]
        model = build_head(model_info, num_classes=len(class_names))
        model.load_state_dict(model_info["model_state"], strict=True)
        model.to(device)
        model.eval()
        models.append(
            {
                "fold_name": fold_artifact.get("fold_name", "fold"),
                "standardizer": fold_artifact["token_standardizer"],
                "fitted_class_indices": np.asarray(model_info["fitted_class_indices"], dtype=np.int32),
                "model": model,
            }
        )
    return artifact, models


def predict_heads(models: Sequence[Dict[str, object]], tokens: np.ndarray, device: torch.device, batch_size: int, fallback_prob: float) -> np.ndarray:
    fold_preds = []
    for fold_artifact in models:
        x = transform_tokens(tokens, fold_artifact["standardizer"])
        pred = np.full((len(tokens), fold_artifact["model"].head[-1].out_features), fallback_prob, dtype=np.float32)
        fitted = np.asarray(fold_artifact["fitted_class_indices"], dtype=np.int32)
        chunks = []
        model = fold_artifact["model"]
        with torch.no_grad():
            for start in range(0, len(x), int(batch_size)):
                batch = torch.from_numpy(x[start:start + int(batch_size)].astype(np.float32, copy=False)).to(device)
                chunks.append(torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32))
        model_pred = np.concatenate(chunks, axis=0)
        pred[:, fitted] = model_pred[:, fitted]
        fold_preds.append(pred)
        print(f"[INFO] Applied {fold_artifact['fold_name']} fitted_classes={len(fitted)}")
    return np.clip(np.mean(fold_preds, axis=0).astype(np.float32), 0.0, 1.0)


def file_level_topk_mean_scale(pred: np.ndarray, filename: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        if len(idx) == 0:
            continue
        p = pred[idx]
        k = max(1, min(int(topk), len(idx)))
        scale = np.sort(p, axis=0)[-k:].mean(axis=0, keepdims=True)
        out[idx] = p * scale
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    total_start = time.time()
    timings: Dict[str, float] = {}

    competition_root = Path(args.competition_root)
    soundscapes_dir = Path(args.soundscapes_dir) if args.soundscapes_dir else competition_root / "test_soundscapes"
    sample_submission_path = (
        Path(args.sample_submission_path) if args.sample_submission_path else competition_root / "sample_submission.csv"
    )
    ckpt_dir = resolve_user_path(args.audiomae_ckpt_dir, competition_root=competition_root)
    model_path = resolve_user_path(args.model_path, competition_root=competition_root)
    output_path = Path(args.output_path)
    device = resolve_device(args.device)
    class_names = load_class_names(sample_submission_path)
    paths = list_soundscape_paths(soundscapes_dir, debug=args.debug, debug_limit=args.debug_limit)

    print("[INFO] AudioMAE token inference")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] files: {len(paths)}")
    print(f"[INFO] ckpt_dir: {ckpt_dir}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] output_path: {output_path}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] file_scale_topk: {0 if args.disable_file_scale else args.file_scale_topk}")

    t0 = time.time()
    encoder = load_audiomae_encoder(ckpt_dir, device=device)
    artifact, head_models = load_head_ensemble(model_path, class_names=class_names, device=device)
    fallback_prob = float(artifact.get("config", {}).get("fallback_prob", 0.5))
    timings["load_models"] = time.time() - t0

    row_ids: List[str] = []
    filenames: List[str] = []
    token_batches: List[np.ndarray] = []

    t0 = time.time()
    current_features: List[torch.Tensor] = []
    current_row_ids: List[str] = []
    current_filenames: List[str] = []
    with torch.no_grad():
        for path in tqdm(paths, desc="AudioMAE", leave=False):
            file_features = build_file_features(path)
            for win_idx in range(N_WINDOWS):
                current_features.append(file_features[win_idx])
                end_sec = (win_idx + 1) * 5
                current_row_ids.append(f"{path.stem}_{end_sec}")
                current_filenames.append(path.name)
            while len(current_features) >= int(args.batch_size):
                batch = torch.stack(current_features[: int(args.batch_size)], dim=0).to(device)
                token_batches.append(features_to_time_tokens(encoder, batch).detach().cpu().numpy().astype(np.float32))
                row_ids.extend(current_row_ids[: int(args.batch_size)])
                filenames.extend(current_filenames[: int(args.batch_size)])
                del current_features[: int(args.batch_size)]
                del current_row_ids[: int(args.batch_size)]
                del current_filenames[: int(args.batch_size)]
        if current_features:
            batch = torch.stack(current_features, dim=0).to(device)
            token_batches.append(features_to_time_tokens(encoder, batch).detach().cpu().numpy().astype(np.float32))
            row_ids.extend(current_row_ids)
            filenames.extend(current_filenames)

    tokens = np.concatenate(token_batches, axis=0).astype(np.float32, copy=False)
    timings["feature_extract"] = time.time() - t0

    t0 = time.time()
    pred = predict_heads(
        models=head_models,
        tokens=tokens,
        device=device,
        batch_size=args.batch_size,
        fallback_prob=fallback_prob,
    )
    timings["head_predict"] = time.time() - t0

    t0 = time.time()
    if not args.disable_file_scale and args.file_scale_topk > 0:
        pred = file_level_topk_mean_scale(pred, filename=np.asarray(filenames, dtype=object), topk=args.file_scale_topk)
    timings["postprocess"] = time.time() - t0

    t0 = time.time()
    submission = pd.DataFrame(pred, columns=class_names)
    submission.insert(0, "row_id", row_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    timings["save_submission"] = time.time() - t0
    timings["total"] = time.time() - total_start

    print("[INFO] Timing summary:")
    for key, value in timings.items():
        print(f"[INFO]   {key}: {value:.1f}s")
    print(f"[INFO]   seconds_per_file_total: {timings['total'] / max(1, len(paths)):.2f}s")
    print(f"[INFO] Saved submission to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
