#!/usr/bin/env python3
"""Cache frozen AudioMAE embeddings for BirdCLEF 2026 soundscape windows.

The downloaded Hugging Face/timm checkpoint in ``ckpt/AudioMAE-HF`` expects
normalized 16 kHz Kaldi fbank features with shape ``[B, 1, 1024, 128]`` and
returns a 768-d embedding.  This script extracts those embeddings for the
same 5s windows used by our 3-fold CNN/Perch OOF pipeline.

No labels are used here.  This is deterministic feature extraction from a
frozen pretrained encoder; leakage control happens in downstream fold training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torchaudio.compliance import kaldi


ORIG_SR = 32000
MODEL_SR = 16000
WINDOW_SEC = 5.0
MODEL_FRAMES = 1024
MODEL_MELS = 128
AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache AudioMAE embeddings for soundscape 5s windows.")
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument(
        "--target-rows-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
        help="CSV with row_id, filename, and end_sec columns. Defaults to the main 3-fold CNN rows.",
    )
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/AudioMAE-HF")
    parser.add_argument("--output-dir", type=str, default="audiomae_soundscape_cache_cnn195634folds_v1")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-rows", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_target_rows(path: Path, limit_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"row_id", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Target rows are missing required columns: {sorted(missing)}")
    df = df.drop_duplicates(subset=["row_id"]).copy()
    df["row_id"] = df["row_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    if "end_sec" not in df.columns:
        df["end_sec"] = df["row_id"].str.rsplit("_", n=1).str[-1].astype(int)
    df["end_sec"] = df["end_sec"].astype(float)
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype(int)
    # Preserve the source CSV order so downstream OOF files are easy to compare
    # with the CNN/Perch runs that provided the target row list.
    df = df.reset_index(drop=True)
    if limit_rows > 0:
        df = df.iloc[: int(limit_rows)].reset_index(drop=True)
    return df


def load_model(ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
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


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def crop_window(audio: np.ndarray, sr: int, end_sec: float) -> np.ndarray:
    start_sec = float(end_sec) - WINDOW_SEC
    start = int(round(start_sec * sr))
    end = int(round(float(end_sec) * sr))
    target_len = int(round(WINDOW_SEC * sr))
    out = np.zeros(target_len, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(len(audio), end)
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        out[dst_start:dst_end] = audio[src_start:src_end]
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


def build_batch_features(batch_df: pd.DataFrame, soundscapes_dir: Path) -> torch.Tensor:
    features: List[torch.Tensor] = []
    cached_filename = None
    cached_audio: np.ndarray | None = None
    cached_sr = ORIG_SR
    for row in batch_df.itertuples(index=False):
        filename = str(row.filename)
        if filename != cached_filename:
            cached_audio, cached_sr = read_audio_mono(soundscapes_dir / filename)
            cached_filename = filename
        if cached_audio is None:
            raise RuntimeError("Internal audio cache was not initialized.")
        window = crop_window(cached_audio, cached_sr, float(row.end_sec))
        features.append(waveform_to_fbank(window, cached_sr))
    return torch.stack(features, dim=0)


def save_meta(meta_df: pd.DataFrame, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.suffix.lower() == ".parquet":
        meta_df.to_parquet(meta_path, index=False)
    elif meta_path.suffix.lower() == ".csv":
        meta_df.to_csv(meta_path, index=False)
    else:
        raise ValueError(f"Unsupported meta suffix: {meta_path.suffix}")


def main() -> None:
    args = parse_args()
    soundscapes_dir = Path(args.soundscapes_dir)
    target_rows_path = Path(args.target_rows_path)
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "audiomae_soundscape_meta.csv"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "audiomae_soundscape_embeddings.npz"

    if args.skip_existing and meta_path.exists() and arrays_path.exists():
        print("[INFO] Skip because cache already exists:")
        print(f"  - {meta_path}")
        print(f"  - {arrays_path}")
        return

    target_df = load_target_rows(target_rows_path, limit_rows=args.limit_rows)
    missing_files = [
        str(soundscapes_dir / filename)
        for filename in target_df["filename"].drop_duplicates().astype(str)
        if not (soundscapes_dir / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError("Missing soundscape files:\n" + "\n".join(missing_files[:20]))

    device = resolve_device(args.device)
    model = load_model(ckpt_dir, device=device)
    batch_size = max(1, int(args.batch_size))
    embeddings = np.empty((len(target_df), 768), dtype=np.float32)

    print("[INFO] AudioMAE soundscape embedding cache")
    print(f"[INFO] target_rows_path: {target_rows_path}")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] ckpt_dir: {ckpt_dir}")
    print(f"[INFO] rows: {len(target_df)} files={target_df['filename'].nunique()}")
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] meta_path: {meta_path}")
    print(f"[INFO] arrays_path: {arrays_path}")

    start_time = time.time()
    with torch.no_grad():
        for start in range(0, len(target_df), batch_size):
            end = min(len(target_df), start + batch_size)
            batch_df = target_df.iloc[start:end]
            x = build_batch_features(batch_df, soundscapes_dir=soundscapes_dir).to(device)
            emb = model(x).detach().cpu().numpy().astype(np.float32, copy=False)
            if emb.shape != (len(batch_df), 768):
                raise RuntimeError(f"Unexpected AudioMAE embedding shape: {emb.shape}")
            embeddings[start:end] = emb
            if end == len(target_df) or end % 128 == 0:
                print(f"[INFO] cached {end}/{len(target_df)} rows | elapsed={time.time() - start_time:.1f}s", flush=True)

    save_meta(target_df, meta_path)
    np.savez_compressed(arrays_path, embeddings=embeddings)
    summary = {
        "rows": int(len(target_df)),
        "files": int(target_df["filename"].nunique()),
        "embedding_shape": list(embeddings.shape),
        "target_rows_path": str(target_rows_path),
        "soundscapes_dir": str(soundscapes_dir),
        "ckpt_dir": str(ckpt_dir),
        "meta_path": str(meta_path),
        "arrays_path": str(arrays_path),
        "elapsed_sec": float(time.time() - start_time),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Done.")
    print(f"[INFO] embeddings shape: {embeddings.shape}")
    print(f"[INFO] elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
