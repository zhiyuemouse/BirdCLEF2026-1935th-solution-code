#!/usr/bin/env python3
"""Cache frozen AudioMAE patch tokens for BirdCLEF 2026 soundscape windows.

AudioMAE ``forward_features`` returns ``[B, 513, 768]`` for our 1024x128 fbank
input: one CLS token plus ``64 x 8`` time-frequency patch tokens.  This script
keeps the temporal structure by averaging over the 8 frequency patches:

``[B, 512, 768] -> [B, 64, 8, 768] -> [B, 64, 768]``.

The resulting tokens can be consumed by a lightweight Mamba/attention head.
No labels are used during caching.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from birdclef2026_cache_audiomae_soundscape_embeddings import (
    load_model,
    load_target_rows,
    build_batch_features,
    resolve_device,
    save_meta,
)


TIME_TOKENS = 64
FREQ_TOKENS = 8
FEATURE_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache AudioMAE time tokens for soundscape 5s windows.")
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument(
        "--target-rows-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
    )
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/AudioMAE-HF")
    parser.add_argument("--output-dir", type=str, default="audiomae_soundscape_token_cache_cnn195634folds_v1")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-rows", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    soundscapes_dir = Path(args.soundscapes_dir)
    target_rows_path = Path(args.target_rows_path)
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "audiomae_soundscape_token_meta.csv"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "audiomae_soundscape_tokens.npz"

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
    tokens = np.empty((len(target_df), TIME_TOKENS, FEATURE_DIM), dtype=np.float32)
    cls_tokens = np.empty((len(target_df), FEATURE_DIM), dtype=np.float32)

    print("[INFO] AudioMAE soundscape token cache")
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
            features = model.forward_features(x)
            if features.shape[1:] != (1 + TIME_TOKENS * FREQ_TOKENS, FEATURE_DIM):
                raise RuntimeError(f"Unexpected AudioMAE feature shape: {tuple(features.shape)}")
            cls = features[:, 0, :]
            patch = features[:, 1:, :].reshape(len(batch_df), TIME_TOKENS, FREQ_TOKENS, FEATURE_DIM)
            time_tokens = patch.mean(dim=2)
            tokens[start:end] = time_tokens.detach().cpu().numpy().astype(np.float32, copy=False)
            cls_tokens[start:end] = cls.detach().cpu().numpy().astype(np.float32, copy=False)
            if end == len(target_df) or end % 128 == 0:
                print(f"[INFO] cached {end}/{len(target_df)} rows | elapsed={time.time() - start_time:.1f}s", flush=True)

    save_meta(target_df, meta_path)
    np.savez_compressed(
        arrays_path,
        tokens=tokens.astype(np.float32, copy=False),
        cls_tokens=cls_tokens.astype(np.float32, copy=False),
    )
    summary = {
        "rows": int(len(target_df)),
        "files": int(target_df["filename"].nunique()),
        "tokens_shape": list(tokens.shape),
        "cls_tokens_shape": list(cls_tokens.shape),
        "target_rows_path": str(target_rows_path),
        "soundscapes_dir": str(soundscapes_dir),
        "ckpt_dir": str(ckpt_dir),
        "meta_path": str(meta_path),
        "arrays_path": str(arrays_path),
        "elapsed_sec": float(time.time() - start_time),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[INFO] Done.")
    print(f"[INFO] tokens shape: {tokens.shape}")
    print(f"[INFO] elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
