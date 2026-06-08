#!/usr/bin/env python3
"""Cache Perch ONNX global embeddings for train_audio clips.

This mirrors ``birdclef2026_cache_perch_audio_spatial_onnx.py`` but saves the
Perch ``embedding`` output with shape ``[B, 1536]``.  The cache is intended for
stage1 pretraining lightweight heads while keeping Perch frozen.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from birdclef2026_cache_perch_audio_spatial_onnx import (
    WINDOW_SAMPLES,
    build_targets,
    load_class_names,
    read_audio_crop,
    save_meta,
    select_audio_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache train_audio Perch embeddings with ONNX.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--train-csv-path", type=str, default="")
    parser.add_argument("--train-audio-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--output-dir", type=str, default="perch_audio_embedding_cache_max100")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--min-rating", type=float, default=-1.0)
    parser.add_argument("--include-secondary-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def infer_embeddings(
    audio_paths: Sequence[Path],
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
    seed: int,
) -> np.ndarray:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "embedding" not in output_names:
        raise RuntimeError(f"ONNX model has no embedding output. Outputs: {output_names}")

    n_rows = len(audio_paths)
    embeddings = np.empty((n_rows, 1536), dtype=np.float32)
    start_time = time.time()
    batch_size = max(1, int(batch_size))
    for start in range(0, n_rows, batch_size):
        batch_paths = list(audio_paths[start:start + batch_size])
        batch_audio = np.empty((len(batch_paths), WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            batch_audio[batch_idx] = read_audio_crop(path, seed=seed)
        emb = session.run(["embedding"], {input_name: batch_audio})[0].astype(np.float32, copy=False)
        if emb.shape[1:] != (1536,):
            raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")
        end = start + len(batch_paths)
        embeddings[start:end] = emb
        if end == n_rows or end % 500 == 0:
            print(f"[INFO] audio embedding cache {end}/{n_rows} rows | elapsed={time.time() - start_time:.1f}s", flush=True)
    return embeddings


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    train_csv_path = Path(args.train_csv_path) if args.train_csv_path else input_dir / "train.csv"
    train_audio_dir = Path(args.train_audio_dir) if args.train_audio_dir else input_dir / "train_audio"
    sample_submission_path = (
        Path(args.sample_submission_path) if args.sample_submission_path else input_dir / "sample_submission.csv"
    )
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "perch_audio_embedding_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "perch_audio_embedding_arrays.npz"

    if args.skip_existing and meta_path.exists() and arrays_path.exists():
        print("[INFO] Skip because output files already exist:")
        print(f"  - {meta_path}")
        print(f"  - {arrays_path}")
        return

    class_names = load_class_names(sample_submission_path)
    audio_df = select_audio_rows(
        train_csv_path=train_csv_path,
        train_audio_dir=train_audio_dir,
        max_per_class=args.max_per_class,
        min_rating=args.min_rating,
        limit_rows=args.limit_rows,
    )
    targets = build_targets(audio_df, class_names=class_names, include_secondary=args.include_secondary_labels)
    audio_paths = [Path(path) for path in audio_df["audio_path"].astype(str).tolist()]

    print("[INFO] Perch audio embedding ONNX cache")
    print(f"[INFO] onnx_path: {onnx_path}")
    print(f"[INFO] train_csv_path: {train_csv_path}")
    print(f"[INFO] train_audio_dir: {train_audio_dir}")
    print(f"[INFO] rows: {len(audio_df)}")
    print(f"[INFO] classes with audio rows: {audio_df['primary_label'].nunique()}")
    print(f"[INFO] target positives: {int(targets.sum())}")
    print(f"[INFO] output meta: {meta_path}")
    print(f"[INFO] output arrays: {arrays_path}")

    embeddings = infer_embeddings(
        audio_paths=audio_paths,
        onnx_path=onnx_path,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    save_meta(audio_df, meta_path)
    np.savez_compressed(
        arrays_path,
        embeddings=embeddings.astype(np.float32, copy=False),
        y=targets.astype(np.float32, copy=False),
    )

    print("[INFO] Done.")
    print(f"[INFO] meta shape: {audio_df.shape}")
    print(f"[INFO] embeddings shape: {embeddings.shape}")
    print(f"[INFO] y shape: {targets.shape}")


if __name__ == "__main__":
    main()
