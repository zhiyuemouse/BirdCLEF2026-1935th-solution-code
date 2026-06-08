#!/usr/bin/env python3
"""Cache Perch ONNX spatial tokens for train_audio clips.

This cache is intended for a lightweight stage1 pretrain of downstream Perch
heads.  Perch remains frozen; we only save deterministic 5s crop features from
``input/train_audio`` plus labels from ``train.csv``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf


SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache train_audio Perch spatial tokens with ONNX.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--train-csv-path", type=str, default="")
    parser.add_argument("--train-audio-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--output-dir", type=str, default="perch_audio_spatial_cache_max20")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--min-rating", type=float, default=-1.0)
    parser.add_argument("--include-secondary-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=-1)
    parser.add_argument("--save-flat64", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def parse_secondary_labels(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item) for item in parsed]


def deterministic_crop_offset(filename: str, n_samples: int, seed: int) -> int:
    if n_samples <= WINDOW_SAMPLES:
        return 0
    span = n_samples - WINDOW_SAMPLES
    digest = hashlib.md5(f"{seed}:{filename}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (span + 1)


def read_audio_crop(path: Path, seed: int) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected sample rate {SR}, got {sr} for {path}")
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        return y.astype(np.float32, copy=False)
    offset = deterministic_crop_offset(str(path), len(y), seed=seed)
    return y[offset:offset + WINDOW_SAMPLES].astype(np.float32, copy=False)


def select_audio_rows(
    train_csv_path: Path,
    train_audio_dir: Path,
    max_per_class: int,
    min_rating: float,
    limit_rows: int,
) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv_path)
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    train_df["filename"] = train_df["filename"].astype(str)
    if min_rating >= 0 and "rating" in train_df.columns:
        train_df = train_df[train_df["rating"].fillna(-1).astype(float) >= float(min_rating)].copy()
    train_df["audio_path"] = train_df["filename"].map(lambda name: str(train_audio_dir / name))
    exists = train_df["audio_path"].map(lambda path: Path(path).exists())
    train_df = train_df.loc[exists].copy()
    train_df = train_df.sort_values(["primary_label", "rating", "filename"], ascending=[True, False, True])
    if max_per_class > 0:
        train_df = train_df.groupby("primary_label", group_keys=False).head(int(max_per_class)).copy()
    if limit_rows > 0:
        train_df = train_df.head(int(limit_rows)).copy()
    train_df = train_df.reset_index(drop=True)
    train_df["row_id"] = [
        f"audio_{idx:06d}_{Path(filename).stem}" for idx, filename in enumerate(train_df["filename"].astype(str))
    ]
    return train_df


def build_targets(train_df: pd.DataFrame, class_names: Sequence[str], include_secondary: bool) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.zeros((len(train_df), len(class_names)), dtype=np.float32)
    for i, row in train_df.iterrows():
        labels = [str(row["primary_label"])]
        if include_secondary:
            labels.extend(parse_secondary_labels(row.get("secondary_labels", "")))
        for label in labels:
            idx = label_to_idx.get(str(label))
            if idx is not None:
                y[i, idx] = 1.0
    return y


def infer_spatial_tokens(
    audio_paths: Sequence[Path],
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
    seed: int,
    save_flat64: bool,
) -> Tuple[np.ndarray, np.ndarray | None]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "spatial_embedding" not in output_names:
        raise RuntimeError(f"ONNX model has no spatial_embedding output. Outputs: {output_names}")

    n_rows = len(audio_paths)
    spatial_tokens = np.empty((n_rows, 16, 1536), dtype=np.float32)
    spatial_tokens_64 = np.empty((n_rows, 64, 1536), dtype=np.float32) if save_flat64 else None
    start_time = time.time()
    batch_size = max(1, int(batch_size))
    for start in range(0, n_rows, batch_size):
        batch_paths = list(audio_paths[start:start + batch_size])
        batch_audio = np.empty((len(batch_paths), WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            batch_audio[batch_idx] = read_audio_crop(path, seed=seed)
        spatial = session.run(["spatial_embedding"], {input_name: batch_audio})[0].astype(np.float32, copy=False)
        if spatial.shape[1:] != (16, 4, 1536):
            raise RuntimeError(f"Unexpected spatial_embedding shape: {spatial.shape}")
        end = start + len(batch_paths)
        spatial_tokens[start:end] = spatial.mean(axis=2).astype(np.float32, copy=False)
        if spatial_tokens_64 is not None:
            spatial_tokens_64[start:end] = spatial.reshape(len(batch_paths), 64, 1536)
        if end == n_rows or end % 500 == 0:
            print(f"[INFO] audio spatial cache {end}/{n_rows} rows | elapsed={time.time() - start_time:.1f}s", flush=True)
    return spatial_tokens, spatial_tokens_64


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
    input_dir = Path(args.input_dir)
    train_csv_path = Path(args.train_csv_path) if args.train_csv_path else input_dir / "train.csv"
    train_audio_dir = Path(args.train_audio_dir) if args.train_audio_dir else input_dir / "train_audio"
    sample_submission_path = (
        Path(args.sample_submission_path) if args.sample_submission_path else input_dir / "sample_submission.csv"
    )
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "perch_audio_spatial_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "perch_audio_spatial_arrays.npz"

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

    print("[INFO] Perch audio spatial ONNX cache")
    print(f"[INFO] onnx_path: {onnx_path}")
    print(f"[INFO] train_csv_path: {train_csv_path}")
    print(f"[INFO] train_audio_dir: {train_audio_dir}")
    print(f"[INFO] rows: {len(audio_df)}")
    print(f"[INFO] classes with audio rows: {audio_df['primary_label'].nunique()}")
    print(f"[INFO] target positives: {int(targets.sum())}")
    print(f"[INFO] output meta: {meta_path}")
    print(f"[INFO] output arrays: {arrays_path}")

    spatial_tokens, spatial_tokens_64 = infer_spatial_tokens(
        audio_paths=audio_paths,
        onnx_path=onnx_path,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        save_flat64=args.save_flat64,
    )
    save_meta(audio_df, meta_path)
    arrays = {"spatial_tokens": spatial_tokens, "y": targets.astype(np.float32, copy=False)}
    if spatial_tokens_64 is not None:
        arrays["spatial_tokens_64"] = spatial_tokens_64
    np.savez_compressed(arrays_path, **arrays)

    print("[INFO] Done.")
    print(f"[INFO] meta shape: {audio_df.shape}")
    print(f"[INFO] spatial_tokens shape: {spatial_tokens.shape}")
    print(f"[INFO] y shape: {targets.shape}")


if __name__ == "__main__":
    main()
