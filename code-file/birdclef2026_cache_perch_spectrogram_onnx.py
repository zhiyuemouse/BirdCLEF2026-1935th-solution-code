#!/usr/bin/env python3
"""Cache Perch ONNX spectrogram outputs for BirdCLEF 2026 soundscapes.

Perch v2 ONNX exposes a frontend ``spectrogram`` output with shape
``[B, 500, 128]`` for each 5s waveform.  This script caches those frozen
frontend images for soundscape windows so downstream CNN experiments can use
Perch's own mel-like representation without re-running ONNX in every epoch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from birdclef2026_run_perch_local import (
    FILE_SAMPLES,
    N_WINDOWS,
    WINDOW_SAMPLES,
    build_target_file_list,
    parse_soundscape_filename,
    read_soundscape_60s,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Perch ONNX spectrogram outputs.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--output-dir", type=str, default="perch_spectrogram_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--file-scope", type=str, choices=["full", "labeled", "all"], default="labeled")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--batch-files", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_meta_rows(paths: Sequence[Path]) -> pd.DataFrame:
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


def save_meta(meta_df: pd.DataFrame, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = meta_path.suffix.lower()
    if suffix == ".parquet":
        meta_df.to_parquet(meta_path, index=False)
    elif suffix == ".csv":
        meta_df.to_csv(meta_path, index=False)
    else:
        raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}")


def infer_spectrograms(
    paths: Sequence[Path],
    onnx_path: Path,
    batch_files: int,
    num_threads: int,
) -> np.ndarray:
    import onnxruntime as ort

    if batch_files < 1:
        raise ValueError("--batch-files must be >= 1")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "spectrogram" not in output_names:
        raise RuntimeError(f"ONNX model has no spectrogram output. Outputs: {output_names}")

    n_rows = len(paths) * N_WINDOWS
    spectrograms = np.empty((n_rows, 500, 128), dtype=np.float32)
    start_time = time.time()

    for start in range(0, len(paths), batch_files):
        batch_paths = list(paths[start:start + batch_files])
        batch_audio = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            if len(y) != FILE_SAMPLES:
                raise ValueError(f"Unexpected audio length for {path}: {len(y)}")
            row_start = batch_idx * N_WINDOWS
            row_end = row_start + N_WINDOWS
            batch_audio[row_start:row_end] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

        spec = session.run(["spectrogram"], {input_name: batch_audio})[0].astype(np.float32, copy=False)
        if spec.shape[1:] != (500, 128):
            raise RuntimeError(f"Unexpected spectrogram shape: {spec.shape}")

        out_start = start * N_WINDOWS
        out_end = out_start + len(batch_paths) * N_WINDOWS
        spectrograms[out_start:out_end] = spec

        done = start + len(batch_paths)
        if done == len(paths) or done % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] spectrogram cache {done}/{len(paths)} files | elapsed={elapsed:.1f}s", flush=True)

    return spectrograms


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    soundscapes_dir = Path(args.soundscapes_dir) if args.soundscapes_dir else input_dir / "train_soundscapes"
    labels_path = Path(args.labels_path) if args.labels_path else input_dir / "train_soundscapes_labels.csv"
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "perch_spectrogram_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "perch_spectrogram_arrays.npz"

    if args.skip_existing and meta_path.exists() and arrays_path.exists():
        print("[INFO] Skip because output files already exist:")
        print(f"  - {meta_path}")
        print(f"  - {arrays_path}")
        return

    paths = build_target_file_list(
        labels_path=labels_path,
        soundscapes_dir=soundscapes_dir,
        file_scope=args.file_scope,
        limit_files=args.limit_files,
    )

    print("[INFO] Perch spectrogram ONNX cache")
    print(f"[INFO] onnx_path: {onnx_path}")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] file_scope: {args.file_scope}")
    print(f"[INFO] files: {len(paths)}")
    print(f"[INFO] meta_path: {meta_path}")
    print(f"[INFO] arrays_path: {arrays_path}")

    meta_df = build_meta_rows(paths)
    spectrograms = infer_spectrograms(
        paths=paths,
        onnx_path=onnx_path,
        batch_files=args.batch_files,
        num_threads=args.num_threads,
    )
    save_meta(meta_df, meta_path)
    np.savez_compressed(arrays_path, spectrogram=spectrograms)
    print("[INFO] Done.")
    print(f"[INFO] meta shape: {meta_df.shape}")
    print(f"[INFO] spectrogram shape: {spectrograms.shape}")
    print(f"[INFO] spectrogram dtype: {spectrograms.dtype}")


if __name__ == "__main__":
    main()
