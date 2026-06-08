#!/usr/bin/env python3
"""Cache Perch ONNX spatial embeddings for BirdCLEF 2026 soundscapes.

The existing Perch caches keep only:

- mapped competition logits: ``scores_full_raw``
- global Perch embeddings: ``emb_full``

This script adds the frozen Perch spatial branch.  Perch v2 ONNX emits
``spatial_embedding`` with shape ``[B, 16, 4, 1536]`` for 5s windows.  We save
compact temporal tokens with shape ``[rows, 16, 1536]``:

- ``spatial_tokens``: mean over the small frequency axis
- ``spatial_tokens_max``: max over the small frequency axis, optional
- ``spatial_tokens_64``: flattened 16 x 4 spatial tokens, optional

No labels are used here.  This is only deterministic feature extraction from
the frozen Perch model, so leakage control happens in downstream fold training.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Cache Perch ONNX spatial embeddings.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--output-dir", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument(
        "--target-rows-path",
        type=str,
        default="",
        help="Optional CSV with row_id and filename columns. If set, only these 5s rows are cached.",
    )
    parser.add_argument(
        "--pseudo-root",
        type=str,
        default="",
        help="Optional fold-specific pseudo root. All fold_*/pseudo_segments.csv rows are unioned and cached.",
    )
    parser.add_argument("--file-scope", type=str, choices=["full", "labeled", "all"], default="labeled")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--batch-files", type=int, default=1)
    parser.add_argument("--clip-offset-seconds", type=float, default=0.0)
    parser.add_argument("--save-freq-max", action="store_true")
    parser.add_argument("--save-flat64", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    soundscapes_dir = Path(args.soundscapes_dir) if args.soundscapes_dir else input_dir / "train_soundscapes"
    labels_path = Path(args.labels_path) if args.labels_path else input_dir / "train_soundscapes_labels.csv"
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "perch_spatial_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "perch_spatial_arrays.npz"
    return {
        "input_dir": input_dir,
        "soundscapes_dir": soundscapes_dir,
        "labels_path": labels_path,
        "onnx_path": Path(args.onnx_path),
        "output_dir": output_dir,
        "meta_path": meta_path,
        "arrays_path": arrays_path,
    }


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


def load_target_rows(target_rows_path: Path | None, pseudo_root: Path | None) -> pd.DataFrame | None:
    frames: List[pd.DataFrame] = []
    if target_rows_path is not None:
        frames.append(pd.read_csv(target_rows_path))
    if pseudo_root is not None:
        fold_paths = sorted(pseudo_root.glob("fold_*/pseudo_segments.csv"))
        if not fold_paths:
            raise FileNotFoundError(f"No fold_*/pseudo_segments.csv files found under {pseudo_root}")
        frames.extend(pd.read_csv(path) for path in fold_paths)
    if not frames:
        return None

    target_df = pd.concat(frames, axis=0, ignore_index=True)
    required = {"row_id", "filename"}
    missing = required - set(target_df.columns)
    if missing:
        raise KeyError(f"Target rows are missing required columns: {sorted(missing)}")
    target_df = target_df.drop_duplicates(subset=["row_id"]).copy()
    target_df["filename"] = target_df["filename"].astype(str)
    target_df["row_id"] = target_df["row_id"].astype(str)
    if "end_sec" not in target_df.columns:
        target_df["end_sec"] = target_df["row_id"].str.rsplit("_", n=1).str[-1].astype(int)
    target_df["end_sec"] = target_df["end_sec"].astype(int)
    target_df = target_df.sort_values(["filename", "end_sec", "row_id"]).reset_index(drop=True)
    return target_df


def build_meta_rows_from_target_df(target_df: pd.DataFrame) -> pd.DataFrame:
    filenames = target_df["filename"].astype(str).to_numpy()
    sites = np.empty(len(target_df), dtype=object)
    hours = np.empty(len(target_df), dtype=np.int16)
    for idx, filename in enumerate(filenames):
        meta = parse_soundscape_filename(filename)
        sites[idx] = meta["site"]
        hours[idx] = int(meta["hour_utc"])
    return pd.DataFrame(
        {
            "row_id": target_df["row_id"].astype(str).to_numpy(),
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )


def save_meta(meta_df: pd.DataFrame, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = meta_path.suffix.lower()
    if suffix == ".parquet":
        meta_df.to_parquet(meta_path, index=False)
    elif suffix == ".csv":
        meta_df.to_csv(meta_path, index=False)
    else:
        raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}")


def build_shifted_windows(audio: np.ndarray, offset_samples: int) -> np.ndarray:
    windows = np.zeros((N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
    for window_idx in range(N_WINDOWS):
        base_start = window_idx * WINDOW_SAMPLES
        src_start = base_start + int(offset_samples)
        src_end = src_start + WINDOW_SAMPLES
        dst_start = max(0, -src_start)
        dst_end = WINDOW_SAMPLES - max(0, src_end - len(audio))
        clipped_start = max(0, src_start)
        clipped_end = min(len(audio), src_end)
        if clipped_end > clipped_start and dst_end > dst_start:
            windows[window_idx, dst_start:dst_end] = audio[clipped_start:clipped_end]
    return windows


def infer_spatial_tokens(
    paths: Sequence[Path],
    onnx_path: Path,
    num_threads: int,
    batch_files: int,
    clip_offset_seconds: float,
    save_freq_max: bool,
    save_flat64: bool,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import onnxruntime as ort

    if batch_files < 1:
        raise ValueError("--batch-files must be >= 1")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "spatial_embedding" not in output_names:
        raise RuntimeError(f"ONNX model has no spatial_embedding output. Outputs: {output_names}")

    n_rows = len(paths) * N_WINDOWS
    spatial_tokens = np.empty((n_rows, 16, 1536), dtype=np.float32)
    spatial_tokens_max = np.empty((n_rows, 16, 1536), dtype=np.float32) if save_freq_max else None
    spatial_tokens_64 = np.empty((n_rows, 64, 1536), dtype=np.float32) if save_flat64 else None

    start_time = time.time()
    offset_samples = int(round(float(clip_offset_seconds) * (WINDOW_SAMPLES // 5)))
    for start in range(0, len(paths), batch_files):
        batch_paths = list(paths[start:start + batch_files])
        batch_audio = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            if len(y) != FILE_SAMPLES:
                raise ValueError(f"Unexpected audio length for {path}: {len(y)}")
            row_start = batch_idx * N_WINDOWS
            row_end = row_start + N_WINDOWS
            if offset_samples == 0:
                batch_audio[row_start:row_end] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            else:
                batch_audio[row_start:row_end] = build_shifted_windows(y, offset_samples=offset_samples)

        spatial = session.run(["spatial_embedding"], {input_name: batch_audio})[0].astype(np.float32, copy=False)
        if spatial.shape[1:] != (16, 4, 1536):
            raise RuntimeError(f"Unexpected spatial_embedding shape: {spatial.shape}")

        token_start = start * N_WINDOWS
        token_end = token_start + len(batch_paths) * N_WINDOWS
        spatial_tokens[token_start:token_end] = spatial.mean(axis=2).astype(np.float32, copy=False)
        if spatial_tokens_max is not None:
            spatial_tokens_max[token_start:token_end] = spatial.max(axis=2).astype(np.float32, copy=False)
        if spatial_tokens_64 is not None:
            spatial_tokens_64[token_start:token_end] = spatial.reshape(len(batch_paths) * N_WINDOWS, 64, 1536)

        done = start + len(batch_paths)
        if done == len(paths) or done % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] spatial cache {done}/{len(paths)} files | elapsed={elapsed:.1f}s", flush=True)

    return spatial_tokens, spatial_tokens_max, spatial_tokens_64


def infer_selected_spatial_tokens(
    target_df: pd.DataFrame,
    soundscapes_dir: Path,
    onnx_path: Path,
    num_threads: int,
    batch_files: int,
    clip_offset_seconds: float,
    save_freq_max: bool,
    save_flat64: bool,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import onnxruntime as ort

    if batch_files < 1:
        raise ValueError("--batch-files must be >= 1")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if "spatial_embedding" not in output_names:
        raise RuntimeError(f"ONNX model has no spatial_embedding output. Outputs: {output_names}")

    target_df = target_df.reset_index(drop=True).copy()
    target_pos = {row_id: idx for idx, row_id in enumerate(target_df["row_id"].astype(str).tolist())}
    file_names = target_df["filename"].drop_duplicates().astype(str).tolist()
    file_paths = [soundscapes_dir / filename for filename in file_names]
    missing = [str(path) for path in file_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Some target soundscapes are missing:\n" + "\n".join(missing[:20]))

    n_rows = len(target_df)
    spatial_tokens = np.empty((n_rows, 16, 1536), dtype=np.float32)
    spatial_tokens_max = np.empty((n_rows, 16, 1536), dtype=np.float32) if save_freq_max else None
    spatial_tokens_64 = np.empty((n_rows, 64, 1536), dtype=np.float32) if save_flat64 else None
    filled = np.zeros(n_rows, dtype=bool)

    start_time = time.time()
    offset_samples = int(round(float(clip_offset_seconds) * (WINDOW_SAMPLES // 5)))
    for start in range(0, len(file_paths), batch_files):
        batch_paths = list(file_paths[start:start + batch_files])
        batch_audio = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            if len(y) != FILE_SAMPLES:
                raise ValueError(f"Unexpected audio length for {path}: {len(y)}")
            row_start = batch_idx * N_WINDOWS
            row_end = row_start + N_WINDOWS
            if offset_samples == 0:
                batch_audio[row_start:row_end] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            else:
                batch_audio[row_start:row_end] = build_shifted_windows(y, offset_samples=offset_samples)

        spatial = session.run(["spatial_embedding"], {input_name: batch_audio})[0].astype(np.float32, copy=False)
        if spatial.shape[1:] != (16, 4, 1536):
            raise RuntimeError(f"Unexpected spatial_embedding shape: {spatial.shape}")

        for batch_idx, path in enumerate(batch_paths):
            stem = path.stem
            batch_row_start = batch_idx * N_WINDOWS
            for window_idx, end_sec in enumerate(range(5, 65, 5)):
                row_id = f"{stem}_{end_sec}"
                out_idx = target_pos.get(row_id)
                if out_idx is None:
                    continue
                spatial_row = spatial[batch_row_start + window_idx]
                spatial_tokens[out_idx] = spatial_row.mean(axis=1).astype(np.float32, copy=False)
                if spatial_tokens_max is not None:
                    spatial_tokens_max[out_idx] = spatial_row.max(axis=1).astype(np.float32, copy=False)
                if spatial_tokens_64 is not None:
                    spatial_tokens_64[out_idx] = spatial_row.reshape(64, 1536)
                filled[out_idx] = True

        done = start + len(batch_paths)
        if done == len(file_paths) or done % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] selected spatial cache {done}/{len(file_paths)} files | elapsed={elapsed:.1f}s", flush=True)

    if not filled.all():
        missing_rows = target_df.loc[~filled, "row_id"].head(10).astype(str).tolist()
        raise RuntimeError(f"Failed to fill {int((~filled).sum())} selected rows. Examples: {missing_rows}")

    return spatial_tokens, spatial_tokens_max, spatial_tokens_64


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    missing = [
        str(path)
        for path in [paths["soundscapes_dir"], paths["labels_path"], paths["onnx_path"]]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    if args.skip_existing and paths["meta_path"].exists() and paths["arrays_path"].exists():
        print("[INFO] Skip because output files already exist:")
        print(f"  - {paths['meta_path']}")
        print(f"  - {paths['arrays_path']}")
        return

    target_rows = load_target_rows(
        target_rows_path=Path(args.target_rows_path) if args.target_rows_path else None,
        pseudo_root=Path(args.pseudo_root) if args.pseudo_root else None,
    )
    if target_rows is None:
        target_paths = build_target_file_list(
            labels_path=paths["labels_path"],
            soundscapes_dir=paths["soundscapes_dir"],
            file_scope=args.file_scope,
            limit_files=args.limit_files,
        )
    else:
        if args.limit_files > 0:
            keep_files = target_rows["filename"].drop_duplicates().iloc[: args.limit_files].tolist()
            target_rows = target_rows[target_rows["filename"].isin(keep_files)].reset_index(drop=True)
        target_paths = [paths["soundscapes_dir"] / name for name in target_rows["filename"].drop_duplicates().tolist()]

    print("[INFO] Perch spatial ONNX cache")
    print(f"[INFO] onnx_path: {paths['onnx_path']}")
    print(f"[INFO] soundscapes_dir: {paths['soundscapes_dir']}")
    print(f"[INFO] file_scope: {args.file_scope}")
    print(f"[INFO] clip_offset_seconds: {args.clip_offset_seconds}")
    print(f"[INFO] files: {len(target_paths)}")
    if target_rows is not None:
        print(f"[INFO] selected rows: {len(target_rows)}")
    print(f"[INFO] output meta: {paths['meta_path']}")
    print(f"[INFO] output arrays: {paths['arrays_path']}")

    if target_rows is None:
        meta_df = build_meta_rows(target_paths)
        spatial_tokens, spatial_tokens_max, spatial_tokens_64 = infer_spatial_tokens(
            paths=target_paths,
            onnx_path=paths["onnx_path"],
            num_threads=args.num_threads,
            batch_files=args.batch_files,
            clip_offset_seconds=args.clip_offset_seconds,
            save_freq_max=args.save_freq_max,
            save_flat64=args.save_flat64,
        )
    else:
        meta_df = build_meta_rows_from_target_df(target_rows)
        spatial_tokens, spatial_tokens_max, spatial_tokens_64 = infer_selected_spatial_tokens(
            target_df=target_rows,
            soundscapes_dir=paths["soundscapes_dir"],
            onnx_path=paths["onnx_path"],
            num_threads=args.num_threads,
            batch_files=args.batch_files,
            clip_offset_seconds=args.clip_offset_seconds,
            save_freq_max=args.save_freq_max,
            save_flat64=args.save_flat64,
        )

    save_meta(meta_df, paths["meta_path"])
    paths["arrays_path"].parent.mkdir(parents=True, exist_ok=True)
    arrays = {"spatial_tokens": spatial_tokens}
    if spatial_tokens_max is not None:
        arrays["spatial_tokens_max"] = spatial_tokens_max
    if spatial_tokens_64 is not None:
        arrays["spatial_tokens_64"] = spatial_tokens_64
    np.savez_compressed(paths["arrays_path"], **arrays)

    print("[INFO] Done.")
    print(f"[INFO] meta shape: {meta_df.shape}")
    print(f"[INFO] spatial_tokens shape: {spatial_tokens.shape}")
    if spatial_tokens_max is not None:
        print(f"[INFO] spatial_tokens_max shape: {spatial_tokens_max.shape}")
    if spatial_tokens_64 is not None:
        print(f"[INFO] spatial_tokens_64 shape: {spatial_tokens_64.shape}")


if __name__ == "__main__":
    main()
