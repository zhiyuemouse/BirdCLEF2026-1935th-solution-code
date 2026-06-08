#!/usr/bin/env python3
"""Cache full-file Perch embeddings and mapped logits with ONNX.

This is the sequence-friendly companion to the spatial cache.  It always saves
complete 12-window soundscapes so downstream SSM models can use unlabeled
windows as context without treating them as negative labels.
"""

from __future__ import annotations

import argparse
import time
import re
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

import birdclef2026_perch_kaggle_infer_spatial_mamba as perch_utils


SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12


def parse_soundscape_filename(name: str) -> dict[str, object]:
    stem = Path(name).stem
    match = re.match(r"^.+?_(?:Train|Test)_.+?_(S\d+)_(\d{8})_(\d{6})$", stem)
    if not match:
        return {"site": "unknown", "hour_utc": -1, "month": -1}
    site, yyyymmdd, hhmmss = match.groups()
    return {"site": site, "hour_utc": int(hhmmss[:2]), "month": int(yyyymmdd[4:6])}


def build_target_file_list(labels_path: Path, soundscapes_dir: Path, file_scope: str, limit_files: int) -> list[Path]:
    if file_scope == "all":
        file_names = sorted(path.name for path in soundscapes_dir.glob("*.ogg"))
    elif file_scope == "labeled":
        raw = pd.read_csv(labels_path)
        file_names = sorted(raw["filename"].drop_duplicates().tolist())
    else:
        raw = pd.read_csv(labels_path)
        sc_clean = raw.drop_duplicates(subset=["filename", "start", "end"]).copy()
        windows_per_file = sc_clean.groupby("filename").size()
        file_names = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    if limit_files > 0:
        file_names = file_names[:limit_files]
    paths = [soundscapes_dir / name for name in file_names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Some target soundscapes are missing:\n" + "\n".join(missing[:20]))
    return paths


def read_soundscape_60s(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected sample rate {SR}, got {sr} for {path}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES].astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache full-file Perch sequence features with ONNX.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--taxonomy-path", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--output-dir", type=str, default="perch_sequence_cache_labeled_all_full")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--file-scope", type=str, choices=["full", "labeled", "all"], default="labeled")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--batch-files", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--proxy-reduce", type=str, choices=["max", "mean"], default="max")
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
    if meta_path.suffix.lower() == ".parquet":
        meta_df.to_parquet(meta_path, index=False)
    elif meta_path.suffix.lower() == ".csv":
        meta_df.to_csv(meta_path, index=False)
    else:
        raise ValueError(f"Unsupported meta suffix: {meta_path.suffix}")


def infer_sequence_features(
    paths: Sequence[Path],
    onnx_path: Path,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: dict[int, np.ndarray],
    proxy_reduce: str,
    batch_files: int,
    num_threads: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    required = {"embedding", "label"}
    missing = required - set(output_names)
    if missing:
        raise RuntimeError(f"Perch ONNX missing outputs {sorted(missing)}. Outputs: {output_names}")

    n_rows = len(paths) * N_WINDOWS
    emb_full = np.empty((n_rows, 1536), dtype=np.float32)
    scores_full_raw = np.empty((n_rows, n_classes), dtype=np.float32)
    start_time = time.time()

    for start in range(0, len(paths), max(1, int(batch_files))):
        batch_paths = list(paths[start:start + max(1, int(batch_files))])
        x = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for batch_idx, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            if len(y) != FILE_SAMPLES:
                raise ValueError(f"Unexpected audio length for {path}: {len(y)}")
            row_start = batch_idx * N_WINDOWS
            x[row_start:row_start + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

        embedding, logits = session.run(["embedding", "label"], {input_name: x})
        embedding = embedding.astype(np.float32, copy=False)
        logits = logits.astype(np.float32, copy=False)
        scores = perch_utils.map_logits_to_competition(
            logits=logits,
            n_classes=n_classes,
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=proxy_reduce,
        )
        out_start = start * N_WINDOWS
        out_end = out_start + len(batch_paths) * N_WINDOWS
        emb_full[out_start:out_end] = embedding
        scores_full_raw[out_start:out_end] = scores

        done = start + len(batch_paths)
        if done == len(paths) or done % 10 == 0:
            print(f"[INFO] sequence cache {done}/{len(paths)} files | elapsed={time.time() - start_time:.1f}s", flush=True)

    return scores_full_raw, emb_full


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    soundscapes_dir = Path(args.soundscapes_dir) if args.soundscapes_dir else input_dir / "train_soundscapes"
    labels_path = Path(args.labels_path) if args.labels_path else input_dir / "train_soundscapes_labels.csv"
    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else input_dir / "taxonomy.csv"
    sample_submission_path = (
        Path(args.sample_submission_path) if args.sample_submission_path else input_dir / "sample_submission.csv"
    )
    perch_dir = Path(args.perch_dir)
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "perch_sequence_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "perch_sequence_arrays.npz"

    if args.skip_existing and meta_path.exists() and arrays_path.exists():
        print("[INFO] Skip because output files already exist:")
        print(f"  - {meta_path}")
        print(f"  - {arrays_path}")
        return

    class_names = perch_utils.load_class_names(sample_submission_path)
    bc_labels = perch_utils.load_perch_label_table(perch_dir=perch_dir, onnx_path=onnx_path)
    bc_indices, mapped_bc_indices, mapping = perch_utils.build_competition_mapping(
        primary_labels=class_names,
        taxonomy_path=taxonomy_path,
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = perch_utils.build_selected_proxy_targets(
        primary_labels=class_names,
        mapping=mapping,
        bc_labels=bc_labels,
    )
    paths = build_target_file_list(
        labels_path=labels_path,
        soundscapes_dir=soundscapes_dir,
        file_scope=args.file_scope,
        limit_files=args.limit_files,
    )

    print("[INFO] Perch sequence ONNX cache")
    print(f"[INFO] onnx_path: {onnx_path}")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] file_scope: {args.file_scope}")
    print(f"[INFO] files: {len(paths)}")
    print(f"[INFO] meta_path: {meta_path}")
    print(f"[INFO] arrays_path: {arrays_path}")

    meta_df = build_meta_rows(paths)
    scores_full_raw, emb_full = infer_sequence_features(
        paths=paths,
        onnx_path=onnx_path,
        n_classes=len(class_names),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        proxy_reduce=args.proxy_reduce,
        batch_files=args.batch_files,
        num_threads=args.num_threads,
    )
    save_meta(meta_df, meta_path)
    np.savez_compressed(arrays_path, scores_full_raw=scores_full_raw, emb_full=emb_full)
    print("[INFO] Done.")
    print(f"[INFO] meta shape: {meta_df.shape}")
    print(f"[INFO] scores_full_raw shape: {scores_full_raw.shape}")
    print(f"[INFO] emb_full shape: {emb_full.shape}")


if __name__ == "__main__":
    main()
