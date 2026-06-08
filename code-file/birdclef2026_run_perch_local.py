#!/usr/bin/env python3
"""Run Perch v2 locally and save BirdCLEF-style cache files.

This script is designed to mirror the public BirdCLEF 2026 Perch notebooks:

1. Read local soundscapes from `input/train_soundscapes`
2. Split each 60s file into 12 x 5s windows
3. Run local Perch v2 (`saved_model.pb`) on CPU
4. Map Perch logits to the competition's 234 classes
5. Save:
   - `full_perch_meta.parquet` / `.csv`
   - `full_perch_arrays.npz`

By default, it only caches `full_files`, i.e. soundscapes that have all 12
windows present in `train_soundscapes_labels.csv`, which is what the public
Perch/ProtoSSM notebooks use. Use `--file-scope labeled` to cache every
manually labeled row, including partial soundscapes.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local Perch cache for BirdCLEF 2026.")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--taxonomy-path", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--output-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--file-scope", type=str, choices=["full", "labeled", "all"], default="full")
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--proxy-reduce", type=str, choices=["max", "mean"], default="max")
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    soundscapes_dir = Path(args.soundscapes_dir) if args.soundscapes_dir else input_dir / "train_soundscapes"
    labels_path = Path(args.labels_path) if args.labels_path else input_dir / "train_soundscapes_labels.csv"
    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else input_dir / "taxonomy.csv"
    sample_submission_path = (
        Path(args.sample_submission_path) if args.sample_submission_path else input_dir / "sample_submission.csv"
    )
    perch_dir = Path(args.perch_dir)
    meta_path = Path(args.meta_path) if args.meta_path else output_dir / "full_perch_meta.parquet"
    arrays_path = Path(args.arrays_path) if args.arrays_path else output_dir / "full_perch_arrays.npz"

    return {
        "input_dir": input_dir,
        "soundscapes_dir": soundscapes_dir,
        "labels_path": labels_path,
        "taxonomy_path": taxonomy_path,
        "sample_submission_path": sample_submission_path,
        "perch_dir": perch_dir,
        "output_dir": output_dir,
        "meta_path": meta_path,
        "arrays_path": arrays_path,
    }


def ensure_required_files(paths: Dict[str, Path]) -> None:
    required = [
        paths["soundscapes_dir"],
        paths["labels_path"],
        paths["taxonomy_path"],
        paths["sample_submission_path"],
        paths["perch_dir"] / "saved_model.pb",
        paths["perch_dir"] / "assets" / "labels.csv",
        paths["perch_dir"] / "variables" / "variables.index",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def ensure_meta_writer_available(meta_path: Path) -> None:
    suffix = meta_path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            try:
                import fastparquet  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Saving parquet requires `pyarrow` or `fastparquet` in the `perch` env. "
                    "Either install one of them, or pass `--meta-path ...csv` for a quick smoke test."
                ) from exc
    elif suffix == ".csv":
        return
    else:
        raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}. Use .parquet or .csv")


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def load_perch_label_table(perch_dir: Path) -> pd.DataFrame:
    bc_labels = (
        pd.read_csv(perch_dir / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    if "scientific_name" not in bc_labels.columns:
        raise KeyError("Perch labels.csv must contain the `inat2024_fsd50k` column.")
    return bc_labels


def build_competition_mapping(
    primary_labels: List[str],
    taxonomy_path: Path,
    bc_labels: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    taxonomy = pd.read_csv(taxonomy_path)
    taxonomy = taxonomy.copy()
    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]

    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = taxonomy.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    no_label_index = len(bc_labels)
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)

    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    bc_indices = np.array([int(label_to_bc_index.get(label, no_label_index)) for label in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label_index
    mapped_bc_indices = bc_indices[mapped_mask].astype(np.int32)
    return bc_indices, mapped_bc_indices, mapping


def build_selected_proxy_targets(
    primary_labels: List[str],
    mapping: pd.DataFrame,
    bc_labels: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    class_name_map = mapping.set_index("primary_label")["class_name"].to_dict()
    label_to_idx = {label: i for i, label in enumerate(primary_labels)}
    no_label_index = len(bc_labels)

    unmapped_df = mapping[mapping["bc_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    proxy_pos_to_bc: Dict[int, np.ndarray] = {}
    scientific_name_series = bc_labels["scientific_name"].astype(str)

    for _, row in unmapped_non_sonotype.iterrows():
        target = str(row["primary_label"])
        if target not in label_to_idx:
            continue
        if class_name_map.get(target) not in PROXY_TAXA:
            continue
        genus = str(row["scientific_name"]).split()[0]
        hits = bc_labels[scientific_name_series.str.match(rf"^{re.escape(genus)}\s", na=False)].copy()
        if len(hits) == 0:
            continue
        proxy_pos_to_bc[label_to_idx[target]] = hits["bc_index"].to_numpy(dtype=np.int32)

    return proxy_pos_to_bc


def parse_soundscape_filename(name: str) -> Dict[str, object]:
    stem = Path(name).stem
    match = re.match(r"^.+?_(?:Train|Test)_.+?_(S\d+)_(\d{8})_(\d{6})$", stem)
    if not match:
        return {"site": "unknown", "hour_utc": -1, "month": -1}
    site, yyyymmdd, hhmmss = match.groups()
    return {"site": site, "hour_utc": int(hhmmss[:2]), "month": int(yyyymmdd[4:6])}


def build_target_file_list(labels_path: Path, soundscapes_dir: Path, file_scope: str, limit_files: int) -> List[Path]:
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

    file_paths = [soundscapes_dir / name for name in file_names]
    missing = [str(path) for path in file_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Some target soundscapes are missing:\n" + "\n".join(missing[:20]))
    return file_paths


def build_labeled_row_id_set(labels_path: Path) -> set[str]:
    raw = pd.read_csv(labels_path)
    sc_clean = raw.drop_duplicates(subset=["filename", "start", "end"]).copy()
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    row_ids = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
    return set(row_ids.tolist())


def read_soundscape_60s(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32, copy=False)


def load_perch_infer_fn(perch_dir: Path):
    saved_model_path = perch_dir / "saved_model.pb"
    if saved_model_path.exists() and b"vhlo.cosine_v2" in saved_model_path.read_bytes():
        raise RuntimeError(
            "This Perch SavedModel contains `vhlo.cosine_v2`, which is known to be incompatible with "
            "the current Kaggle TensorFlow/StableHLO runtime. Upload and use the locally validated "
            "`Perch/` SavedModel folder instead of the Kaggle Models `google/bird-vocalization-classifier` path."
        )
    model = tf.saved_model.load(str(perch_dir))
    return model.signatures["serving_default"]


def infer_perch_with_embeddings(
    paths: List[Path],
    infer_fn,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    batch_files: int,
    proxy_reduce: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    total_batches = (n_files + batch_files - 1) // batch_files
    iterator: Iterable[int] = range(0, n_files, batch_files)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_batches, desc="Perch batches")
    else:
        print(
            f"[INFO] Perch batches: total={total_batches}, files={n_files}, batch_files={batch_files}",
            flush=True,
        )

    progress_start_time = time.time()
    for batch_idx, start in enumerate(iterator, start=1):
        batch_paths = paths[start:start + batch_files]
        batch_n = len(batch_paths)
        batch_row_start = write_row
        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)

        x_pos = 0
        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(path.name)
            stem = path.stem
            row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32, copy=False)
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

        scores[batch_row_start:write_row, mapped_pos] = logits[:, mapped_bc_indices]
        embeddings[batch_row_start:write_row] = emb

        for pos, bc_idx_arr in proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32, copy=False)

        if tqdm is None and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0):
            elapsed = time.time() - progress_start_time
            files_done = min(start + batch_n, n_files)
            files_per_sec = files_done / max(elapsed, 1e-6)
            remaining_files = max(n_files - files_done, 0)
            eta_sec = remaining_files / max(files_per_sec, 1e-6)
            print(
                "[INFO] Perch batches "
                f"{batch_idx}/{total_batches} | files={files_done}/{n_files} "
                f"| elapsed={elapsed / 60.0:.1f}m | speed={files_per_sec:.2f} files/s "
                f"| eta={eta_sec / 60.0:.1f}m",
                flush=True,
            )

    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )
    return meta_df, scores, embeddings


def save_meta(meta_df: pd.DataFrame, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = meta_path.suffix.lower()
    if suffix == ".parquet":
        meta_df.to_parquet(meta_path, index=False)
    elif suffix == ".csv":
        meta_df.to_csv(meta_path, index=False)
    else:
        raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    ensure_required_files(paths)
    ensure_meta_writer_available(paths["meta_path"])

    if args.skip_existing and paths["meta_path"].exists() and paths["arrays_path"].exists():
        print("[INFO] Skip because output files already exist:")
        print(f"  - {paths['meta_path']}")
        print(f"  - {paths['arrays_path']}")
        return

    primary_labels = load_class_names(paths["sample_submission_path"])
    bc_labels = load_perch_label_table(paths["perch_dir"])
    bc_indices, mapped_bc_indices, mapping = build_competition_mapping(
        primary_labels=primary_labels,
        taxonomy_path=paths["taxonomy_path"],
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = build_selected_proxy_targets(primary_labels=primary_labels, mapping=mapping, bc_labels=bc_labels)
    target_paths = build_target_file_list(
        labels_path=paths["labels_path"],
        soundscapes_dir=paths["soundscapes_dir"],
        file_scope=args.file_scope,
        limit_files=args.limit_files,
    )

    print("[INFO] Local Perch cache build")
    print(f"[INFO] Perch model dir: {paths['perch_dir']}")
    print(f"[INFO] Soundscapes dir: {paths['soundscapes_dir']}")
    print(f"[INFO] File scope: {args.file_scope}")
    print(f"[INFO] Number of files: {len(target_paths)}")
    print(f"[INFO] Mapped competition classes: {len(mapped_pos)} / {len(primary_labels)}")
    print(f"[INFO] Proxy competition classes: {len(proxy_pos_to_bc)}")
    print(f"[INFO] Proxy reduce: {args.proxy_reduce}")
    print(f"[INFO] Meta output: {paths['meta_path']}")
    print(f"[INFO] Arrays output: {paths['arrays_path']}")

    infer_fn = load_perch_infer_fn(paths["perch_dir"])
    meta_df, scores_full_raw, emb_full = infer_perch_with_embeddings(
        paths=target_paths,
        infer_fn=infer_fn,
        n_classes=len(primary_labels),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        batch_files=args.batch_files,
        proxy_reduce=args.proxy_reduce,
    )

    if args.file_scope == "labeled":
        labeled_row_ids = build_labeled_row_id_set(paths["labels_path"])
        keep_mask = meta_df["row_id"].isin(labeled_row_ids).to_numpy()
        encoded_rows = len(meta_df)
        meta_df = meta_df.loc[keep_mask].reset_index(drop=True)
        scores_full_raw = scores_full_raw[keep_mask]
        emb_full = emb_full[keep_mask]
        print(f"[INFO] Labeled-row filter: kept {len(meta_df)} / {encoded_rows} encoded windows")

    save_meta(meta_df, paths["meta_path"])
    paths["arrays_path"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["arrays_path"], scores_full_raw=scores_full_raw, emb_full=emb_full)

    print("[INFO] Done.")
    print(f"[INFO] meta shape: {meta_df.shape}")
    print(f"[INFO] scores_full_raw shape: {scores_full_raw.shape}")
    print(f"[INFO] emb_full shape: {emb_full.shape}")


if __name__ == "__main__":
    main()
