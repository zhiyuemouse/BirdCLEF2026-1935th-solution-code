#!/usr/bin/env python3
"""Kaggle inference for deployable Perch context artifacts.

Supported artifact model types:

- `perch_context_logreg`
- `perch_context_mlp`
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd

from birdclef2026_run_perch_local import (
    N_WINDOWS,
    WINDOW_SAMPLES,
    build_competition_mapping,
    build_selected_proxy_targets,
    infer_perch_with_embeddings,
    load_class_names,
    load_perch_infer_fn,
    load_perch_label_table,
    parse_soundscape_filename,
    read_soundscape_60s,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Perch context LogReg Kaggle inference.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--taxonomy-path", type=str, default="")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--perch-backend", type=str, choices=["auto", "saved_model", "onnx", "tflite"], default="auto")
    parser.add_argument("--perch-onnx-path", type=str, default="")
    parser.add_argument("--perch-tflite-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--batch-files", type=int, default=32)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--proxy-reduce", type=str, choices=["max", "mean"], default="max")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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


def discover_model_path(model_path_arg: str) -> Path:
    if model_path_arg:
        model_path = Path(model_path_arg)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Explicit model artifact does not exist: {model_path}")
        return model_path

    search_roots = [Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working")]
    candidates: List[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("perch_context_logreg_artifacts.joblib"))
            candidates.extend(root.rglob("perch_context_mlp_artifacts.joblib"))
    if not candidates:
        raise FileNotFoundError(
            "No perch_context_logreg_artifacts.joblib or perch_context_mlp_artifacts.joblib found. "
            "Pass --model-path explicitly."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"[INFO] Auto-discovered model artifact: {candidates[0]}")
    return candidates[0]


def list_soundscape_files(soundscapes_dir: Path, debug: bool, debug_limit: int) -> List[Path]:
    files = sorted(path for path in soundscapes_dir.iterdir() if path.suffix == ".ogg")
    if debug:
        files = files[:debug_limit]
    return files


def discover_existing_path(explicit_path: str, candidates: Sequence[str], competition_root: Path) -> Path | None:
    if explicit_path:
        path = resolve_user_path(explicit_path, competition_root=competition_root)
        if not path.exists():
            raise FileNotFoundError(f"Explicit runtime model path does not exist: {path}")
        return path
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def map_perch_outputs_to_competition(
    logits: np.ndarray,
    embeddings: np.ndarray,
    paths: Sequence[Path],
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    emb_full = np.zeros((n_rows, 1536), dtype=np.float32)

    for file_idx, path in enumerate(paths):
        row_start = file_idx * N_WINDOWS
        row_end = row_start + N_WINDOWS
        file_logits = logits[row_start:row_end]
        scores[row_start:row_end, mapped_pos] = file_logits[:, mapped_bc_indices]
        emb_full[row_start:row_end] = embeddings[row_start:row_end]
        for pos, bc_idx_arr in proxy_pos_to_bc.items():
            sub = file_logits[:, bc_idx_arr]
            proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
            scores[row_start:row_end, pos] = proxy_score.astype(np.float32, copy=False)

        meta = parse_soundscape_filename(path.name)
        stem = path.stem
        row_ids[row_start:row_end] = [f"{stem}_{t}" for t in range(5, 65, 5)]
        filenames[row_start:row_end] = path.name
        sites[row_start:row_end] = meta["site"]
        hours[row_start:row_end] = int(meta["hour_utc"])

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours})
    return meta_df, scores, emb_full


def infer_perch_onnx(
    paths: Sequence[Path],
    onnx_path: Path,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
    num_threads: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import time

    import onnxruntime as ort

    start_time = time.time()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(num_threads)
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    paths = [Path(path) for path in paths]
    logits_parts: List[np.ndarray] = []
    emb_parts: List[np.ndarray] = []
    print(f"[INFO] Using Perch ONNX: {onnx_path}")

    for file_idx, path in enumerate(paths):
        y = read_soundscape_60s(path)
        x = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32, copy=False)
        outputs = session.run(None, {input_name: x})
        emb = None
        logits = None
        for output in outputs:
            if output.ndim == 2 and output.shape[-1] == 1536:
                emb = output.astype(np.float32, copy=False)
            elif output.ndim == 2 and output.shape[-1] >= mapped_bc_indices.max(initial=0) + 1:
                logits = output.astype(np.float32, copy=False)
        if emb is None or logits is None:
            shapes = [tuple(output.shape) for output in outputs]
            raise RuntimeError(f"Could not identify ONNX Perch outputs from shapes: {shapes}")
        logits_parts.append(logits)
        emb_parts.append(emb)
        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] ONNX Perch {file_idx + 1}/{len(paths)} files | elapsed={elapsed:.1f}s")

    logits_full = np.concatenate(logits_parts, axis=0)
    emb_full_raw = np.concatenate(emb_parts, axis=0)
    meta_df, scores, emb_full = map_perch_outputs_to_competition(
        logits=logits_full,
        embeddings=emb_full_raw,
        paths=paths,
        n_classes=n_classes,
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        proxy_reduce=proxy_reduce,
    )
    print(f"[INFO] ONNX Perch done: {len(paths)} files in {time.time() - start_time:.1f}s")
    return meta_df, scores, emb_full


def infer_perch_tflite(
    paths: Sequence[Path],
    tflite_path: Path,
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
    proxy_reduce: str,
    num_threads: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import time

    import tensorflow as tf

    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=int(num_threads))
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]["index"], [N_WINDOWS, WINDOW_SAMPLES])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    emb_index = None
    logits_index = None
    for output in output_details:
        shape = output["shape"]
        if len(shape) == 2 and shape[-1] == 1536:
            emb_index = output["index"]
        elif len(shape) == 2 and shape[-1] >= mapped_bc_indices.max(initial=0) + 1:
            logits_index = output["index"]
    if emb_index is None or logits_index is None:
        shapes = [tuple(output["shape"]) for output in output_details]
        raise RuntimeError(f"Could not identify TFLite Perch outputs from shapes: {shapes}")

    paths = [Path(path) for path in paths]
    logits_parts: List[np.ndarray] = []
    emb_parts: List[np.ndarray] = []
    print(f"[INFO] Using Perch TFLite: {tflite_path}")

    for file_idx, path in enumerate(paths):
        y = read_soundscape_60s(path)
        x = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32, copy=False)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        emb_parts.append(interpreter.get_tensor(emb_index).astype(np.float32, copy=False))
        logits_parts.append(interpreter.get_tensor(logits_index).astype(np.float32, copy=False))
        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] TFLite Perch {file_idx + 1}/{len(paths)} files | elapsed={elapsed:.1f}s")

    logits_full = np.concatenate(logits_parts, axis=0)
    emb_full_raw = np.concatenate(emb_parts, axis=0)
    meta_df, scores, emb_full = map_perch_outputs_to_competition(
        logits=logits_full,
        embeddings=emb_full_raw,
        paths=paths,
        n_classes=n_classes,
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        proxy_reduce=proxy_reduce,
    )
    print(f"[INFO] TFLite Perch done: {len(paths)} files in {time.time() - start_time:.1f}s")
    return meta_df, scores, emb_full


def infer_perch_auto_backend(
    args: argparse.Namespace,
    competition_root: Path,
    perch_dir: Path,
    soundscape_files: Sequence[Path],
    n_classes: int,
    mapped_pos: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_pos_to_bc: Dict[int, np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    onnx_path = discover_existing_path(
        args.perch_onnx_path,
        candidates=[
            "PerchV2Onnx/perch_v2.onnx",
            "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
            "/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx",
        ],
        competition_root=competition_root,
    )
    tflite_path = discover_existing_path(
        args.perch_tflite_path,
        candidates=[
            "/kaggle/input/datasets/needless090/birdclef2026-perch-tflite/perch_v2.tflite",
            "/kaggle/input/birdclef2026-perch-tflite/perch_v2.tflite",
        ],
        competition_root=competition_root,
    )

    if args.perch_backend in {"auto", "onnx"} and onnx_path is not None:
        return infer_perch_onnx(
            paths=soundscape_files,
            onnx_path=onnx_path,
            n_classes=n_classes,
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=args.proxy_reduce,
            num_threads=args.runtime_num_threads,
        )
    if args.perch_backend == "onnx":
        raise FileNotFoundError("Perch backend is `onnx`, but no ONNX model was found. Pass --perch-onnx-path.")

    if args.perch_backend in {"auto", "tflite"} and tflite_path is not None:
        return infer_perch_tflite(
            paths=soundscape_files,
            tflite_path=tflite_path,
            n_classes=n_classes,
            mapped_pos=mapped_pos,
            mapped_bc_indices=mapped_bc_indices,
            proxy_pos_to_bc=proxy_pos_to_bc,
            proxy_reduce=args.proxy_reduce,
            num_threads=args.runtime_num_threads,
        )
    if args.perch_backend == "tflite":
        raise FileNotFoundError("Perch backend is `tflite`, but no TFLite model was found. Pass --perch-tflite-path.")

    print(f"[INFO] Using Perch SavedModel backend with batch_files={args.batch_files}")
    infer_fn = load_perch_infer_fn(perch_dir)
    return infer_perch_with_embeddings(
        paths=list(soundscape_files),
        infer_fn=infer_fn,
        n_classes=n_classes,
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        batch_files=args.batch_files,
        proxy_reduce=args.proxy_reduce,
    )


def parse_end_seconds(row_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(str(row_id).rsplit("_", 1)[-1]) for row_id in row_ids], dtype=np.int64)


def build_position_features(end_seconds: np.ndarray) -> np.ndarray:
    pos = end_seconds.astype(np.float32) / 60.0
    angle = 2.0 * np.pi * pos
    return np.stack(
        [
            pos,
            np.sin(angle).astype(np.float32, copy=False),
            np.cos(angle).astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def build_metadata_features(meta_df: pd.DataFrame, include_hour_features: bool) -> np.ndarray:
    if not include_hour_features:
        return np.zeros((len(meta_df), 0), dtype=np.float32)
    hour = meta_df["hour_utc"].to_numpy(dtype=np.float32, copy=False)
    hour_phase = 2.0 * np.pi * (hour / 24.0)
    return np.stack(
        [
            hour / 24.0,
            np.sin(hour_phase).astype(np.float32, copy=False),
            np.cos(hour_phase).astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def previous_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.empty_like(values)
    out[:steps] = values[:1]
    out[steps:] = values[:-steps]
    return out


def next_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.empty_like(values)
    out[:-steps] = values[steps:]
    out[-steps:] = values[-1:]
    return out


def build_context_tensor(meta_df: pd.DataFrame, scores_full_raw: np.ndarray) -> np.ndarray:
    end_seconds = parse_end_seconds(meta_df["row_id"].tolist())
    n_rows, n_classes = scores_full_raw.shape
    context = np.zeros((n_rows, n_classes, 13), dtype=np.float32)
    eps = 1e-6

    for _, row_indices in meta_df.groupby("filename", sort=False).indices.items():
        idx = np.asarray(row_indices, dtype=np.int64)
        order = np.argsort(end_seconds[idx], kind="stable")
        idx_sorted = idx[order]
        scores = scores_full_raw[idx_sorted].astype(np.float32, copy=False)

        prev1 = previous_with_edge(scores, steps=1)
        next1 = next_with_edge(scores, steps=1)
        prev2 = previous_with_edge(scores, steps=2)
        next2 = next_with_edge(scores, steps=2)
        file_mean = np.repeat(scores.mean(axis=0, keepdims=True), len(idx_sorted), axis=0)
        file_max = np.repeat(scores.max(axis=0, keepdims=True), len(idx_sorted), axis=0)
        file_std = np.repeat(scores.std(axis=0, keepdims=True), len(idx_sorted), axis=0)
        neighbor_mean = 0.5 * (prev1 + next1)
        neighbor_max = np.maximum(prev1, next1)
        centered = scores - file_mean
        delta_prev1 = scores - prev1
        delta_next1 = scores - next1
        relative_to_file_max = scores / (file_max + eps)

        context[idx_sorted] = np.stack(
            [
                prev1,
                next1,
                prev2,
                next2,
                file_mean,
                file_max,
                file_std,
                neighbor_mean,
                neighbor_max,
                centered,
                delta_prev1,
                delta_next1,
                relative_to_file_max,
            ],
            axis=2,
        ).astype(np.float32, copy=False)

    return context


def build_base_features(
    emb_part: np.ndarray,
    raw_scores: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    parts = [emb_part, raw_scores.astype(np.float32, copy=False), position_features]
    if metadata_features.shape[1] > 0:
        parts.append(metadata_features)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


MLP_CONTEXT_CORE = [
    "prev1",
    "next1",
    "file_mean",
    "file_max",
    "file_std",
    "neighbor_mean",
    "neighbor_max",
    "centered",
    "delta_prev1",
    "delta_next1",
]


def select_mlp_context_indices(context_feature_names: Sequence[str], mode: str) -> List[int]:
    if mode == "all":
        return list(range(len(context_feature_names)))
    wanted = set(MLP_CONTEXT_CORE)
    return [idx for idx, name in enumerate(context_feature_names) if name in wanted]


def build_mlp_feature_matrix(
    emb_proj: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    context_indices: Sequence[int],
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    if feature_set in {"base", "full_context"}:
        parts.append(emb_proj.astype(np.float32, copy=False))
    parts.append(raw_scores.astype(np.float32, copy=False))
    if feature_set in {"raw_context", "full_context"}:
        ctx = context[:, :, list(context_indices)].reshape(len(raw_scores), -1)
        parts.append(ctx.astype(np.float32, copy=False))
    parts.append(position_features.astype(np.float32, copy=False))
    if metadata_features.shape[1] > 0:
        parts.append(metadata_features.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def predict_binary_logreg_proba(model, x: np.ndarray) -> np.ndarray:
    """Version-stable binary LogisticRegression probability.

    Kaggle may load our sklearn-1.8 artifact with sklearn-1.6. Calling
    `model.predict_proba()` can then fail on missing compatibility attrs such
    as `multi_class`. The learned binary model is fully represented by
    `coef_` and `intercept_`, so compute the positive-class sigmoid directly.
    """
    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)
    if coef.shape[0] != 1:
        raise ValueError(f"Expected binary LogisticRegression coef shape (1, n_features), got {coef.shape}")
    logits = x @ coef[0] + float(intercept[0])
    proba = sigmoid_np(logits).astype(np.float32, copy=False)

    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2 and int(classes[1]) != 1:
        proba = 1.0 - proba
    return proba


def transform_embedding_projector(emb: np.ndarray, fold_artifact: Dict[str, object]) -> np.ndarray:
    embedding_scaler = fold_artifact["embedding_scaler"]
    embedding_pca = fold_artifact["embedding_pca"]
    if embedding_scaler is None:
        return emb.astype(np.float32, copy=False)
    emb_scaled = embedding_scaler.transform(emb).astype(np.float32)
    if embedding_pca is None:
        return emb_scaled
    return embedding_pca.transform(emb_scaled).astype(np.float32)


def predict_context_artifact(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    emb_proj = transform_embedding_projector(emb, fold_artifact=fold_artifact)
    base = build_base_features(
        emb_part=emb_proj,
        raw_scores=raw_scores,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    base_scaled = fold_artifact["base_scaler"].transform(base).astype(np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32)

    class_models = fold_artifact["class_models"]
    context_mean = fold_artifact["context_mean"]
    context_std = fold_artifact["context_std"]
    for class_idx in fold_artifact["fitted_class_indices"]:
        class_idx = int(class_idx)
        model = class_models[class_idx]
        ctx = context[:, class_idx, :].astype(np.float32, copy=False)
        ctx_scaled = ((ctx - context_mean[class_idx]) / context_std[class_idx]).astype(np.float32, copy=False)
        x = np.concatenate([base_scaled, ctx_scaled], axis=1).astype(np.float32, copy=False)
        pred[:, class_idx] = predict_binary_logreg_proba(model, x)
    return pred.astype(np.float32, copy=False)


def predict_mlp_fold_artifact(
    fold_artifact: Dict[str, object],
    artifact_config: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch
    from torch import nn

    class PerchMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                hidden_dim = int(hidden_dim)
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(float(dropout)))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, int(output_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    feature_set = str(artifact_config.get("feature_set", "base"))
    context_mode = str(artifact_config.get("context_mode", "core"))
    context_feature_names = artifact_config.get("context_feature_names", [])
    context_indices = artifact_config.get("context_indices")
    if context_indices is None:
        context_indices = select_mlp_context_indices(
            context_feature_names=context_feature_names,
            mode=context_mode,
        )
    context_indices = [int(idx) for idx in context_indices]

    if feature_set in {"base", "full_context"}:
        emb_proj = transform_embedding_projector(emb, fold_artifact=fold_artifact)
    else:
        emb_proj = np.zeros((len(emb), 0), dtype=np.float32)

    x = build_mlp_feature_matrix(
        emb_proj=emb_proj,
        raw_scores=raw_scores,
        context=context,
        context_indices=context_indices,
        position_features=position_features,
        metadata_features=metadata_features,
        feature_set=feature_set,
    )
    x = fold_artifact["feature_scaler"].transform(x).astype(np.float32)

    model_artifact = fold_artifact["model"]
    model = PerchMLP(
        input_dim=int(model_artifact["input_dim"]),
        hidden_dims=[int(dim) for dim in model_artifact["hidden_dims"]],
        output_dim=int(model_artifact["output_dim"]),
        dropout=float(model_artifact["dropout"]),
    )
    model.load_state_dict(model_artifact["model_state"])
    model.eval()

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size])
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    mlp_all = np.concatenate(preds, axis=0)

    pred = sigmoid_np(raw_scores).astype(np.float32)
    fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int32)
    pred[:, fitted] = mlp_all[:, fitted]
    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0)


def predict_ensemble(
    artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    config = artifact["config"]
    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    metadata_features = build_metadata_features(
        meta_df=meta_df,
        include_hour_features=bool(config.get("include_hour_features", False)),
    )
    context = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    model_type = str(artifact.get("model_type", "perch_context_logreg"))
    fold_preds = []
    for fold_artifact in artifact["folds"]:
        if model_type == "perch_context_mlp":
            fold_pred = predict_mlp_fold_artifact(
                fold_artifact=fold_artifact,
                artifact_config=config,
                emb=emb_full,
                raw_scores=scores_full_raw,
                context=context,
                position_features=position_features,
                metadata_features=metadata_features,
                batch_size=batch_size,
            )
            fitted_count = len(fold_artifact["model"]["fitted_class_indices"])
        elif model_type == "perch_context_logreg":
            fold_pred = predict_context_artifact(
                fold_artifact=fold_artifact,
                emb=emb_full,
                raw_scores=scores_full_raw,
                context=context,
                position_features=position_features,
                metadata_features=metadata_features,
            )
            fitted_count = len(fold_artifact["fitted_class_indices"])
        else:
            raise ValueError(f"Unsupported artifact model_type: {model_type}")
        fold_preds.append(fold_pred)
        print(
            f"[INFO] Applied {fold_artifact.get('fold_name', 'fold')} "
            f"fitted_classes={fitted_count}"
        )

    pred = np.mean(fold_preds, axis=0).astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    competition_root = Path(args.competition_root)
    sample_submission_path = (
        resolve_user_path(args.sample_submission_path, competition_root)
        if args.sample_submission_path
        else competition_root / "sample_submission.csv"
    )
    taxonomy_path = (
        resolve_user_path(args.taxonomy_path, competition_root)
        if args.taxonomy_path
        else competition_root / "taxonomy.csv"
    )
    soundscapes_dir = (
        resolve_user_path(args.soundscapes_dir, competition_root)
        if args.soundscapes_dir
        else competition_root / ("train_soundscapes" if args.debug else "test_soundscapes")
    )
    perch_dir = resolve_user_path(args.perch_dir, competition_root)
    model_path = discover_model_path(args.model_path)
    output_path = Path(args.output_path)

    artifact = joblib.load(model_path)
    class_names = load_class_names(sample_submission_path)
    if list(artifact["class_names"]) != list(class_names):
        raise ValueError("Artifact class_names do not match sample_submission columns.")

    soundscape_files = list_soundscape_files(soundscapes_dir, debug=args.debug, debug_limit=args.debug_limit)
    if not soundscape_files:
        raise FileNotFoundError(f"No .ogg files found under {soundscapes_dir}")

    print("[INFO] Perch context LogReg inference")
    print(f"[INFO] soundscapes_dir: {soundscapes_dir}")
    print(f"[INFO] files: {len(soundscape_files)}")
    print(f"[INFO] perch_dir: {perch_dir}")
    print(f"[INFO] perch_backend: {args.perch_backend}")
    print(f"[INFO] artifact_model_type: {artifact.get('model_type', 'perch_context_logreg')}")
    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] output_path: {output_path}")
    print(f"[INFO] seed: {args.seed}")

    bc_labels = load_perch_label_table(perch_dir)
    bc_indices, mapped_bc_indices, mapping = build_competition_mapping(
        primary_labels=class_names,
        taxonomy_path=taxonomy_path,
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = build_selected_proxy_targets(primary_labels=class_names, mapping=mapping, bc_labels=bc_labels)
    meta_df, scores_full_raw, emb_full = infer_perch_auto_backend(
        args=args,
        competition_root=competition_root,
        perch_dir=perch_dir,
        soundscape_files=soundscape_files,
        n_classes=len(class_names),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
    )

    pred = predict_ensemble(
        artifact=artifact,
        meta_df=meta_df,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
        batch_size=args.batch_files * N_WINDOWS,
    )
    submission = pd.concat(
        [pd.DataFrame({"row_id": meta_df["row_id"].to_numpy()}), pd.DataFrame(pred, columns=class_names)],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved submission to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
