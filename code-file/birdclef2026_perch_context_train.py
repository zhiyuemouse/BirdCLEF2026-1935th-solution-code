#!/usr/bin/env python3
"""Train deployable Perch + temporal context LogisticRegression folds.

This is the deployment-oriented companion to `birdclef2026_perch_context_logreg.py`.
It keeps the same leakage-aware local CV, but also saves all fold artifacts
needed by a Kaggle inference script:

- embedding scaler + PCA
- base feature scaler
- per-class context scalers
- per-class LogisticRegression models

The saved model is intentionally small and CPU-friendly. Perch produces the
heavy features; these fold models only re-rank them with lightweight context.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deployable Perch context LogReg folds.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_context_deploy")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="",
        help="Optional CSV with row_id and fold columns. Use this to align Perch folds with a CNN run.",
    )
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--embedding-pca-dim", type=int, default=128)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument("--logreg-min-pos", type=int, default=8)
    parser.add_argument("--include-hour-features", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_label_cell(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def union_labels(series: Sequence[object]) -> List[str]:
    merged = set()
    for value in series:
        merged.update(parse_label_cell(value))
    return sorted(merged)


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def load_meta(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() == ".parquet":
        return pd.read_parquet(meta_path)
    if meta_path.suffix.lower() == ".csv":
        return pd.read_csv(meta_path)
    raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}")


def load_cache(cache_dir: Path, meta_path_arg: str, arrays_path_arg: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_candidates: List[Path] = []
    arrays_candidates: List[Path] = []

    if meta_path_arg:
        meta_candidates.append(Path(meta_path_arg))
    else:
        meta_candidates.extend([cache_dir / "perch_meta.parquet", cache_dir / "full_perch_meta.parquet"])

    if arrays_path_arg:
        arrays_candidates.append(Path(arrays_path_arg))
    else:
        arrays_candidates.extend([cache_dir / "perch_arrays.npz", cache_dir / "full_perch_arrays.npz"])

    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find Perch meta file under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find Perch arrays file under {cache_dir}")

    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    scores_full_raw = arrays["scores_full_raw"].astype(np.float32, copy=False)
    emb_full = arrays["emb_full"].astype(np.float32, copy=False)
    return meta_df, scores_full_raw, emb_full


def load_fold_assignments(fold_assignment_path: Path, meta_df: pd.DataFrame) -> np.ndarray:
    fold_df = pd.read_csv(fold_assignment_path)
    required = {"row_id", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise KeyError(f"Fold assignment file is missing columns: {sorted(missing)}")
    fold_map = fold_df.drop_duplicates(subset=["row_id"]).set_index("row_id")["fold"]
    folds = meta_df["row_id"].map(fold_map)
    missing_rows = meta_df.loc[folds.isna(), "row_id"].astype(str).tolist()
    if missing_rows:
        raise ValueError(
            f"Fold assignment file does not cover {len(missing_rows)} Perch rows. "
            f"Examples: {missing_rows[:5]}"
        )
    return folds.astype(int).to_numpy()


def build_aligned_labels(labels_path: Path, class_names: Sequence[str], meta_df: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(labels_path)
    sc_clean = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean = sc_clean.reset_index(names="orig_index")
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.zeros((len(sc_clean), len(class_names)), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[label] for label in labels if label in label_to_idx]
        if idxs:
            y[i, idxs] = 1

    aligned = sc_clean.set_index("row_id").loc[meta_df["row_id"]].reset_index()
    if not np.all(aligned["filename"].values == meta_df["filename"].values):
        raise AssertionError("Meta and label filename order mismatch after row_id alignment.")
    return y[aligned["orig_index"].to_numpy()]


def limit_by_files(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
    limit_files: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if limit_files <= 0:
        return meta_df, y_true, scores_full_raw, emb_full
    keep_files = meta_df["filename"].drop_duplicates().iloc[:limit_files].tolist()
    keep_mask = meta_df["filename"].isin(keep_files).values
    return (
        meta_df.loc[keep_mask].reset_index(drop=True),
        y_true[keep_mask],
        scores_full_raw[keep_mask],
        emb_full[keep_mask],
    )


def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true.sum(axis=0)
    neg = y_true.shape[0] - pos
    keep = (pos > 0) & (neg > 0)
    if keep.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


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


def build_metadata_features(meta_df: pd.DataFrame, include_hour_features: bool) -> Tuple[np.ndarray, List[str]]:
    if not include_hour_features:
        return np.zeros((len(meta_df), 0), dtype=np.float32), []

    hour = meta_df["hour_utc"].to_numpy(dtype=np.float32, copy=False)
    hour_phase = 2.0 * np.pi * (hour / 24.0)
    hour_feat = np.stack(
        [
            hour / 24.0,
            np.sin(hour_phase).astype(np.float32, copy=False),
            np.cos(hour_phase).astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return hour_feat, ["hour_norm", "hour_sin", "hour_cos"]


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


def build_context_tensor(meta_df: pd.DataFrame, scores_full_raw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    end_seconds = parse_end_seconds(meta_df["row_id"].tolist())
    n_rows, n_classes = scores_full_raw.shape
    feature_names = [
        "prev1",
        "next1",
        "prev2",
        "next2",
        "file_mean",
        "file_max",
        "file_std",
        "neighbor_mean",
        "neighbor_max",
        "centered",
        "delta_prev1",
        "delta_next1",
        "relative_to_file_max",
    ]
    context = np.zeros((n_rows, n_classes, len(feature_names)), dtype=np.float32)
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

    return context, feature_names


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


def fit_embedding_projector(emb_train: np.ndarray, pca_dim: int, seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    if pca_dim <= 0:
        return emb_train.astype(np.float32, copy=False), {
            "embedding_scaler": None,
            "embedding_pca": None,
            "actual_embedding_dim": int(emb_train.shape[1]),
        }

    embedding_scaler = StandardScaler()
    emb_scaled = embedding_scaler.fit_transform(emb_train).astype(np.float32)
    max_dim = min(pca_dim, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
    if max_dim < 1:
        return emb_scaled, {
            "embedding_scaler": embedding_scaler,
            "embedding_pca": None,
            "actual_embedding_dim": int(emb_scaled.shape[1]),
        }

    embedding_pca = PCA(n_components=max_dim, random_state=seed)
    emb_proj = embedding_pca.fit_transform(emb_scaled).astype(np.float32)
    return emb_proj, {
        "embedding_scaler": embedding_scaler,
        "embedding_pca": embedding_pca,
        "actual_embedding_dim": int(max_dim),
    }


def transform_embedding_projector(emb: np.ndarray, fold_artifact: Dict[str, object]) -> np.ndarray:
    embedding_scaler = fold_artifact["embedding_scaler"]
    embedding_pca = fold_artifact["embedding_pca"]
    if embedding_scaler is None:
        return emb.astype(np.float32, copy=False)
    emb_scaled = embedding_scaler.transform(emb).astype(np.float32)
    if embedding_pca is None:
        return emb_scaled
    return embedding_pca.transform(emb_scaled).astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_context_artifact(
    emb_train: np.ndarray,
    raw_scores_train: np.ndarray,
    context_train: np.ndarray,
    position_train: np.ndarray,
    metadata_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
    min_pos: int,
    embedding_pca_dim: int,
    seed: int,
    fold_name: str,
) -> Dict[str, object]:
    emb_train_proj, projector = fit_embedding_projector(emb_train=emb_train, pca_dim=embedding_pca_dim, seed=seed)
    base_train = build_base_features(
        emb_part=emb_train_proj,
        raw_scores=raw_scores_train,
        position_features=position_train,
        metadata_features=metadata_train,
    )
    base_scaler = StandardScaler()
    base_train_scaled = base_scaler.fit_transform(base_train).astype(np.float32)

    n_classes = y_train.shape[1]
    n_context = context_train.shape[2]
    class_models: List[object] = [None] * n_classes
    context_mean = np.zeros((n_classes, n_context), dtype=np.float32)
    context_std = np.ones((n_classes, n_context), dtype=np.float32)
    fitted_class_indices: List[int] = []

    for class_idx in range(n_classes):
        target = y_train[:, class_idx]
        pos = int(target.sum())
        neg = int(len(target) - pos)
        if pos < min_pos or neg == 0:
            continue

        ctx_train = context_train[:, class_idx, :].astype(np.float32, copy=False)
        mean = ctx_train.mean(axis=0, keepdims=True)
        std = ctx_train.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        ctx_train_scaled = ((ctx_train - mean) / std).astype(np.float32, copy=False)
        x_train = np.concatenate([base_train_scaled, ctx_train_scaled], axis=1).astype(np.float32, copy=False)

        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        model.fit(x_train, target)

        class_models[class_idx] = model
        context_mean[class_idx] = mean.reshape(-1).astype(np.float32, copy=False)
        context_std[class_idx] = std.reshape(-1).astype(np.float32, copy=False)
        fitted_class_indices.append(class_idx)

    return {
        "fold_name": fold_name,
        "embedding_scaler": projector["embedding_scaler"],
        "embedding_pca": projector["embedding_pca"],
        "actual_embedding_dim": int(projector["actual_embedding_dim"]),
        "base_scaler": base_scaler,
        "class_models": class_models,
        "context_mean": context_mean,
        "context_std": context_std,
        "fitted_class_indices": np.asarray(fitted_class_indices, dtype=np.int32),
    }


def predict_context_artifact(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    sigmoid_fallback: bool,
) -> np.ndarray:
    emb_proj = transform_embedding_projector(emb, fold_artifact=fold_artifact)
    base = build_base_features(
        emb_part=emb_proj,
        raw_scores=raw_scores,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    base_scaled = fold_artifact["base_scaler"].transform(base).astype(np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32) if sigmoid_fallback else raw_scores.astype(np.float32, copy=True)

    class_models = fold_artifact["class_models"]
    context_mean = fold_artifact["context_mean"]
    context_std = fold_artifact["context_std"]
    for class_idx in fold_artifact["fitted_class_indices"]:
        class_idx = int(class_idx)
        model = class_models[class_idx]
        ctx = context[:, class_idx, :].astype(np.float32, copy=False)
        ctx_scaled = ((ctx - context_mean[class_idx]) / context_std[class_idx]).astype(np.float32, copy=False)
        x = np.concatenate([base_scaled, ctx_scaled], axis=1).astype(np.float32, copy=False)
        pred[:, class_idx] = model.predict_proba(x)[:, 1].astype(np.float32, copy=False)

    return pred.astype(np.float32, copy=False)


@dataclass
class FoldResult:
    fold: int
    raw_valid_auc: float
    probe_valid_auc: float
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    actual_embedding_dim: int
    fitted_classes: int


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df, scores_full_raw, emb_full = load_cache(
        cache_dir=Path(args.cache_dir),
        meta_path_arg=args.meta_path,
        arrays_path_arg=args.arrays_path,
    )
    class_names = load_class_names(Path(args.sample_submission_path))
    y_true = build_aligned_labels(
        labels_path=Path(args.labels_path),
        class_names=class_names,
        meta_df=meta_df,
    )
    meta_df, y_true, scores_full_raw, emb_full = limit_by_files(
        meta_df=meta_df,
        y_true=y_true,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
        limit_files=args.limit_files,
    )

    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    metadata_features, metadata_feature_names = build_metadata_features(
        meta_df=meta_df,
        include_hour_features=args.include_hour_features,
    )
    context_tensor, context_feature_names = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need {args.n_folds}.")

    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    if args.fold_assignment_path:
        row_folds = load_fold_assignments(Path(args.fold_assignment_path), meta_df=meta_df)
        fold_values = sorted(pd.Index(row_folds).unique().tolist())
        if len(fold_values) != args.n_folds:
            raise ValueError(
                f"--n-folds={args.n_folds}, but fold assignment has {len(fold_values)} folds: {fold_values}"
            )
        fold_splits = []
        all_idx = np.arange(len(meta_df), dtype=np.int64)
        for fold_value in fold_values:
            valid_idx = np.where(row_folds == int(fold_value))[0]
            train_idx = all_idx[row_folds != int(fold_value)]
            fold_splits.append((int(fold_value), train_idx, valid_idx))
        fold_source = f"assignment:{args.fold_assignment_path}"
    else:
        gkf = GroupKFold(n_splits=args.n_folds)
        fold_splits = [
            (fold, np.asarray(train_idx), np.asarray(valid_idx))
            for fold, (train_idx, valid_idx) in enumerate(gkf.split(meta_df, groups=groups), start=1)
        ]
        fold_source = "GroupKFold(filename)"

    print("[INFO] Train deployable Perch + context LogReg")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] embedding_pca_dim: {args.embedding_pca_dim}")
    print(f"[INFO] logreg_c: {args.logreg_c}")
    print(f"[INFO] logreg_min_pos: {args.logreg_min_pos}")
    print(f"[INFO] include_hour_features: {args.include_hour_features}")
    print(f"[INFO] fold_source: {fold_source}")

    for fold, train_idx, valid_idx in fold_splits:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        fold_artifact = fit_context_artifact(
            emb_train=emb_full[train_idx],
            raw_scores_train=scores_full_raw[train_idx],
            context_train=context_tensor[train_idx],
            position_train=position_features[train_idx],
            metadata_train=metadata_features[train_idx],
            y_train=y_true[train_idx],
            c_value=args.logreg_c,
            max_iter=args.logreg_max_iter,
            min_pos=args.logreg_min_pos,
            embedding_pca_dim=args.embedding_pca_dim,
            seed=args.seed,
            fold_name=f"fold_{fold}",
        )
        fold_pred = predict_context_artifact(
            fold_artifact=fold_artifact,
            emb=emb_full[valid_idx],
            raw_scores=scores_full_raw[valid_idx],
            context=context_tensor[valid_idx],
            position_features=position_features[valid_idx],
            metadata_features=metadata_features[valid_idx],
            sigmoid_fallback=False,
        )
        oof_pred[valid_idx] = fold_pred
        fold_artifacts.append(fold_artifact)

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], scores_full_raw[valid_idx])
        probe_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        fitted_classes = int(len(fold_artifact["fitted_class_indices"]))
        fold_result = FoldResult(
            fold=fold,
            raw_valid_auc=float(raw_valid_auc),
            probe_valid_auc=float(probe_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            actual_embedding_dim=int(fold_artifact["actual_embedding_dim"]),
            fitted_classes=fitted_classes,
        )
        fold_results.append(fold_result)
        print(
            f"[FOLD {fold}] raw_auc={raw_valid_auc:.6f} probe_auc={probe_valid_auc:.6f} "
            f"embed_dim={fold_result.actual_embedding_dim} fitted_classes={fitted_classes} "
            f"train_files={fold_result.n_train_files} valid_files={fold_result.n_valid_files}"
        )

    probe_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_valid_auc for item in fold_results]))
    mean_fold_probe_auc = float(np.mean([item.probe_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_probe_auc - probe_oof_auc)

    artifact = {
        "artifact_version": 1,
        "model_type": "perch_context_logreg",
        "class_names": class_names,
        "config": {
            "n_folds": int(args.n_folds),
            "embedding_pca_dim": int(args.embedding_pca_dim),
            "logreg_c": float(args.logreg_c),
            "logreg_max_iter": int(args.logreg_max_iter),
            "logreg_min_pos": int(args.logreg_min_pos),
            "include_hour_features": bool(args.include_hour_features),
            "seed": int(args.seed),
            "fold_source": fold_source,
            "fold_assignment_path": str(args.fold_assignment_path),
            "context_feature_names": context_feature_names,
            "metadata_feature_names": metadata_feature_names,
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_context_logreg_artifacts.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    fold_metrics_df = pd.DataFrame([item.__dict__ for item in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    oof_pred_path = output_dir / "oof_predictions.npz"
    np.savez_compressed(
        oof_pred_path,
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=scores_full_raw.astype(np.float32, copy=False),
        oof_pred=oof_pred.astype(np.float32, copy=False),
        row_id=meta_df["row_id"].to_numpy(),
        filename=meta_df["filename"].to_numpy(),
    )

    summary = {
        "rows": int(len(meta_df)),
        "files": int(len(unique_files)),
        "classes": int(len(class_names)),
        "raw_perch_auc": float(raw_perch_auc),
        "probe_oof_auc": float(probe_oof_auc),
        "mean_fold_raw_auc": float(mean_fold_raw_auc),
        "mean_fold_probe_auc": float(mean_fold_probe_auc),
        "fold_gap": float(fold_gap),
        "embedding_pca_dim": int(args.embedding_pca_dim),
        "logreg_c": float(args.logreg_c),
        "logreg_max_iter": int(args.logreg_max_iter),
        "logreg_min_pos": int(args.logreg_min_pos),
        "include_hour_features": bool(args.include_hour_features),
        "fold_source": fold_source,
        "fold_assignment_path": str(args.fold_assignment_path),
        "artifact_path": str(artifact_path),
        "context_feature_names": context_feature_names,
        "metadata_feature_names": metadata_feature_names,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] probe_oof_auc: {probe_oof_auc:.6f}")
    print(f"[INFO] mean_fold_probe_auc: {mean_fold_probe_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")
    print(f"[INFO] Saved artifact to: {artifact_path}")
    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF predictions to: {oof_pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
