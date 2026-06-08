#!/usr/bin/env python3
"""Leakage-aware local CV for Perch + temporal context + per-class LogReg.

This script keeps the same honest evaluation philosophy as our earlier Perch
probe scripts:

- labels are aligned strictly by `row_id`
- folds are split by `filename` with GroupKFold
- all context features are built only from Perch outputs inside the same
  soundscape file

The main idea is simple:

1. Use Perch embeddings as a global representation (optionally compressed by
   PCA fitted on the train fold only).
2. Keep the full 234-dim raw Perch class scores as row-level base features.
3. For each target class, append lightweight temporal/file-level context
   features derived from that class's score trajectory inside the same
   60-second soundscape.
4. Train one binary LogisticRegression per class.

This is intentionally much cheaper than training a new neural net, while still
being closer to the "context features" tricks seen in strong BirdCLEF-style
solutions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdclef2026_perch_probe_cv import (
    build_aligned_labels,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perch + temporal context + per-class LogReg local CV.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_context_logreg")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--embedding-pca-dim", type=int, default=128)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument("--logreg-min-pos", type=int, default=8)
    parser.add_argument("--include-site-onehot", action="store_true")
    parser.add_argument("--include-hour-features", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


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


def build_metadata_features(
    meta_df: pd.DataFrame,
    include_site_onehot: bool,
    include_hour_features: bool,
) -> Tuple[np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    feature_names: List[str] = []

    if include_site_onehot:
        site_codes = pd.Categorical(meta_df["site"]).codes
        n_sites = int(site_codes.max()) + 1 if len(site_codes) else 0
        if n_sites > 0:
            site_onehot = np.eye(n_sites, dtype=np.float32)[site_codes]
            features.append(site_onehot)
            feature_names.extend([f"site_{idx}" for idx in range(n_sites)])

    if include_hour_features:
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
        features.append(hour_feat)
        feature_names.extend(["hour_norm", "hour_sin", "hour_cos"])

    if not features:
        return np.zeros((len(meta_df), 0), dtype=np.float32), feature_names

    return np.concatenate(features, axis=1).astype(np.float32, copy=False), feature_names


def previous_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return values.copy()
    out = np.empty_like(values)
    out[:steps] = values[:1]
    out[steps:] = values[:-steps]
    return out


def next_with_edge(values: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return values.copy()
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

    file_groups = meta_df.groupby("filename", sort=False).indices
    eps = 1e-6

    for _, row_indices in file_groups.items():
        idx = np.asarray(row_indices, dtype=np.int64)
        file_end_sec = end_seconds[idx]
        order = np.argsort(file_end_sec, kind="stable")
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

        stacked = np.stack(
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

        context[idx_sorted] = stacked

    return context, feature_names


def fit_embedding_projection(
    emb_train: np.ndarray,
    emb_valid: np.ndarray,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if pca_dim <= 0:
        return (
            emb_train.astype(np.float32, copy=False),
            emb_valid.astype(np.float32, copy=False),
            emb_train.shape[1],
        )

    scaler = StandardScaler()
    emb_train_scaled = scaler.fit_transform(emb_train).astype(np.float32)
    emb_valid_scaled = scaler.transform(emb_valid).astype(np.float32)

    max_dim = min(pca_dim, emb_train_scaled.shape[0] - 1, emb_train_scaled.shape[1])
    if max_dim < 1:
        return emb_train_scaled, emb_valid_scaled, emb_train_scaled.shape[1]

    pca = PCA(n_components=max_dim, random_state=seed)
    emb_train_pca = pca.fit_transform(emb_train_scaled).astype(np.float32)
    emb_valid_pca = pca.transform(emb_valid_scaled).astype(np.float32)
    return emb_train_pca, emb_valid_pca, int(max_dim)


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


def scale_block(
    train_block: np.ndarray,
    valid_block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_block.shape[1] == 0:
        return train_block.astype(np.float32, copy=False), valid_block.astype(np.float32, copy=False)
    mean = train_block.mean(axis=0, keepdims=True)
    std = train_block.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = ((train_block - mean) / std).astype(np.float32, copy=False)
    valid_scaled = ((valid_block - mean) / std).astype(np.float32, copy=False)
    return train_scaled, valid_scaled


def train_logreg_context_one_fold(
    emb_train: np.ndarray,
    emb_valid: np.ndarray,
    raw_scores_train: np.ndarray,
    raw_scores_valid: np.ndarray,
    context_train: np.ndarray,
    context_valid: np.ndarray,
    position_train: np.ndarray,
    position_valid: np.ndarray,
    metadata_train: np.ndarray,
    metadata_valid: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
    min_pos: int,
    embedding_pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    emb_train_proj, emb_valid_proj, actual_embedding_dim = fit_embedding_projection(
        emb_train=emb_train,
        emb_valid=emb_valid,
        pca_dim=embedding_pca_dim,
        seed=seed,
    )

    base_train = build_base_features(
        emb_part=emb_train_proj,
        raw_scores=raw_scores_train,
        position_features=position_train,
        metadata_features=metadata_train,
    )
    base_valid = build_base_features(
        emb_part=emb_valid_proj,
        raw_scores=raw_scores_valid,
        position_features=position_valid,
        metadata_features=metadata_valid,
    )

    base_scaler = StandardScaler()
    base_train_scaled = base_scaler.fit_transform(base_train).astype(np.float32)
    base_valid_scaled = base_scaler.transform(base_valid).astype(np.float32)

    n_classes = y_train.shape[1]
    pred = raw_scores_valid.astype(np.float32, copy=True)
    fitted_classes = 0

    for class_idx in range(n_classes):
        target = y_train[:, class_idx]
        pos = int(target.sum())
        neg = int(len(target) - pos)
        if pos < min_pos or neg == 0:
            continue

        ctx_train = context_train[:, class_idx, :]
        ctx_valid = context_valid[:, class_idx, :]
        ctx_train_scaled, ctx_valid_scaled = scale_block(ctx_train, ctx_valid)

        x_train = np.concatenate([base_train_scaled, ctx_train_scaled], axis=1).astype(np.float32, copy=False)
        x_valid = np.concatenate([base_valid_scaled, ctx_valid_scaled], axis=1).astype(np.float32, copy=False)

        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        model.fit(x_train, target)
        pred[:, class_idx] = model.predict_proba(x_valid)[:, 1].astype(np.float32, copy=False)
        fitted_classes += 1

    return pred, actual_embedding_dim, fitted_classes


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
        include_site_onehot=args.include_site_onehot,
        include_hour_features=args.include_hour_features,
    )
    context_tensor, context_feature_names = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(
            f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need at least {args.n_folds}."
        )

    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_results: List[FoldResult] = []

    print("[INFO] Perch + context + LogReg")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] embedding_pca_dim(requested): {args.embedding_pca_dim}")
    print(f"[INFO] logreg_c: {args.logreg_c}")
    print(f"[INFO] logreg_min_pos: {args.logreg_min_pos}")
    print(f"[INFO] include_site_onehot: {args.include_site_onehot}")
    print(f"[INFO] include_hour_features: {args.include_hour_features}")
    print(f"[INFO] context_features: {', '.join(context_feature_names)}")

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(meta_df, groups=groups), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)

        fold_pred, actual_embedding_dim, fitted_classes = train_logreg_context_one_fold(
            emb_train=emb_full[train_idx],
            emb_valid=emb_full[valid_idx],
            raw_scores_train=scores_full_raw[train_idx],
            raw_scores_valid=scores_full_raw[valid_idx],
            context_train=context_tensor[train_idx],
            context_valid=context_tensor[valid_idx],
            position_train=position_features[train_idx],
            position_valid=position_features[valid_idx],
            metadata_train=metadata_features[train_idx],
            metadata_valid=metadata_features[valid_idx],
            y_train=y_true[train_idx],
            c_value=args.logreg_c,
            max_iter=args.logreg_max_iter,
            min_pos=args.logreg_min_pos,
            embedding_pca_dim=args.embedding_pca_dim,
            seed=args.seed,
        )

        oof_pred[valid_idx] = fold_pred
        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], scores_full_raw[valid_idx])
        probe_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)

        fold_result = FoldResult(
            fold=fold,
            raw_valid_auc=float(raw_valid_auc),
            probe_valid_auc=float(probe_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            actual_embedding_dim=int(actual_embedding_dim),
        )
        fold_results.append(fold_result)

        print(
            f"[FOLD {fold}] raw_auc={raw_valid_auc:.6f} "
            f"probe_auc={probe_valid_auc:.6f} "
            f"embed_dim={actual_embedding_dim} "
            f"fitted_classes={fitted_classes} "
            f"train_files={fold_result.n_train_files} "
            f"valid_files={fold_result.n_valid_files}"
        )

    probe_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_valid_auc for item in fold_results]))
    mean_fold_probe_auc = float(np.mean([item.probe_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_probe_auc - probe_oof_auc)

    print(f"[INFO] probe_oof_auc: {probe_oof_auc:.6f}")
    print(f"[INFO] mean_fold_raw_auc: {mean_fold_raw_auc:.6f}")
    print(f"[INFO] mean_fold_probe_auc: {mean_fold_probe_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")

    fold_metrics_df = pd.DataFrame(
        [
            {
                "fold": item.fold,
                "raw_valid_auc": item.raw_valid_auc,
                "probe_valid_auc": item.probe_valid_auc,
                "n_train_rows": item.n_train_rows,
                "n_valid_rows": item.n_valid_rows,
                "n_train_files": item.n_train_files,
                "n_valid_files": item.n_valid_files,
                "actual_embedding_dim": item.actual_embedding_dim,
            }
            for item in fold_results
        ]
    )
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    meta_out = meta_df.copy()
    meta_out["raw_max"] = scores_full_raw.max(axis=1)
    meta_out["oof_max"] = oof_pred.max(axis=1)
    meta_out_path = output_dir / "oof_meta.csv"
    meta_out.to_csv(meta_out_path, index=False)

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
        "embedding_pca_dim_requested": int(args.embedding_pca_dim),
        "logreg_c": float(args.logreg_c),
        "logreg_max_iter": int(args.logreg_max_iter),
        "logreg_min_pos": int(args.logreg_min_pos),
        "include_site_onehot": bool(args.include_site_onehot),
        "include_hour_features": bool(args.include_hour_features),
        "context_feature_names": context_feature_names,
        "metadata_feature_names": metadata_feature_names,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF metadata to: {meta_out_path}")
    print(f"[INFO] Saved OOF predictions to: {oof_pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
