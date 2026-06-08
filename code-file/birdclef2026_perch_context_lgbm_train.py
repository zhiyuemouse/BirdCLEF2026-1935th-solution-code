#!/usr/bin/env python3
"""Train deployable Perch + temporal context LightGBM folds.

This is a Perch-only single-model experiment that replaces the previous
per-class LogisticRegression head with small, strongly regularized LightGBM
binary heads. Perch remains frozen; the training data and leakage-aware
GroupKFold split are the same as `birdclef2026_perch_context_train.py`.

The saved OOF predictions use the deploy/online-like probability convention:

- classes fitted by the fold's LightGBM head use `predict_proba`
- classes not fitted in that fold fall back to `sigmoid(raw Perch logits)`

This keeps the reported local CV comparable to Kaggle submission outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover - fail fast in user envs.
    raise ImportError(
        "LightGBM is required for this script. Install `lightgbm` in the active env."
    ) from exc

from birdclef2026_perch_context_train import (
    build_aligned_labels,
    build_base_features,
    build_context_tensor,
    build_metadata_features,
    build_position_features,
    fit_embedding_projector,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    parse_end_seconds,
    seed_everything,
    sigmoid_np,
    transform_embedding_projector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deployable Perch context LightGBM folds.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_context_lgbm_labeled_all_v1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--embedding-pca-dim", type=int, default=128)
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["full", "no_embedding", "target", "target_with_embedding"],
        default="full",
        help=(
            "`full`: embedding PCA + all Perch logits + position + context. "
            "`no_embedding`: all Perch logits + position + context. "
            "`target`: target-class Perch logit + position + context only. "
            "`target_with_embedding`: embedding PCA + target-class Perch logit + position + context."
        ),
    )
    parser.add_argument("--lgbm-min-pos", type=int, default=8)
    parser.add_argument("--lgbm-n-estimators", type=int, default=120)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.035)
    parser.add_argument("--lgbm-num-leaves", type=int, default=7)
    parser.add_argument("--lgbm-max-depth", type=int, default=3)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=12)
    parser.add_argument("--lgbm-subsample", type=float, default=0.85)
    parser.add_argument("--lgbm-subsample-freq", type=int, default=1)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.65)
    parser.add_argument("--lgbm-reg-alpha", type=float, default=0.05)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=2.0)
    parser.add_argument("--lgbm-min-split-gain", type=float, default=0.0)
    parser.add_argument("--lgbm-n-jobs", type=int, default=2)
    parser.add_argument("--lgbm-class-weight", type=str, choices=["balanced", "none"], default="balanced")
    parser.add_argument("--include-hour-features", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def build_lgbm_params(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": int(args.lgbm_n_estimators),
        "learning_rate": float(args.lgbm_learning_rate),
        "num_leaves": int(args.lgbm_num_leaves),
        "max_depth": int(args.lgbm_max_depth),
        "min_child_samples": int(args.lgbm_min_child_samples),
        "subsample": float(args.lgbm_subsample),
        "subsample_freq": int(args.lgbm_subsample_freq),
        "colsample_bytree": float(args.lgbm_colsample_bytree),
        "reg_alpha": float(args.lgbm_reg_alpha),
        "reg_lambda": float(args.lgbm_reg_lambda),
        "min_split_gain": float(args.lgbm_min_split_gain),
        "random_state": int(seed),
        "n_jobs": int(args.lgbm_n_jobs),
        "verbosity": -1,
    }
    if args.lgbm_class_weight == "balanced":
        params["class_weight"] = "balanced"
    return params


def make_features_for_class(
    emb_proj: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    class_idx: int,
    feature_set: str,
) -> np.ndarray:
    if feature_set in {"full", "no_embedding"}:
        raw_part = raw_scores
    elif feature_set in {"target", "target_with_embedding"}:
        raw_part = raw_scores[:, class_idx:class_idx + 1]
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    if feature_set in {"full", "target_with_embedding"}:
        emb_part = emb_proj
    else:
        emb_part = np.zeros((len(raw_scores), 0), dtype=np.float32)

    base = build_base_features(
        emb_part=emb_part,
        raw_scores=raw_part,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    ctx = context[:, class_idx, :].astype(np.float32, copy=False)
    return np.concatenate([base, ctx], axis=1).astype(np.float32, copy=False)


def fit_context_artifact(
    emb_train: np.ndarray,
    raw_scores_train: np.ndarray,
    context_train: np.ndarray,
    position_train: np.ndarray,
    metadata_train: np.ndarray,
    y_train: np.ndarray,
    embedding_pca_dim: int,
    min_pos: int,
    lgbm_params: Dict[str, object],
    feature_set: str,
    seed: int,
    fold_name: str,
) -> Dict[str, object]:
    if feature_set in {"full", "target_with_embedding"}:
        emb_train_proj, projector = fit_embedding_projector(emb_train=emb_train, pca_dim=embedding_pca_dim, seed=seed)
    else:
        emb_train_proj = np.zeros((len(emb_train), 0), dtype=np.float32)
        projector = {
            "embedding_scaler": None,
            "embedding_pca": None,
            "actual_embedding_dim": 0,
        }

    n_classes = y_train.shape[1]
    class_models: List[object] = [None] * n_classes
    fitted_class_indices: List[int] = []

    for class_idx in range(n_classes):
        target = y_train[:, class_idx]
        pos = int(target.sum())
        neg = int(len(target) - pos)
        if pos < min_pos or neg == 0:
            continue

        x_train = make_features_for_class(
            emb_proj=emb_train_proj,
            raw_scores=raw_scores_train,
            context=context_train,
            position_features=position_train,
            metadata_features=metadata_train,
            class_idx=class_idx,
            feature_set=feature_set,
        )
        params = dict(lgbm_params)
        params["random_state"] = int(seed + class_idx)
        model = LGBMClassifier(**params)
        model.fit(x_train, target)

        class_models[class_idx] = model
        fitted_class_indices.append(class_idx)

    return {
        "fold_name": fold_name,
        "embedding_scaler": projector["embedding_scaler"],
        "embedding_pca": projector["embedding_pca"],
        "actual_embedding_dim": int(projector["actual_embedding_dim"]),
        "feature_set": feature_set,
        "class_models": class_models,
        "fitted_class_indices": np.asarray(fitted_class_indices, dtype=np.int32),
    }


def predict_context_artifact(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    feature_set = str(fold_artifact.get("feature_set", "full"))
    if feature_set in {"full", "target_with_embedding"}:
        emb_proj = transform_embedding_projector(emb, fold_artifact=fold_artifact)
    else:
        emb_proj = np.zeros((len(emb), 0), dtype=np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32)

    class_models = fold_artifact["class_models"]
    for class_idx in fold_artifact["fitted_class_indices"]:
        class_idx = int(class_idx)
        model = class_models[class_idx]
        x = make_features_for_class(
            emb_proj=emb_proj,
            raw_scores=raw_scores,
            context=context,
            position_features=position_features,
            metadata_features=metadata_features,
            class_idx=class_idx,
            feature_set=feature_set,
        )
        pred[:, class_idx] = model.predict_proba(x)[:, 1].astype(np.float32, copy=False)

    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0)


@dataclass
class FoldResult:
    fold: int
    raw_sigmoid_valid_auc: float
    lgbm_valid_auc: float
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

    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32)
    raw_sigmoid_auc = macro_auc_skip_empty(y_true, raw_sigmoid)
    oof_pred = raw_sigmoid.copy()
    fold_artifacts: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    lgbm_params = build_lgbm_params(args=args, seed=args.seed)
    gkf = GroupKFold(n_splits=args.n_folds)

    print("[INFO] Train deployable Perch + context LightGBM")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] raw_sigmoid_auc: {raw_sigmoid_auc:.6f}")
    print(f"[INFO] embedding_pca_dim: {args.embedding_pca_dim}")
    print(f"[INFO] feature_set: {args.feature_set}")
    print(f"[INFO] lgbm_min_pos: {args.lgbm_min_pos}")
    print(f"[INFO] lgbm_params: {json.dumps(lgbm_params, sort_keys=True)}")
    print(f"[INFO] include_hour_features: {args.include_hour_features}")

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(meta_df, groups=groups), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        fold_artifact = fit_context_artifact(
            emb_train=emb_full[train_idx],
            raw_scores_train=scores_full_raw[train_idx],
            context_train=context_tensor[train_idx],
            position_train=position_features[train_idx],
            metadata_train=metadata_features[train_idx],
            y_train=y_true[train_idx],
            embedding_pca_dim=args.embedding_pca_dim,
            min_pos=args.lgbm_min_pos,
            lgbm_params=lgbm_params,
            feature_set=args.feature_set,
            seed=args.seed + fold * 1000,
            fold_name=f"fold_{fold}",
        )
        fold_pred = predict_context_artifact(
            fold_artifact=fold_artifact,
            emb=emb_full[valid_idx],
            raw_scores=scores_full_raw[valid_idx],
            context=context_tensor[valid_idx],
            position_features=position_features[valid_idx],
            metadata_features=metadata_features[valid_idx],
        )
        oof_pred[valid_idx] = fold_pred
        fold_artifacts.append(fold_artifact)

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], raw_sigmoid[valid_idx])
        lgbm_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)
        fitted_classes = int(len(fold_artifact["fitted_class_indices"]))
        fold_result = FoldResult(
            fold=fold,
            raw_sigmoid_valid_auc=float(raw_valid_auc),
            lgbm_valid_auc=float(lgbm_valid_auc),
            n_train_rows=int(len(train_idx)),
            n_valid_rows=int(len(valid_idx)),
            n_train_files=int(len(pd.Index(groups[train_idx]).unique())),
            n_valid_files=int(len(pd.Index(groups[valid_idx]).unique())),
            actual_embedding_dim=int(fold_artifact["actual_embedding_dim"]),
            fitted_classes=fitted_classes,
        )
        fold_results.append(fold_result)
        print(
            f"[FOLD {fold}] raw_sigmoid_auc={raw_valid_auc:.6f} lgbm_auc={lgbm_valid_auc:.6f} "
            f"embed_dim={fold_result.actual_embedding_dim} fitted_classes={fitted_classes} "
            f"train_files={fold_result.n_train_files} valid_files={fold_result.n_valid_files}"
        )

    lgbm_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([item.raw_sigmoid_valid_auc for item in fold_results]))
    mean_fold_lgbm_auc = float(np.mean([item.lgbm_valid_auc for item in fold_results]))
    fold_gap = float(mean_fold_lgbm_auc - lgbm_oof_auc)

    artifact = {
        "artifact_version": 1,
        "model_type": "perch_context_lgbm",
        "class_names": class_names,
        "config": {
            "n_folds": int(args.n_folds),
            "embedding_pca_dim": int(args.embedding_pca_dim),
            "feature_set": str(args.feature_set),
            "lgbm_min_pos": int(args.lgbm_min_pos),
            "lgbm_params": lgbm_params,
            "include_hour_features": bool(args.include_hour_features),
            "seed": int(args.seed),
            "context_feature_names": context_feature_names,
            "metadata_feature_names": metadata_feature_names,
        },
        "folds": fold_artifacts,
    }
    artifact_path = output_dir / "perch_context_lgbm_artifacts.joblib"
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
        "raw_sigmoid_auc": float(raw_sigmoid_auc),
        "lgbm_oof_auc": float(lgbm_oof_auc),
        "mean_fold_raw_sigmoid_auc": float(mean_fold_raw_auc),
        "mean_fold_lgbm_auc": float(mean_fold_lgbm_auc),
        "fold_gap": float(fold_gap),
        "embedding_pca_dim": int(args.embedding_pca_dim),
        "feature_set": str(args.feature_set),
        "lgbm_min_pos": int(args.lgbm_min_pos),
        "lgbm_params": lgbm_params,
        "include_hour_features": bool(args.include_hour_features),
        "artifact_path": str(artifact_path),
        "context_feature_names": context_feature_names,
        "metadata_feature_names": metadata_feature_names,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] lgbm_oof_auc: {lgbm_oof_auc:.6f}")
    print(f"[INFO] mean_fold_lgbm_auc: {mean_fold_lgbm_auc:.6f}")
    print(f"[INFO] fold_gap: {fold_gap:.6f}")
    print(f"[INFO] Saved artifact to: {artifact_path}")
    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved OOF predictions to: {oof_pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
