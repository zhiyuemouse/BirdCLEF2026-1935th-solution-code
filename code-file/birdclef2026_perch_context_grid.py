#!/usr/bin/env python3
"""Grid search for Perch + context + per-class LogReg.

This script extends the honest local CV setup from
`birdclef2026_perch_context_logreg.py` and sweeps:

- metadata mode: none / site / hour / site_hour
- embedding PCA dim
- LogisticRegression C

All splits remain grouped by `filename`, and all context features are derived
only from the same soundscape file's raw Perch outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from birdclef2026_perch_context_logreg import (
    build_context_tensor,
    build_metadata_features,
    build_position_features,
    parse_end_seconds,
    train_logreg_context_one_fold,
)
from birdclef2026_perch_probe_cv import (
    build_aligned_labels,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search Perch + context + metadata + LogReg.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_context_grid")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--metadata-modes", type=str, default="none,site,hour,site_hour")
    parser.add_argument("--pca-dims", type=str, default="64,128,256")
    parser.add_argument("--logreg-c-values", type=str, default="0.125,0.25,0.5")
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument("--logreg-min-pos", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    return [int(value) for value in values]


def parse_float_list(text: str) -> List[float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    return [float(value) for value in values]


def parse_metadata_modes(text: str) -> List[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {"none", "site", "hour", "site_hour"}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported metadata modes: {invalid}. Allowed: {sorted(allowed)}")
    return values


def metadata_mode_to_flags(mode: str) -> Tuple[bool, bool]:
    if mode == "none":
        return False, False
    if mode == "site":
        return True, False
    if mode == "hour":
        return False, True
    if mode == "site_hour":
        return True, True
    raise ValueError(f"Unsupported metadata mode: {mode}")


def run_one_combo(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    emb_full: np.ndarray,
    scores_full_raw: np.ndarray,
    context_tensor: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    metadata_mode: str,
    groups: np.ndarray,
    n_folds: int,
    embedding_pca_dim: int,
    c_value: float,
    max_iter: int,
    min_pos: int,
    seed: int,
) -> Tuple[Dict[str, float | int | str], List[Dict[str, float | int | str]]]:
    gkf = GroupKFold(n_splits=n_folds)
    oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_rows: List[Dict[str, float | int | str]] = []

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
            c_value=c_value,
            max_iter=max_iter,
            min_pos=min_pos,
            embedding_pca_dim=embedding_pca_dim,
            seed=seed,
        )

        oof_pred[valid_idx] = fold_pred
        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], scores_full_raw[valid_idx])
        probe_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)

        fold_rows.append(
            {
                "metadata_mode": metadata_mode,
                "embedding_pca_dim": int(embedding_pca_dim),
                "logreg_c": float(c_value),
                "fold": int(fold),
                "raw_valid_auc": float(raw_valid_auc),
                "probe_valid_auc": float(probe_valid_auc),
                "n_train_rows": int(len(train_idx)),
                "n_valid_rows": int(len(valid_idx)),
                "n_train_files": int(len(pd.Index(groups[train_idx]).unique())),
                "n_valid_files": int(len(pd.Index(groups[valid_idx]).unique())),
                "actual_embedding_dim": int(actual_embedding_dim),
                "fitted_classes": int(fitted_classes),
            }
        )

    probe_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([row["raw_valid_auc"] for row in fold_rows]))
    mean_fold_probe_auc = float(np.mean([row["probe_valid_auc"] for row in fold_rows]))
    mean_actual_embedding_dim = float(np.mean([row["actual_embedding_dim"] for row in fold_rows]))
    mean_fitted_classes = float(np.mean([row["fitted_classes"] for row in fold_rows]))

    summary = {
        "metadata_mode": metadata_mode,
        "embedding_pca_dim": int(embedding_pca_dim),
        "logreg_c": float(c_value),
        "probe_oof_auc": float(probe_oof_auc),
        "mean_fold_raw_auc": mean_fold_raw_auc,
        "mean_fold_probe_auc": mean_fold_probe_auc,
        "fold_gap": float(mean_fold_probe_auc - probe_oof_auc),
        "mean_actual_embedding_dim": mean_actual_embedding_dim,
        "mean_fitted_classes": mean_fitted_classes,
    }
    return summary, fold_rows


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

    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(
            f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need at least {args.n_folds}."
        )

    metadata_modes = parse_metadata_modes(args.metadata_modes)
    pca_dims = parse_int_list(args.pca_dims)
    c_values = parse_float_list(args.logreg_c_values)

    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    position_features = build_position_features(parse_end_seconds(meta_df["row_id"].tolist()))
    context_tensor, context_feature_names = build_context_tensor(meta_df=meta_df, scores_full_raw=scores_full_raw)

    metadata_feature_cache: Dict[str, np.ndarray] = {}
    metadata_feature_name_cache: Dict[str, List[str]] = {}
    for mode in metadata_modes:
        include_site_onehot, include_hour_features = metadata_mode_to_flags(mode)
        metadata_features, metadata_feature_names = build_metadata_features(
            meta_df=meta_df,
            include_site_onehot=include_site_onehot,
            include_hour_features=include_hour_features,
        )
        metadata_feature_cache[mode] = metadata_features
        metadata_feature_name_cache[mode] = metadata_feature_names

    total = len(metadata_modes) * len(pca_dims) * len(c_values)
    combo_idx = 0
    result_rows: List[Dict[str, float | int | str]] = []
    fold_rows_all: List[Dict[str, float | int | str]] = []

    print("[INFO] Perch + context grid search")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] classes: {len(class_names)}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] metadata_modes: {metadata_modes}")
    print(f"[INFO] pca_dims: {pca_dims}")
    print(f"[INFO] logreg_c_values: {c_values}")
    print(f"[INFO] context_features: {', '.join(context_feature_names)}")

    for metadata_mode in metadata_modes:
        metadata_features = metadata_feature_cache[metadata_mode]
        metadata_feature_names = metadata_feature_name_cache[metadata_mode]
        for pca_dim in pca_dims:
            for c_value in c_values:
                combo_idx += 1
                print(
                    f"[GRID] {combo_idx}/{total} | metadata_mode={metadata_mode} "
                    f"| pca_dim={pca_dim} | C={c_value}"
                )
                summary, fold_rows = run_one_combo(
                    meta_df=meta_df,
                    y_true=y_true,
                    emb_full=emb_full,
                    scores_full_raw=scores_full_raw,
                    context_tensor=context_tensor,
                    position_features=position_features,
                    metadata_features=metadata_features,
                    metadata_mode=metadata_mode,
                    groups=groups,
                    n_folds=args.n_folds,
                    embedding_pca_dim=pca_dim,
                    c_value=c_value,
                    max_iter=args.logreg_max_iter,
                    min_pos=args.logreg_min_pos,
                    seed=args.seed,
                )
                row = {
                    "metadata_mode": metadata_mode,
                    "embedding_pca_dim": int(pca_dim),
                    "logreg_c": float(c_value),
                    "raw_perch_auc": float(raw_perch_auc),
                    "metadata_feature_dim": int(metadata_features.shape[1]),
                    "metadata_feature_names": ",".join(metadata_feature_names),
                    **summary,
                }
                result_rows.append(row)
                fold_rows_all.extend(fold_rows)

                print(
                    f"[RESULT] metadata_mode={metadata_mode} pca_dim={pca_dim} C={c_value} "
                    f"probe_oof_auc={summary['probe_oof_auc']:.6f} "
                    f"mean_fold_probe_auc={summary['mean_fold_probe_auc']:.6f} "
                    f"fold_gap={summary['fold_gap']:.6f}"
                )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["probe_oof_auc", "mean_fold_probe_auc", "fold_gap"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    results_df = results_df[
        [
            "rank",
            "metadata_mode",
            "embedding_pca_dim",
            "logreg_c",
            "raw_perch_auc",
            "probe_oof_auc",
            "mean_fold_raw_auc",
            "mean_fold_probe_auc",
            "fold_gap",
            "mean_actual_embedding_dim",
            "mean_fitted_classes",
            "metadata_feature_dim",
            "metadata_feature_names",
        ]
    ]

    folds_df = pd.DataFrame(fold_rows_all)

    results_path = output_dir / "grid_results.csv"
    folds_path = output_dir / "grid_fold_metrics.csv"
    best_path = output_dir / "best_config.json"

    results_df.to_csv(results_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    best_row = results_df.iloc[0].to_dict()
    best_path.write_text(json.dumps(best_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved grid results to: {results_path}")
    print(f"[INFO] Saved grid fold metrics to: {folds_path}")
    print(f"[INFO] Saved best config to: {best_path}")
    print("[INFO] Top 10 configs:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
