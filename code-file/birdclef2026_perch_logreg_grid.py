#!/usr/bin/env python3
"""Grid search for Perch LogReg probe hyperparameters.

Focuses on cheap, honest local CV for:

- feature_mode: usually `pca_embedding_plus_scores`
- pca_dim
- logistic regression C

All splits are grouped by `filename` to avoid soundscape leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from birdclef2026_perch_probe_cv import (
    build_aligned_labels,
    build_features,
    limit_by_files,
    load_cache,
    load_class_names,
    macro_auc_skip_empty,
    seed_everything,
    train_logreg_one_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search pca_dim and C for Perch LogReg probe.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_logreg_grid")
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["embedding", "embedding_plus_scores", "pca_embedding_plus_scores"],
        default="pca_embedding_plus_scores",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--pca-dims", type=str, default="128,256,384,512")
    parser.add_argument("--logreg-c-values", type=str, default="0.25,0.5,1,2,4")
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


def run_one_combo(
    x: np.ndarray,
    y_true: np.ndarray,
    scores_full_raw: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    pca_dim: int,
    c_value: float,
    max_iter: int,
    min_pos: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    gkf = GroupKFold(n_splits=n_folds)
    oof_pred = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_rows: List[Dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(x, groups=groups), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], scores_full_raw[valid_idx])
        fold_pred = train_logreg_one_fold(
            x_train=x[train_idx],
            y_train=y_true[train_idx],
            x_valid=x[valid_idx],
            raw_scores_valid=scores_full_raw[valid_idx],
            c_value=c_value,
            max_iter=max_iter,
            min_pos=min_pos,
            pca_dim=pca_dim,
            seed=seed,
        )
        oof_pred[valid_idx] = fold_pred
        probe_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)

        fold_rows.append(
            {
                "fold": int(fold),
                "raw_valid_auc": float(raw_valid_auc),
                "probe_valid_auc": float(probe_valid_auc),
                "n_train_rows": int(len(train_idx)),
                "n_valid_rows": int(len(valid_idx)),
                "n_train_files": int(len(pd.Index(groups[train_idx]).unique())),
                "n_valid_files": int(len(pd.Index(groups[valid_idx]).unique())),
            }
        )

    probe_oof_auc = macro_auc_skip_empty(y_true, oof_pred)
    mean_fold_raw_auc = float(np.mean([row["raw_valid_auc"] for row in fold_rows]))
    mean_fold_probe_auc = float(np.mean([row["probe_valid_auc"] for row in fold_rows]))

    summary = {
        "probe_oof_auc": float(probe_oof_auc),
        "mean_fold_raw_auc": mean_fold_raw_auc,
        "mean_fold_probe_auc": mean_fold_probe_auc,
        "fold_gap": float(mean_fold_probe_auc - probe_oof_auc),
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

    x = build_features(emb_full=emb_full, scores_full_raw=scores_full_raw, feature_mode=args.feature_mode)
    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(
            f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need at least {args.n_folds}."
        )

    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    pca_dims = parse_int_list(args.pca_dims)
    c_values = parse_float_list(args.logreg_c_values)

    print("[INFO] Perch LogReg grid search")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] feature_mode: {args.feature_mode}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] pca_dims: {pca_dims}")
    print(f"[INFO] logreg_c_values: {c_values}")

    result_rows: List[Dict[str, float | int | str]] = []
    fold_rows_all: List[Dict[str, float | int | str]] = []

    total = len(pca_dims) * len(c_values)
    combo_idx = 0
    for pca_dim in pca_dims:
        for c_value in c_values:
            combo_idx += 1
            print(f"[GRID] {combo_idx}/{total} | pca_dim={pca_dim} | C={c_value}")
            summary, fold_rows = run_one_combo(
                x=x,
                y_true=y_true,
                scores_full_raw=scores_full_raw,
                groups=groups,
                n_folds=args.n_folds,
                pca_dim=pca_dim,
                c_value=c_value,
                max_iter=args.logreg_max_iter,
                min_pos=args.logreg_min_pos,
                seed=args.seed,
            )

            row = {
                "feature_mode": args.feature_mode,
                "pca_dim": int(pca_dim),
                "logreg_c": float(c_value),
                "raw_perch_auc": float(raw_perch_auc),
                **summary,
            }
            result_rows.append(row)
            print(
                f"[RESULT] pca_dim={pca_dim} C={c_value} "
                f"probe_oof_auc={summary['probe_oof_auc']:.6f} "
                f"mean_fold_probe_auc={summary['mean_fold_probe_auc']:.6f} "
                f"fold_gap={summary['fold_gap']:.6f}"
            )

            for fold_row in fold_rows:
                fold_rows_all.append(
                    {
                        "feature_mode": args.feature_mode,
                        "pca_dim": int(pca_dim),
                        "logreg_c": float(c_value),
                        **fold_row,
                    }
                )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["probe_oof_auc", "mean_fold_probe_auc", "fold_gap"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    results_df = results_df[
        [
            "rank",
            "feature_mode",
            "pca_dim",
            "logreg_c",
            "raw_perch_auc",
            "probe_oof_auc",
            "mean_fold_raw_auc",
            "mean_fold_probe_auc",
            "fold_gap",
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
    print("[INFO] Top 5 configs:")
    print(results_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
