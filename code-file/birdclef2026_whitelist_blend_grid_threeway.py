#!/usr/bin/env python3
"""Leak-safe OOF grid for Perch + Stage3 CNN + base CNN blending.

Only saved OOF predictions and train labels are used.  This is intentionally a
small whitelist search: global blend weights plus the same file-level scaling
and temporal smoothing families already used by the two-way blend.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


EPS = 1e-6


@dataclass(frozen=True)
class EvalResult:
    name: str
    auc: float
    scored_classes: int
    delta_vs_base: float
    n_improved_classes: int
    n_worse_classes: int
    mean_class_delta: float
    median_class_delta: float
    min_class_delta: float
    max_class_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-safe three-way OOF blend/post-process grid.")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--perch-dir", type=str, default="outputs/perch_context_deploy_labeled_all_v1")
    parser.add_argument(
        "--stage3-oof-path",
        type=str,
        default=(
            "outputs/birdclef2026_gm_stage3_perchcnn_white_v1/"
            "20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo/"
            "soundscape_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--base-cnn-oof-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_oof_predictions.csv",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/whitelist_blend_threeway_perch_stage3_cnn195634_v1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--top-k-weights", type=int, default=50)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def binary_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.float32, copy=False)
    n_pos = int((y_true > 0.5).sum())
    n = len(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("binary_auc_from_scores requires both positive and negative samples.")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end
    pos_rank_sum = ranks[sorted_true > 0.5].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc_and_class_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    scores: list[float] = []
    class_indices: list[int] = []
    for idx in range(y_true.shape[1]):
        y_col = y_true[:, idx]
        if y_col.max() <= 0 or y_col.min() >= 1:
            continue
        scores.append(binary_auc_from_scores(y_col, y_score[:, idx]))
        class_indices.append(idx)
    if not scores:
        raise ValueError("No scored classes for macro AUC.")
    return float(np.mean(scores)), np.asarray(class_indices, dtype=np.int32), np.asarray(scores, dtype=np.float32)


def average_ranks(values: np.ndarray) -> np.ndarray:
    n, c = values.shape
    ranks = np.empty((n, c), dtype=np.float32)
    for j in range(c):
        order = np.argsort(values[:, j], kind="mergesort")
        sorted_values = values[order, j]
        col_ranks = np.empty(n, dtype=np.float32)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_values[end] == sorted_values[start]:
                end += 1
            avg_rank = (start + end - 1) / 2.0
            col_ranks[order[start:end]] = avg_rank
            start = end
        ranks[:, j] = col_ranks / max(n - 1, 1)
    return ranks


def load_class_names(path: Path) -> list[str]:
    sample = pd.read_csv(path, nrows=0)
    return [col for col in sample.columns if col != "row_id"]


def reconstruct_perch_strict_oof(perch_dir: Path, n_folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(perch_dir / "oof_predictions.npz", allow_pickle=True)
    y_true = npz["y_true"].astype(np.float32)
    raw_scores = npz["raw_scores"].astype(np.float32)
    mixed_oof = npz["oof_pred"].astype(np.float32)
    row_id = npz["row_id"].astype(str)
    filename = npz["filename"].astype(str)

    artifact = joblib.load(perch_dir / "perch_context_logreg_artifacts.joblib")
    folds = artifact["folds"]
    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} Perch folds, got {len(folds)}")

    strict = sigmoid(raw_scores).astype(np.float32)
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (_, valid_idx) in enumerate(gkf.split(np.zeros(len(filename)), groups=filename)):
        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        fitted = np.asarray(folds[fold_idx]["fitted_class_indices"], dtype=np.int32)
        strict[np.ix_(valid_idx, fitted)] = mixed_oof[np.ix_(valid_idx, fitted)]

    return y_true, strict.astype(np.float32), row_id, filename


def load_cnn_oof(path: Path, class_names: list[str], row_id: np.ndarray, y_true: np.ndarray, name: str) -> np.ndarray:
    df = pd.read_csv(path)
    pred_cols = class_names
    target_cols = [f"target_{label}" for label in class_names]
    required = ["row_id", *target_cols, *pred_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{name} OOF is missing columns: {missing[:10]}")

    aligned = pd.DataFrame({"row_id": row_id}).merge(
        df[["row_id", *target_cols, *pred_cols]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[pred_cols].isna().any().any():
        raise ValueError(f"{name} OOF missing predictions after row_id alignment.")

    cnn_y = aligned[target_cols].to_numpy(dtype=np.float32)
    if not np.array_equal(cnn_y, y_true):
        raise ValueError(f"{name} target columns do not match Perch y_true after row_id alignment.")
    return aligned[pred_cols].to_numpy(dtype=np.float32)


def blend_three(
    perch: np.ndarray,
    stage3: np.ndarray,
    base_cnn: np.ndarray,
    method: str,
    weights: tuple[float, float, float],
    rank_mats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    wp, ws, wb = weights
    if method == "prob":
        out = wp * perch + ws * stage3 + wb * base_cnn
    elif method == "logit":
        out = sigmoid(wp * logit(perch) + ws * logit(stage3) + wb * logit(base_cnn))
    elif method == "rank":
        if rank_mats is None:
            rank_mats = (average_ranks(perch), average_ranks(stage3), average_ranks(base_cnn))
        pr, sr, br = rank_mats
        out = wp * pr + ws * sr + wb * br
    else:
        raise ValueError(f"Unknown blend method: {method}")
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def temporal_smooth(pred: np.ndarray, filename: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        if len(idx) <= 1:
            continue
        p = pred[idx]
        prev_p = np.concatenate([p[:1], p[:-1]], axis=0)
        next_p = np.concatenate([p[1:], p[-1:]], axis=0)
        out[idx] = (1.0 - alpha) * p + 0.5 * alpha * (prev_p + next_p)
    return np.clip(out, 0.0, 1.0)


def adaptive_temporal_smooth(pred: np.ndarray, filename: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        if len(idx) <= 2:
            continue
        p = pred[idx]
        new_p = p.copy()
        for pos in range(1, len(idx) - 1):
            conf = float(p[pos].max())
            a = alpha * (1.0 - conf)
            new_p[pos] = (1.0 - a) * p[pos] + 0.5 * a * (p[pos - 1] + p[pos + 1])
        out[idx] = new_p
    return np.clip(out, 0.0, 1.0)


def file_level_scale(pred: np.ndarray, filename: np.ndarray, mode: str, value: float | int) -> np.ndarray:
    if mode == "none":
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        p = pred[idx]
        if mode == "max_power":
            scale = np.power(np.maximum(p.max(axis=0, keepdims=True), EPS), float(value))
        elif mode == "topk_mean":
            k = int(value)
            k = max(1, min(k, len(idx)))
            scale = np.sort(p, axis=0)[-k:].mean(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown file scale mode: {mode}")
        out[idx] = p * scale
    return np.clip(out, 0.0, 1.0)


def iter_weight_grid() -> Iterable[tuple[float, float, float]]:
    # Search around the known strong two-way point: Perch 0.74 / Stage3 0.26.
    for base_weight in np.arange(0.0, 0.121, 0.005):
        remaining = 1.0 - float(base_weight)
        for perch_ratio in np.arange(0.68, 0.851, 0.005):
            perch_weight = remaining * float(perch_ratio)
            stage3_weight = remaining - perch_weight
            yield (round(perch_weight, 6), round(stage3_weight, 6), round(float(base_weight), 6))


def iter_postprocess_grid() -> Iterable[dict[str, object]]:
    file_scales: list[tuple[str, float | int]] = [
        ("none", 0),
        ("max_power", 0.20),
        ("max_power", 0.40),
        ("topk_mean", 1),
        ("topk_mean", 2),
    ]
    plain_alphas = [0.0, 0.05, 0.10, 0.15, 0.20]
    adaptive_alphas = [0.0, 0.10, 0.20]
    for method in ["logit", "prob", "rank"]:
        for file_scale_mode, file_scale_value in file_scales:
            for alpha in plain_alphas:
                if alpha > 0:
                    yield {
                        "method": method,
                        "file_scale_mode": file_scale_mode,
                        "file_scale_value": file_scale_value,
                        "smooth_mode": "plain",
                        "smooth_alpha": alpha,
                    }
                else:
                    for adaptive_alpha in adaptive_alphas:
                        yield {
                            "method": method,
                            "file_scale_mode": file_scale_mode,
                            "file_scale_value": file_scale_value,
                            "smooth_mode": "adaptive" if adaptive_alpha > 0 else "none",
                            "smooth_alpha": adaptive_alpha,
                        }


def apply_postprocess(pred: np.ndarray, filename: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    pred = file_level_scale(pred, filename=filename, mode=str(cfg["file_scale_mode"]), value=cfg["file_scale_value"])
    if cfg["smooth_mode"] == "plain":
        pred = temporal_smooth(pred, filename=filename, alpha=float(cfg["smooth_alpha"]))
    elif cfg["smooth_mode"] == "adaptive":
        pred = adaptive_temporal_smooth(pred, filename=filename, alpha=float(cfg["smooth_alpha"]))
    elif cfg["smooth_mode"] != "none":
        raise ValueError(f"Unknown smooth mode: {cfg['smooth_mode']}")
    return pred


def eval_prediction(
    name: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    base_class_scores: np.ndarray,
    base_auc: float,
    class_indices: np.ndarray,
) -> EvalResult:
    auc, scored_indices, class_scores = macro_auc_and_class_scores(y_true, pred)
    if not np.array_equal(scored_indices, class_indices):
        raise ValueError("Scored class set changed unexpectedly.")
    delta = class_scores - base_class_scores
    return EvalResult(
        name=name,
        auc=float(auc),
        scored_classes=int(len(class_scores)),
        delta_vs_base=float(auc - base_auc),
        n_improved_classes=int((delta > 1e-12).sum()),
        n_worse_classes=int((delta < -1e-12).sum()),
        mean_class_delta=float(delta.mean()),
        median_class_delta=float(np.median(delta)),
        min_class_delta=float(delta.min()),
        max_class_delta=float(delta.max()),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.sample_submission_path))
    y_true, perch, row_id, filename = reconstruct_perch_strict_oof(Path(args.perch_dir), n_folds=args.n_folds)
    stage3 = load_cnn_oof(Path(args.stage3_oof_path), class_names, row_id, y_true, name="stage3")
    base_cnn = load_cnn_oof(Path(args.base_cnn_oof_path), class_names, row_id, y_true, name="base_cnn")

    print("[INFO] Precomputing rank matrices once...")
    rank_mats = (average_ranks(perch), average_ranks(stage3), average_ranks(base_cnn))

    base_pred = blend_three(
        perch,
        stage3,
        base_cnn,
        method="logit",
        weights=(0.74, 0.26, 0.0),
        rank_mats=rank_mats,
    )
    base_pred = file_level_scale(base_pred, filename=filename, mode="max_power", value=0.4)
    base_pred = temporal_smooth(base_pred, filename=filename, alpha=0.15)
    base_auc, class_indices, base_class_scores = macro_auc_and_class_scores(y_true, base_pred)

    base_rows = [
        eval_prediction("perch_strict", y_true, perch, base_class_scores, base_auc, class_indices),
        eval_prediction("stage3", y_true, stage3, base_class_scores, base_auc, class_indices),
        eval_prediction("base_cnn", y_true, base_cnn, base_class_scores, base_auc, class_indices),
        eval_prediction("baseline_perch_stage3", y_true, base_pred, base_class_scores, base_auc, class_indices),
    ]

    weight_stage_rows: list[dict[str, object]] = []
    known_post_cfg = {
        "file_scale_mode": "max_power",
        "file_scale_value": 0.4,
        "smooth_mode": "plain",
        "smooth_alpha": 0.15,
    }
    print("[INFO] Stage 1: weight scan with known two-way post-processing...")
    for method in ["logit", "prob", "rank"]:
        for weights in iter_weight_grid():
            pred = blend_three(perch, stage3, base_cnn, method=method, weights=weights, rank_mats=rank_mats)
            pred = apply_postprocess(pred, filename=filename, cfg=known_post_cfg)
            result = eval_prediction("weight_scan", y_true, pred, base_class_scores, base_auc, class_indices)
            weight_stage_rows.append(
                {
                    "method": method,
                    "perch_weight": weights[0],
                    "stage3_weight": weights[1],
                    "base_cnn_weight": weights[2],
                    **known_post_cfg,
                    **asdict(result),
                }
            )
    weight_stage_df = pd.DataFrame(weight_stage_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    weight_stage_df.insert(0, "rank", np.arange(1, len(weight_stage_df) + 1))

    top_weight_cfgs = (
        weight_stage_df[["method", "perch_weight", "stage3_weight", "base_cnn_weight"]]
        .head(int(args.top_k_weights))
        .drop_duplicates()
        .to_dict("records")
    )

    rows: list[dict[str, object]] = []
    best_pred = base_pred
    best_config: dict[str, object] | None = None
    best_auc = -np.inf

    print(f"[INFO] Stage 2: post-processing scan for top {len(top_weight_cfgs)} weight configs...")
    for weight_cfg in top_weight_cfgs:
        weights = (
            float(weight_cfg["perch_weight"]),
            float(weight_cfg["stage3_weight"]),
            float(weight_cfg["base_cnn_weight"]),
        )
        method = str(weight_cfg["method"])
        blended = blend_three(perch, stage3, base_cnn, method=method, weights=weights, rank_mats=rank_mats)
        for post_cfg in iter_postprocess_grid():
            if str(post_cfg["method"]) != method:
                continue
            cfg = {**weight_cfg, **post_cfg}
            pred = apply_postprocess(blended, filename=filename, cfg=cfg)

            result = eval_prediction("grid", y_true, pred, base_class_scores, base_auc, class_indices)
            row = {**cfg, **asdict(result)}
            rows.append(row)
            if result.auc > best_auc:
                best_auc = result.auc
                best_config = row
                best_pred = pred

    results_df = pd.DataFrame(rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))

    baseline_df = pd.DataFrame([asdict(row) for row in base_rows])
    best_auc_final, scored_indices, best_class_scores = macro_auc_and_class_scores(y_true, best_pred)
    class_delta_df = pd.DataFrame(
        {
            "class_index": scored_indices,
            "class_name": [class_names[int(idx)] for idx in scored_indices],
            "baseline_auc": base_class_scores,
            "best_auc": best_class_scores,
            "delta": best_class_scores - base_class_scores,
            "positives": y_true[:, scored_indices].sum(axis=0).astype(int),
        }
    ).sort_values("delta", ascending=False)

    summary = {
        "inputs": {
            "perch_dir": str(Path(args.perch_dir)),
            "stage3_oof_path": str(Path(args.stage3_oof_path)),
            "base_cnn_oof_path": str(Path(args.base_cnn_oof_path)),
            "sample_submission_path": str(Path(args.sample_submission_path)),
        },
        "leakage_policy": (
            "Only saved OOF predictions and train labels are used. Perch strict OOF uses validation-fold "
            "LogReg outputs for fitted classes and sigmoid(raw Perch logits) fallback for unfitted classes. "
            "Stage3 and base CNN inputs are saved OOF predictions aligned by row_id. No hidden-test "
            "predictions or leaderboard feedback are used for parameter selection."
        ),
        "rows": int(len(row_id)),
        "files": int(len(pd.unique(filename))),
        "classes": int(len(class_names)),
        "scored_classes": int(len(class_indices)),
        "baseline_perch_stage3_auc": float(base_auc),
        "best_auc_recomputed": float(best_auc_final),
        "best": best_config,
    }

    baseline_df.to_csv(output_dir / "baseline_scores.csv", index=False)
    weight_stage_df.to_csv(output_dir / "weight_scan_results.csv", index=False)
    results_df.to_csv(output_dir / "grid_results.csv", index=False)
    class_delta_df.to_csv(output_dir / "best_class_deltas.csv", index=False)
    np.savez_compressed(
        output_dir / "best_oof_predictions.npz",
        row_id=row_id,
        filename=filename,
        y_true=y_true.astype(np.uint8),
        pred=best_pred.astype(np.float32),
        baseline=base_pred.astype(np.float32),
        perch=perch.astype(np.float32),
        stage3=stage3.astype(np.float32),
        base_cnn=base_cnn.astype(np.float32),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[INFO] Baselines")
    print(baseline_df.to_string(index=False))
    print("[INFO] Top 20 grid rows")
    print(results_df.head(20).to_string(index=False))
    print("[INFO] Best class deltas")
    print(class_delta_df.head(15).to_string(index=False))
    print("[INFO] Worst class deltas")
    print(class_delta_df.tail(15).to_string(index=False))
    print(f"[INFO] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
