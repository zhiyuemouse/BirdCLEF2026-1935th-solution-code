#!/usr/bin/env python3
"""Leak-safe OOF grid for unified Perch+Stage3 ensemble plus raw waveform.

This script only reads saved out-of-fold predictions and training labels.  It
is intended to answer one narrow question: does the raw waveform Transformer
branch add useful diversity to the current unified PerchLR + PerchMamba +
Stage3 CNN + PerchAttention ensemble?
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


EPS = 1e-6


def install_numpy_core_compat() -> None:
    """Let numpy-1.x environments read numpy-2 pickled object arrays."""

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.numerictypes", np.core.numerictypes)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)


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
    parser = argparse.ArgumentParser(description="OOF blend grid for unified Perch+Stage3 plus raw waveform.")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--perch-lr-dir", type=str, default="outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1")
    parser.add_argument(
        "--mamba-dir",
        type=str,
        default="outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1",
    )
    parser.add_argument(
        "--attention-dir",
        type=str,
        default="outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1",
    )
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
        "--raw-wave-oof-path",
        type=str,
        default=(
            "outputs/birdclef2026_raw_waveform_transformer/"
            "20260512_013731_raw_wave_conv_tokenizer_base_long_n32_d768/"
            "soundscape_oof_predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/whitelist_blend_unified_raw_waveform_20260512")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--top-k-weights", type=int, default=80)
    parser.add_argument("--file-scale-topk", type=int, default=2)
    parser.add_argument("--raw-scan-max", type=float, default=0.25)
    parser.add_argument("--raw-scan-step", type=float, default=0.0025)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


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
    class_indices = np.asarray(
        [idx for idx in range(y_true.shape[1]) if y_true[:, idx].max() > 0 and y_true[:, idx].min() < 1],
        dtype=np.int32,
    )
    if len(class_indices) == 0:
        raise ValueError("No scored classes for macro AUC.")
    class_scores = roc_auc_score(
        y_true[:, class_indices],
        y_score[:, class_indices],
        average=None,
    ).astype(np.float32)
    return float(np.mean(class_scores)), class_indices, class_scores


def load_class_names(path: Path) -> list[str]:
    sample = pd.read_csv(path, nrows=0)
    return [col for col in sample.columns if col != "row_id"]


def load_fold_assignments(fold_assignment_path: Path, row_id: np.ndarray) -> np.ndarray:
    fold_df = pd.read_csv(fold_assignment_path)
    required = {"row_id", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise KeyError(f"Fold assignment is missing columns: {sorted(missing)}")
    fold_map = fold_df.drop_duplicates(subset=["row_id"]).set_index("row_id")["fold"]
    folds = pd.Series(row_id).map(fold_map)
    if folds.isna().any():
        examples = pd.Series(row_id)[folds.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {examples}")
    return folds.astype(int).to_numpy()


def reconstruct_perch_lr_online_like_oof(perch_dir: Path, n_folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(perch_dir / "oof_predictions.npz", allow_pickle=True)
    y_true = npz["y_true"].astype(np.float32)
    raw_scores = npz["raw_scores"].astype(np.float32)
    mixed_oof = npz["oof_pred"].astype(np.float32)
    row_id = npz["row_id"].astype(str)
    filename = npz["filename"].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = joblib.load(perch_dir / "perch_context_logreg_artifacts.joblib")
    folds = artifact["folds"]
    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} Perch LR folds, got {len(folds)}")

    strict = sigmoid(raw_scores).astype(np.float32)
    config = artifact.get("config", {})
    fold_assignment_path = str(config.get("fold_assignment_path", "") or "")
    if fold_assignment_path:
        row_folds = load_fold_assignments(Path(fold_assignment_path), row_id=row_id)
        fold_values = sorted(pd.Index(row_folds).unique().tolist())
        if len(fold_values) != len(folds):
            raise ValueError(f"Fold assignment has {len(fold_values)} folds but artifact has {len(folds)} folds.")
        for fold_artifact, fold_value in zip(folds, fold_values):
            valid_idx = np.where(row_folds == int(fold_value))[0]
            fitted = np.asarray(fold_artifact["fitted_class_indices"], dtype=np.int32)
            strict[np.ix_(valid_idx, fitted)] = mixed_oof[np.ix_(valid_idx, fitted)]
    else:
        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (_, valid_idx) in enumerate(gkf.split(np.zeros(len(filename)), groups=filename)):
            valid_idx = np.asarray(valid_idx, dtype=np.int64)
            fitted = np.asarray(folds[fold_idx]["fitted_class_indices"], dtype=np.int32)
            strict[np.ix_(valid_idx, fitted)] = mixed_oof[np.ix_(valid_idx, fitted)]

    return y_true, strict.astype(np.float32), row_id, filename


def load_npz_oof(
    path: Path,
    row_id: np.ndarray,
    y_true: np.ndarray,
    name: str,
) -> np.ndarray:
    npz = np.load(path / "oof_predictions.npz", allow_pickle=True)
    pred = npz["oof_pred"].astype(np.float32)
    source_row_id = npz["row_id"].astype(str)
    source_y_true = npz["y_true"].astype(np.float32)
    if not np.array_equal(source_row_id, row_id):
        order = pd.DataFrame({"row_id": row_id}).merge(
            pd.DataFrame({"row_id": source_row_id, "_pos": np.arange(len(source_row_id), dtype=np.int64)}),
            on="row_id",
            how="left",
            validate="one_to_one",
        )["_pos"]
        if order.isna().any():
            examples = pd.Series(row_id)[order.isna()].astype(str).head(5).tolist()
            raise ValueError(f"{name} OOF misses rows after row_id alignment. Examples: {examples}")
        order_arr = order.to_numpy(dtype=np.int64)
        pred = pred[order_arr]
        source_y_true = source_y_true[order_arr]
    if not np.array_equal(source_y_true, y_true):
        raise ValueError(f"{name} y_true does not match Perch LR y_true after row_id alignment.")
    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0)


def load_csv_oof(path: Path, class_names: list[str], row_id: np.ndarray, y_true: np.ndarray, name: str) -> np.ndarray:
    df = pd.read_csv(path)
    target_cols = [f"target_{label}" for label in class_names]
    required = ["row_id", *target_cols, *class_names]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{name} OOF is missing columns: {missing[:10]}")

    aligned = pd.DataFrame({"row_id": row_id}).merge(
        df[["row_id", *target_cols, *class_names]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[class_names].isna().any().any():
        raise ValueError(f"{name} OOF missing predictions after row_id alignment.")
    source_y = aligned[target_cols].to_numpy(dtype=np.float32)
    if not np.array_equal(source_y, y_true):
        raise ValueError(f"{name} target columns do not match Perch LR y_true after row_id alignment.")
    return np.clip(aligned[class_names].to_numpy(dtype=np.float32), 0.0, 1.0)


def file_level_topk_mean_scale(pred: np.ndarray, filename: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        if len(idx) == 0:
            continue
        p = pred[idx]
        k = max(1, min(int(topk), len(idx)))
        scale = np.sort(p, axis=0)[-k:].mean(axis=0, keepdims=True)
        out[idx] = p * scale
    return np.clip(out.astype(np.float32, copy=False), 0.0, 1.0)


def blend_logit(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Weights must sum to a positive value, got {weights}")
    fused = None
    for key, weight in weights.items():
        term = (float(weight) / total) * logit(preds[key])
        fused = term if fused is None else fused + term
    return np.clip(sigmoid(fused).astype(np.float32, copy=False), 0.0, 1.0)


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


def iter_raw_weight_grid(raw_scan_max: float, raw_scan_step: float) -> Iterable[dict[str, float]]:
    base = {"perch_lr": 0.25, "mamba": 0.30, "stage3": 0.15, "attention": 0.30}
    if raw_scan_max < 0:
        raise ValueError(f"raw_scan_max must be non-negative, got {raw_scan_max}")
    if raw_scan_step <= 0:
        raise ValueError(f"raw_scan_step must be positive, got {raw_scan_step}")
    for raw_weight in np.arange(0.0, raw_scan_max + raw_scan_step * 0.5, raw_scan_step):
        scale = 1.0 - float(raw_weight)
        if scale < 0:
            continue
        yield {
            "perch_lr": round(base["perch_lr"] * scale, 6),
            "mamba": round(base["mamba"] * scale, 6),
            "stage3": round(base["stage3"] * scale, 6),
            "attention": round(base["attention"] * scale, 6),
            "raw_wave": round(float(raw_weight), 6),
        }


def iter_local_weight_grid() -> Iterable[dict[str, float]]:
    # Tiny perturbation whitelist around the uploaded 4-way blend.  The raw
    # branch has weak standalone CV, so we only test small transfers rather
    # than a broad overfit-prone grid.
    base = {"perch_lr": 0.25, "mamba": 0.30, "stage3": 0.15, "attention": 0.30, "raw_wave": 0.0}
    transfers = ["all_scaled", "mamba", "attention", "stage3", "perch_lr", "mamba_attention"]
    for raw_weight in np.arange(0.0, 0.0601, 0.0025):
        raw_weight = float(raw_weight)
        for mode in transfers:
            weights = dict(base)
            weights["raw_wave"] = raw_weight
            if mode == "all_scaled":
                scale = 1.0 - raw_weight
                for key in ["perch_lr", "mamba", "stage3", "attention"]:
                    weights[key] = base[key] * scale
            elif mode in ["mamba", "attention", "stage3", "perch_lr"]:
                weights[mode] = max(0.0, base[mode] - raw_weight)
            elif mode == "mamba_attention":
                weights["mamba"] = max(0.0, base["mamba"] - raw_weight * 0.5)
                weights["attention"] = max(0.0, base["attention"] - raw_weight * 0.5)
            yield {key: round(float(value), 6) for key, value in weights.items()}


def main() -> None:
    install_numpy_core_compat()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.sample_submission_path))
    y_true, perch_lr, row_id, filename = reconstruct_perch_lr_online_like_oof(
        Path(args.perch_lr_dir),
        n_folds=args.n_folds,
    )
    mamba = load_npz_oof(Path(args.mamba_dir), row_id=row_id, y_true=y_true, name="mamba")
    attention = load_npz_oof(Path(args.attention_dir), row_id=row_id, y_true=y_true, name="attention")
    stage3 = load_csv_oof(Path(args.stage3_oof_path), class_names, row_id=row_id, y_true=y_true, name="stage3")
    raw_wave = load_csv_oof(Path(args.raw_wave_oof_path), class_names, row_id=row_id, y_true=y_true, name="raw_wave")

    preds = {
        "perch_lr": perch_lr,
        "mamba": mamba,
        "stage3": stage3,
        "attention": attention,
        "raw_wave": raw_wave,
    }
    uploaded_weights = {"perch_lr": 0.25, "mamba": 0.30, "stage3": 0.15, "attention": 0.30, "raw_wave": 0.0}
    base_pred_raw = blend_logit(preds, uploaded_weights)
    base_pred = file_level_topk_mean_scale(base_pred_raw, filename=filename, topk=args.file_scale_topk)
    base_auc, class_indices, base_class_scores = macro_auc_and_class_scores(y_true, base_pred)

    baseline_rows = []
    for name, pred in preds.items():
        pred_scaled = file_level_topk_mean_scale(pred, filename=filename, topk=args.file_scale_topk)
        baseline_rows.append(eval_prediction(name, y_true, pred_scaled, base_class_scores, base_auc, class_indices))
    baseline_rows.append(eval_prediction("uploaded_4way", y_true, base_pred, base_class_scores, base_auc, class_indices))
    baseline_df = pd.DataFrame([asdict(row) for row in baseline_rows])

    raw_rows: list[dict[str, object]] = []
    best_pred = base_pred
    best_auc = base_auc
    best_config: dict[str, object] | None = None

    print("[INFO] Stage 1: raw small-weight scan around uploaded 4-way weights...")
    for weights in iter_raw_weight_grid(raw_scan_max=float(args.raw_scan_max), raw_scan_step=float(args.raw_scan_step)):
        pred = blend_logit(preds, weights)
        pred = file_level_topk_mean_scale(pred, filename=filename, topk=args.file_scale_topk)
        result = eval_prediction("raw_weight_scan", y_true, pred, base_class_scores, base_auc, class_indices)
        row = {**weights, "file_scale_topk": int(args.file_scale_topk), **asdict(result)}
        raw_rows.append(row)
        if result.auc > best_auc:
            best_auc = float(result.auc)
            best_config = row
            best_pred = pred
    raw_scan_df = pd.DataFrame(raw_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    raw_scan_df.insert(0, "rank", np.arange(1, len(raw_scan_df) + 1))

    local_rows: list[dict[str, object]] = []
    print("[INFO] Stage 2: narrow 5-way local grid...")
    for weights in iter_local_weight_grid():
        pred = blend_logit(preds, weights)
        pred = file_level_topk_mean_scale(pred, filename=filename, topk=args.file_scale_topk)
        result = eval_prediction("local_5way_grid", y_true, pred, base_class_scores, base_auc, class_indices)
        row = {**weights, "file_scale_topk": int(args.file_scale_topk), **asdict(result)}
        local_rows.append(row)
        if result.auc > best_auc:
            best_auc = float(result.auc)
            best_config = row
            best_pred = pred
    grid_df = pd.DataFrame(local_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grid_df.insert(0, "rank", np.arange(1, len(grid_df) + 1))

    best_auc_final, scored_indices, best_class_scores = macro_auc_and_class_scores(y_true, best_pred)
    class_delta_df = pd.DataFrame(
        {
            "class_index": scored_indices,
            "class_name": [class_names[int(idx)] for idx in scored_indices],
            "uploaded_4way_auc": base_class_scores,
            "best_auc": best_class_scores,
            "delta": best_class_scores - base_class_scores,
            "positives": y_true[:, scored_indices].sum(axis=0).astype(int),
        }
    ).sort_values("delta", ascending=False)

    summary = {
        "inputs": {
            "sample_submission_path": str(Path(args.sample_submission_path)),
            "perch_lr_dir": str(Path(args.perch_lr_dir)),
            "mamba_dir": str(Path(args.mamba_dir)),
            "attention_dir": str(Path(args.attention_dir)),
            "stage3_oof_path": str(Path(args.stage3_oof_path)),
            "raw_wave_oof_path": str(Path(args.raw_wave_oof_path)),
        },
        "leakage_policy": (
            "Only saved OOF predictions and train labels are used. PerchLR is reconstructed in online-like form: "
            "sigmoid(raw Perch logits) fallback for unfitted classes, fitted validation classes from fold OOF. "
            "Spatial heads, Stage3, and raw waveform are saved OOF predictions aligned by row_id. No hidden-test "
            "predictions or leaderboard feedback are used for this grid."
        ),
        "rows": int(len(row_id)),
        "files": int(len(pd.unique(filename))),
        "classes": int(len(class_names)),
        "scored_classes": int(len(class_indices)),
        "uploaded_4way_weights": uploaded_weights,
        "file_scale_topk": int(args.file_scale_topk),
        "uploaded_4way_auc": float(base_auc),
        "best_auc_recomputed": float(best_auc_final),
        "best": best_config,
    }

    baseline_df.to_csv(output_dir / "baseline_scores.csv", index=False)
    raw_scan_df.to_csv(output_dir / "raw_weight_scan_results.csv", index=False)
    grid_df.to_csv(output_dir / "grid_results.csv", index=False)
    class_delta_df.to_csv(output_dir / "best_class_deltas.csv", index=False)
    np.savez_compressed(
        output_dir / "best_oof_predictions.npz",
        row_id=row_id,
        filename=filename,
        y_true=y_true.astype(np.uint8),
        pred=best_pred.astype(np.float32),
        uploaded_4way=base_pred.astype(np.float32),
        perch_lr=perch_lr.astype(np.float32),
        mamba=mamba.astype(np.float32),
        stage3=stage3.astype(np.float32),
        attention=attention.astype(np.float32),
        raw_wave=raw_wave.astype(np.float32),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[INFO] Baselines")
    print(baseline_df.to_string(index=False))
    print("[INFO] Top raw-weight scan rows")
    print(raw_scan_df.head(20).to_string(index=False))
    print("[INFO] Top local 5-way grid rows")
    print(grid_df.head(20).to_string(index=False))
    print("[INFO] Best class deltas")
    print(class_delta_df.head(15).to_string(index=False))
    print("[INFO] Worst class deltas")
    print(class_delta_df.tail(15).to_string(index=False))
    print(f"[INFO] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
