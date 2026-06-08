#!/usr/bin/env python3
"""Probe safe final-ensemble branch replacements on saved OOF predictions.

This script mirrors the current online-safe submission shape:

PerchLR + Mamba + Attention + SSM + Stage3, no RawWave, no TTA,
BLEND_MODE=family3, FILE_SCALE_MODE=topk_mean, FILE_SCALE_VALUE=2.

It only reads existing OOF files and tries branch replacements/additions that do
not increase online cost materially.  It is meant for late-stage triage, not a
broad leaderboard-tuned search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import birdclef2026_whitelist_blend_unified_raw_waveform as base


EPS = 1e-6

DEFAULT_SAFE_WEIGHTS = {
    "perch_lr": 0.22625,
    "mamba": 0.14650,
    "stage3": 0.13575,
    "attention": 0.14650,
    "ssm": 0.25000,
}

RETUNED_SAFE_WEIGHTS = {
    "perch_lr": 0.20000,
    "mamba": 0.11200,
    "stage3": 0.13750,
    "attention": 0.20800,
    "ssm": 0.34250,
}

CURRENT = {
    "perch_lr": "outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1",
    "mamba": "outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1",
    "attention": "outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1",
    "ssm": "outputs/perch_sequence_ssm_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1",
    "stage3": (
        "outputs/birdclef2026_gm_stage3_perchcnn_white_v1/"
        "20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo/"
        "soundscape_oof_predictions.csv"
    ),
    "stage3_tta_m05p05": "outputs/cnn_shift_tta_sweep_stage3_white_20260526/stage3_tta_m0p5_p0p5_oof_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF probe for safe final branch replacements.")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/safe_oof_branch_probe_20260603")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--top-stage3", type=int, default=20)
    parser.add_argument("--top-perch", type=int, default=40)
    return parser.parse_args()


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0.0:
        raise ValueError(f"Non-positive weights: {weights}")
    return {k: max(0.0, float(v)) / total for k, v in weights.items() if max(0.0, float(v)) > 1e-12}


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32, copy=False)


def rank01_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return (ranks / float(n - 1)).astype(np.float32)


def rank01_matrix(pred: np.ndarray) -> np.ndarray:
    out = np.empty_like(pred, dtype=np.float32)
    for class_idx in range(pred.shape[1]):
        out[:, class_idx] = rank01_1d(pred[:, class_idx])
    return out


def weighted_logit_blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    weights = normalize_weights(weights)
    fused = None
    for name, weight in weights.items():
        if name not in preds:
            continue
        term = float(weight) * logit_np(preds[name])
        fused = term if fused is None else fused + term
    if fused is None:
        raise ValueError(f"No available weighted branches: {weights}")
    return np.clip(sigmoid_np(fused), 0.0, 1.0)


def weighted_rank_blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    weights = normalize_weights(weights)
    fused = None
    for name, weight in weights.items():
        if name not in preds:
            continue
        term = float(weight) * rank01_matrix(preds[name])
        fused = term if fused is None else fused + term
    if fused is None:
        raise ValueError(f"No available weighted branches: {weights}")
    return np.clip(fused.astype(np.float32, copy=False), 0.0, 1.0)


def family3_blend(preds: dict[str, np.ndarray], weights: dict[str, float], class_names: list[str]) -> np.ndarray:
    logit_pred = weighted_logit_blend(preds, weights)
    rank_pred = weighted_rank_blend(preds, weights)
    out = logit_pred.astype(np.float32, copy=True)
    son_mask = np.asarray([name.startswith("47158son") for name in class_names], dtype=bool)
    out[:, son_mask] = rank_pred[:, son_mask]
    return np.clip(out, 0.0, 1.0)


def final_pred(preds: dict[str, np.ndarray], weights: dict[str, float], class_names: list[str], filename: np.ndarray) -> np.ndarray:
    fused = family3_blend(preds, weights=weights, class_names=class_names)
    return base.file_level_topk_mean_scale(fused, filename=filename, topk=2)


def eval_auc(name: str, y_true: np.ndarray, pred: np.ndarray, base_auc: float, base_scores: np.ndarray, class_indices: np.ndarray) -> dict[str, object]:
    result = base.eval_prediction(name, y_true, pred, base_scores, base_auc, class_indices)
    row = result.__dict__.copy()
    return row


def load_npz_candidate(path: Path, row_id: np.ndarray, y_true: np.ndarray, name: str) -> np.ndarray:
    return base.load_npz_oof(path, row_id=row_id, y_true=y_true, name=name)


def load_csv_candidate(path: Path, class_names: list[str], row_id: np.ndarray, y_true: np.ndarray, name: str) -> np.ndarray:
    return base.load_csv_oof(path, class_names=class_names, row_id=row_id, y_true=y_true, name=name)


def iter_existing_npz_dirs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path("outputs").glob(pattern))
    dirs = sorted({p for p in paths if (p / "oof_predictions.npz").exists()})
    return [p for p in dirs if "smoke" not in p.name.lower()]


def iter_stage3_csvs() -> list[Path]:
    paths = []
    for path in Path("outputs").glob("**/soundscape_oof_predictions.csv"):
        text = str(path)
        if "smoke" in text.lower():
            continue
        if "stage3" not in text:
            continue
        paths.append(path)
    paths.append(Path(CURRENT["stage3_tta_m05p05"]))
    return sorted({p for p in paths if p.exists()})


def branch_single_auc(y_true: np.ndarray, pred: np.ndarray, filename: np.ndarray) -> float:
    scaled = base.file_level_topk_mean_scale(pred, filename=filename, topk=2)
    auc, _, _ = base.macro_auc_and_class_scores(y_true, scaled)
    return float(auc)


def try_replacements(
    branch_name: str,
    candidate_paths: list[Path],
    loader,
    preds_base: dict[str, np.ndarray],
    weights: dict[str, float],
    class_names: list[str],
    row_id: np.ndarray,
    y_true: np.ndarray,
    filename: np.ndarray,
    base_auc: float,
    base_scores: np.ndarray,
    class_indices: np.ndarray,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = []
    for path in candidate_paths:
        try:
            pred = loader(path, str(path))
        except Exception as exc:
            rows.append(
                {
                    "name": f"replace_{branch_name}",
                    "branch": branch_name,
                    "path": str(path),
                    "auc": np.nan,
                    "single_auc": np.nan,
                    "delta_vs_base": np.nan,
                    "error": repr(exc)[:500],
                }
            )
            continue
        preds = dict(preds_base)
        preds[branch_name] = pred
        fused = final_pred(preds, weights=weights, class_names=class_names, filename=filename)
        row = eval_auc(f"replace_{branch_name}", y_true, fused, base_auc, base_scores, class_indices)
        row.update(
            {
                "branch": branch_name,
                "path": str(path),
                "single_auc": branch_single_auc(y_true, pred, filename),
                "error": "",
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["auc", "single_auc"], ascending=[False, False], na_position="last").reset_index(drop=True)
    if limit is not None:
        return df.head(limit).copy()
    return df


def temporal_add_grid(
    temporal_paths: list[Path],
    preds_base: dict[str, np.ndarray],
    weights_base: dict[str, float],
    class_names: list[str],
    row_id: np.ndarray,
    y_true: np.ndarray,
    filename: np.ndarray,
    base_auc: float,
    base_scores: np.ndarray,
    class_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    transfer_modes = ["all", "ssm", "mamba_attention", "attention"]
    weights_base = normalize_weights(weights_base)
    for path in temporal_paths:
        try:
            temporal = load_npz_candidate(path, row_id=row_id, y_true=y_true, name=str(path))
        except Exception as exc:
            rows.append({"name": "add_temporal", "path": str(path), "auc": np.nan, "delta_vs_base": np.nan, "error": repr(exc)[:500]})
            continue
        for weight in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]:
            for mode in transfer_modes:
                weights = dict(weights_base)
                weights["temporal"] = float(weight)
                if mode == "all":
                    scale = 1.0 - float(weight)
                    for key in list(weights_base):
                        weights[key] = weights_base[key] * scale
                elif mode == "ssm":
                    weights["ssm"] = max(0.0, weights_base.get("ssm", 0.0) - float(weight))
                elif mode == "mamba_attention":
                    weights["mamba"] = max(0.0, weights_base.get("mamba", 0.0) - float(weight) * 0.5)
                    weights["attention"] = max(0.0, weights_base.get("attention", 0.0) - float(weight) * 0.5)
                elif mode == "attention":
                    weights["attention"] = max(0.0, weights_base.get("attention", 0.0) - float(weight))
                preds = dict(preds_base)
                preds["temporal"] = temporal
                fused = final_pred(preds, weights=weights, class_names=class_names, filename=filename)
                row = eval_auc("add_temporal", y_true, fused, base_auc, base_scores, class_indices)
                row.update(
                    {
                        "branch": "temporal",
                        "path": str(path),
                        "single_auc": branch_single_auc(y_true, temporal, filename),
                        "temporal_weight": float(weight),
                        "transfer_mode": mode,
                        "weights_json": json.dumps(normalize_weights(weights), sort_keys=True),
                        "error": "",
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["auc", "single_auc"], ascending=[False, False], na_position="last").reset_index(drop=True)


def main() -> None:
    base.install_numpy_core_compat()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = base.load_class_names(Path(args.sample_submission_path))
    y_true, perch_lr, row_id, filename = base.reconstruct_perch_lr_online_like_oof(
        Path(CURRENT["perch_lr"]),
        n_folds=args.n_folds,
    )
    preds_base = {
        "perch_lr": perch_lr,
        "mamba": load_npz_candidate(Path(CURRENT["mamba"]), row_id=row_id, y_true=y_true, name="current_mamba"),
        "attention": load_npz_candidate(Path(CURRENT["attention"]), row_id=row_id, y_true=y_true, name="current_attention"),
        "ssm": load_npz_candidate(Path(CURRENT["ssm"]), row_id=row_id, y_true=y_true, name="current_ssm"),
        "stage3": load_csv_candidate(Path(CURRENT["stage3"]), class_names=class_names, row_id=row_id, y_true=y_true, name="current_stage3"),
    }

    baselines = []
    baseline_preds = {}
    for label, weights in [("default_safe", DEFAULT_SAFE_WEIGHTS), ("retuned_safe", RETUNED_SAFE_WEIGHTS)]:
        pred = final_pred(preds_base, weights=weights, class_names=class_names, filename=filename)
        auc, class_indices, class_scores = base.macro_auc_and_class_scores(y_true, pred)
        baseline_preds[label] = (pred, auc, class_indices, class_scores, normalize_weights(weights))
        baselines.append(
            {
                "label": label,
                "auc": auc,
                "weights_json": json.dumps(normalize_weights(weights), sort_keys=True),
            }
        )
    pd.DataFrame(baselines).to_csv(output_dir / "baseline_safe_scores.csv", index=False)

    # Use the retuned line as reference for replacement scans because it is the
    # strongest no-RawWave/no-TTA local baseline, while also writing default rows.
    _, base_auc, class_indices, base_scores, retuned_weights = baseline_preds["retuned_safe"]

    def npz_loader(path: Path, label: str) -> np.ndarray:
        return load_npz_candidate(path, row_id=row_id, y_true=y_true, name=label)

    def csv_loader(path: Path, label: str) -> np.ndarray:
        return load_csv_candidate(path, class_names=class_names, row_id=row_id, y_true=y_true, name=label)

    mamba_paths = iter_existing_npz_dirs(
        [
            "perch_spatial_mamba*/",
            "perch_embedding_mlp*/",
        ]
    )
    attention_paths = iter_existing_npz_dirs(["perch_spatial_attention*/", "perch_spatial_mhattn*/"])
    ssm_paths = iter_existing_npz_dirs(["perch_sequence_ssm*/"])
    temporal_paths = iter_existing_npz_dirs(["perch_temporal_head*/"])
    stage3_paths = iter_stage3_csvs()

    scans = {
        "replace_mamba": try_replacements(
            "mamba",
            mamba_paths,
            npz_loader,
            preds_base,
            retuned_weights,
            class_names,
            row_id,
            y_true,
            filename,
            base_auc,
            base_scores,
            class_indices,
            limit=args.top_perch,
        ),
        "replace_attention": try_replacements(
            "attention",
            attention_paths,
            npz_loader,
            preds_base,
            retuned_weights,
            class_names,
            row_id,
            y_true,
            filename,
            base_auc,
            base_scores,
            class_indices,
            limit=args.top_perch,
        ),
        "replace_ssm": try_replacements(
            "ssm",
            ssm_paths,
            npz_loader,
            preds_base,
            retuned_weights,
            class_names,
            row_id,
            y_true,
            filename,
            base_auc,
            base_scores,
            class_indices,
            limit=args.top_perch,
        ),
        "replace_stage3": try_replacements(
            "stage3",
            stage3_paths,
            csv_loader,
            preds_base,
            retuned_weights,
            class_names,
            row_id,
            y_true,
            filename,
            base_auc,
            base_scores,
            class_indices,
            limit=args.top_stage3,
        ),
        "add_temporal": temporal_add_grid(
            temporal_paths,
            preds_base,
            retuned_weights,
            class_names,
            row_id,
            y_true,
            filename,
            base_auc,
            base_scores,
            class_indices,
        ).head(args.top_perch),
    }

    for name, df in scans.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    summary_rows = []
    for name, df in scans.items():
        if len(df) == 0:
            continue
        best = df.iloc[0].to_dict()
        best["scan"] = name
        summary_rows.append(best)
    summary = pd.DataFrame(summary_rows).sort_values(["auc", "single_auc"], ascending=[False, False], na_position="last")
    summary.to_csv(output_dir / "summary_best.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "baseline": baselines,
                "best_by_scan": summary_rows,
                "current_paths": CURRENT,
            },
            fp,
            indent=2,
        )
    print("[INFO] Baselines")
    print(pd.DataFrame(baselines).to_string(index=False))
    print("[INFO] Best by scan")
    print(summary[["scan", "auc", "delta_vs_base", "single_auc", "path"]].to_string(index=False))


if __name__ == "__main__":
    main()
