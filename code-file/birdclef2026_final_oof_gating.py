#!/usr/bin/env python3
"""OOF-only file-level and label-family gating experiments.

This is a late-stage validation script.  It starts from the current strongest
non-prior OOF candidate, reconstructs it from saved model OOF predictions, and
then tries two cheap post-processing families:

1. File-level presence gating: use the 12 windows inside each soundscape to
   estimate whether a class is present in the whole file, then lightly gate
   each window score for that class.
2. Label-family gating: choose a small set of global candidate predictions per
   coarse label family, not per class.

No hidden-test predictions, public LB feedback, or validation labels outside
the ordinary OOF metric are used to create predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import birdclef2026_whitelist_blend_unified_raw_waveform as base
from birdclef2026_final_oof_tricks import blend_rank, rank01_matrix


EPS = 1e-6

DEFAULT_WEIGHTS = {
    "perch_lr": 0.22625,
    "mamba": 0.14650,
    "stage3": 0.13575,
    "attention": 0.14650,
    "raw_wave": 0.09500,
    "ssm": 0.25000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF file-level and label-family gating experiments.")
    parser.add_argument(
        "--input-npz",
        type=str,
        default="outputs/final_oof_tricks_mambatta_branch025_20260528/final_oof_predictions.npz",
    )
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/final_oof_gating_20260530")
    parser.add_argument("--alpha-logit", type=float, default=0.70)
    return parser.parse_args()


def macro_auc(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return base.macro_auc_and_class_scores(y_true, pred)


def eval_prediction(
    name: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    base_auc: float,
    base_class_scores: np.ndarray,
    class_indices: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result = base.eval_prediction(name, y_true, pred, base_class_scores, base_auc, class_indices)
    row = asdict(result)
    if extra:
        row.update(extra)
    return row


def topk_stat(p: np.ndarray, topk: int, method: str) -> np.ndarray:
    k = max(1, min(int(topk), len(p)))
    if method == "topk_mean":
        return np.sort(p, axis=0)[-k:].mean(axis=0)
    if method == "topk_max":
        return np.max(p, axis=0)
    if method == "topk_geom":
        top = np.sort(np.clip(p, EPS, 1.0), axis=0)[-k:]
        return np.exp(np.log(top).mean(axis=0))
    if method == "mean":
        return np.mean(p, axis=0)
    if method == "mean_max":
        return 0.5 * np.mean(p, axis=0) + 0.5 * np.max(p, axis=0)
    if method == "topk_mean_max":
        return 0.5 * np.sort(p, axis=0)[-k:].mean(axis=0) + 0.5 * np.max(p, axis=0)
    if method == "noisy_or":
        return 1.0 - np.prod(1.0 - np.clip(p, 0.0, 1.0), axis=0)
    raise ValueError(f"Unknown file presence method: {method}")


def file_presence_gate(
    pred: np.ndarray,
    filename: np.ndarray,
    method: str,
    topk: int,
    gamma: float,
    strength: float,
) -> np.ndarray:
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        p = pred[idx].astype(np.float32, copy=False)
        presence = np.clip(topk_stat(p, topk=topk, method=method), EPS, 1.0)
        gate = (1.0 - float(strength)) + float(strength) * np.power(presence, float(gamma))
        out[idx] = p * gate[None, :]
    return np.clip(out, 0.0, 1.0)


def file_presence_logit_gate(
    pred: np.ndarray,
    filename: np.ndarray,
    method: str,
    topk: int,
    beta: float,
) -> np.ndarray:
    out = pred.astype(np.float32, copy=True)
    for name in pd.Index(filename).unique():
        idx = np.where(filename == name)[0]
        p = pred[idx].astype(np.float32, copy=False)
        presence = np.clip(topk_stat(p, topk=topk, method=method), EPS, 1.0 - EPS)
        out[idx] = base.sigmoid(base.logit(p) + float(beta) * base.logit(presence)[None, :])
    return np.clip(out, 0.0, 1.0)


def iter_file_gate_configs() -> Iterable[dict[str, object]]:
    methods = ["topk_mean", "topk_max", "topk_geom", "mean", "mean_max", "topk_mean_max", "noisy_or"]
    topks = [1, 2, 3, 4]
    gammas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    strengths = [0.35, 0.50, 0.65, 0.80, 1.00]
    seen: set[str] = set()
    for method in methods:
        for topk in topks:
            for gamma in gammas:
                for strength in strengths:
                    if method in {"topk_max", "noisy_or", "mean"} and topk != 1:
                        continue
                    cfg = {
                        "gate_type": "multiply",
                        "method": method,
                        "topk": int(topk),
                        "gamma": float(gamma),
                        "strength": float(strength),
                        "beta": np.nan,
                    }
                    key = json.dumps(cfg, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        yield cfg
    for method in ["topk_mean", "topk_max", "topk_geom", "mean_max", "topk_mean_max"]:
        for topk in topks:
            for beta in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
                if method == "topk_max" and topk != 1:
                    continue
                cfg = {
                    "gate_type": "logit",
                    "method": method,
                    "topk": int(topk),
                    "gamma": np.nan,
                    "strength": np.nan,
                    "beta": float(beta),
                }
                key = json.dumps(cfg, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    yield cfg


def apply_file_gate(pred: np.ndarray, filename: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    if cfg["gate_type"] == "multiply":
        return file_presence_gate(
            pred,
            filename=filename,
            method=str(cfg["method"]),
            topk=int(cfg["topk"]),
            gamma=float(cfg["gamma"]),
            strength=float(cfg["strength"]),
        )
    if cfg["gate_type"] == "logit":
        return file_presence_logit_gate(
            pred,
            filename=filename,
            method=str(cfg["method"]),
            topk=int(cfg["topk"]),
            beta=float(cfg["beta"]),
        )
    raise ValueError(f"Unknown gate_type: {cfg['gate_type']}")


def make_family_masks(class_names: list[str], y_true: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    labels = np.asarray(class_names)
    counts = y_true.sum(axis=0)
    is_son = np.asarray([name.startswith("47158son") for name in labels])
    is_numeric = np.asarray([name.isdigit() for name in labels])
    is_y = np.asarray([name.startswith("y") for name in labels])

    if mode == "family3":
        return {
            "son_47158": is_son,
            "numeric": is_numeric & ~is_son,
            "alpha_code": ~(is_son | is_numeric),
        }
    if mode == "family4":
        return {
            "son_47158": is_son,
            "numeric": is_numeric & ~is_son,
            "y_prefix": is_y & ~(is_son | is_numeric),
            "other_alpha": ~(is_son | is_numeric | is_y),
        }
    if mode == "family_plus_rare":
        rare = counts <= 2
        return {
            "rare_le2": rare,
            "son_47158_nonrare": is_son & ~rare,
            "numeric_nonrare": is_numeric & ~is_son & ~rare,
            "alpha_nonrare": ~(is_son | is_numeric | rare),
        }
    raise ValueError(f"Unknown family mode: {mode}")


def family_select_grid(
    candidates: dict[str, np.ndarray],
    class_names: list[str],
    y_true: np.ndarray,
    base_pred: np.ndarray,
    base_auc: float,
    base_class_scores: np.ndarray,
    class_indices: np.ndarray,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    masks = make_family_masks(class_names, y_true=y_true, mode=mode)
    family_names = [name for name, mask in masks.items() if bool(mask.any())]
    candidate_names = list(candidates.keys())

    rows: list[dict[str, object]] = []
    best_pred = base_pred
    best_auc = base_auc
    best_config = {family: "baseline" for family in family_names}

    def rec(pos: int, config: dict[str, str]) -> None:
        nonlocal best_pred, best_auc, best_config
        if pos == len(family_names):
            pred = base_pred.copy()
            for family in family_names:
                mask = masks[family]
                pred[:, mask] = candidates[config[family]][:, mask]
            row = eval_prediction(
                "label_family_select",
                y_true,
                pred,
                base_auc=base_auc,
                base_class_scores=base_class_scores,
                class_indices=class_indices,
                extra={"family_mode": mode, **config},
            )
            rows.append(row)
            if float(row["auc"]) > best_auc:
                best_auc = float(row["auc"])
                best_pred = pred
                best_config = dict(config)
            return
        family = family_names[pos]
        for candidate in candidate_names:
            config[family] = candidate
            rec(pos + 1, config)

    rec(0, {})
    df = pd.DataFrame(rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df, best_pred, best_config


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = base.load_class_names(Path(args.sample_submission_path))
    npz = np.load(args.input_npz, allow_pickle=True)
    row_id = npz["row_id"].astype(str)
    filename = npz["filename"].astype(str)
    y_true = npz["y_true"].astype(np.float32)
    preds = {
        "perch_lr": npz["model_perch_lr"].astype(np.float32),
        "mamba": npz["model_mamba"].astype(np.float32),
        "stage3": npz["model_stage3"].astype(np.float32),
        "attention": npz["model_attention"].astype(np.float32),
        "raw_wave": npz["model_raw_wave"].astype(np.float32),
        "ssm": npz["model_ssm"].astype(np.float32),
    }

    branch_logit = base.blend_logit(preds, DEFAULT_WEIGHTS)
    ranked = {key: rank01_matrix(value) for key, value in preds.items()}
    branch_rank = blend_rank(ranked, DEFAULT_WEIGHTS)
    mixed = np.clip(
        float(args.alpha_logit) * branch_logit + (1.0 - float(args.alpha_logit)) * branch_rank,
        0.0,
        1.0,
    ).astype(np.float32)
    baseline = base.file_level_topk_mean_scale(mixed, filename=filename, topk=2)
    base_auc, class_indices, base_class_scores = macro_auc(y_true, baseline)

    baseline_rows = [
        eval_prediction("mixed_no_file_gate", y_true, mixed, base_auc, base_class_scores, class_indices),
        eval_prediction("mixed_topk2_existing", y_true, baseline, base_auc, base_class_scores, class_indices),
        eval_prediction(
            "branch_logit_topk2",
            y_true,
            base.file_level_topk_mean_scale(branch_logit, filename=filename, topk=2),
            base_auc,
            base_class_scores,
            class_indices,
        ),
        eval_prediction(
            "branch_rank_topk2",
            y_true,
            base.file_level_topk_mean_scale(branch_rank, filename=filename, topk=2),
            base_auc,
            base_class_scores,
            class_indices,
        ),
    ]
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(output_dir / "baseline_scores.csv", index=False)

    print("[INFO] Step 1: file-level presence gating...")
    gate_rows: list[dict[str, object]] = []
    gate_predictions: dict[str, np.ndarray] = {
        "baseline": baseline,
        "no_file_gate": mixed,
        "branch_logit_topk2": base.file_level_topk_mean_scale(branch_logit, filename=filename, topk=2),
        "branch_rank_topk2": base.file_level_topk_mean_scale(branch_rank, filename=filename, topk=2),
    }
    best_gate_pred = baseline
    best_gate_auc = base_auc
    best_gate_config: dict[str, object] = {"candidate": "baseline"}
    for cfg in iter_file_gate_configs():
        gated = apply_file_gate(mixed, filename=filename, cfg=cfg)
        row = eval_prediction(
            "file_presence_gate",
            y_true,
            gated,
            base_auc=base_auc,
            base_class_scores=base_class_scores,
            class_indices=class_indices,
            extra=cfg,
        )
        gate_rows.append(row)
        if float(row["auc"]) > best_gate_auc:
            best_gate_auc = float(row["auc"])
            best_gate_pred = gated
            best_gate_config = dict(cfg)
    gate_df = pd.DataFrame(gate_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    gate_df.insert(0, "rank", np.arange(1, len(gate_df) + 1))
    gate_df.to_csv(output_dir / "file_gate_results.csv", index=False)

    top_gate_names = []
    for idx, row in gate_df.head(6).iterrows():
        name = f"file_gate_rank{int(row['rank'])}"
        cfg = {
            "gate_type": row["gate_type"],
            "method": row["method"],
            "topk": int(row["topk"]),
            "gamma": float(row["gamma"]) if not pd.isna(row["gamma"]) else np.nan,
            "strength": float(row["strength"]) if not pd.isna(row["strength"]) else np.nan,
            "beta": float(row["beta"]) if not pd.isna(row["beta"]) else np.nan,
        }
        gate_predictions[name] = apply_file_gate(mixed, filename=filename, cfg=cfg)
        top_gate_names.append(name)
    gate_predictions["best_file_gate"] = best_gate_pred

    print("[INFO] Step 2: label-family gating...")
    # Keep a small candidate set.  This is intentionally coarse and less
    # overfit-prone than per-class weights.
    candidates = {
        "baseline": baseline,
        "best_file_gate": best_gate_pred,
        "branch_logit_topk2": gate_predictions["branch_logit_topk2"],
        "branch_rank_topk2": gate_predictions["branch_rank_topk2"],
    }
    for name in top_gate_names[:3]:
        candidates[name] = gate_predictions[name]

    family_frames = []
    best_family_pred = baseline
    best_family_auc = base_auc
    best_family_config: dict[str, object] = {}
    for mode in ["family3", "family4", "family_plus_rare"]:
        df, pred, config = family_select_grid(
            candidates=candidates,
            class_names=class_names,
            y_true=y_true,
            base_pred=baseline,
            base_auc=base_auc,
            base_class_scores=base_class_scores,
            class_indices=class_indices,
            mode=mode,
        )
        df.to_csv(output_dir / f"label_family_{mode}_results.csv", index=False)
        family_frames.append(df)
        if float(df.iloc[0]["auc"]) > best_family_auc:
            best_family_auc = float(df.iloc[0]["auc"])
            best_family_pred = pred
            best_family_config = {"family_mode": mode, **config}
    family_df = pd.concat(family_frames, ignore_index=True).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    family_df.insert(0, "overall_rank", np.arange(1, len(family_df) + 1))
    family_df.to_csv(output_dir / "label_family_all_results.csv", index=False)

    final_candidates = {
        "baseline_mixed_topk2": baseline,
        "best_file_gate": best_gate_pred,
        "best_label_family": best_family_pred,
    }
    final_rows = [
        eval_prediction(name, y_true, pred, base_auc, base_class_scores, class_indices)
        for name, pred in final_candidates.items()
    ]
    final_df = pd.DataFrame(final_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_df.to_csv(output_dir / "final_candidate_scores.csv", index=False)
    best_name = str(final_df.iloc[0]["name"])
    best_pred = final_candidates[best_name]
    best_auc_final, scored_indices, best_class_scores = macro_auc(y_true, best_pred)
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
    class_delta_df.to_csv(output_dir / "best_class_deltas.csv", index=False)

    np.savez_compressed(
        output_dir / "gating_oof_predictions.npz",
        row_id=row_id,
        filename=filename,
        y_true=y_true.astype(np.uint8),
        mixed=mixed.astype(np.float32),
        baseline=baseline.astype(np.float32),
        best_file_gate=best_gate_pred.astype(np.float32),
        best_label_family=best_family_pred.astype(np.float32),
        best=best_pred.astype(np.float32),
    )

    summary = {
        "leakage_policy": (
            "File-level gates use only predictions inside the same soundscape file. "
            "Label-family gates choose among coarse global candidates using saved OOF labels; "
            "no hidden-test predictions or public LB feedback are used."
        ),
        "input_npz": str(Path(args.input_npz)),
        "rows": int(len(row_id)),
        "files": int(len(pd.unique(filename))),
        "classes": int(len(class_names)),
        "scored_classes": int(len(class_indices)),
        "weights": DEFAULT_WEIGHTS,
        "alpha_logit": float(args.alpha_logit),
        "baseline_auc": float(base_auc),
        "best_file_gate_auc": float(best_gate_auc),
        "best_file_gate_config": best_gate_config,
        "best_label_family_auc": float(best_family_auc),
        "best_label_family_config": best_family_config,
        "best_final_name": best_name,
        "best_final_auc_recomputed": float(best_auc_final),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[INFO] Baselines")
    print(baseline_df.to_string(index=False))
    print("[INFO] Top file gates")
    print(gate_df.head(20).to_string(index=False))
    print("[INFO] Top label-family gates")
    print(family_df.head(20).to_string(index=False))
    print("[INFO] Final candidates")
    print(final_df.to_string(index=False))
    print("[INFO] Best class deltas")
    print(class_delta_df.head(12).to_string(index=False))
    print("[INFO] Worst class deltas")
    print(class_delta_df.tail(12).to_string(index=False))
    print(f"[INFO] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
