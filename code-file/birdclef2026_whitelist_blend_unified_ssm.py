#!/usr/bin/env python3
"""Leak-safe OOF grid for adding Perch sequence SSM to the current ensemble."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import birdclef2026_whitelist_blend_unified_raw_waveform as base


DEFAULT_5WAY_WEIGHTS = {
    "perch_lr": 0.2275,
    "mamba": 0.273,
    "stage3": 0.1365,
    "attention": 0.273,
    "raw_wave": 0.09,
}

DEFAULT_6WAY_TEMPORAL_WEIGHTS = {
    "perch_lr": 0.2275,
    "mamba": 0.273,
    "stage3": 0.1365,
    "attention": 0.193,
    "raw_wave": 0.09,
    "temporal": 0.08,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF blend grid for unified ensemble plus Perch sequence SSM.")
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
            "outputs/birdclef2026_raw_waveform_transformer_strict_teacher/"
            "20260514_164133_raw_wave_conv_tokenizer_base_strictteacher_w100/"
            "soundscape_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--temporal-oof-dir",
        type=str,
        default="outputs/perch_temporal_head_flat64_nolocal_drop035_labeled_all_cnn195634folds_v1",
    )
    parser.add_argument(
        "--ssm-oof-dir",
        type=str,
        default="outputs/perch_sequence_ssm_d192_l2_crossattn_cnn195634folds_v1",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/whitelist_blend_unified_ssm_20260515")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--file-scale-topk", type=int, default=2)
    return parser.parse_args()


def iter_transfer_grid(base_weights: dict[str, float], ssm_max: float) -> Iterable[dict[str, float]]:
    transfer_sources = [
        ("attention",),
        ("mamba",),
        ("temporal",),
        ("mamba", "attention"),
        ("perch_lr",),
        ("raw_wave",),
        ("stage3",),
        ("attention", "temporal"),
    ]
    for ssm_weight in np.arange(0.0, ssm_max + 0.0001, 0.0025):
        ssm_weight = float(ssm_weight)
        for sources in transfer_sources:
            weights = dict(base_weights)
            weights["ssm"] = ssm_weight
            delta = ssm_weight / len(sources)
            valid = True
            for source in sources:
                if source not in weights:
                    valid = False
                    break
                weights[source] = weights[source] - delta
                if weights[source] < -1e-9:
                    valid = False
            if valid:
                yield {key: round(float(value), 6) for key, value in weights.items()}


def eval_grid(
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    filename: np.ndarray,
    class_names: list[str],
    weights_base: dict[str, float],
    file_scale_topk: int,
    label: str,
    ssm_max: float,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, float, np.ndarray, np.ndarray]:
    base_raw = base.blend_logit(preds, weights_base)
    base_pred = base.file_level_topk_mean_scale(base_raw, filename=filename, topk=file_scale_topk)
    base_auc, class_indices, base_class_scores = base.macro_auc_and_class_scores(y_true, base_pred)
    best_pred = base_pred
    best_auc = float(base_auc)
    best_config: dict[str, object] = {
        **weights_base,
        "ssm": 0.0,
        "file_scale_topk": int(file_scale_topk),
        "name": label,
        "auc": float(base_auc),
        "scored_classes": int(len(class_indices)),
        "delta_vs_base": 0.0,
    }
    rows = []
    for weights in iter_transfer_grid(weights_base, ssm_max=ssm_max):
        pred = base.blend_logit(preds, weights)
        pred = base.file_level_topk_mean_scale(pred, filename=filename, topk=file_scale_topk)
        result = base.eval_prediction(f"{label}_ssm_transfer", y_true, pred, base_class_scores, base_auc, class_indices)
        row = {**weights, "file_scale_topk": int(file_scale_topk), **asdict(result)}
        rows.append(row)
        if result.auc > best_auc:
            best_auc = float(result.auc)
            best_config = row
            best_pred = pred
    grid_df = pd.DataFrame(rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grid_df.insert(0, "rank", np.arange(1, len(grid_df) + 1))
    return grid_df, best_config, best_pred, float(base_auc), class_indices, base_class_scores


def main() -> None:
    base.install_numpy_core_compat()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = base.load_class_names(Path(args.sample_submission_path))
    y_true, perch_lr, row_id, filename = base.reconstruct_perch_lr_online_like_oof(
        Path(args.perch_lr_dir),
        n_folds=args.n_folds,
    )
    preds = {
        "perch_lr": perch_lr,
        "mamba": base.load_npz_oof(Path(args.mamba_dir), row_id=row_id, y_true=y_true, name="mamba"),
        "attention": base.load_npz_oof(Path(args.attention_dir), row_id=row_id, y_true=y_true, name="attention"),
        "stage3": base.load_csv_oof(Path(args.stage3_oof_path), class_names, row_id=row_id, y_true=y_true, name="stage3"),
        "raw_wave": base.load_csv_oof(Path(args.raw_wave_oof_path), class_names, row_id=row_id, y_true=y_true, name="raw_wave"),
        "temporal": base.load_npz_oof(Path(args.temporal_oof_dir), row_id=row_id, y_true=y_true, name="temporal"),
        "ssm": base.load_npz_oof(Path(args.ssm_oof_dir), row_id=row_id, y_true=y_true, name="ssm"),
    }

    baseline_rows = []
    current_5way = base.file_level_topk_mean_scale(
        base.blend_logit(preds, DEFAULT_5WAY_WEIGHTS),
        filename=filename,
        topk=args.file_scale_topk,
    )
    current_5way_auc, class_indices, current_5way_class_scores = base.macro_auc_and_class_scores(y_true, current_5way)
    for name, pred in preds.items():
        scaled = base.file_level_topk_mean_scale(pred, filename=filename, topk=args.file_scale_topk)
        baseline_rows.append(base.eval_prediction(name, y_true, scaled, current_5way_class_scores, current_5way_auc, class_indices))
    baseline_rows.append(base.eval_prediction("current_5way", y_true, current_5way, current_5way_class_scores, current_5way_auc, class_indices))
    baseline_df = pd.DataFrame([asdict(row) for row in baseline_rows])

    grid5, best5, pred5, auc5, _, _ = eval_grid(
        preds=preds,
        y_true=y_true,
        filename=filename,
        class_names=class_names,
        weights_base=DEFAULT_5WAY_WEIGHTS,
        file_scale_topk=args.file_scale_topk,
        label="current_5way",
        ssm_max=0.12,
    )
    grid6, best6, pred6, auc6, _, _ = eval_grid(
        preds=preds,
        y_true=y_true,
        filename=filename,
        class_names=class_names,
        weights_base=DEFAULT_6WAY_TEMPORAL_WEIGHTS,
        file_scale_topk=args.file_scale_topk,
        label="current_6way_temporal",
        ssm_max=0.12,
    )
    best_pred = pred5 if float(best5["auc"]) >= float(best6["auc"]) else pred6
    best_config = best5 if float(best5["auc"]) >= float(best6["auc"]) else best6
    best_auc_final, scored_indices, best_class_scores = base.macro_auc_and_class_scores(y_true, best_pred)
    class_delta_df = pd.DataFrame(
        {
            "class_index": scored_indices,
            "class_name": [class_names[int(idx)] for idx in scored_indices],
            "current_5way_auc": current_5way_class_scores,
            "best_auc": best_class_scores,
            "delta": best_class_scores - current_5way_class_scores,
            "positives": y_true[:, scored_indices].sum(axis=0).astype(int),
        }
    ).sort_values("delta", ascending=False)
    summary = {
        "inputs": {
            "perch_lr_dir": str(Path(args.perch_lr_dir)),
            "mamba_dir": str(Path(args.mamba_dir)),
            "attention_dir": str(Path(args.attention_dir)),
            "stage3_oof_path": str(Path(args.stage3_oof_path)),
            "raw_wave_oof_path": str(Path(args.raw_wave_oof_path)),
            "temporal_oof_dir": str(Path(args.temporal_oof_dir)),
            "ssm_oof_dir": str(Path(args.ssm_oof_dir)),
        },
        "leakage_policy": "Only saved OOF predictions and train labels are used. No hidden-test predictions or LB feedback.",
        "rows": int(len(row_id)),
        "files": int(len(pd.unique(filename))),
        "classes": int(len(class_names)),
        "scored_classes": int(len(scored_indices)),
        "current_5way_auc": float(auc5),
        "current_6way_temporal_auc": float(auc6),
        "best_auc_recomputed": float(best_auc_final),
        "best": best_config,
    }
    baseline_df.to_csv(output_dir / "baseline_scores.csv", index=False)
    grid5.to_csv(output_dir / "ssm_grid_from_5way.csv", index=False)
    grid6.to_csv(output_dir / "ssm_grid_from_6way_temporal.csv", index=False)
    class_delta_df.to_csv(output_dir / "best_class_deltas.csv", index=False)
    np.savez_compressed(
        output_dir / "best_oof_predictions.npz",
        row_id=row_id,
        filename=filename,
        y_true=y_true.astype(np.uint8),
        pred=best_pred.astype(np.float32),
        current_5way=current_5way.astype(np.float32),
        **{key: value.astype(np.float32) for key, value in preds.items()},
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[INFO] Baselines vs current 5-way")
    print(baseline_df.to_string(index=False))
    print("[INFO] Top SSM from 5-way")
    print(grid5.head(20).to_string(index=False))
    print("[INFO] Top SSM from 6-way temporal")
    print(grid6.head(20).to_string(index=False))
    print("[INFO] Best class deltas")
    print(class_delta_df.head(15).to_string(index=False))
    print("[INFO] Worst class deltas")
    print(class_delta_df.tail(15).to_string(index=False))
    print(f"[INFO] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
