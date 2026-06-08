#!/usr/bin/env python3
"""Final leak-safe OOF trick experiments for BirdCLEF 2026.

The script intentionally only reads saved out-of-fold predictions, train
labels, and train metadata.  It is meant for late-stage, low-risk validation:

1. Add Perch temporal / sequence SSM branches back into the current final
   ensemble and re-search small weights.
2. Compare logit blend, rank blend, and mixed blend on the exact same OOF set.
3. Try a light fold-safe site/hour prior.
4. Try very conservative class-group branch selection.

No hidden-test predictions or leaderboard feedback are used.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import birdclef2026_whitelist_blend_unified_raw_waveform as base


DEFAULT_FINAL_5WAY_WEIGHTS = {
    "perch_lr": 0.22625,
    "mamba": 0.2715,
    "stage3": 0.13575,
    "attention": 0.2715,
    "raw_wave": 0.095,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final OOF-only trick experiments.")
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
        default="outputs/cnn_shift_tta_sweep_stage3_white_20260526/stage3_tta_m0p5_p0p5_oof_predictions.csv",
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
        default="outputs/perch_sequence_ssm_protoclr_w005_d192_l2_crossattn_cnn195634folds_v1",
    )
    parser.add_argument(
        "--fold-metadata-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/final_oof_tricks_20260528")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--file-scale-topk", type=int, default=2)
    parser.add_argument("--branch-max-weight", type=float, default=0.12)
    parser.add_argument("--branch-step", type=float, default=0.0025)
    return parser.parse_args()


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Non-positive weights: {weights}")
    return {key: float(value) / total for key, value in weights.items() if abs(float(value)) > 1e-12}


def sorted_weight_json(weights: dict[str, float]) -> str:
    return json.dumps({k: round(float(weights[k]), 8) for k in sorted(weights)}, sort_keys=True)


def rank01_1d(values: np.ndarray) -> np.ndarray:
    """Average-rank transform to [0, 1] for one class."""

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
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return (ranks / float(n - 1)).astype(np.float32)


def rank01_matrix(pred: np.ndarray) -> np.ndarray:
    out = np.empty_like(pred, dtype=np.float32)
    for class_idx in range(pred.shape[1]):
        out[:, class_idx] = rank01_1d(pred[:, class_idx])
    return out


def blend_rank(preds_ranked: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    weights = normalize_weights(weights)
    fused = None
    for key, weight in weights.items():
        term = float(weight) * preds_ranked[key]
        fused = term if fused is None else fused + term
    return np.clip(fused.astype(np.float32, copy=False), 0.0, 1.0)


def blend_logit_unscaled(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return base.blend_logit(preds, weights)


def apply_file_scale(pred: np.ndarray, filename: np.ndarray, topk: int) -> np.ndarray:
    return base.file_level_topk_mean_scale(pred, filename=filename, topk=topk)


def eval_row(
    name: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    base_class_scores: np.ndarray,
    base_auc: float,
    class_indices: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result = base.eval_prediction(name, y_true, pred, base_class_scores, base_auc, class_indices)
    row = asdict(result)
    if extra:
        row.update(extra)
    return row


def iter_branch_weight_grid(
    base_weights: dict[str, float],
    branch_names: list[str],
    max_weight: float,
    step: float,
) -> Iterable[dict[str, float]]:
    """Small transfer-only grid; avoids broad public-LB-style weight fitting."""

    if max_weight < 0:
        raise ValueError("--branch-max-weight must be non-negative")
    if step <= 0:
        raise ValueError("--branch-step must be positive")

    transfer_modes = [
        ("attention",),
        ("mamba",),
        ("mamba", "attention"),
        ("raw_wave",),
        ("stage3",),
        ("perch_lr",),
        ("perch_lr", "mamba", "attention", "stage3", "raw_wave"),
    ]
    branch_allocations: list[dict[str, float]] = []
    if "temporal" in branch_names:
        branch_allocations.append({"temporal": 1.0})
    if "ssm" in branch_names:
        branch_allocations.append({"ssm": 1.0})
    if "temporal" in branch_names and "ssm" in branch_names:
        branch_allocations.extend(
            [
                {"temporal": 0.5, "ssm": 0.5},
                {"temporal": 0.25, "ssm": 0.75},
                {"temporal": 0.75, "ssm": 0.25},
            ]
        )

    yielded: set[str] = set()
    for total_branch_weight in np.arange(0.0, max_weight + step * 0.5, step):
        total_branch_weight = float(total_branch_weight)
        for allocation in branch_allocations:
            for sources in transfer_modes:
                weights = dict(base_weights)
                for branch, share in allocation.items():
                    weights[branch] = weights.get(branch, 0.0) + total_branch_weight * float(share)
                delta = total_branch_weight / float(len(sources))
                ok = True
                for source in sources:
                    weights[source] = weights[source] - delta
                    if weights[source] < -1e-10:
                        ok = False
                        break
                if not ok:
                    continue
                weights = {key: round(max(0.0, float(value)), 8) for key, value in weights.items()}
                key = sorted_weight_json(weights)
                if key in yielded:
                    continue
                yielded.add(key)
                yield weights


def parse_hour_from_filename(filename: str) -> int:
    match = re.search(r"_(\d{8})_(\d{6})", str(filename))
    if not match:
        return -1
    return int(match.group(2)[:2])


def load_metadata(row_id: np.ndarray, metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    required = {"row_id", "filename", "site", "fold"}
    missing = required - set(meta.columns)
    if missing:
        raise KeyError(f"Metadata missing columns: {sorted(missing)}")
    meta = meta.drop_duplicates(subset=["row_id"]).copy()
    meta["hour"] = meta["filename"].map(parse_hour_from_filename).astype(int)
    aligned = pd.DataFrame({"row_id": row_id}).merge(
        meta[["row_id", "filename", "site", "fold", "hour"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[["site", "fold", "hour"]].isna().any().any():
        missing_rows = aligned.loc[aligned["site"].isna(), "row_id"].astype(str).head(5).tolist()
        raise ValueError(f"Metadata missing aligned rows. Examples: {missing_rows}")
    aligned["fold"] = aligned["fold"].astype(int)
    aligned["hour"] = aligned["hour"].astype(int)
    return aligned


def smoothed_group_mean(
    y_train: np.ndarray,
    train_values: np.ndarray,
    valid_values: np.ndarray,
    global_prob: np.ndarray,
    strength: float,
) -> np.ndarray:
    out = np.tile(global_prob[None, :], (len(valid_values), 1)).astype(np.float32)
    for value in pd.Index(train_values).unique():
        train_mask = train_values == value
        valid_mask = valid_values == value
        if not np.any(valid_mask):
            continue
        n = int(train_mask.sum())
        if n <= 0:
            continue
        summed = y_train[train_mask].sum(axis=0).astype(np.float32)
        prior = (summed + float(strength) * global_prob) / (float(n) + float(strength))
        out[valid_mask] = prior.astype(np.float32)
    return out


def build_fold_safe_context_adjustments(
    y_true: np.ndarray,
    metadata: pd.DataFrame,
    site_strength: float = 24.0,
    hour_strength: float = 36.0,
) -> dict[str, np.ndarray]:
    folds = metadata["fold"].to_numpy(dtype=int)
    sites = metadata["site"].astype(str).to_numpy()
    hours = metadata["hour"].to_numpy(dtype=int)
    n, c = y_true.shape
    adjustments = {
        "site": np.zeros((n, c), dtype=np.float32),
        "hour": np.zeros((n, c), dtype=np.float32),
        "site_hour": np.zeros((n, c), dtype=np.float32),
        "site_hour_mean": np.zeros((n, c), dtype=np.float32),
    }

    site_hour = np.asarray([f"{site}_{hour:02d}" for site, hour in zip(sites, hours)], dtype=object)
    for fold in sorted(pd.Index(folds).unique().tolist()):
        valid_mask = folds == int(fold)
        train_mask = ~valid_mask
        y_train = y_true[train_mask].astype(np.float32)
        # Beta smoothing around the training-fold global rate keeps this light.
        global_prob = (y_train.sum(axis=0) + 1.0) / (float(y_train.shape[0]) + 2.0)
        global_logit = base.logit(global_prob)

        site_prior = smoothed_group_mean(
            y_train=y_train,
            train_values=sites[train_mask],
            valid_values=sites[valid_mask],
            global_prob=global_prob,
            strength=site_strength,
        )
        hour_prior = smoothed_group_mean(
            y_train=y_train,
            train_values=hours[train_mask],
            valid_values=hours[valid_mask],
            global_prob=global_prob,
            strength=hour_strength,
        )
        site_hour_prior = smoothed_group_mean(
            y_train=y_train,
            train_values=site_hour[train_mask],
            valid_values=site_hour[valid_mask],
            global_prob=global_prob,
            strength=site_strength + hour_strength,
        )
        mean_prior = np.clip(0.5 * site_prior + 0.5 * hour_prior, base.EPS, 1.0 - base.EPS)

        adjustments["site"][valid_mask] = base.logit(site_prior) - global_logit
        adjustments["hour"][valid_mask] = base.logit(hour_prior) - global_logit
        adjustments["site_hour"][valid_mask] = base.logit(site_hour_prior) - global_logit
        adjustments["site_hour_mean"][valid_mask] = base.logit(mean_prior) - global_logit
    return adjustments


def apply_context_prior(pred: np.ndarray, adjustment: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base.sigmoid(base.logit(pred) + float(alpha) * adjustment).astype(np.float32), 0.0, 1.0)


def positive_count_groups(y_true: np.ndarray) -> dict[str, np.ndarray]:
    counts = y_true.sum(axis=0)
    return {
        "rare_le2": counts <= 2,
        "mid_3_5": (counts >= 3) & (counts <= 5),
        "common_ge6": counts >= 6,
    }


def class_group_select(
    base_pred: np.ndarray,
    candidate_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    filename: np.ndarray,
    file_scale_topk: int,
    base_auc: float,
    base_class_scores: np.ndarray,
    class_indices: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Try coarse per-class-group candidate selection.

    This is deliberately conservative: groups are based only on positive count,
    and each group can choose from a small whitelist of already-tested global
    predictions.
    """

    groups = positive_count_groups(y_true)
    group_names = ["rare_le2", "mid_3_5", "common_ge6"]
    candidate_names = ["base", *[name for name in candidate_preds if name != "base"]]
    all_preds = {"base": base_pred, **candidate_preds}

    rows: list[dict[str, object]] = []
    best_pred = base_pred
    best_auc = base_auc
    best_config: dict[str, object] = {"rare_le2": "base", "mid_3_5": "base", "common_ge6": "base"}

    for rare_name in candidate_names:
        for mid_name in candidate_names:
            for common_name in candidate_names:
                pred = base_pred.copy()
                selection = {
                    "rare_le2": rare_name,
                    "mid_3_5": mid_name,
                    "common_ge6": common_name,
                }
                for group_name in group_names:
                    mask = groups[group_name]
                    pred[:, mask] = all_preds[selection[group_name]][:, mask]
                pred = apply_file_scale(pred, filename=filename, topk=file_scale_topk)
                row = eval_row(
                    "class_group_select",
                    y_true,
                    pred,
                    base_class_scores,
                    base_auc,
                    class_indices,
                    extra=selection,
                )
                rows.append(row)
                if float(row["auc"]) > best_auc:
                    best_auc = float(row["auc"])
                    best_pred = pred
                    best_config = selection

    df = pd.DataFrame(rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df, {"class_group": best_pred}, best_config


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
        "stage3": base.load_csv_oof(Path(args.stage3_oof_path), class_names, row_id=row_id, y_true=y_true, name="stage3"),
        "attention": base.load_npz_oof(Path(args.attention_dir), row_id=row_id, y_true=y_true, name="attention"),
        "raw_wave": base.load_csv_oof(Path(args.raw_wave_oof_path), class_names, row_id=row_id, y_true=y_true, name="raw_wave"),
        "temporal": base.load_npz_oof(Path(args.temporal_oof_dir), row_id=row_id, y_true=y_true, name="temporal"),
        "ssm": base.load_npz_oof(Path(args.ssm_oof_dir), row_id=row_id, y_true=y_true, name="ssm"),
    }

    base_weights = dict(DEFAULT_FINAL_5WAY_WEIGHTS)
    base_unscaled = blend_logit_unscaled(preds, base_weights)
    base_pred = apply_file_scale(base_unscaled, filename=filename, topk=args.file_scale_topk)
    base_auc, class_indices, base_class_scores = base.macro_auc_and_class_scores(y_true, base_pred)

    baseline_rows = []
    for name, pred in preds.items():
        scaled = apply_file_scale(pred, filename=filename, topk=args.file_scale_topk)
        baseline_rows.append(eval_row(name, y_true, scaled, base_class_scores, base_auc, class_indices))
    baseline_rows.append(eval_row("final_5way_stage3tta", y_true, base_pred, base_class_scores, base_auc, class_indices))
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(output_dir / "baseline_scores.csv", index=False)

    # 1. Temporal / SSM branch search with logit blend.
    branch_rows: list[dict[str, object]] = []
    best_branch_pred = base_pred
    best_branch_unscaled = base_unscaled
    best_branch_auc = base_auc
    best_branch_weights = dict(base_weights)
    print("[INFO] Step 1: Temporal/SSM transfer grid...")
    for weights in iter_branch_weight_grid(
        base_weights=base_weights,
        branch_names=["temporal", "ssm"],
        max_weight=float(args.branch_max_weight),
        step=float(args.branch_step),
    ):
        pred_unscaled = blend_logit_unscaled(preds, weights)
        pred = apply_file_scale(pred_unscaled, filename=filename, topk=args.file_scale_topk)
        row = eval_row(
            "branch_logit_grid",
            y_true,
            pred,
            base_class_scores,
            base_auc,
            class_indices,
            extra={**weights, "weights_json": sorted_weight_json(weights), "file_scale_topk": int(args.file_scale_topk)},
        )
        branch_rows.append(row)
        if float(row["auc"]) > best_branch_auc:
            best_branch_auc = float(row["auc"])
            best_branch_pred = pred
            best_branch_unscaled = pred_unscaled
            best_branch_weights = dict(weights)
    branch_df = pd.DataFrame(branch_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    branch_df.insert(0, "rank", np.arange(1, len(branch_df) + 1))
    branch_df.to_csv(output_dir / "branch_grid_results.csv", index=False)

    # 2. Rank blend vs logit blend vs mixed blend.
    print("[INFO] Step 2: rank/logit/mixed blend comparison...")
    ranked_preds = {key: rank01_matrix(value) for key, value in preds.items()}
    blend_rows: list[dict[str, object]] = []
    best_blend_pred = best_branch_pred
    best_blend_unscaled = best_branch_unscaled
    best_blend_auc = best_branch_auc
    best_blend_name = "best_branch_logit"
    best_rank_unscaled = blend_rank(ranked_preds, best_branch_weights)
    best_rank_pred = apply_file_scale(best_rank_unscaled, filename=filename, topk=args.file_scale_topk)
    for blend_name, pred_unscaled in [
        ("base_logit", base_unscaled),
        ("best_branch_logit", best_branch_unscaled),
        ("best_branch_rank", best_rank_unscaled),
    ]:
        pred = apply_file_scale(pred_unscaled, filename=filename, topk=args.file_scale_topk)
        row = eval_row(
            blend_name,
            y_true,
            pred,
            base_class_scores,
            base_auc,
            class_indices,
            extra={"alpha_logit": np.nan, "weights_json": sorted_weight_json(best_branch_weights)},
        )
        blend_rows.append(row)
        if float(row["auc"]) > best_blend_auc:
            best_blend_auc = float(row["auc"])
            best_blend_pred = pred
            best_blend_unscaled = pred_unscaled
            best_blend_name = blend_name

    for alpha in np.arange(0.0, 1.0001, 0.025):
        mixed_unscaled = np.clip(
            float(alpha) * best_branch_unscaled + (1.0 - float(alpha)) * best_rank_unscaled,
            0.0,
            1.0,
        ).astype(np.float32)
        mixed_pred = apply_file_scale(mixed_unscaled, filename=filename, topk=args.file_scale_topk)
        row = eval_row(
            "mixed_logit_rank",
            y_true,
            mixed_pred,
            base_class_scores,
            base_auc,
            class_indices,
            extra={"alpha_logit": round(float(alpha), 4), "weights_json": sorted_weight_json(best_branch_weights)},
        )
        blend_rows.append(row)
        if float(row["auc"]) > best_blend_auc:
            best_blend_auc = float(row["auc"])
            best_blend_pred = mixed_pred
            best_blend_unscaled = mixed_unscaled
            best_blend_name = f"mixed_logit_rank_alpha_{alpha:.3f}"
    blend_df = pd.DataFrame(blend_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    blend_df.insert(0, "rank", np.arange(1, len(blend_df) + 1))
    blend_df.to_csv(output_dir / "blend_type_results.csv", index=False)

    # 3. Fold-safe site/hour prior.
    print("[INFO] Step 3: fold-safe site/hour prior...")
    metadata = load_metadata(row_id=row_id, metadata_path=Path(args.fold_metadata_path))
    adjustments = build_fold_safe_context_adjustments(y_true=y_true, metadata=metadata)
    prior_rows: list[dict[str, object]] = []
    best_prior_pred = best_blend_pred
    best_prior_auc = best_blend_auc
    best_prior_name = "none"
    for prior_name, adjustment in adjustments.items():
        for alpha in [0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.125, 0.15]:
            pred_unscaled = apply_context_prior(best_blend_unscaled, adjustment=adjustment, alpha=float(alpha))
            pred = apply_file_scale(pred_unscaled, filename=filename, topk=args.file_scale_topk)
            row = eval_row(
                "fold_safe_context_prior",
                y_true,
                pred,
                base_class_scores,
                base_auc,
                class_indices,
                extra={"prior": prior_name, "alpha": float(alpha), "source_blend": best_blend_name},
            )
            prior_rows.append(row)
            if float(row["auc"]) > best_prior_auc:
                best_prior_auc = float(row["auc"])
                best_prior_pred = pred
                best_prior_name = f"{prior_name}_alpha_{alpha:g}"
    prior_df = pd.DataFrame(prior_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    prior_df.insert(0, "rank", np.arange(1, len(prior_df) + 1))
    prior_df.to_csv(output_dir / "prior_results.csv", index=False)

    # 4. Conservative class-group candidate selection.
    print("[INFO] Step 4: class-group candidate selection...")
    candidate_preds = {
        "branch_logit": best_branch_unscaled,
        "rank": best_rank_unscaled,
        "mixed_or_best_blend": best_blend_unscaled,
    }
    class_group_df, class_group_best, class_group_config = class_group_select(
        base_pred=base_unscaled,
        candidate_preds=candidate_preds,
        y_true=y_true,
        filename=filename,
        file_scale_topk=int(args.file_scale_topk),
        base_auc=base_auc,
        base_class_scores=base_class_scores,
        class_indices=class_indices,
    )
    class_group_df.to_csv(output_dir / "class_group_results.csv", index=False)
    best_group_pred = class_group_best["class_group"]
    best_group_auc = float(class_group_df.iloc[0]["auc"]) if len(class_group_df) else base_auc

    final_candidates = {
        "base_5way": base_pred,
        "branch_logit": best_branch_pred,
        "blend_type": best_blend_pred,
        "prior": best_prior_pred,
        "class_group": best_group_pred,
    }
    final_rows = []
    for name, pred in final_candidates.items():
        final_rows.append(eval_row(name, y_true, pred, base_class_scores, base_auc, class_indices))
    final_df = pd.DataFrame(final_rows).sort_values(
        ["auc", "median_class_delta", "n_improved_classes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_df.to_csv(output_dir / "final_candidate_scores.csv", index=False)

    best_name = str(final_df.iloc[0]["name"])
    best_pred = final_candidates[best_name]
    best_auc_final, scored_indices, best_class_scores = base.macro_auc_and_class_scores(y_true, best_pred)
    class_delta_df = pd.DataFrame(
        {
            "class_index": scored_indices,
            "class_name": [class_names[int(idx)] for idx in scored_indices],
            "base_auc": base_class_scores,
            "best_auc": best_class_scores,
            "delta": best_class_scores - base_class_scores,
            "positives": y_true[:, scored_indices].sum(axis=0).astype(int),
        }
    ).sort_values("delta", ascending=False)
    class_delta_df.to_csv(output_dir / "best_class_deltas.csv", index=False)

    np.savez_compressed(
        output_dir / "final_oof_predictions.npz",
        row_id=row_id,
        filename=filename,
        y_true=y_true.astype(np.uint8),
        base_5way=base_pred.astype(np.float32),
        branch_logit=best_branch_pred.astype(np.float32),
        blend_type=best_blend_pred.astype(np.float32),
        prior=best_prior_pred.astype(np.float32),
        class_group=best_group_pred.astype(np.float32),
        best=best_pred.astype(np.float32),
        **{f"model_{key}": value.astype(np.float32) for key, value in preds.items()},
    )

    summary = {
        "leakage_policy": (
            "All experiments use saved OOF predictions aligned by row_id. "
            "The site/hour prior is fold-safe: each validation fold only sees label priors "
            "estimated from the other folds. No hidden-test predictions or leaderboard feedback are used."
        ),
        "inputs": {
            "sample_submission_path": str(Path(args.sample_submission_path)),
            "perch_lr_dir": str(Path(args.perch_lr_dir)),
            "mamba_dir": str(Path(args.mamba_dir)),
            "attention_dir": str(Path(args.attention_dir)),
            "stage3_oof_path": str(Path(args.stage3_oof_path)),
            "raw_wave_oof_path": str(Path(args.raw_wave_oof_path)),
            "temporal_oof_dir": str(Path(args.temporal_oof_dir)),
            "ssm_oof_dir": str(Path(args.ssm_oof_dir)),
            "fold_metadata_path": str(Path(args.fold_metadata_path)),
        },
        "rows": int(len(row_id)),
        "files": int(len(pd.unique(filename))),
        "classes": int(len(class_names)),
        "scored_classes": int(len(class_indices)),
        "file_scale_topk": int(args.file_scale_topk),
        "base_5way_weights": base_weights,
        "base_5way_auc": float(base_auc),
        "best_branch_auc": float(best_branch_auc),
        "best_branch_weights": best_branch_weights,
        "best_blend_auc": float(best_blend_auc),
        "best_blend_name": best_blend_name,
        "best_prior_auc": float(best_prior_auc),
        "best_prior_name": best_prior_name,
        "best_class_group_auc": float(best_group_auc),
        "best_class_group_config": class_group_config,
        "best_final_name": best_name,
        "best_final_auc_recomputed": float(best_auc_final),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[INFO] Baselines")
    print(baseline_df.to_string(index=False))
    print("[INFO] Top branch grid")
    print(branch_df.head(20).to_string(index=False))
    print("[INFO] Top blend type")
    print(blend_df.head(20).to_string(index=False))
    print("[INFO] Top prior")
    print(prior_df.head(20).to_string(index=False))
    print("[INFO] Top class-group")
    print(class_group_df.head(20).to_string(index=False))
    print("[INFO] Final candidates")
    print(final_df.to_string(index=False))
    print("[INFO] Best class deltas")
    print(class_delta_df.head(12).to_string(index=False))
    print("[INFO] Worst class deltas")
    print(class_delta_df.tail(12).to_string(index=False))
    print(f"[INFO] Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
