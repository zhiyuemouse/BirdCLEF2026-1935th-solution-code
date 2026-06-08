#!/usr/bin/env python3
"""Local evaluation for the Kaggle-style BirdCLEF 2026 inference pipeline.

This script reuses the current ensemble inference utilities so that we can
evaluate the *same* inference-time logic locally on labeled train_soundscapes:

- model ensembling
- row-level TTA
- temporal smoothing
- soundscape-level top-k postprocess

Important caveat:
This is useful for debugging and tuning inference/postprocess settings, but it
is not an honest model-CV score if the evaluated soundscapes were seen during
training. Treat it as an inference-debug metric, not as a replacement for the
leakage-aware fold CV we use for model development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from birdclef2026_gm_kaggle_infer_ensemble import (
    aggregate_tta_predictions,
    apply_soundscape_postprocess,
    apply_temporal_smoothing,
    build_segments_for_file,
    discover_model_roots,
    list_soundscape_files,
    load_class_names,
    load_model_bundle,
    load_soundscape_audio,
    parse_float_list,
    parse_multi_string_args,
    predict_file_segments,
    resolve_user_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local evaluation for birdclef2026_gm Kaggle inference flow.")
    parser.add_argument("--competition-root", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_local_infer_eval")
    parser.add_argument(
        "--model-root",
        type=str,
        action="append",
        default=None,
        help="Model run directory. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="")
    parser.add_argument("--sample-submission-path", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument(
        "--tta-offsets",
        type=str,
        default="0",
        help="Comma-separated time shifts in seconds for row-level TTA, e.g. '0,-1.25,1.25'.",
    )
    parser.add_argument(
        "--smoothing-kernel",
        type=str,
        default="",
        help="Optional temporal smoothing kernel, e.g. '0.1,0.8,0.1'.",
    )
    parser.add_argument(
        "--soundscape-top-k",
        type=int,
        default=0,
        help="Optional soundscape-level scaling using the mean of top-k chunk probabilities per class.",
    )
    return parser.parse_args()


def parse_label_cell(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def union_labels(series: Iterable[object]) -> List[str]:
    merged = set()
    for value in series:
        merged.update(parse_label_cell(value))
    return sorted(merged)


def time_to_seconds(value: str) -> int:
    hours, minutes, seconds = str(value).split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos_mask = y_true > 0.5
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    n = len(sorted_scores)
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    pos_ranks = ranks[sorted_true > 0.5].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc_skip_missing(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    scores: List[float] = []
    for class_idx in range(y_true.shape[1]):
        auc = binary_auc(y_true[:, class_idx], y_pred[:, class_idx])
        if auc is not None:
            scores.append(auc)
    if not scores:
        raise ValueError("No positive classes in evaluation data; cannot compute macro ROC-AUC.")
    return float(np.mean(scores)), int(len(scores))


def resolve_labels_path(args: argparse.Namespace, competition_root: Path) -> Path:
    if args.labels_path:
        return resolve_user_path(args.labels_path, competition_root=competition_root)
    return competition_root / "train_soundscapes_labels.csv"


def resolve_sample_submission_path(args: argparse.Namespace, competition_root: Path) -> Path:
    if args.sample_submission_path:
        return resolve_user_path(args.sample_submission_path, competition_root=competition_root)
    return competition_root / "sample_submission.csv"


def build_solution_df(
    labels_path: Path,
    class_names: Sequence[str],
) -> pd.DataFrame:
    raw = pd.read_csv(labels_path)
    grouped = (
        raw.groupby(["filename", "start", "end"], as_index=False)["primary_label"]
        .apply(union_labels)
        .rename(columns={"primary_label": "label_list"})
    )
    grouped["end_sec"] = grouped["end"].map(time_to_seconds).astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_true = np.zeros((len(grouped), len(class_names)), dtype=np.uint8)
    for row_idx, labels in enumerate(grouped["label_list"]):
        for label in labels:
            class_idx = label_to_idx.get(label)
            if class_idx is not None:
                y_true[row_idx, class_idx] = 1

    solution_df = pd.DataFrame(y_true, columns=class_names)
    solution_df.insert(0, "row_id", grouped["row_id"].values)
    solution_df.insert(1, "filename", grouped["filename"].values)
    solution_df.insert(2, "end_sec", grouped["end_sec"].values)
    return solution_df


def select_eval_files(
    soundscapes_dir: Path,
    labeled_filenames: Sequence[str],
    debug: bool,
    debug_limit: int,
    limit_files: int,
) -> List[Path]:
    labeled_set = set(labeled_filenames)
    all_files = list_soundscape_files(soundscapes_dir, debug=False, debug_limit=debug_limit)
    eval_files = [path for path in all_files if path.name in labeled_set]
    if debug:
        eval_files = eval_files[:debug_limit]
    if limit_files > 0:
        eval_files = eval_files[:limit_files]
    return eval_files


def summarize_prediction_spread(pred_df: pd.DataFrame, class_names: Sequence[str]) -> Dict[str, float]:
    values = pred_df[list(class_names)].to_numpy(dtype=np.float32, copy=False)
    row_max = values.max(axis=1)
    row_mean = values.mean(axis=1)
    return {
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
        "prediction_mean": float(values.mean()),
        "row_max_mean": float(row_max.mean()),
        "row_max_median": float(np.median(row_max)),
        "row_mean_mean": float(row_mean.mean()),
    }


def main() -> None:
    args = parse_args()

    competition_root = Path(args.competition_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.soundscapes_dir:
        soundscapes_dir = resolve_user_path(args.soundscapes_dir, competition_root=competition_root)
    else:
        soundscapes_dir = competition_root / "train_soundscapes"

    labels_path = resolve_labels_path(args, competition_root=competition_root)
    sample_submission_path = resolve_sample_submission_path(args, competition_root=competition_root)

    class_names = load_class_names(sample_submission_path)
    solution_df = build_solution_df(labels_path=labels_path, class_names=class_names)
    labeled_filenames = solution_df["filename"].drop_duplicates().tolist()

    model_roots = discover_model_roots(parse_multi_string_args(args.model_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Using soundscapes dir: {soundscapes_dir}")
    print(f"[INFO] Labels path: {labels_path}")
    print(f"[INFO] TTA offsets (sec): {parse_float_list(args.tta_offsets, default=[0.0])}")
    print(f"[INFO] Smoothing kernel: {parse_float_list(args.smoothing_kernel, default=[] ) if str(args.smoothing_kernel).strip() else 'disabled'}")
    print(f"[INFO] Soundscape top-k postprocess: {args.soundscape_top_k}")

    bundles = [load_model_bundle(model_root=model_root, class_names=class_names, device=device) for model_root in model_roots]
    total_models = sum(len(bundle.models) for bundle in bundles)
    print(f"[INFO] Loaded {len(bundles)} model run(s), total fold models = {total_models}")

    eval_files = select_eval_files(
        soundscapes_dir=soundscapes_dir,
        labeled_filenames=labeled_filenames,
        debug=args.debug,
        debug_limit=args.debug_limit,
        limit_files=args.limit_files,
    )
    if not eval_files:
        raise FileNotFoundError(
            f"No labeled .ogg files found under {soundscapes_dir}. "
            f"Expected overlap with labels from {labels_path}."
        )

    solution_subset = solution_df[solution_df["filename"].isin([path.name for path in eval_files])].copy()
    print(f"[INFO] Labeled files available: {len(labeled_filenames)}")
    print(f"[INFO] Eval files selected: {len(eval_files)}")
    print(f"[INFO] Labeled rows selected: {len(solution_subset)}")

    tta_offsets = parse_float_list(args.tta_offsets, default=[0.0])
    smoothing_kernel = parse_float_list(args.smoothing_kernel, default=[])

    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(eval_files, total=len(eval_files), desc="Local infer eval", dynamic_ncols=True)
    for audio_path in progress:
        row_ids = None
        bundle_preds = []
        for bundle in bundles:
            audio = load_soundscape_audio(audio_path, sample_rate=bundle.sample_rate)
            segments, bundle_row_ids, row_indices = build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=bundle.sample_rate,
                clip_seconds=bundle.clip_seconds,
                tta_offsets=tta_offsets,
            )
            window_preds = predict_file_segments(
                segments=segments,
                models=bundle.models,
                renderer=bundle.renderer,
                device=device,
                segment_batch_size=args.segment_batch_size,
            )
            pred_matrix = aggregate_tta_predictions(window_preds, row_indices=row_indices, n_rows=len(bundle_row_ids))
            bundle_preds.append(pred_matrix)
            if row_ids is None:
                row_ids = bundle_row_ids
            elif row_ids != bundle_row_ids:
                raise ValueError("Row ids mismatch across model bundles.")

        preds = np.mean(np.stack(bundle_preds, axis=0), axis=0)
        preds = apply_temporal_smoothing(preds, smoothing_kernel)
        preds = apply_soundscape_postprocess(preds, args.soundscape_top_k)
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    prediction_matrix = np.concatenate(all_preds, axis=0)
    pred_df = pd.DataFrame(prediction_matrix, columns=class_names)
    pred_df.insert(0, "row_id", all_row_ids)

    common_row_ids = solution_subset["row_id"][solution_subset["row_id"].isin(pred_df["row_id"])].tolist()
    if not common_row_ids:
        raise ValueError("No common row_id found between local predictions and labels.")

    solution_aligned = solution_subset.set_index("row_id").loc[common_row_ids].reset_index()
    pred_aligned = pred_df.set_index("row_id").loc[common_row_ids].reset_index()

    y_true = solution_aligned[class_names].to_numpy(dtype=np.uint8, copy=False)
    y_pred = pred_aligned[class_names].to_numpy(dtype=np.float32, copy=False)
    macro_auc, scored_classes = macro_auc_skip_missing(y_true, y_pred)

    print(f"[INFO] Scored row_ids: {len(common_row_ids)}")
    print(f"[INFO] Scored classes: {scored_classes}")
    print(f"[INFO] Local macro ROC-AUC: {macro_auc:.6f}")

    submission_like_path = output_dir / "submission_like.csv"
    solution_like_path = output_dir / "solution_like.csv"
    summary_path = output_dir / "summary.json"

    pred_aligned.to_csv(submission_like_path, index=False)
    solution_aligned.to_csv(solution_like_path, index=False)

    summary = {
        "macro_auc": float(macro_auc),
        "scored_classes": int(scored_classes),
        "num_eval_files": int(len(eval_files)),
        "num_pred_rows_total": int(len(pred_df)),
        "num_scored_rows": int(len(common_row_ids)),
        "competition_root": str(competition_root),
        "soundscapes_dir": str(soundscapes_dir),
        "labels_path": str(labels_path),
        "model_roots": [str(path) for path in model_roots],
        "tta_offsets": tta_offsets,
        "smoothing_kernel": smoothing_kernel,
        "soundscape_top_k": int(args.soundscape_top_k),
        **summarize_prediction_spread(pred_aligned, class_names),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved aligned predictions to: {submission_like_path}")
    print(f"[INFO] Saved aligned solution to: {solution_like_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
