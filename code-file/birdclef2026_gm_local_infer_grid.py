#!/usr/bin/env python3
"""Grid search for local inference-time BirdCLEF 2026 postprocess settings.

This script is meant for *cheap local ranking* of inference-side tricks on
train_soundscapes labels:

- row-level TTA
- temporal smoothing
- soundscape-level top-k postprocess

It shares the same warning as `birdclef2026_gm_local_infer_eval.py`: this is
not an honest model-CV score if the soundscapes were seen during training.
Treat it as a local debugging/tuning signal only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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
from birdclef2026_gm_local_infer_eval import (
    build_solution_df,
    macro_auc_skip_missing,
    resolve_labels_path,
    resolve_sample_submission_path,
    select_eval_files,
    summarize_prediction_spread,
)


def parse_grid_strings(text: str, default_items: List[str]) -> List[str]:
    if text is None:
        return list(default_items)
    raw = str(text).strip()
    if not raw:
        return list(default_items)
    items = []
    seen = set()
    for part in raw.split(";"):
        item = part.strip()
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
    return items if items else list(default_items)


def parse_top_k_grid(text: str, default_items: List[int]) -> List[int]:
    if text is None:
        return list(default_items)
    raw = str(text).strip()
    if not raw:
        return list(default_items)
    values = []
    seen = set()
    for part in raw.split(","):
        item = int(part.strip())
        if item in seen:
            continue
        seen.add(item)
        values.append(item)
    return values if values else list(default_items)


def format_config_name(tta_text: str, smoothing_text: str, top_k: int) -> str:
    def normalize(text: str, empty_name: str) -> str:
        value = str(text).strip()
        if not value:
            return empty_name
        return value.replace(",", "_").replace(".", "p").replace("-", "m")

    return (
        f"tta_{normalize(tta_text, 'base')}"
        f"__smooth_{normalize(smoothing_text, 'none')}"
        f"__topk_{top_k}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local grid search for birdclef2026 inference-time settings.")
    parser.add_argument("--competition-root", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_local_infer_grid")
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
        "--tta-offsets-grid",
        type=str,
        default="0;0,-1.25,1.25;0,-1.0,1.0",
        help="Semicolon-separated list of TTA offset strings.",
    )
    parser.add_argument(
        "--smoothing-kernel-grid",
        type=str,
        default=";0.1,0.8,0.1;0.2,0.6,0.2",
        help="Semicolon-separated list of smoothing-kernel strings. Empty item means disabled.",
    )
    parser.add_argument(
        "--soundscape-top-k-grid",
        type=str,
        default="0,1,2",
        help="Comma-separated list of soundscape-top-k values.",
    )
    return parser.parse_args()


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
    model_roots = discover_model_roots(parse_multi_string_args(args.model_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = [load_model_bundle(model_root=model_root, class_names=class_names, device=device) for model_root in model_roots]
    total_models = sum(len(bundle.models) for bundle in bundles)

    tta_grid = parse_grid_strings(args.tta_offsets_grid, default_items=["0"])
    smoothing_grid = parse_grid_strings(args.smoothing_kernel_grid, default_items=[""])
    top_k_grid = parse_top_k_grid(args.soundscape_top_k_grid, default_items=[0])
    grid = [(tta_text, smoothing_text, top_k) for tta_text in tta_grid for smoothing_text in smoothing_grid for top_k in top_k_grid]

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Using soundscapes dir: {soundscapes_dir}")
    print(f"[INFO] Labels path: {labels_path}")
    print(f"[INFO] Eval files selected: {len(eval_files)}")
    print(f"[INFO] Loaded {len(bundles)} model run(s), total fold models = {total_models}")
    print(f"[INFO] Grid size: {len(grid)}")

    results = []
    for grid_idx, (tta_text, smoothing_text, top_k) in enumerate(grid, start=1):
        tta_offsets = parse_float_list(tta_text, default=[0.0])
        smoothing_kernel = parse_float_list(smoothing_text, default=[])
        print(
            f"[GRID] {grid_idx}/{len(grid)} | "
            f"tta_offsets={tta_text or '0'} | "
            f"smoothing={smoothing_text or 'disabled'} | "
            f"top_k={top_k}"
        )

        all_row_ids: List[str] = []
        all_preds: List[np.ndarray] = []
        progress = tqdm(eval_files, total=len(eval_files), desc=f"Grid {grid_idx}", dynamic_ncols=True, leave=False)
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
            preds = apply_soundscape_postprocess(preds, top_k)
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

        result = {
            "tta_offsets": tta_text or "0",
            "smoothing_kernel": smoothing_text,
            "soundscape_top_k": int(top_k),
            "macro_auc": float(macro_auc),
            "scored_classes": int(scored_classes),
            **summarize_prediction_spread(pred_aligned, class_names),
        }
        results.append(result)
        print(
            f"[RESULT] tta_offsets={result['tta_offsets']} "
            f"smoothing={smoothing_text or 'disabled'} "
            f"top_k={top_k} macro_auc={macro_auc:.6f}"
        )

        config_name = format_config_name(tta_text, smoothing_text, top_k)
        pred_aligned.to_csv(output_dir / f"{config_name}_submission_like.csv", index=False)

    results_df = pd.DataFrame(results).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "grid_results.csv", index=False)

    summary = {
        "competition_root": str(competition_root),
        "soundscapes_dir": str(soundscapes_dir),
        "labels_path": str(labels_path),
        "model_roots": [str(path) for path in model_roots],
        "num_eval_files": int(len(eval_files)),
        "grid_size": int(len(grid)),
        "best_config": results_df.iloc[0].to_dict() if not results_df.empty else {},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved grid results to: {output_dir / 'grid_results.csv'}")
    print(f"[INFO] Saved summary to: {output_dir / 'summary.json'}")
    if not results_df.empty:
        print("[INFO] Top configs:")
        print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
