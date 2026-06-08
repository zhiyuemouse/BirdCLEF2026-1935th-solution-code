#!/usr/bin/env python3
"""Evaluate shifted-window CNN TTA with fold-safe local OOF predictions."""

from __future__ import annotations

import argparse
import ast
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import birdclef2026_gm_kaggle_infer as base_infer
import birdclef2026_gm_kaggle_infer_stage3 as stage3_infer
from birdclef2026_perch_context_train import load_class_names, macro_auc_skip_empty


EPS = 1e-6


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN shifted-window TTA on local soundscape OOF folds.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=(
            "outputs/birdclef2026_gm_stage3_perchcnn_white_v1/"
            "20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo"
        ),
    )
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="",
        help=(
            "CSV with row_id/fold/target_* columns. Defaults to model_root/soundscape_segments_with_folds.csv "
            "when present, otherwise the CNN 20260505_195634 fold assignment."
        ),
    )
    parser.add_argument("--offsets", type=str, default="0,-1,1", help="Comma-separated clip offsets in seconds.")
    parser.add_argument("--segment-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/cnn_shift_tta_eval")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("At least one offset is required.")
    return values


def offset_name(offset: float) -> str:
    if abs(offset) < 1e-8:
        return "exact"
    sign = "p" if offset > 0 else "m"
    value = abs(float(offset))
    if abs(value - round(value)) < 1e-8:
        value_text = str(int(round(value)))
    else:
        value_text = f"{value:g}".replace(".", "p")
    return f"{sign}{value_text}"


def default_fold_assignment_path(model_root: Path) -> Path:
    own_path = model_root / "soundscape_segments_with_folds.csv"
    if own_path.exists():
        return own_path
    return Path("outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv")


def load_segment_table(path: Path, class_names: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"row_id", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Fold assignment file misses required columns: {sorted(missing)}")

    target_cols = [f"target_{name}" for name in class_names]
    if not all(col in df.columns for col in target_cols):
        y = np.zeros((len(df), len(class_names)), dtype=np.float32)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        if "label_indices" in df.columns:
            for row_idx, value in enumerate(df["label_indices"]):
                indices = ast.literal_eval(str(value))
                y[row_idx, np.asarray(indices, dtype=np.int64)] = 1.0
        elif "labels" in df.columns:
            for row_idx, value in enumerate(df["labels"]):
                labels = ast.literal_eval(str(value))
                for label in labels:
                    if label in class_to_idx:
                        y[row_idx, class_to_idx[label]] = 1.0
        else:
            raise KeyError("Need target_* columns, label_indices, or labels to build y_true.")
        target_df = pd.DataFrame(y, columns=target_cols, index=df.index)
        df = pd.concat([df, target_df], axis=1).copy()

    if "filename" not in df.columns:
        df["filename"] = df["row_id"].astype(str).str.rsplit("_", n=1).str[0] + ".ogg"
    if "audio_path" not in df.columns:
        df["audio_path"] = ""
    return df


def checkpoint_paths_by_fold(spec: stage3_infer.ResolvedModelSpec) -> Dict[int, Path]:
    if spec.run_kind == "stage3":
        paths = sorted(spec.model_root.glob("fold_*/stage3_best.pth"))
    else:
        paths = sorted(spec.model_root.glob("fold_*/stage2_fold*_best.pth"))
    out: Dict[int, Path] = {}
    for path in paths:
        match = re.search(r"fold_(\d+)", str(path))
        if not match:
            raise ValueError(f"Could not parse fold from checkpoint path: {path}")
        fold = int(match.group(1))
        out[fold] = path
    if not out:
        raise FileNotFoundError(f"No fold checkpoints found under {spec.model_root}")
    return out


def load_one_fold_model(
    checkpoint_path: Path,
    spec: stage3_infer.ResolvedModelSpec,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = base_infer.BirdCLEFNet(
        model_name=spec.model_name,
        num_classes=num_classes,
        dropout=spec.dropout,
        drop_path=spec.drop_path,
        head_type=spec.head_type,
    )
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = base_infer.extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def resolve_audio_path(group: pd.DataFrame, soundscapes_dir: Path) -> Path:
    audio_path = str(group["audio_path"].iloc[0]).strip()
    if audio_path and Path(audio_path).exists():
        return Path(audio_path)
    return soundscapes_dir / str(group["filename"].iloc[0])


def predict_offset_oof(
    segment_df: pd.DataFrame,
    class_names: Sequence[str],
    spec: stage3_infer.ResolvedModelSpec,
    checkpoint_paths: Dict[int, Path],
    soundscapes_dir: Path,
    offset: float,
    renderer: base_infer.SpectrogramRenderer,
    device: torch.device,
    segment_batch_size: int,
) -> np.ndarray:
    row_to_idx = {row_id: idx for idx, row_id in enumerate(segment_df["row_id"].astype(str).tolist())}
    pred = np.full((len(segment_df), len(class_names)), np.nan, dtype=np.float32)
    for fold in sorted(checkpoint_paths):
        fold_df = segment_df[segment_df["fold"].astype(int) == int(fold)]
        if fold_df.empty:
            continue
        model = load_one_fold_model(
            checkpoint_path=checkpoint_paths[fold],
            spec=spec,
            num_classes=len(class_names),
            device=device,
        )
        print(f"[INFO] Loaded fold_{fold} checkpoint for offset={offset:g}: {checkpoint_paths[fold]}", flush=True)
        file_groups = list(fold_df.groupby("filename", sort=True))
        progress = tqdm(
            file_groups,
            total=len(file_groups),
            desc=f"CNN offset {offset_name(offset)} fold{fold}",
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
        )
        for _, group in progress:
            audio_path = resolve_audio_path(group=group, soundscapes_dir=soundscapes_dir)
            audio = base_infer.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
            segments, row_ids = base_infer.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=spec.sample_rate,
                clip_seconds=spec.clip_seconds,
                clip_offset_seconds=float(offset),
            )
            file_pred = base_infer.predict_file_segments(
                segments=segments,
                models=[model],
                renderer=renderer,
                device=device,
                segment_batch_size=segment_batch_size,
            )
            for row_id, row_pred in zip(row_ids, file_pred):
                idx = row_to_idx.get(str(row_id))
                if idx is not None:
                    pred[idx] = row_pred.astype(np.float32, copy=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    missing = np.isnan(pred).any(axis=1)
    if missing.any():
        examples = segment_df.loc[missing, "row_id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for {int(missing.sum())} rows. Examples: {examples}")
    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)


def logit_np(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(prob / (1.0 - prob)).astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def main() -> None:
    args = parse_args()
    base_infer.seed_everything(args.seed)
    model_root = Path(args.model_root)
    fold_assignment_path = Path(args.fold_assignment_path) if args.fold_assignment_path else default_fold_assignment_path(model_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.sample_submission_path))
    segment_df = load_segment_table(fold_assignment_path, class_names=class_names)
    target_cols = [f"target_{name}" for name in class_names]
    y_true = segment_df[target_cols].to_numpy(dtype=np.float32)
    offsets = parse_float_list(args.offsets)
    offset_names = [offset_name(offset) for offset in offsets]
    if len(set(offset_names)) != len(offset_names):
        raise ValueError(f"Duplicate offset names from offsets {offsets}: {offset_names}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    spec = stage3_infer.resolve_model_spec(model_root)
    checkpoint_paths = checkpoint_paths_by_fold(spec)
    renderer = base_infer.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
        spectrogram_variant=spec.spectrogram_variant,
    )

    print("[INFO] CNN shifted-window TTA local OOF evaluation")
    print(f"[INFO] model_root: {model_root}")
    print(f"[INFO] fold_assignment_path: {fold_assignment_path}")
    print(f"[INFO] soundscapes_dir: {args.soundscapes_dir}")
    print(f"[INFO] run_kind: {spec.run_kind}")
    print(f"[INFO] config_source: {spec.config_source}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] rows: {len(segment_df)} | folds: {sorted(segment_df['fold'].astype(int).unique().tolist())}")
    print(f"[INFO] offsets: {dict(zip(offset_names, offsets))}")

    preds: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    for name, offset in zip(offset_names, offsets):
        pred = predict_offset_oof(
            segment_df=segment_df,
            class_names=class_names,
            spec=spec,
            checkpoint_paths=checkpoint_paths,
            soundscapes_dir=Path(args.soundscapes_dir),
            offset=offset,
            renderer=renderer,
            device=device,
            segment_batch_size=args.segment_batch_size,
        )
        preds[name] = pred
        auc = float(macro_auc_skip_empty(y_true, pred))
        rows.append({"name": name, "kind": "single", "auc": auc})
        print(f"[INFO] {name}: auc={auc:.9f}", flush=True)

    names = list(preds)
    for combo_size in range(2, len(names) + 1):
        for combo_names in itertools.combinations(names, combo_size):
            combo_name = "_".join(combo_names)
            stack = np.stack([preds[name] for name in combo_names], axis=0)
            prob_mean = stack.mean(axis=0)
            logit_mean = sigmoid_np(np.stack([logit_np(preds[name]) for name in combo_names], axis=0).mean(axis=0))
            prob_auc = float(macro_auc_skip_empty(y_true, prob_mean))
            logit_auc = float(macro_auc_skip_empty(y_true, logit_mean))
            rows.append({"name": combo_name, "kind": "prob_mean", "auc": prob_auc})
            rows.append({"name": combo_name, "kind": "logit_mean", "auc": logit_auc})
            print(f"[INFO] {combo_name}: prob_auc={prob_auc:.9f} logit_auc={logit_auc:.9f}", flush=True)

    result_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    result_df.to_csv(output_dir / "shift_tta_results.csv", index=False)
    np.savez_compressed(output_dir / "shift_tta_predictions.npz", row_ids=segment_df["row_id"].astype(str).to_numpy(), **preds)
    print("[INFO] Results:")
    print(result_df.to_string(index=False))
    print(f"[INFO] Saved results to {output_dir}")


if __name__ == "__main__":
    main()
