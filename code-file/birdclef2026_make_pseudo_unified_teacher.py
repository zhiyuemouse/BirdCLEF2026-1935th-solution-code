#!/usr/bin/env python3
"""Generate soft pseudo labels from the current unified five-branch teacher.

Leakage policy:
- Labeled soundscape files from train_soundscapes_labels.csv are excluded by default.
- The saved pseudo package is intended for train-only use. Validation folds remain
  the original labeled soundscape rows.
- Probabilities are saved as full soft vectors by default; filtering only decides
  which rows are kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import birdclef2026_kaggle_infer_unified_perch_stage3 as unified
import birdclef2026_perch_kaggle_infer_spatial_mamba as spatial_infer


DEFAULT_OUTPUT_DIR = "outputs/pseudo_labels"
DEFAULT_OUTPUT_NAME = "unified_teacher_0921_softpseudo"
DEFAULT_PERCH_LR = "outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib"
DEFAULT_MAMBA = (
    "outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1/"
    "perch_spatial_mamba_artifacts.joblib"
)
DEFAULT_ATTENTION = (
    "outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1/"
    "perch_spatial_mamba_artifacts.joblib"
)
DEFAULT_STAGE3 = "outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo"
DEFAULT_RAW_WAVE = (
    "outputs/birdclef2026_raw_waveform_transformer_strict_teacher/"
    "20260514_164133_raw_wave_conv_tokenizer_base_strictteacher_w100"
)


def install_numpy_core_compat() -> None:
    """Let numpy-1.x environments read numpy-2 pickled object arrays."""

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.numerictypes", np.core.numerictypes)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified-teacher soft pseudo labels for BirdCLEF 2026.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--taxonomy-path", type=str, default="input/taxonomy.csv")
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--perch-onnx-path", type=str, default="PerchV2Onnx/perch_v2.onnx")
    parser.add_argument("--perch-lr-model-path", type=str, default=DEFAULT_PERCH_LR)
    parser.add_argument("--mamba-model-path", type=str, default=DEFAULT_MAMBA)
    parser.add_argument("--attention-model-path", type=str, default=DEFAULT_ATTENTION)
    parser.add_argument("--stage3-model-root", type=str, default=DEFAULT_STAGE3)
    parser.add_argument("--raw-wave-model-root", type=str, default=DEFAULT_RAW_WAVE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--stage3-segment-batch-size", type=int, default=12)
    parser.add_argument("--raw-wave-segment-batch-size", type=int, default=12)
    parser.add_argument("--proxy-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--perch-lr-weight", type=float, default=0.2275)
    parser.add_argument("--mamba-weight", type=float, default=0.273)
    parser.add_argument("--stage3-weight", type=float, default=0.1365)
    parser.add_argument("--attention-weight", type=float, default=0.273)
    parser.add_argument("--raw-wave-weight", type=float, default=0.09)
    parser.add_argument("--file-scale-topk", type=int, default=2)
    parser.add_argument("--disable-file-scale", action="store_true")
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--max-files", type=int, default=1024)
    parser.add_argument("--select-mode", choices=["random", "sorted"], default="random")
    parser.add_argument("--row-min-max-prob", type=float, default=0.70)
    parser.add_argument("--min-top1-top2-margin", type=float, default=0.0)
    parser.add_argument("--max-topk-entropy", type=float, default=-1.0)
    parser.add_argument("--entropy-top-k", type=int, default=5)
    parser.add_argument("--zero-out-nontopk", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else root / path


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seconds_to_clock(seconds: int) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def extract_site(filename: str) -> str:
    for part in str(filename).split("_"):
        if part.startswith("S") and part[1:].isdigit():
            return part
    return "unknown"


def list_target_files(
    soundscapes_dir: Path,
    labels_path: Path,
    include_labeled: bool,
    max_files: int,
    select_mode: str,
    seed: int,
) -> Tuple[List[Path], set[str]]:
    labeled_df = pd.read_csv(labels_path)
    labeled_files = set(labeled_df["filename"].astype(str).unique().tolist())
    files = sorted(soundscapes_dir.glob("*.ogg"))
    if not include_labeled:
        files = [path for path in files if path.name not in labeled_files]
    if max_files > 0 and len(files) > max_files:
        if select_mode == "random":
            rng = random.Random(seed)
            files = sorted(rng.sample(files, k=int(max_files)))
        else:
            files = files[: int(max_files)]
    return files, labeled_files


def normalized_topk_entropy(probs: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 1:
        return np.zeros(len(probs), dtype=np.float32)
    k = max(1, min(int(top_k), probs.shape[1]))
    top = np.partition(probs, kth=probs.shape[1] - k, axis=1)[:, -k:].astype(np.float32, copy=False)
    q = top / (top.sum(axis=1, keepdims=True) + 1e-8)
    return (-(q * np.log(q + 1e-8)).sum(axis=1) / np.log(k)).astype(np.float32, copy=False)


def build_keep_metadata(
    meta_df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    row_min_max_prob: float,
    min_top1_top2_margin: float,
    max_topk_entropy: float,
    entropy_top_k: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    row_max = probs.max(axis=1)
    top1_idx = probs.argmax(axis=1)
    top2_values = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2:]
    top2 = top2_values.min(axis=1).astype(np.float32, copy=False)
    margin = (row_max - top2).astype(np.float32, copy=False)
    entropy = normalized_topk_entropy(probs, top_k=entropy_top_k)

    keep = row_max >= float(row_min_max_prob)
    if min_top1_top2_margin > 0:
        keep &= margin >= float(min_top1_top2_margin)
    if max_topk_entropy >= 0:
        keep &= entropy <= float(max_topk_entropy)

    rows = []
    meta_kept = meta_df.loc[keep].reset_index(drop=True)
    kept_indices = np.flatnonzero(keep)
    for out_idx, src_idx in enumerate(kept_indices):
        row = meta_kept.iloc[out_idx]
        row_id = str(row["row_id"])
        end_sec = int(row_id.rsplit("_", 1)[1])
        start_sec = end_sec - 5
        filename = str(row["filename"])
        top1 = int(top1_idx[src_idx])
        rows.append(
            {
                "row_id": row_id,
                "filename": filename,
                "site": extract_site(filename),
                "start": seconds_to_clock(start_sec),
                "end": seconds_to_clock(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "teacher_fold": -1,
                "max_prob": float(row_max[src_idx]),
                "top1_label": class_names[top1],
                "top1_prob": float(row_max[src_idx]),
                "top2_prob": float(top2[src_idx]),
                "top1_top2_margin": float(margin[src_idx]),
                "topk_entropy": float(entropy[src_idx]),
            }
        )
    return pd.DataFrame(rows), probs[keep].astype(np.float32, copy=False), keep


def maybe_zero_out_nontopk(probs: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= probs.shape[1]:
        return probs.astype(np.float32, copy=False)
    top_indices = np.argpartition(-probs, kth=top_k - 1, axis=1)[:, :top_k]
    out = np.zeros_like(probs, dtype=np.float32)
    out[np.arange(len(probs))[:, None], top_indices] = probs[np.arange(len(probs))[:, None], top_indices]
    return out


def run_teacher(args: argparse.Namespace, soundscape_files: Sequence[Path]):
    root = Path(args.root).resolve()
    input_dir = resolve_path(root, args.input_dir)
    sample_submission_path = resolve_path(root, args.sample_submission_path)
    taxonomy_path = resolve_path(root, args.taxonomy_path)
    perch_dir = resolve_path(root, args.perch_dir)
    onnx_path = resolve_path(root, args.perch_onnx_path)
    perch_lr_path = resolve_path(root, args.perch_lr_model_path)
    mamba_path = resolve_path(root, args.mamba_model_path)
    attention_path = resolve_path(root, args.attention_model_path)
    stage3_model_root = resolve_path(root, args.stage3_model_root)
    raw_wave_model_root = resolve_path(root, args.raw_wave_model_root) if args.raw_wave_model_root else None

    class_names = spatial_infer.load_class_names(sample_submission_path)
    perch_lr_artifact = unified.load_artifact(perch_lr_path, class_names=class_names, label="Perch LR")
    mamba_artifact = unified.load_artifact(mamba_path, class_names=class_names, label="Mamba")
    attention_artifact = unified.load_artifact(attention_path, class_names=class_names, label="Attention")
    freq_pools = unified.required_freq_pools(mamba_artifact, attention_artifact)
    weights = unified.normalize_weights(
        {
            "perch_lr": args.perch_lr_weight,
            "mamba": args.mamba_weight,
            "stage3": args.stage3_weight,
            "attention": args.attention_weight,
            "raw_wave": args.raw_wave_weight if raw_wave_model_root is not None else 0.0,
        }
    )

    bc_labels = spatial_infer.load_perch_label_table(perch_dir=perch_dir, onnx_path=onnx_path)
    bc_indices, mapped_bc_indices, mapping = spatial_infer.build_competition_mapping(
        primary_labels=class_names,
        taxonomy_path=taxonomy_path,
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = spatial_infer.build_selected_proxy_targets(
        primary_labels=class_names,
        mapping=mapping,
        bc_labels=bc_labels,
    )

    print("[INFO] Unified teacher pseudo generation")
    print(f"[INFO] files: {len(soundscape_files)}")
    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] perch_onnx_path: {onnx_path}")
    print(f"[INFO] perch_lr_path: {perch_lr_path}")
    print(f"[INFO] mamba_path: {mamba_path}")
    print(f"[INFO] attention_path: {attention_path}")
    print(f"[INFO] stage3_model_root: {stage3_model_root}")
    print(f"[INFO] raw_wave_model_root: {raw_wave_model_root}")
    print(f"[INFO] weights: {weights}")
    print(f"[INFO] file_scale_topk: {0 if args.disable_file_scale else args.file_scale_topk}")

    meta_df, spatial_tokens_by_pool, raw_scores, embedding = unified.infer_perch_shared_onnx(
        paths=soundscape_files,
        onnx_path=onnx_path,
        n_classes=len(class_names),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
        proxy_reduce=args.proxy_reduce,
        num_threads=args.runtime_num_threads,
        batch_files=args.batch_files,
        freq_pools=freq_pools,
    )
    row_ids = meta_df["row_id"].to_numpy()
    filenames = meta_df["filename"].to_numpy()

    perch_lr_pred = unified.predict_context_logreg_ensemble(
        artifact=perch_lr_artifact,
        meta_df=meta_df,
        scores_full_raw=raw_scores,
        emb_full=embedding,
    )
    mamba_pool = str(mamba_artifact.get("config", {}).get("freq_pool", "mean"))
    attention_pool = str(attention_artifact.get("config", {}).get("freq_pool", "mean"))
    mamba_pred = spatial_infer.predict_ensemble(
        artifact=mamba_artifact,
        spatial_tokens=spatial_tokens_by_pool[mamba_pool],
        raw_scores=raw_scores,
        embedding=embedding,
        batch_size=args.batch_files * spatial_infer.N_WINDOWS,
    )
    attention_pred = spatial_infer.predict_ensemble(
        artifact=attention_artifact,
        spatial_tokens=spatial_tokens_by_pool[attention_pool],
        raw_scores=raw_scores,
        embedding=embedding,
        batch_size=args.batch_files * spatial_infer.N_WINDOWS,
    )
    stage3_row_ids, stage3_pred_raw = unified.predict_stage3_cnn(
        model_root=stage3_model_root,
        soundscape_files=soundscape_files,
        class_names=class_names,
        competition_root=input_dir,
        debug=False,
        debug_limit=0,
        segment_batch_size=args.stage3_segment_batch_size,
        seed=args.seed,
    )
    stage3_pred = unified.align_prediction_by_row_id(
        source_row_ids=stage3_row_ids,
        source_pred=stage3_pred_raw,
        target_row_ids=row_ids,
        class_names=class_names,
        label="Stage3 CNN",
    )

    raw_wave_pred = None
    if raw_wave_model_root is not None:
        raw_wave_row_ids, raw_wave_pred_raw = unified.predict_raw_wave(
            model_root=raw_wave_model_root,
            soundscape_files=soundscape_files,
            class_names=class_names,
            segment_batch_size=args.raw_wave_segment_batch_size,
        )
        raw_wave_pred = unified.align_prediction_by_row_id(
            source_row_ids=raw_wave_row_ids,
            source_pred=raw_wave_pred_raw,
            target_row_ids=row_ids,
            class_names=class_names,
            label="Raw waveform",
        )

    fused_logit = (
        weights["perch_lr"] * unified.logit_np(perch_lr_pred)
        + weights["mamba"] * unified.logit_np(mamba_pred)
        + weights["stage3"] * unified.logit_np(stage3_pred)
        + weights["attention"] * unified.logit_np(attention_pred)
    )
    if raw_wave_pred is not None and weights.get("raw_wave", 0.0) > 0:
        fused_logit = fused_logit + weights["raw_wave"] * unified.logit_np(raw_wave_pred)
    fused = unified.sigmoid_np(fused_logit).astype(np.float32, copy=False)
    if not args.disable_file_scale:
        fused = unified.file_level_topk_mean_scale(fused, filename=filenames, topk=args.file_scale_topk)
    return meta_df, fused.astype(np.float32, copy=False), class_names, weights


def main() -> None:
    install_numpy_core_compat()
    args = parse_args()
    total_start = time.perf_counter()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, int(args.runtime_num_threads)))

    root = Path(args.root).resolve()
    soundscapes_dir = resolve_path(root, args.soundscapes_dir)
    labels_path = resolve_path(root, args.labels_path)
    output_root = resolve_path(root, args.output_dir)
    run_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.output_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    soundscape_files, labeled_files = list_target_files(
        soundscapes_dir=soundscapes_dir,
        labels_path=labels_path,
        include_labeled=args.include_labeled,
        max_files=args.max_files,
        select_mode=args.select_mode,
        seed=args.seed,
    )
    if not soundscape_files:
        raise FileNotFoundError(f"No target .ogg files found under {soundscapes_dir}")

    print(f"[INFO] Output dir: {run_dir}")
    print(f"[INFO] Soundscapes dir: {soundscapes_dir}")
    print(f"[INFO] Target files: {len(soundscape_files)}")
    print(f"[INFO] Include labeled: {args.include_labeled}")
    print(f"[INFO] Labeled files excluded: {0 if args.include_labeled else len(labeled_files)}")
    meta_df, probs, class_names, weights = run_teacher(args, soundscape_files=soundscape_files)
    probs = maybe_zero_out_nontopk(probs, top_k=args.zero_out_nontopk)

    pseudo_df, pseudo_probs, keep_mask = build_keep_metadata(
        meta_df=meta_df,
        probs=probs,
        class_names=class_names,
        row_min_max_prob=args.row_min_max_prob,
        min_top1_top2_margin=args.min_top1_top2_margin,
        max_topk_entropy=args.max_topk_entropy,
        entropy_top_k=args.entropy_top_k,
    )
    pseudo_df.to_csv(run_dir / "pseudo_segments.csv", index=False)
    np.save(run_dir / "pseudo_probs.npy", pseudo_probs.astype(np.float16))
    pd.DataFrame({"filename": [path.name for path in soundscape_files]}).to_csv(run_dir / "source_files.csv", index=False)
    save_json(run_dir / "config.json", vars(args))
    save_json(
        run_dir / "summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "teacher": "unified_perch_lr_mamba_stage3_attention_raw_wave",
            "weights": weights,
            "n_labeled_files": len(labeled_files),
            "include_labeled": bool(args.include_labeled),
            "leakage_policy": (
                "By default, files present in train_soundscapes_labels.csv are excluded. "
                "This package should be used only as train-only pseudo data; validation remains labeled folds."
            ),
            "n_source_files": int(len(soundscape_files)),
            "n_total_rows": int(len(meta_df)),
            "n_kept_rows": int(len(pseudo_df)),
            "keep_rate": float(len(pseudo_df) / max(len(meta_df), 1)),
            "row_min_max_prob": float(args.row_min_max_prob),
            "min_top1_top2_margin": float(args.min_top1_top2_margin),
            "max_topk_entropy": float(args.max_topk_entropy),
            "entropy_top_k": int(args.entropy_top_k),
            "zero_out_nontopk": int(args.zero_out_nontopk),
            "elapsed_sec": float(time.perf_counter() - total_start),
        },
    )
    print(f"[INFO] Saved pseudo package: {run_dir}")
    print(f"[INFO] kept_rows={len(pseudo_df)} / total_rows={len(meta_df)}")
    print(f"[INFO] elapsed={time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
