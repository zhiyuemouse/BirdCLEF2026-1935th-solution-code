#!/usr/bin/env python3
"""Generate fold-specific pseudo labels from Perch LogReg + CNN blend.

Leakage policy:
- Pseudo labels are generated only for unlabeled soundscape files by default.
- In fold-specific mode, fold `k` uses only CNN checkpoint `fold_k` and Perch
  artifact `fold_k`; it does not average teacher folds.
- Blend and post-processing defaults are frozen from the leak-safe OOF grid in
  `outputs/whitelist_blend_cnn195634_perch_logreg_v1`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

warnings.filterwarnings("ignore", message="Failed to load image Python extension:.*")

from birdclef2026_blend_submissions_postprocess import file_level_scale, logit, sigmoid, temporal_smooth


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()


class TeeStream:
    """Mirror output to console and log, compacting tqdm carriage returns."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.text_buffer = ""
        self.progress_buffer = None

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())

    def fileno(self):
        return self.console_stream.fileno()

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.console_stream.write(text)
        self._write_to_log(text)
        return len(text)

    def flush(self) -> None:
        self.console_stream.flush()
        self.log_stream.flush()

    def flush_pending(self) -> None:
        if self.progress_buffer:
            self.log_stream.write(self.progress_buffer.rstrip() + "\n")
            self.progress_buffer = None
        elif self.text_buffer:
            self.log_stream.write(self.text_buffer)
            self.text_buffer = ""
        self.log_stream.flush()

    def close(self) -> None:
        # Some libraries (notably absl logging) call close() on sys.stderr at exit.
        self.flush_pending()

    def _write_to_log(self, text: str) -> None:
        for char in text:
            if char == "\r":
                if self.text_buffer:
                    self.log_stream.write(self.text_buffer)
                    self.text_buffer = ""
                self.progress_buffer = ""
            elif char == "\n":
                if self.progress_buffer is not None:
                    self.log_stream.write(self.progress_buffer.rstrip())
                    self.progress_buffer = None
                else:
                    self.log_stream.write(self.text_buffer)
                    self.text_buffer = ""
                self.log_stream.write("\n")
                self.log_stream.flush()
            else:
                if self.progress_buffer is not None:
                    self.progress_buffer += char
                else:
                    self.text_buffer += char


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._stdout = None
        self._stderr = None
        self._fp = None
        self._stdout_proxy = None
        self._stderr_proxy = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._fp = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._stdout_proxy = TeeStream(self._stdout, self._fp)
        self._stderr_proxy = TeeStream(self._stderr, self._fp)
        sys.stdout = self._stdout_proxy
        sys.stderr = self._stderr_proxy
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stdout_proxy is not None:
            self._stdout_proxy.flush_pending()
        if self._stderr_proxy is not None:
            self._stderr_proxy.flush_pending()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self._fp is not None:
            self._fp.close()


@dataclass
class PseudoBlendConfig:
    mode: str = "both"
    root: str = "."
    input_dir: str = "input"
    output_dir: str = "outputs/pseudo_labels"
    output_name: str = "perch_cnn_blend_white_v1"
    perch_cache_dir: str = ""
    soundscapes_dir: str = "input/train_soundscapes"
    labels_csv: str = "input/train_soundscapes_labels.csv"
    cnn_model_root: str = "outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k"
    perch_model_path: str = "outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib"
    perch_dir: str = "Perch"
    perch_backend: str = "auto"
    perch_onnx_path: str = ""
    perch_tflite_path: str = ""
    runtime_num_threads: int = 4
    batch_files: int = 32
    segment_batch_size: int = 12
    proxy_reduce: str = "max"
    pseudo_scope: str = "fold-specific"
    teacher_folds: Optional[List[int]] = None
    max_perch_folds: int = 3
    include_labeled: bool = False
    debug: bool = False
    debug_limit: int = 16
    perch_weight: float = 0.83
    file_scale_mode: str = "topk_mean"
    file_scale_value: float = 2.0
    smooth_mode: str = "adaptive"
    smooth_alpha: float = 0.10
    prob_threshold: float = 0.35
    row_min_max_prob: float = 0.85
    top_k_labels: int = 2
    min_top1_top2_margin: float = 0.0
    max_topk_entropy: float = -1.0
    entropy_top_k: int = 5
    seed: int = 2026


def parse_int_list(text: str) -> Optional[List[int]]:
    text = str(text).strip()
    if not text:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values if values else None


def import_cnn_modules():
    import birdclef2026_gm_kaggle_infer_ensemble as cnn_lib
    import birdclef2026_gm_train as base_train

    return cnn_lib, base_train, base_train.torch


def import_perch_modules():
    from birdclef2026_perch_kaggle_infer_context_logreg import (
        build_context_tensor,
        build_metadata_features,
        build_position_features,
        infer_perch_auto_backend,
        parse_end_seconds,
        predict_context_artifact,
    )
    from birdclef2026_run_perch_local import (
        build_competition_mapping,
        build_selected_proxy_targets,
        load_class_names,
        load_perch_label_table,
    )

    return {
        "build_competition_mapping": build_competition_mapping,
        "build_context_tensor": build_context_tensor,
        "build_metadata_features": build_metadata_features,
        "build_position_features": build_position_features,
        "build_selected_proxy_targets": build_selected_proxy_targets,
        "infer_perch_auto_backend": infer_perch_auto_backend,
        "load_class_names": load_class_names,
        "load_perch_label_table": load_perch_label_table,
        "parse_end_seconds": parse_end_seconds,
        "predict_context_artifact": predict_context_artifact,
    }


def parse_args() -> PseudoBlendConfig:
    parser = argparse.ArgumentParser(description="Generate Perch+CNN blend pseudo labels.")
    parser.add_argument("--mode", choices=["both", "perch-cache", "pseudo-from-cache"], default="both")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/pseudo_labels")
    parser.add_argument("--output-name", type=str, default="perch_cnn_blend_white_v1")
    parser.add_argument("--perch-cache-dir", type=str, default="")
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument("--labels-csv", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--cnn-model-root", type=str, default=PseudoBlendConfig.cnn_model_root)
    parser.add_argument("--perch-model-path", type=str, default=PseudoBlendConfig.perch_model_path)
    parser.add_argument("--perch-dir", type=str, default="Perch")
    parser.add_argument("--perch-backend", choices=["auto", "saved_model", "onnx", "tflite"], default="auto")
    parser.add_argument("--perch-onnx-path", type=str, default="")
    parser.add_argument("--perch-tflite-path", type=str, default="")
    parser.add_argument("--runtime-num-threads", type=int, default=4)
    parser.add_argument("--batch-files", type=int, default=32)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument("--proxy-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--pseudo-scope", choices=["fold-specific", "global"], default="fold-specific")
    parser.add_argument("--teacher-folds", type=str, default="")
    parser.add_argument("--max-perch-folds", type=int, default=3)
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=16)
    parser.add_argument("--perch-weight", type=float, default=0.83)
    parser.add_argument("--file-scale-mode", choices=["none", "topk_mean", "max_power"], default="topk_mean")
    parser.add_argument("--file-scale-value", type=float, default=2.0)
    parser.add_argument("--smooth-mode", choices=["none", "plain", "adaptive"], default="adaptive")
    parser.add_argument("--smooth-alpha", type=float, default=0.10)
    parser.add_argument("--prob-threshold", type=float, default=0.35)
    parser.add_argument("--row-min-max-prob", type=float, default=0.85)
    parser.add_argument("--top-k-labels", type=int, default=2)
    parser.add_argument("--min-top1-top2-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-topk-entropy",
        type=float,
        default=-1.0,
        help="Optional normalized entropy upper bound over top-k probabilities. Negative disables it.",
    )
    parser.add_argument("--entropy-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return PseudoBlendConfig(
        mode=args.mode,
        root=args.root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        perch_cache_dir=args.perch_cache_dir,
        soundscapes_dir=args.soundscapes_dir,
        labels_csv=args.labels_csv,
        cnn_model_root=args.cnn_model_root,
        perch_model_path=args.perch_model_path,
        perch_dir=args.perch_dir,
        perch_backend=args.perch_backend,
        perch_onnx_path=args.perch_onnx_path,
        perch_tflite_path=args.perch_tflite_path,
        runtime_num_threads=args.runtime_num_threads,
        batch_files=args.batch_files,
        segment_batch_size=args.segment_batch_size,
        proxy_reduce=args.proxy_reduce,
        pseudo_scope=args.pseudo_scope,
        teacher_folds=parse_int_list(args.teacher_folds),
        max_perch_folds=args.max_perch_folds,
        include_labeled=args.include_labeled,
        debug=args.debug,
        debug_limit=args.debug_limit,
        perch_weight=args.perch_weight,
        file_scale_mode=args.file_scale_mode,
        file_scale_value=args.file_scale_value,
        smooth_mode=args.smooth_mode,
        smooth_alpha=args.smooth_alpha,
        prob_threshold=args.prob_threshold,
        row_min_max_prob=args.row_min_max_prob,
        top_k_labels=args.top_k_labels,
        min_top1_top2_margin=args.min_top1_top2_margin,
        max_topk_entropy=args.max_topk_entropy,
        entropy_top_k=args.entropy_top_k,
        seed=args.seed,
    )


def seed_everything(seed: int, torch_module=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def resolve_path(root: Path, path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seconds_to_clock(seconds: int) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def extract_site(filename: str) -> str:
    parts = filename.split("_")
    for part in parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    return ""


def list_target_files(soundscapes_dir: Path, labels_csv: Path, include_labeled: bool, debug: bool, debug_limit: int) -> Tuple[List[Path], set[str]]:
    labeled_df = pd.read_csv(labels_csv)
    labeled_filenames = set(labeled_df["filename"].astype(str).unique().tolist())
    files = sorted(path for path in soundscapes_dir.glob("*.ogg"))
    if not include_labeled:
        files = [path for path in files if path.name not in labeled_filenames]
    if debug:
        files = files[:debug_limit]
    return files, labeled_filenames


def discover_available_cnn_folds(model_root: Path) -> List[int]:
    folds: List[int] = []
    for fold_dir in sorted(model_root.glob("fold_*")):
        try:
            fold = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (fold_dir / f"stage2_fold{fold}_best.pth").exists() or (fold_dir / "stage3_best.pth").exists():
            folds.append(fold)
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints found under {model_root}")
    return folds


def load_cnn_fold_model(cnn_lib, torch_module, model_root: Path, fold: int, class_names: Sequence[str], device):
    spec = cnn_lib.resolve_model_spec(model_root)
    renderer = cnn_lib.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )
    checkpoint_name = "stage3_best.pth" if spec.run_kind == "stage3" else f"stage2_fold{fold}_best.pth"
    checkpoint_path = model_root / f"fold_{fold}" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CNN fold checkpoint: {checkpoint_path}")
    model = cnn_lib.BirdCLEFNet(
        model_name=spec.model_name,
        num_classes=len(class_names),
        dropout=spec.dropout,
        drop_path=spec.drop_path,
        head_type=spec.head_type,
    )
    checkpoint_obj = torch_module.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(cnn_lib.extract_state_dict(checkpoint_obj), strict=True)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded CNN fold {fold}: {checkpoint_path}")
    return spec, renderer, model


def predict_cnn_files(
    cnn_lib,
    paths: Sequence[Path],
    spec,
    renderer,
    model,
    device,
    segment_batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    all_row_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    progress = tqdm(
        paths,
        total=len(paths),
        desc="CNN fold inference",
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for audio_path in progress:
        audio = cnn_lib.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
        segments, row_ids, row_indices = cnn_lib.build_segments_for_file(
            audio=audio,
            file_stem=audio_path.stem,
            sample_rate=spec.sample_rate,
            clip_seconds=spec.clip_seconds,
            tta_offsets=[0.0],
        )
        window_preds = cnn_lib.predict_file_segments(
            segments=segments,
            models=[model],
            renderer=renderer,
            device=device,
            segment_batch_size=segment_batch_size,
        )
        pred_matrix = cnn_lib.aggregate_tta_predictions(window_preds, row_indices=row_indices, n_rows=len(row_ids))
        all_row_ids.extend(row_ids)
        all_preds.append(pred_matrix.astype(np.float32, copy=False))
    return all_row_ids, np.concatenate(all_preds, axis=0)


def make_perch_args(cfg: PseudoBlendConfig):
    return argparse.Namespace(
        perch_backend=cfg.perch_backend,
        perch_onnx_path=cfg.perch_onnx_path,
        perch_tflite_path=cfg.perch_tflite_path,
        proxy_reduce=cfg.proxy_reduce,
        runtime_num_threads=cfg.runtime_num_threads,
        batch_files=cfg.batch_files,
    )


def run_perch_for_paths(
    cfg: PseudoBlendConfig,
    root: Path,
    input_dir: Path,
    perch_dir: Path,
    paths: Sequence[Path],
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    perch = import_perch_modules()
    bc_labels = perch["load_perch_label_table"](perch_dir)
    bc_indices, mapped_bc_indices, mapping = perch["build_competition_mapping"](
        primary_labels=list(class_names),
        taxonomy_path=input_dir / "taxonomy.csv",
        bc_labels=bc_labels,
    )
    mapped_pos = np.where(bc_indices != len(bc_labels))[0].astype(np.int32)
    proxy_pos_to_bc = perch["build_selected_proxy_targets"](primary_labels=list(class_names), mapping=mapping, bc_labels=bc_labels)
    return perch["infer_perch_auto_backend"](
        args=make_perch_args(cfg),
        competition_root=root,
        perch_dir=perch_dir,
        soundscape_files=paths,
        n_classes=len(class_names),
        mapped_pos=mapped_pos,
        mapped_bc_indices=mapped_bc_indices,
        proxy_pos_to_bc=proxy_pos_to_bc,
    )


def predict_perch_fold(
    fold_artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    raw_scores: np.ndarray,
    emb: np.ndarray,
    artifact_config: Dict[str, object],
) -> np.ndarray:
    perch = import_perch_modules()
    position_features = perch["build_position_features"](perch["parse_end_seconds"](meta_df["row_id"].tolist()))
    metadata_features = perch["build_metadata_features"](
        meta_df=meta_df,
        include_hour_features=bool(artifact_config.get("include_hour_features", False)),
    )
    context = perch["build_context_tensor"](meta_df=meta_df, scores_full_raw=raw_scores)
    return perch["predict_context_artifact"](
        fold_artifact=fold_artifact,
        emb=emb,
        raw_scores=raw_scores,
        context=context,
        position_features=position_features,
        metadata_features=metadata_features,
    )


def normalized_topk_entropy(raw: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 1:
        return np.zeros(raw.shape[0], dtype=np.float32)
    k = max(1, min(int(top_k), raw.shape[1]))
    top = np.partition(raw, kth=raw.shape[1] - k, axis=1)[:, -k:].astype(np.float32, copy=False)
    q = top / (top.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -(q * np.log(q + 1e-8)).sum(axis=1) / np.log(k)
    return entropy.astype(np.float32, copy=False)


def filter_prediction_matrix(
    pred_matrix: np.ndarray,
    prob_threshold: float,
    row_min_max_prob: float,
    top_k_labels: int,
    min_top1_top2_margin: float,
    max_topk_entropy: float,
    entropy_top_k: int,
):
    raw = np.asarray(pred_matrix, dtype=np.float32)
    filtered = raw.copy()
    if top_k_labels > 0 and top_k_labels < filtered.shape[1]:
        top_indices = np.argpartition(-filtered, kth=top_k_labels - 1, axis=1)[:, :top_k_labels]
        top_mask = np.zeros_like(filtered, dtype=bool)
        top_mask[np.arange(filtered.shape[0])[:, None], top_indices] = True
        filtered = np.where(top_mask, filtered, 0.0)
    filtered = np.where(filtered >= prob_threshold, filtered, 0.0).astype(np.float32)
    raw_max = raw.max(axis=1)
    raw_top1_idx = raw.argmax(axis=1)
    top2_values = np.partition(raw, kth=raw.shape[1] - 2, axis=1)[:, -2:]
    raw_top2 = top2_values.min(axis=1).astype(np.float32, copy=False)
    top1_top2_margin = (raw_max - raw_top2).astype(np.float32, copy=False)
    topk_entropy = normalized_topk_entropy(raw, top_k=entropy_top_k)
    positive_counts = (filtered > 0).sum(axis=1)
    keep_mask = (raw_max >= row_min_max_prob) & (positive_counts > 0)
    if min_top1_top2_margin > 0:
        keep_mask &= top1_top2_margin >= float(min_top1_top2_margin)
    if max_topk_entropy >= 0:
        keep_mask &= topk_entropy <= float(max_topk_entropy)
    return filtered, keep_mask, raw_max, raw_top1_idx, positive_counts, raw_top2, top1_top2_margin, topk_entropy


def build_metadata_rows(
    row_ids: Sequence[str],
    filenames: Sequence[str],
    pred_matrix: np.ndarray,
    keep_mask: np.ndarray,
    raw_max: np.ndarray,
    raw_top1_idx: np.ndarray,
    positive_counts: np.ndarray,
    raw_top2: np.ndarray,
    top1_top2_margin: np.ndarray,
    topk_entropy: np.ndarray,
    class_names: Sequence[str],
    teacher_fold: Optional[int],
) -> List[dict]:
    rows: List[dict] = []
    for row_idx, row_id in enumerate(row_ids):
        if not keep_mask[row_idx]:
            continue
        end_sec = int(str(row_id).rsplit("_", 1)[1])
        start_sec = end_sec - 5
        filename = str(filenames[row_idx])
        positive_indices = np.flatnonzero(pred_matrix[row_idx] > 0).tolist()
        positive_labels = [class_names[idx] for idx in positive_indices]
        top1_idx = int(raw_top1_idx[row_idx])
        rows.append(
            {
                "row_id": row_id,
                "filename": filename,
                "site": extract_site(filename),
                "start": seconds_to_clock(start_sec),
                "end": seconds_to_clock(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "teacher_fold": -1 if teacher_fold is None else int(teacher_fold),
                "max_prob": float(raw_max[row_idx]),
                "top1_label": class_names[top1_idx],
                "top1_prob": float(raw_max[row_idx]),
                "top2_prob": float(raw_top2[row_idx]),
                "top1_top2_margin": float(top1_top2_margin[row_idx]),
                "topk_entropy": float(topk_entropy[row_idx]),
                "positive_count": int(positive_counts[row_idx]),
                "positive_labels": ";".join(positive_labels),
            }
        )
    return rows


def pseudo_metadata_columns() -> List[str]:
    return [
        "row_id",
        "filename",
        "site",
        "start",
        "end",
        "start_sec",
        "end_sec",
        "teacher_fold",
        "max_prob",
        "top1_label",
        "top1_prob",
        "top2_prob",
        "top1_top2_margin",
        "topk_entropy",
        "positive_count",
        "positive_labels",
    ]


def apply_blend_postprocess(
    perch_pred: np.ndarray,
    cnn_pred: np.ndarray,
    row_ids: Sequence[str],
    cfg: PseudoBlendConfig,
) -> np.ndarray:
    pred = sigmoid(cfg.perch_weight * logit(perch_pred) + (1.0 - cfg.perch_weight) * logit(cnn_pred))
    file_keys = np.asarray([str(row_id).rpartition("_")[0] for row_id in row_ids], dtype=object)
    pred = file_level_scale(pred, file_keys=file_keys, mode=cfg.file_scale_mode, value=cfg.file_scale_value)
    pred = temporal_smooth(pred, file_keys=file_keys, mode=cfg.smooth_mode, alpha=cfg.smooth_alpha)
    return np.clip(pred.astype(np.float32), 0.0, 1.0)


def save_perch_cache(
    cache_dir: Path,
    meta_df: pd.DataFrame,
    raw_scores: np.ndarray,
    emb: np.ndarray,
    perch_fold_preds: Dict[int, np.ndarray],
    class_names: Sequence[str],
    cfg: PseudoBlendConfig,
    soundscape_files: Sequence[Path],
) -> None:
    ensure_dir(cache_dir)
    meta_df.to_parquet(cache_dir / "perch_meta.parquet", index=False)
    meta_df.to_csv(cache_dir / "perch_meta.csv", index=False)
    np.savez_compressed(
        cache_dir / "perch_arrays.npz",
        scores_full_raw=raw_scores.astype(np.float32, copy=False),
        emb_full=emb.astype(np.float32, copy=False),
    )
    if perch_fold_preds:
        fold_payload = {f"fold_{fold}": pred.astype(np.float32, copy=False) for fold, pred in perch_fold_preds.items()}
        np.savez_compressed(cache_dir / "perch_fold_preds.npz", **fold_payload)
    save_json(
        cache_dir / "perch_cache_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": cfg.mode,
            "perch_backend": cfg.perch_backend,
            "perch_dir": cfg.perch_dir,
            "perch_onnx_path": cfg.perch_onnx_path,
            "perch_tflite_path": cfg.perch_tflite_path,
            "n_soundscape_files": int(len(soundscape_files)),
            "n_rows": int(len(meta_df)),
            "n_classes": int(len(class_names)),
            "perch_pred_folds": sorted(int(fold) for fold in perch_fold_preds),
            "columns": meta_df.columns.tolist(),
            "leakage_policy": (
                "This cache contains Perch raw mapped scores, embeddings, and optional fold_k Perch "
                "LogReg predictions. Fold_k predictions are computed only from the corresponding "
                "fold_k artifact and are used with CNN fold_k for fold-specific pseudo labels."
            ),
        },
    )
    print(f"[INFO] Saved Perch cache to {cache_dir}")
    print(f"[INFO] perch rows={len(meta_df)} raw_scores={raw_scores.shape} emb={emb.shape}")


def load_perch_cache(cache_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    meta_path = cache_dir / "perch_meta.parquet"
    meta_csv_path = cache_dir / "perch_meta.csv"
    arrays_path = cache_dir / "perch_arrays.npz"
    if not arrays_path.exists() or (not meta_path.exists() and not meta_csv_path.exists()):
        raise FileNotFoundError(f"Missing Perch cache files under {cache_dir}")
    if meta_path.exists():
        try:
            meta_df = pd.read_parquet(meta_path)
        except ImportError:
            if not meta_csv_path.exists():
                raise
            meta_df = pd.read_csv(meta_csv_path)
    else:
        meta_df = pd.read_csv(meta_csv_path)
    arrays = np.load(arrays_path)
    raw_scores = arrays["scores_full_raw"].astype(np.float32, copy=False)
    emb = arrays["emb_full"].astype(np.float32, copy=False)
    if len(meta_df) != len(raw_scores) or len(meta_df) != len(emb):
        raise RuntimeError(f"Perch cache row mismatch under {cache_dir}")
    fold_preds: Dict[int, np.ndarray] = {}
    fold_pred_path = cache_dir / "perch_fold_preds.npz"
    if fold_pred_path.exists():
        fold_arrays = np.load(fold_pred_path)
        for key in fold_arrays.files:
            if not key.startswith("fold_"):
                continue
            fold = int(key.split("_", 1)[1])
            pred = fold_arrays[key].astype(np.float32, copy=False)
            if len(pred) != len(meta_df):
                raise RuntimeError(f"Perch fold prediction row mismatch for {key} under {cache_dir}")
            fold_preds[fold] = pred
    return meta_df, raw_scores, emb, fold_preds


def save_pseudo_package(
    output_dir: Path,
    meta_df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    teacher_fold: Optional[int],
    cfg: PseudoBlendConfig,
) -> None:
    (
        filtered,
        keep_mask,
        raw_max,
        raw_top1_idx,
        positive_counts,
        raw_top2,
        top1_top2_margin,
        topk_entropy,
    ) = filter_prediction_matrix(
        probs,
        prob_threshold=cfg.prob_threshold,
        row_min_max_prob=cfg.row_min_max_prob,
        top_k_labels=cfg.top_k_labels,
        min_top1_top2_margin=cfg.min_top1_top2_margin,
        max_topk_entropy=cfg.max_topk_entropy,
        entropy_top_k=cfg.entropy_top_k,
    )
    metadata_rows = build_metadata_rows(
        row_ids=meta_df["row_id"].astype(str).tolist(),
        filenames=meta_df["filename"].astype(str).tolist(),
        pred_matrix=filtered,
        keep_mask=keep_mask,
        raw_max=raw_max,
        raw_top1_idx=raw_top1_idx,
        positive_counts=positive_counts,
        raw_top2=raw_top2,
        top1_top2_margin=top1_top2_margin,
        topk_entropy=topk_entropy,
        class_names=class_names,
        teacher_fold=teacher_fold,
    )
    pseudo_df = pd.DataFrame(metadata_rows, columns=pseudo_metadata_columns())
    pseudo_probs = filtered[keep_mask].astype(np.float16)
    if len(pseudo_df) != len(pseudo_probs):
        raise RuntimeError("Pseudo metadata/prob row count mismatch.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df.to_csv(output_dir / "pseudo_segments.csv", index=False)
    np.save(output_dir / "pseudo_probs.npy", pseudo_probs)
    save_json(
        output_dir / "summary.json",
        {
            "teacher_fold": teacher_fold,
            "n_soundscape_files": int(meta_df["filename"].nunique()),
            "n_total_rows": int(len(meta_df)),
            "n_kept_rows": int(len(pseudo_df)),
            "keep_rate": float(len(pseudo_df) / max(len(meta_df), 1)),
            "prob_threshold": cfg.prob_threshold,
            "row_min_max_prob": cfg.row_min_max_prob,
            "top_k_labels": cfg.top_k_labels,
            "min_top1_top2_margin": cfg.min_top1_top2_margin,
            "max_topk_entropy": cfg.max_topk_entropy,
            "entropy_top_k": cfg.entropy_top_k,
            "perch_weight": cfg.perch_weight,
            "file_scale_mode": cfg.file_scale_mode,
            "file_scale_value": cfg.file_scale_value,
            "smooth_mode": cfg.smooth_mode,
            "smooth_alpha": cfg.smooth_alpha,
        },
    )
    print(f"[INFO] Saved pseudo labels to {output_dir}")
    print(f"[INFO] kept_rows={len(pseudo_df)} / total_rows={len(meta_df)}")


def make_run_dir(output_root: Path, output_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{output_name}"
    ensure_dir(run_dir)
    return run_dir


def prepare_common_inputs(cfg: PseudoBlendConfig):
    root = Path(cfg.root).resolve()
    input_dir = resolve_path(root, cfg.input_dir)
    output_root = resolve_path(root, cfg.output_dir)
    labels_csv = resolve_path(root, cfg.labels_csv)
    soundscapes_dir = resolve_path(root, cfg.soundscapes_dir)
    soundscape_files, labeled_filenames = list_target_files(
        soundscapes_dir=soundscapes_dir,
        labels_csv=labels_csv,
        include_labeled=cfg.include_labeled,
        debug=cfg.debug,
        debug_limit=cfg.debug_limit,
    )
    if not soundscape_files:
        raise FileNotFoundError("No target soundscapes found for pseudo generation.")
    class_names = pd.read_csv(input_dir / "sample_submission.csv", nrows=0).columns.tolist()
    class_names = [column for column in class_names if column != "row_id"]
    return root, input_dir, output_root, labels_csv, soundscapes_dir, soundscape_files, labeled_filenames, class_names


def run_perch_cache_mode(cfg: PseudoBlendConfig) -> Path:
    seed_everything(cfg.seed)
    root, input_dir, output_root, labels_csv, soundscapes_dir, soundscape_files, labeled_filenames, class_names = prepare_common_inputs(cfg)
    perch_dir = resolve_path(root, cfg.perch_dir)
    perch_model_path = resolve_path(root, cfg.perch_model_path)
    cache_dir = resolve_path(root, cfg.perch_cache_dir) if cfg.perch_cache_dir else make_run_dir(output_root, cfg.output_name) / "perch_cache"
    ensure_dir(cache_dir)

    log_path = cache_dir / "make_perch_cache.log"
    with RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        print(f"[INFO] Mode: perch-cache")
        print(f"[INFO] Target files: {len(soundscape_files)}")
        print(f"[INFO] Include labeled: {cfg.include_labeled}")
        print(f"[INFO] Soundscapes dir: {soundscapes_dir}")
        print(f"[INFO] Perch dir: {perch_dir}")
        print(f"[INFO] Perch backend: {cfg.perch_backend}")
        print(f"[INFO] Perch artifact: {perch_model_path}")
        print("[INFO] Running Perch feature extraction...")
        meta_df, raw_scores, emb = run_perch_for_paths(
            cfg=cfg,
            root=root,
            input_dir=input_dir,
            perch_dir=perch_dir,
            paths=soundscape_files,
            class_names=class_names,
        )
        perch_fold_preds: Dict[int, np.ndarray] = {}
        if perch_model_path.exists() and cfg.max_perch_folds != 0:
            import joblib

            artifact = joblib.load(perch_model_path)
            if artifact.get("model_type") != "perch_context_logreg":
                raise ValueError(f"Expected perch_context_logreg artifact, got {artifact.get('model_type')}")
            max_folds = len(artifact["folds"]) if cfg.max_perch_folds < 0 else min(cfg.max_perch_folds, len(artifact["folds"]))
            requested_folds = cfg.teacher_folds or list(range(max_folds))
            print(f"[INFO] Computing Perch LogReg fold predictions for folds: {requested_folds}")
            for fold in requested_folds:
                if fold >= len(artifact["folds"]):
                    raise ValueError(f"Requested Perch fold {fold}, but artifact has {len(artifact['folds'])} folds")
                perch_fold_preds[int(fold)] = predict_perch_fold(
                    fold_artifact=artifact["folds"][fold],
                    meta_df=meta_df,
                    raw_scores=raw_scores,
                    emb=emb,
                    artifact_config=artifact["config"],
                )
                print(f"[INFO] Perch fold {fold} pred shape: {perch_fold_preds[int(fold)].shape}")
        save_json(cache_dir / "config.json", asdict(cfg))
        save_json(
            cache_dir / "data_scope.json",
            {
                "soundscapes_dir": str(soundscapes_dir),
                "labels_csv": str(labels_csv),
                "n_soundscape_files": len(soundscape_files),
                "n_labeled_filenames": len(labeled_filenames),
                "include_labeled": cfg.include_labeled,
            },
        )
        save_perch_cache(cache_dir, meta_df, raw_scores, emb, perch_fold_preds, class_names, cfg, soundscape_files)
    return cache_dir


def run_pseudo_from_cache_mode(cfg: PseudoBlendConfig) -> Path:
    cnn_lib, base_train, torch_module = import_cnn_modules()
    seed_everything(cfg.seed, torch_module=torch_module)
    base_train.require_training_dependencies()

    root, input_dir, output_root, labels_csv, soundscapes_dir, soundscape_files, labeled_filenames, class_names = prepare_common_inputs(cfg)
    cnn_model_root = resolve_path(root, cfg.cnn_model_root)
    perch_model_path = resolve_path(root, cfg.perch_model_path)
    perch_dir = resolve_path(root, cfg.perch_dir)
    if not cfg.perch_cache_dir:
        raise ValueError("--perch-cache-dir is required in pseudo-from-cache mode.")
    perch_cache_dir = resolve_path(root, cfg.perch_cache_dir)

    available_folds = discover_available_cnn_folds(cnn_model_root)
    teacher_folds = cfg.teacher_folds or available_folds
    missing = sorted(set(teacher_folds) - set(available_folds))
    if missing:
        raise ValueError(f"Requested teacher folds missing from CNN model root: {missing}")

    run_dir = make_run_dir(output_root, cfg.output_name)
    save_json(run_dir / "config.json", asdict(cfg))
    save_json(
        run_dir / "data_scope.json",
        {
            "soundscapes_dir": str(soundscapes_dir),
            "labels_csv": str(labels_csv),
            "n_soundscape_files": len(soundscape_files),
            "n_labeled_filenames": len(labeled_filenames),
            "include_labeled": cfg.include_labeled,
        },
    )
    save_json(
        run_dir / "teacher_runs.json",
        {
            "cnn_model_root": str(cnn_model_root),
            "perch_model_path": str(perch_model_path),
            "perch_dir": str(perch_dir),
            "perch_cache_dir": str(perch_cache_dir),
            "teacher_folds": teacher_folds,
            "leakage_policy": (
                "fold-specific mode uses only CNN fold_k and Perch fold_k. "
                "Targets are unlabeled soundscapes unless --include-labeled is passed."
            ),
        },
    )

    log_path = run_dir / "make_pseudo.log"
    device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    with RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        print(f"[INFO] Mode: pseudo-from-cache")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Perch cache: {perch_cache_dir}")
        print(f"[INFO] Target files: {len(soundscape_files)}")
        print(f"[INFO] Teacher folds: {teacher_folds}")
        meta_df, raw_scores, emb, perch_fold_preds = load_perch_cache(perch_cache_dir)
        expected_row_ids = []
        for path in soundscape_files:
            expected_row_ids.extend([f"{path.stem}_{t}" for t in range(5, 65, 5)])
        cached_row_ids = meta_df["row_id"].astype(str).tolist()
        if cached_row_ids != expected_row_ids:
            row_id_to_idx = {row_id: idx for idx, row_id in enumerate(cached_row_ids)}
            missing_row_ids = [row_id for row_id in expected_row_ids if row_id not in row_id_to_idx]
            if missing_row_ids:
                raise ValueError(
                    "Perch cache row_ids do not match current target soundscape list. "
                    f"Missing examples: {missing_row_ids[:5]}"
                )
            subset_idx = np.asarray([row_id_to_idx[row_id] for row_id in expected_row_ids], dtype=np.int64)
            meta_df = meta_df.iloc[subset_idx].reset_index(drop=True)
            raw_scores = raw_scores[subset_idx]
            emb = emb[subset_idx]
            perch_fold_preds = {fold: pred[subset_idx] for fold, pred in perch_fold_preds.items()}
            print(f"[INFO] Subset Perch cache rows for current target list: {len(subset_idx)} / {len(cached_row_ids)}")
        missing_perch_folds = sorted(set(teacher_folds) - set(perch_fold_preds))
        if missing_perch_folds:
            raise FileNotFoundError(
                f"Perch cache is missing fold predictions for {missing_perch_folds}. "
                "Regenerate cache with --mode perch-cache and matching --teacher-folds."
            )

        if cfg.pseudo_scope == "global":
            fold_preds = [perch_fold_preds[fold] for fold in teacher_folds]
            perch_pred = np.mean(fold_preds, axis=0).astype(np.float32)
            cnn_fold_preds = []
            for fold in teacher_folds:
                spec, renderer, cnn_model = load_cnn_fold_model(cnn_lib, torch_module, cnn_model_root, fold=fold, class_names=class_names, device=device)
                cnn_row_ids, cnn_pred = predict_cnn_files(
                    cnn_lib=cnn_lib,
                    paths=soundscape_files,
                    spec=spec,
                    renderer=renderer,
                    model=cnn_model,
                    device=device,
                    segment_batch_size=cfg.segment_batch_size,
                )
                if cnn_row_ids != meta_df["row_id"].astype(str).tolist():
                    raise ValueError("CNN row_ids do not match Perch row_ids.")
                cnn_fold_preds.append(cnn_pred)
                del cnn_model
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
            cnn_pred = np.mean(cnn_fold_preds, axis=0).astype(np.float32)
            probs = apply_blend_postprocess(perch_pred, cnn_pred, meta_df["row_id"].astype(str).tolist(), cfg)
            save_pseudo_package(run_dir / "global", meta_df, probs, class_names, teacher_fold=None, cfg=cfg)
        else:
            for fold in teacher_folds:
                print(f"[INFO] Generating fold-specific pseudo: fold={fold}")
                perch_pred = perch_fold_preds[fold]
                spec, renderer, cnn_model = load_cnn_fold_model(cnn_lib, torch_module, cnn_model_root, fold=fold, class_names=class_names, device=device)
                cnn_row_ids, cnn_pred = predict_cnn_files(
                    cnn_lib=cnn_lib,
                    paths=soundscape_files,
                    spec=spec,
                    renderer=renderer,
                    model=cnn_model,
                    device=device,
                    segment_batch_size=cfg.segment_batch_size,
                )
                if cnn_row_ids != meta_df["row_id"].astype(str).tolist():
                    raise ValueError("CNN row_ids do not match Perch row_ids.")
                probs = apply_blend_postprocess(perch_pred, cnn_pred, meta_df["row_id"].astype(str).tolist(), cfg)
                save_pseudo_package(run_dir / f"fold_{fold}", meta_df, probs, class_names, teacher_fold=fold, cfg=cfg)
                del cnn_model
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

    print(f"[INFO] Done. Pseudo root: {run_dir}")
    return run_dir


def main() -> None:
    cfg = parse_args()
    if cfg.mode == "perch-cache":
        cache_dir = run_perch_cache_mode(cfg)
        print(f"[INFO] Done. Perch cache: {cache_dir}")
    elif cfg.mode == "pseudo-from-cache":
        run_pseudo_from_cache_mode(cfg)
    elif cfg.mode == "both":
        if cfg.perch_cache_dir:
            cache_dir = resolve_path(Path(cfg.root).resolve(), cfg.perch_cache_dir)
        else:
            cache_dir = run_perch_cache_mode(cfg)
            cfg.perch_cache_dir = str(cache_dir)
        run_pseudo_from_cache_mode(cfg)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
