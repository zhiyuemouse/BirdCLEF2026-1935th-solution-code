from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Failed to load image Python extension:.*")

import birdclef2026_gm_kaggle_infer_ensemble as infer_lib
import birdclef2026_gm_train as base_train


torch = base_train.torch


@dataclass
class PseudoConfig:
    root: str = "."
    input_dir: str = "input"
    output_dir: str = "outputs/pseudo_labels"
    model_roots: Optional[List[str]] = None
    soundscapes_dir: str = "input/train_soundscapes"
    labels_csv: str = "input/train_soundscapes_labels.csv"
    debug: bool = False
    debug_limit: int = 16
    segment_batch_size: int = 12
    tta_offsets: Optional[List[float]] = None
    smoothing_kernel: Optional[List[float]] = None
    soundscape_top_k: int = 0
    prob_threshold: float = 0.15
    row_min_max_prob: float = 0.55
    top_k_labels: int = 6
    include_labeled: bool = False
    pseudo_scope: str = "fold-specific"
    teacher_folds: Optional[List[int]] = None
    output_name: str = ""


@dataclass
class TeacherBundle:
    root: Path
    run_kind: str
    model_name: str
    sample_rate: int
    clip_seconds: float
    image_height: int
    image_width: int
    renderer: infer_lib.SpectrogramRenderer
    models: List
    fold: Optional[int]


def parse_args() -> PseudoConfig:
    parser = argparse.ArgumentParser(description="Generate leakage-aware pseudo labels for BirdCLEF 2026 soundscapes.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/pseudo_labels")
    parser.add_argument(
        "--model-root",
        type=str,
        action="append",
        default=None,
        help="Teacher run directory. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument("--labels-csv", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=16)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument("--tta-offsets", type=str, default="0")
    parser.add_argument("--smoothing-kernel", type=str, default="")
    parser.add_argument("--soundscape-top-k", type=int, default=0)
    parser.add_argument("--prob-threshold", type=float, default=0.15)
    parser.add_argument("--row-min-max-prob", type=float, default=0.55)
    parser.add_argument("--top-k-labels", type=int, default=6)
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument(
        "--pseudo-scope",
        type=str,
        choices=["fold-specific", "global"],
        default="fold-specific",
        help="fold-specific keeps local CV cleaner by generating one pseudo set per fold teacher.",
    )
    parser.add_argument(
        "--teacher-folds",
        type=str,
        default="",
        help="Optional comma-separated fold ids to generate, e.g. '0,2'. Only used in fold-specific mode.",
    )
    parser.add_argument("--output-name", type=str, default="")
    args = parser.parse_args()
    return PseudoConfig(
        root=args.root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_roots=infer_lib.parse_multi_string_args(args.model_root),
        soundscapes_dir=args.soundscapes_dir,
        labels_csv=args.labels_csv,
        debug=args.debug,
        debug_limit=args.debug_limit,
        segment_batch_size=args.segment_batch_size,
        tta_offsets=infer_lib.parse_float_list(args.tta_offsets, default=[0.0]),
        smoothing_kernel=infer_lib.parse_float_list(args.smoothing_kernel, default=[]),
        soundscape_top_k=args.soundscape_top_k,
        prob_threshold=args.prob_threshold,
        row_min_max_prob=args.row_min_max_prob,
        top_k_labels=args.top_k_labels,
        include_labeled=args.include_labeled,
        pseudo_scope=args.pseudo_scope,
        teacher_folds=parse_int_list(args.teacher_folds),
        output_name=args.output_name,
    )


def parse_int_list(text: str) -> Optional[List[int]]:
    text = str(text).strip()
    if not text:
        return None
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values if values else None


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def seconds_to_clock(seconds: int) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remain = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def extract_site(filename: str) -> str:
    parts = filename.split("_")
    for part in parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    return ""


def resolve_path(root: Path, path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def list_target_files(soundscapes_dir: Path, labels_csv: Path, include_labeled: bool, debug: bool, debug_limit: int) -> Tuple[List[Path], set]:
    labeled_df = pd.read_csv(labels_csv)
    labeled_filenames = set(labeled_df["filename"].astype(str).unique().tolist())
    files = sorted(path for path in soundscapes_dir.glob("*.ogg"))
    if not include_labeled:
        files = [path for path in files if path.name not in labeled_filenames]
    if debug:
        files = files[:debug_limit]
    return files, labeled_filenames


def build_teacher_tag(model_roots: Sequence[Path]) -> str:
    names = [path.name for path in model_roots]
    if not names:
        return "teacher"
    tag = "__".join(names)
    return tag[:160]


def discover_available_folds(model_roots: Sequence[Path]) -> List[int]:
    common_folds = None
    for model_root in model_roots:
        fold_ids = []
        for fold_dir in sorted(model_root.glob("fold_*")):
            name = fold_dir.name
            try:
                fold_ids.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        fold_set = set(fold_ids)
        common_folds = fold_set if common_folds is None else (common_folds & fold_set)
    if not common_folds:
        raise FileNotFoundError("No common fold checkpoints found under the provided teacher model roots.")
    return sorted(common_folds)


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
        return checkpoint_obj["model"]
    return checkpoint_obj


def load_teacher_bundle(model_root: Path, class_names: List[str], device, fold: Optional[int]) -> TeacherBundle:
    spec = infer_lib.resolve_model_spec(model_root)
    renderer = infer_lib.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )

    if fold is None:
        if spec.run_kind == "stage3":
            checkpoint_paths = sorted(model_root.glob("fold_*/stage3_best.pth"))
        else:
            checkpoint_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No {spec.run_kind} fold checkpoints found under {model_root}")
    else:
        checkpoint_name = "stage3_best.pth" if spec.run_kind == "stage3" else f"stage2_fold{fold}_best.pth"
        checkpoint_path = model_root / f"fold_{fold}" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]

    models = []
    for checkpoint_path in checkpoint_paths:
        model = infer_lib.BirdCLEFNet(
            model_name=spec.model_name,
            num_classes=len(class_names),
            dropout=spec.dropout,
            drop_path=spec.drop_path,
        )
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint_obj)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded teacher {spec.run_kind} checkpoint: {checkpoint_path}")

    return TeacherBundle(
        root=model_root,
        run_kind=spec.run_kind,
        model_name=spec.model_name,
        sample_rate=spec.sample_rate,
        clip_seconds=spec.clip_seconds,
        image_height=spec.image_height,
        image_width=spec.image_width,
        renderer=renderer,
        models=models,
        fold=fold,
    )


def filter_prediction_matrix(pred_matrix: np.ndarray, prob_threshold: float, row_min_max_prob: float, top_k_labels: int):
    raw = np.asarray(pred_matrix, dtype=np.float32)
    filtered = raw.copy()

    if top_k_labels > 0 and top_k_labels < filtered.shape[1]:
        top_indices = np.argpartition(-filtered, kth=top_k_labels - 1, axis=1)[:, :top_k_labels]
        top_mask = np.zeros_like(filtered, dtype=bool)
        row_indices = np.arange(filtered.shape[0])[:, None]
        top_mask[row_indices, top_indices] = True
        filtered = np.where(top_mask, filtered, 0.0)

    filtered = np.where(filtered >= prob_threshold, filtered, 0.0).astype(np.float32)
    raw_max = raw.max(axis=1)
    raw_top1_idx = raw.argmax(axis=1)
    positive_counts = (filtered > 0).sum(axis=1)
    keep_mask = (raw_max >= row_min_max_prob) & (positive_counts > 0)
    return filtered, keep_mask, raw_max, raw_top1_idx, positive_counts


def build_metadata_rows(
    audio_path: Path,
    row_ids: Sequence[str],
    pred_matrix: np.ndarray,
    keep_mask: np.ndarray,
    raw_max: np.ndarray,
    raw_top1_idx: np.ndarray,
    positive_counts: np.ndarray,
    class_names: Sequence[str],
    teacher_fold: Optional[int],
) -> List[dict]:
    rows = []
    site = extract_site(audio_path.name)
    for row_idx, row_id in enumerate(row_ids):
        if not keep_mask[row_idx]:
            continue
        end_sec = int(row_id.rsplit("_", 1)[1])
        start_sec = end_sec - 5
        positive_indices = np.flatnonzero(pred_matrix[row_idx] > 0).tolist()
        positive_labels = [class_names[idx] for idx in positive_indices]
        top1_idx = int(raw_top1_idx[row_idx])
        rows.append(
            {
                "row_id": row_id,
                "filename": audio_path.name,
                "site": site,
                "start": seconds_to_clock(start_sec),
                "end": seconds_to_clock(end_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "teacher_fold": -1 if teacher_fold is None else int(teacher_fold),
                "max_prob": float(raw_max[row_idx]),
                "top1_label": class_names[top1_idx],
                "top1_prob": float(raw_max[row_idx]),
                "positive_count": int(positive_counts[row_idx]),
                "positive_labels": ";".join(positive_labels),
            }
        )
    return rows


def generate_for_scope(
    cfg: PseudoConfig,
    class_names: List[str],
    model_roots: List[Path],
    soundscape_files: List[Path],
    output_dir: Path,
    device,
    teacher_fold: Optional[int],
) -> None:
    bundles = [load_teacher_bundle(model_root=model_root, class_names=class_names, device=device, fold=teacher_fold) for model_root in model_roots]
    metadata_rows: List[dict] = []
    prob_blocks = []
    total_rows = 0
    kept_rows = 0

    progress_desc = f"Pseudo fold={teacher_fold}" if teacher_fold is not None else "Pseudo global"
    progress = tqdm(soundscape_files, total=len(soundscape_files), desc=progress_desc, dynamic_ncols=True)
    for audio_path in progress:
        row_ids = None
        bundle_preds = []
        for bundle in bundles:
            audio = infer_lib.load_soundscape_audio(audio_path, sample_rate=bundle.sample_rate)
            segments, bundle_row_ids, row_indices = infer_lib.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=bundle.sample_rate,
                clip_seconds=bundle.clip_seconds,
                tta_offsets=cfg.tta_offsets or [0.0],
            )
            window_preds = infer_lib.predict_file_segments(
                segments=segments,
                models=bundle.models,
                renderer=bundle.renderer,
                device=device,
                segment_batch_size=cfg.segment_batch_size,
            )
            pred_matrix = infer_lib.aggregate_tta_predictions(window_preds, row_indices=row_indices, n_rows=len(bundle_row_ids))
            bundle_preds.append(pred_matrix)
            if row_ids is None:
                row_ids = bundle_row_ids
            elif row_ids != bundle_row_ids:
                raise ValueError("Row ids mismatch across teacher bundles.")

        preds = np.mean(np.stack(bundle_preds, axis=0), axis=0)
        preds = infer_lib.apply_temporal_smoothing(preds, cfg.smoothing_kernel or [])
        preds = infer_lib.apply_soundscape_postprocess(preds, cfg.soundscape_top_k)

        filtered_preds, keep_mask, raw_max, raw_top1_idx, positive_counts = filter_prediction_matrix(
            preds,
            prob_threshold=cfg.prob_threshold,
            row_min_max_prob=cfg.row_min_max_prob,
            top_k_labels=cfg.top_k_labels,
        )

        total_rows += len(row_ids)
        kept_rows += int(keep_mask.sum())
        if keep_mask.any():
            metadata_rows.extend(
                build_metadata_rows(
                    audio_path=audio_path,
                    row_ids=row_ids,
                    pred_matrix=filtered_preds,
                    keep_mask=keep_mask,
                    raw_max=raw_max,
                    raw_top1_idx=raw_top1_idx,
                    positive_counts=positive_counts,
                    class_names=class_names,
                    teacher_fold=teacher_fold,
                )
            )
            prob_blocks.append(filtered_preds[keep_mask].astype(np.float16))

        progress.set_postfix(kept=f"{kept_rows}/{total_rows}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.DataFrame(metadata_rows)
    probs = np.concatenate(prob_blocks, axis=0) if prob_blocks else np.zeros((0, len(class_names)), dtype=np.float16)

    if len(metadata_df) != len(probs):
        raise RuntimeError("Pseudo metadata row count does not match pseudo probability rows.")

    metadata_df.to_csv(output_dir / "pseudo_segments.csv", index=False)
    np.save(output_dir / "pseudo_probs.npy", probs)
    save_json(
        output_dir / "summary.json",
        {
            "teacher_fold": teacher_fold,
            "n_soundscape_files": len(soundscape_files),
            "n_total_rows": total_rows,
            "n_kept_rows": kept_rows,
            "keep_rate": float(kept_rows / max(total_rows, 1)),
            "prob_threshold": cfg.prob_threshold,
            "row_min_max_prob": cfg.row_min_max_prob,
            "top_k_labels": cfg.top_k_labels,
            "soundscape_top_k": cfg.soundscape_top_k,
            "tta_offsets": cfg.tta_offsets,
            "smoothing_kernel": cfg.smoothing_kernel,
            "teacher_model_roots": [str(path) for path in model_roots],
        },
    )
    print(f"[INFO] Saved pseudo labels to {output_dir}")
    print(f"[INFO] kept_rows={kept_rows} / total_rows={total_rows}")

    for bundle in bundles:
        for model in bundle.models:
            del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    cfg = parse_args()
    base_train.require_training_dependencies()

    root = Path(cfg.root).resolve()
    input_dir = resolve_path(root, cfg.input_dir)
    labels_csv = resolve_path(root, cfg.labels_csv)
    soundscapes_dir = resolve_path(root, cfg.soundscapes_dir)
    output_root = resolve_path(root, cfg.output_dir)

    model_roots = [Path(path) for path in infer_lib.discover_model_roots(cfg.model_roots or [])]
    class_names = base_train.load_class_names(input_dir)
    soundscape_files, labeled_filenames = list_target_files(
        soundscapes_dir=soundscapes_dir,
        labels_csv=labels_csv,
        include_labeled=cfg.include_labeled,
        debug=cfg.debug,
        debug_limit=cfg.debug_limit,
    )
    if not soundscape_files:
        raise FileNotFoundError("No target soundscape files found for pseudo label generation.")

    available_folds = discover_available_folds(model_roots)
    teacher_folds = cfg.teacher_folds or available_folds
    missing_requested_folds = sorted(set(teacher_folds) - set(available_folds))
    if missing_requested_folds:
        raise ValueError(f"Requested teacher folds are missing from teacher runs: {missing_requested_folds}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_tag = build_teacher_tag(model_roots)
    folder_suffix = cfg.output_name.strip() or f"{cfg.pseudo_scope}_{teacher_tag}"
    run_dir = output_root / f"{timestamp}_{folder_suffix}"
    base_train.ensure_dir(run_dir)

    save_json(run_dir / "config.json", asdict(cfg))
    save_json(run_dir / "teacher_runs.json", {"model_roots": [str(path) for path in model_roots]})
    save_json(run_dir / "class_names.json", {"class_names": class_names})
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

    log_path = run_dir / "make_pseudo.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with base_train.RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Teacher model roots: {[str(path) for path in model_roots]}")
        print(f"[INFO] Pseudo scope: {cfg.pseudo_scope}")
        print(f"[INFO] Target soundscapes: {len(soundscape_files)}")
        if cfg.pseudo_scope == "fold-specific":
            print(f"[INFO] Teacher folds: {teacher_folds}")
            for fold in teacher_folds:
                generate_for_scope(
                    cfg=cfg,
                    class_names=class_names,
                    model_roots=model_roots,
                    soundscape_files=soundscape_files,
                    output_dir=run_dir / f"fold_{fold}",
                    device=device,
                    teacher_fold=fold,
                )
        else:
            generate_for_scope(
                cfg=cfg,
                class_names=class_names,
                model_roots=model_roots,
                soundscape_files=soundscape_files,
                output_dir=run_dir / "global",
                device=device,
                teacher_fold=None,
            )


if __name__ == "__main__":
    main()
