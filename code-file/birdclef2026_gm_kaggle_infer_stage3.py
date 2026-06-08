import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

import birdclef2026_gm_kaggle_infer as base_infer

warnings.filterwarnings("ignore", message="Failed to load image Python extension:.*")


@dataclass
class InferConfig:
    competition_root: str = "/kaggle/input/competitions/birdclef-2026"
    output_path: str = "/kaggle/working/submission.csv"
    model_root: str = ""
    soundscapes_dir: str = ""
    debug: bool = False
    debug_limit: int = 4
    segment_batch_size: int = 12
    segment_offsets: str = ""
    seed: int = 2026


@dataclass
class ResolvedModelSpec:
    model_root: Path
    run_kind: str
    checkpoint_name: str
    model_name: str
    sample_rate: int
    clip_seconds: float
    image_height: int
    image_width: int
    spectrogram_variant: str
    dropout: float
    drop_path: float
    head_type: str
    config_source: str
    student_run_dir: Optional[Path]


DEFAULT_MODEL_CONFIG = {
    "sample_rate": 32000,
    "clip_seconds": 5.0,
    "image_height": 256,
    "image_width": 320,
    "spectrogram_variant": "logmel",
    "dropout": 0.2,
    "drop_path": 0.1,
    "head_type": "linear",
}


def parse_args() -> InferConfig:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Kaggle inference for stage2/stage3 birdclef2026_gm runs.")
    parser.add_argument("--competition-root", type=str, default="/kaggle/input/competitions/birdclef-2026")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--model-root", type=str, default="")
    parser.add_argument("--soundscapes-dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=4)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument(
        "--segment-offsets",
        type=str,
        default="",
        help="Comma-separated shifted 5s offsets in seconds, e.g. '-1,1'. Empty uses exact windows.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return InferConfig(
        competition_root=args.competition_root,
        output_path=args.output_path,
        model_root=args.model_root,
        soundscapes_dir=args.soundscapes_dir,
        debug=args.debug,
        debug_limit=args.debug_limit,
        segment_batch_size=args.segment_batch_size,
        segment_offsets=args.segment_offsets,
        seed=args.seed,
    )


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_search_roots(model_root: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []
    if model_root is not None:
        roots.append(model_root)
        roots.extend(model_root.parents)
    roots.extend(
        [
            Path.cwd(),
            Path("/kaggle/working"),
            Path("/kaggle/input/models"),
            Path("/kaggle/input"),
        ]
    )

    deduped = []
    seen = set()
    for root in roots:
        root = Path(root)
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def resolve_existing_path(path_str: str, model_root: Optional[Path] = None) -> Optional[Path]:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    for root in build_search_roots(model_root=model_root):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def discover_model_root(explicit_model_root: str) -> Path:
    if explicit_model_root:
        resolved = resolve_existing_path(explicit_model_root)
        if resolved is None:
            raise FileNotFoundError(f"Explicit model root does not exist: {explicit_model_root}")
        return resolved

    candidates = []
    for root in build_search_roots():
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            candidate_root = config_path.parent
            stage3_paths = sorted(candidate_root.glob("fold_*/stage3_best.pth"))
            stage2_paths = sorted(candidate_root.glob("fold_*/stage2_fold*_best.pth"))

            if stage3_paths:
                candidates.append((candidate_root, 2, len(stage3_paths), config_path.stat().st_mtime))
            elif stage2_paths:
                candidates.append((candidate_root, 1, len(stage2_paths), config_path.stat().st_mtime))

    if not candidates:
        raise FileNotFoundError(
            "No candidate model directory found under /kaggle/input. "
            "Please upload the trained model folder or pass --model-root."
        )

    candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    chosen_root = candidates[0][0]
    print(f"[INFO] Auto-discovered model root: {chosen_root}")
    return chosen_root


def detect_run_kind(model_root: Path) -> Tuple[str, str]:
    stage3_paths = sorted(model_root.glob("fold_*/stage3_best.pth"))
    if stage3_paths:
        return "stage3", "stage3_best.pth"

    stage2_paths = sorted(model_root.glob("fold_*/stage2_fold*_best.pth"))
    if stage2_paths:
        return "stage2", "stage2_fold*_best.pth"

    raise FileNotFoundError(f"No supported fold checkpoints found under {model_root}")


def infer_model_name_from_run_name(run_name: str) -> Optional[str]:
    patterns = [
        r"^\d{8}_\d{6}_(.+)_stage3_pseudo$",
        r"^\d{8}_\d{6}_(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, run_name)
        if match:
            return match.group(1)
    return None


def resolve_student_run_config(model_root: Path, run_cfg: dict) -> Tuple[Optional[Path], Optional[dict]]:
    candidate_paths = []
    for key in ["student_run_dir"]:
        value = run_cfg.get(key)
        if value:
            candidate_paths.append(str(value))

    metrics_path = model_root / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        value = metrics.get("student_run_dir")
        if value:
            candidate_paths.append(str(value))

    for candidate_path in candidate_paths:
        resolved = resolve_existing_path(candidate_path, model_root=model_root)
        if resolved is None or not resolved.is_dir():
            continue
        config_path = resolved / "config.json"
        if config_path.exists():
            return resolved, load_json(config_path)

    return None, None


def build_fallback_model_config(model_root: Path, run_cfg: dict) -> dict:
    model_name = str(run_cfg.get("model_name", "")).strip() or infer_model_name_from_run_name(model_root.name)
    if model_name is None:
        student_run_name = Path(str(run_cfg.get("student_run_dir", ""))).name
        model_name = infer_model_name_from_run_name(student_run_name)
    if model_name is None:
        raise ValueError(
            "Could not infer model_name for this stage3 run. "
            "Please upload the original stage2 run directory alongside the stage3 run, or pass a model root with config."
        )

    fallback = dict(DEFAULT_MODEL_CONFIG)
    fallback["model_name"] = model_name
    for key in [
        "sample_rate",
        "clip_seconds",
        "image_height",
        "image_width",
        "spectrogram_variant",
        "dropout",
        "drop_path",
        "head_type",
    ]:
        if key in run_cfg:
            fallback[key] = run_cfg[key]
    return fallback


def resolve_model_spec(model_root: Path) -> ResolvedModelSpec:
    run_kind, checkpoint_name = detect_run_kind(model_root)
    run_cfg = load_json(model_root / "config.json")

    student_run_dir = None
    if run_kind == "stage2":
        model_cfg = run_cfg
        config_source = "stage2 config.json"
    else:
        student_run_dir, student_cfg = resolve_student_run_config(model_root, run_cfg)
        if student_cfg is not None:
            model_cfg = student_cfg
            config_source = f"student config from {student_run_dir}"
        else:
            model_cfg = build_fallback_model_config(model_root, run_cfg)
            config_source = "stage3 fallback defaults"
            print(
                "[WARN] Could not resolve student_run_dir for this stage3 run. "
                "Falling back to inferred model_name + default audio/image settings. "
                "For the safest Kaggle submission, upload the original stage2 run folder alongside the stage3 run."
            )

    required = ["model_name", "sample_rate", "clip_seconds", "image_height", "image_width", "dropout", "drop_path"]
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise KeyError(f"Missing required model config keys for inference: {missing}")

    return ResolvedModelSpec(
        model_root=model_root,
        run_kind=run_kind,
        checkpoint_name=checkpoint_name,
        model_name=str(model_cfg["model_name"]),
        sample_rate=int(model_cfg["sample_rate"]),
        clip_seconds=float(model_cfg["clip_seconds"]),
        image_height=int(model_cfg["image_height"]),
        image_width=int(model_cfg["image_width"]),
        spectrogram_variant=str(model_cfg.get("spectrogram_variant", "logmel")),
        dropout=float(model_cfg["dropout"]),
        drop_path=float(model_cfg["drop_path"]),
        head_type=str(model_cfg.get("head_type", "linear")),
        config_source=config_source,
        student_run_dir=student_run_dir,
    )


def load_models(spec: ResolvedModelSpec, num_classes: int, device: torch.device):
    if spec.run_kind == "stage3":
        fold_paths = sorted(spec.model_root.glob("fold_*/stage3_best.pth"))
    else:
        fold_paths = sorted(spec.model_root.glob("fold_*/stage2_fold*_best.pth"))

    if not fold_paths:
        raise FileNotFoundError(f"No {spec.run_kind} fold checkpoints found under {spec.model_root}")

    models = []
    for fold_path in fold_paths:
        model = base_infer.BirdCLEFNet(
            model_name=spec.model_name,
            num_classes=num_classes,
            dropout=spec.dropout,
            drop_path=spec.drop_path,
            head_type=spec.head_type,
        )
        checkpoint_obj = torch.load(fold_path, map_location="cpu")
        state_dict = base_infer.extract_state_dict(checkpoint_obj)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded {spec.run_kind} checkpoint: {fold_path}")
    return models


def run_inference(cfg: InferConfig):
    base_infer.seed_everything(cfg.seed)
    model_root = discover_model_root(cfg.model_root)
    spec = resolve_model_spec(model_root)

    competition_root = Path(cfg.competition_root)
    if cfg.soundscapes_dir:
        test_dir = base_infer.resolve_user_path(cfg.soundscapes_dir, competition_root=competition_root)
    else:
        test_dir = competition_root / ("train_soundscapes" if cfg.debug else "test_soundscapes")
    sample_submission_path = competition_root / "sample_submission.csv"
    output_path = Path(cfg.output_path)

    class_names = base_infer.load_class_names(sample_submission_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Using soundscapes dir: {test_dir}")
    determinism = base_infer.get_determinism_status()
    print(
        f"[INFO] Seed={cfg.seed} | deterministic_algorithms={determinism['deterministic_algorithms']} | "
        f"cudnn_deterministic={determinism['cudnn_deterministic']} | "
        f"cudnn_benchmark={determinism['cudnn_benchmark']}"
    )
    print(f"[INFO] Model root: {model_root}")
    print(f"[INFO] Run kind: {spec.run_kind}")
    print(f"[INFO] Config source: {spec.config_source}")
    if spec.student_run_dir is not None:
        print(f"[INFO] Student run dir: {spec.student_run_dir}")
    segment_offsets = parse_float_list(cfg.segment_offsets) or [0.0]
    print(f"[INFO] Segment offsets: {segment_offsets}")

    renderer = base_infer.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
        spectrogram_variant=spec.spectrogram_variant,
    )
    print(f"[INFO] Spectrogram variant: {renderer.spectrogram_variant}")
    models = load_models(
        spec=spec,
        num_classes=len(class_names),
        device=device,
    )

    soundscape_files = base_infer.list_soundscape_files(test_dir, debug=cfg.debug, debug_limit=cfg.debug_limit)
    if not soundscape_files:
        raise FileNotFoundError(f"No .ogg files found under {test_dir}")

    all_row_ids = []
    all_preds = []
    progress = tqdm(soundscape_files, total=len(soundscape_files), desc="Infer soundscapes", dynamic_ncols=True)
    for audio_path in progress:
        audio = base_infer.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
        offset_preds = []
        row_ids = None
        for offset in segment_offsets:
            segments, current_row_ids = base_infer.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=spec.sample_rate,
                clip_seconds=spec.clip_seconds,
                clip_offset_seconds=float(offset),
            )
            preds = base_infer.predict_file_segments(
                segments=segments,
                models=models,
                renderer=renderer,
                device=device,
                segment_batch_size=cfg.segment_batch_size,
            )
            if row_ids is None:
                row_ids = current_row_ids
            elif row_ids != current_row_ids:
                raise RuntimeError(f"Segment TTA row_id mismatch for {audio_path}")
            offset_preds.append(preds)
        preds = base_infer.np.mean(base_infer.np.stack(offset_preds, axis=0), axis=0)
        if row_ids is None:
            raise RuntimeError(f"No segment offsets were evaluated for {audio_path}")
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    prediction_matrix = base_infer.np.concatenate(all_preds, axis=0)
    prediction_df = pd.DataFrame(prediction_matrix, columns=class_names)
    submission = pd.concat([pd.DataFrame({"row_id": all_row_ids}), prediction_df], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved submission to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    run_inference(parse_args())
