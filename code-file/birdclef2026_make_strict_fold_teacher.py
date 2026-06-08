from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import birdclef2026_gm_kaggle_infer as cnn_infer
import birdclef2026_gm_kaggle_infer_stage3 as stage3_infer
import birdclef2026_kaggle_infer_unified_perch_stage3 as unified_infer
import birdclef2026_perch_context_train as context_train
import birdclef2026_perch_kaggle_infer_spatial_mamba as spatial_infer
import birdclef2026_whitelist_blend_unified_raw_waveform as blend_util


def install_numpy_core_compat() -> None:
    """Let numpy-1.x environments read numpy-2 pickled joblib artifacts."""

    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.numerictypes", np.core.numerictypes)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict per-student-fold teacher predictions.")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--soundscapes-dir", type=str, default="input/train_soundscapes")
    parser.add_argument("--fold-assignment-path", type=str, default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv")
    parser.add_argument("--perch-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--spatial-cache-dir", type=str, default="perch_spatial_cache_labeled_all_flat64")
    parser.add_argument("--perch-lr-dir", type=str, default="outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1")
    parser.add_argument("--mamba-dir", type=str, default="outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1")
    parser.add_argument("--attention-dir", type=str, default="outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1")
    parser.add_argument("--stage3-model-root", type=str, default="outputs/birdclef2026_gm_stage3_perchcnn_white_v1/20260507_173716_convnextv2_atto.fcmae_ft_in1k_stage3_pseudo")
    parser.add_argument("--raw-wave-model-root", type=str, default="outputs/birdclef2026_raw_waveform_transformer/20260512_013731_raw_wave_conv_tokenizer_base_long_n32_d768")
    parser.add_argument("--output-dir", type=str, default="outputs/strict_fold_teacher_20260514")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--perch-lr-weight", type=float, default=0.221875)
    parser.add_argument("--mamba-weight", type=float, default=0.26625)
    parser.add_argument("--stage3-weight", type=float, default=0.133125)
    parser.add_argument("--attention-weight", type=float, default=0.26625)
    parser.add_argument("--raw-wave-weight", type=float, default=0.1125)
    parser.add_argument("--file-scale-topk", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--segment-batch-size", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()


def load_class_names(path: Path) -> List[str]:
    sample = pd.read_csv(path, nrows=0)
    return [col for col in sample.columns if col != "row_id"]


def align_by_row_id(source_row_id: Sequence[str], source_pred: np.ndarray, target_row_id: Sequence[str], label: str) -> np.ndarray:
    order = pd.DataFrame({"row_id": [str(x) for x in target_row_id]}).merge(
        pd.DataFrame({"row_id": [str(x) for x in source_row_id], "_pos": np.arange(len(source_row_id), dtype=np.int64)}),
        on="row_id",
        how="left",
        validate="one_to_one",
    )["_pos"]
    if order.isna().any():
        examples = pd.Series(target_row_id)[order.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{label} misses {order.isna().sum()} rows after alignment. Examples: {examples}")
    return source_pred[order.to_numpy(dtype=np.int64)].astype(np.float32, copy=False)


def load_fold_assignments(path: Path, row_id: np.ndarray) -> np.ndarray:
    fold_df = pd.read_csv(path)
    if "row_id" not in fold_df.columns or "fold" not in fold_df.columns:
        raise KeyError(f"Fold assignment must contain row_id and fold: {path}")
    fold_map = fold_df.drop_duplicates("row_id").set_index("row_id")["fold"]
    folds = pd.Series(row_id).map(fold_map)
    if folds.isna().any():
        examples = pd.Series(row_id)[folds.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {examples}")
    return folds.to_numpy(dtype=np.int64)


def load_perch_base(args: argparse.Namespace, class_names: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta_df, raw_scores, emb = context_train.load_cache(Path(args.perch_cache_dir), "", "")
    y_true = context_train.build_aligned_labels(Path(args.labels_path), class_names, meta_df).astype(np.uint8)
    if args.smoke_test:
        keep_files = meta_df["filename"].drop_duplicates().head(9).tolist()
        keep = meta_df["filename"].isin(keep_files).to_numpy()
        meta_df = meta_df.loc[keep].reset_index(drop=True)
        raw_scores = raw_scores[keep]
        emb = emb[keep]
        y_true = y_true[keep]
    return meta_df, y_true, raw_scores.astype(np.float32, copy=False), emb.astype(np.float32, copy=False)


def load_spatial_tokens(meta_df: pd.DataFrame, args: argparse.Namespace, pool: str) -> np.ndarray:
    spatial_meta, mean_tokens, max_tokens, flat64_tokens = spatial_infer_train_load_cache(args.spatial_cache_dir)
    if pool == "flat64":
        if flat64_tokens is None:
            raise KeyError("Spatial cache does not contain spatial_tokens_64 required by flat64 pool")
        tokens = flat64_tokens
    elif pool == "meanmax":
        if max_tokens is None:
            raise KeyError("Spatial cache does not contain spatial_tokens_max required by meanmax pool")
        tokens = np.stack([mean_tokens, max_tokens], axis=1).astype(np.float32, copy=False)
    else:
        tokens = mean_tokens
    return align_by_row_id(
        source_row_id=spatial_meta["row_id"].astype(str).tolist(),
        source_pred=tokens,
        target_row_id=meta_df["row_id"].astype(str).tolist(),
        label=f"spatial_{pool}",
    )


def spatial_infer_train_load_cache(spatial_cache_dir: str):
    from birdclef2026_perch_spatial_mamba_train import load_spatial_cache_from_paths

    return load_spatial_cache_from_paths(cache_dir_arg=spatial_cache_dir)


def predict_perch_lr_folds(
    artifact: Dict[str, object],
    meta_df: pd.DataFrame,
    raw_scores: np.ndarray,
    emb: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    config = artifact["config"]
    position = context_train.build_position_features(context_train.parse_end_seconds(meta_df["row_id"].tolist()))
    metadata, _ = context_train.build_metadata_features(
        meta_df=meta_df,
        include_hour_features=bool(config.get("include_hour_features", False)),
    )
    context, _ = context_train.build_context_tensor(meta_df=meta_df, scores_full_raw=raw_scores)
    pred = np.zeros((n_folds, len(meta_df), raw_scores.shape[1]), dtype=np.float32)
    for fold_idx in range(n_folds):
        fold_pred = predict_context_logreg_fold(
            fold_artifact=artifact["folds"][fold_idx],
            emb=emb,
            raw_scores=raw_scores,
            context=context,
            position_features=position,
            metadata_features=metadata,
        )
        pred[fold_idx] = np.clip(fold_pred, 0.0, 1.0)
        print(f"[INFO] Strict Perch LR fold_{fold_idx}: {pred[fold_idx].shape}")
    return pred


def predict_binary_logreg_proba(model, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)
    if coef.shape[0] != 1:
        raise ValueError(f"Expected binary LogisticRegression coef shape (1, n_features), got {coef.shape}")
    logits = x @ coef[0] + float(intercept[0])
    proba = sigmoid_np(logits).astype(np.float32, copy=False)
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2 and int(classes[1]) != 1:
        proba = 1.0 - proba
    return proba


def predict_context_logreg_fold(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
) -> np.ndarray:
    emb_proj = context_train.transform_embedding_projector(emb, fold_artifact=fold_artifact)
    base = context_train.build_base_features(
        emb_part=emb_proj,
        raw_scores=raw_scores,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    base_scaled = fold_artifact["base_scaler"].transform(base).astype(np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32)
    class_models = fold_artifact["class_models"]
    context_mean = fold_artifact["context_mean"]
    context_std = fold_artifact["context_std"]
    for class_idx in fold_artifact["fitted_class_indices"]:
        class_idx = int(class_idx)
        model = class_models[class_idx]
        ctx = context[:, class_idx, :].astype(np.float32, copy=False)
        ctx_scaled = ((ctx - context_mean[class_idx]) / context_std[class_idx]).astype(np.float32, copy=False)
        x = np.concatenate([base_scaled, ctx_scaled], axis=1).astype(np.float32, copy=False)
        pred[:, class_idx] = predict_binary_logreg_proba(model, x)
    return pred.astype(np.float32, copy=False)


def predict_spatial_folds(
    artifact: Dict[str, object],
    tokens: np.ndarray,
    raw_scores: np.ndarray,
    emb: np.ndarray,
    n_folds: int,
    batch_size: int,
    name: str,
) -> np.ndarray:
    pred = np.zeros((n_folds, len(raw_scores), raw_scores.shape[1]), dtype=np.float32)
    for fold_idx in range(n_folds):
        fold_pred, fitted = spatial_infer.predict_fold(
            fold_artifact=artifact["folds"][fold_idx],
            spatial_tokens=tokens,
            raw_scores=raw_scores,
            embedding=emb,
            batch_size=batch_size,
        )
        pred[fold_idx] = np.clip(fold_pred, 0.0, 1.0)
        print(f"[INFO] Strict {name} fold_{fold_idx}: fitted_classes={fitted}")
    return pred


def build_soundscape_file_list(soundscapes_dir: Path, filenames: Sequence[str]) -> List[Path]:
    return [soundscapes_dir / name for name in pd.Index(filenames).drop_duplicates().tolist()]


def predict_stage3_folds(
    model_root: Path,
    soundscape_files: Sequence[Path],
    target_row_id: np.ndarray,
    class_names: Sequence[str],
    n_folds: int,
    segment_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    spec = stage3_infer.resolve_model_spec(model_root)
    renderer = cnn_infer.SpectrogramRenderer(
        sample_rate=spec.sample_rate,
        image_height=spec.image_height,
        image_width=spec.image_width,
    )
    if spec.run_kind == "stage3":
        fold_paths = sorted(spec.model_root.glob("fold_*/stage3_best.pth"))
    else:
        fold_paths = sorted(spec.model_root.glob("fold_*/stage2_fold*_best.pth"))
    if len(fold_paths) < n_folds:
        raise FileNotFoundError(f"Expected at least {n_folds} stage3 fold checkpoints under {model_root}")

    pred = np.zeros((n_folds, len(target_row_id), len(class_names)), dtype=np.float32)
    for fold_idx in range(n_folds):
        model = cnn_infer.BirdCLEFNet(
            model_name=spec.model_name,
            num_classes=len(class_names),
            dropout=spec.dropout,
            drop_path=spec.drop_path,
            head_type=spec.head_type,
        )
        checkpoint = torch.load(fold_paths[fold_idx], map_location="cpu")
        model.load_state_dict(cnn_infer.extract_state_dict(checkpoint), strict=True)
        model.to(device).eval()

        row_ids: List[str] = []
        preds: List[np.ndarray] = []
        for audio_path in tqdm(
            soundscape_files,
            desc=f"Stage3 fold_{fold_idx}",
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
        ):
            audio = cnn_infer.load_soundscape_audio(audio_path, sample_rate=spec.sample_rate)
            segments, file_row_ids = cnn_infer.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=spec.sample_rate,
                clip_seconds=spec.clip_seconds,
            )
            file_preds = cnn_infer.predict_file_segments(
                segments=segments,
                models=[model],
                renderer=renderer,
                device=device,
                segment_batch_size=segment_batch_size,
            )
            row_ids.extend(file_row_ids)
            preds.append(file_preds.astype(np.float32, copy=False))
        pred[fold_idx] = align_by_row_id(row_ids, np.concatenate(preds, axis=0), target_row_id, f"stage3_fold_{fold_idx}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] Strict Stage3 fold_{fold_idx}: {pred[fold_idx].shape}")
    return pred


def predict_raw_wave_folds(
    model_root: Path,
    soundscape_files: Sequence[Path],
    target_row_id: np.ndarray,
    class_names: Sequence[str],
    n_folds: int,
    segment_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    models, sample_rate, clip_seconds = unified_infer.load_raw_wave_models(
        model_root=model_root,
        num_classes=len(class_names),
        device=device,
    )
    if len(models) < n_folds:
        raise FileNotFoundError(f"Expected at least {n_folds} raw wave fold checkpoints under {model_root}")
    pred = np.zeros((n_folds, len(target_row_id), len(class_names)), dtype=np.float32)
    for fold_idx in range(n_folds):
        model = models[fold_idx]
        row_ids: List[str] = []
        preds: List[np.ndarray] = []
        for audio_path in tqdm(
            soundscape_files,
            desc=f"RawWave fold_{fold_idx}",
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
        ):
            audio = cnn_infer.load_soundscape_audio(audio_path, sample_rate=sample_rate)
            segments, file_row_ids = cnn_infer.build_segments_for_file(
                audio=audio,
                file_stem=audio_path.stem,
                sample_rate=sample_rate,
                clip_seconds=clip_seconds,
            )
            file_preds = []
            for start in range(0, len(segments), int(segment_batch_size)):
                batch = torch.from_numpy(segments[start:start + int(segment_batch_size)]).float().to(device)
                with torch.inference_mode():
                    file_preds.append(torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32))
                del batch
            row_ids.extend(file_row_ids)
            preds.append(np.concatenate(file_preds, axis=0).astype(np.float32, copy=False))
        pred[fold_idx] = align_by_row_id(row_ids, np.concatenate(preds, axis=0), target_row_id, f"raw_wave_fold_{fold_idx}")
        print(f"[INFO] Strict RawWave fold_{fold_idx}: {pred[fold_idx].shape}")
    return pred


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return {key: float(value) / total for key, value in weights.items()}


def blend_fold_predictions(preds: Dict[str, np.ndarray], weights: Dict[str, float], filename: np.ndarray, topk: int) -> np.ndarray:
    n_folds = next(iter(preds.values())).shape[0]
    blended = []
    for fold_idx in range(n_folds):
        fused_logit = None
        for name, pred_by_fold in preds.items():
            term = float(weights[name]) * logit_np(pred_by_fold[fold_idx])
            fused_logit = term if fused_logit is None else fused_logit + term
        fused = sigmoid_np(fused_logit).astype(np.float32, copy=False)
        fused = unified_infer.file_level_topk_mean_scale(fused, filename=filename, topk=topk)
        blended.append(fused.astype(np.float32, copy=False))
    return np.stack(blended, axis=0).astype(np.float32, copy=False)


def main() -> None:
    install_numpy_core_compat()
    args = parse_args()
    spatial_infer.seed_everything(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(Path(args.sample_submission_path))
    meta_df, y_true, raw_scores, emb = load_perch_base(args, class_names)
    row_id = meta_df["row_id"].astype(str).to_numpy()
    filename = meta_df["filename"].astype(str).to_numpy()
    row_folds = load_fold_assignments(Path(args.fold_assignment_path), row_id=row_id)
    soundscape_files = build_soundscape_file_list(Path(args.soundscapes_dir), filename)

    print("[INFO] Strict fold teacher generation")
    print(f"[INFO] rows={len(row_id)} files={len(soundscape_files)} classes={len(class_names)} device={device}")
    print(f"[INFO] output_dir={output_dir}")

    perch_lr_artifact = joblib.load(Path(args.perch_lr_dir) / "perch_context_logreg_artifacts.joblib")
    mamba_artifact = joblib.load(Path(args.mamba_dir) / "perch_spatial_mamba_artifacts.joblib")
    attention_artifact = joblib.load(Path(args.attention_dir) / "perch_spatial_mamba_artifacts.joblib")

    preds: Dict[str, np.ndarray] = {}
    preds["perch_lr"] = predict_perch_lr_folds(perch_lr_artifact, meta_df, raw_scores, emb, args.n_folds)
    mamba_pool = str(mamba_artifact.get("config", {}).get("freq_pool", "mean"))
    attention_pool = str(attention_artifact.get("config", {}).get("freq_pool", "mean"))
    spatial_cache: Dict[str, np.ndarray] = {}
    for pool in sorted({mamba_pool, attention_pool}):
        spatial_cache[pool] = load_spatial_tokens(meta_df, args, pool)
    preds["mamba"] = predict_spatial_folds(
        mamba_artifact,
        spatial_cache[mamba_pool],
        raw_scores,
        emb,
        args.n_folds,
        args.batch_size,
        "Mamba",
    )
    preds["attention"] = predict_spatial_folds(
        attention_artifact,
        spatial_cache[attention_pool],
        raw_scores,
        emb,
        args.n_folds,
        args.batch_size,
        "Attention",
    )
    preds["stage3"] = predict_stage3_folds(
        Path(args.stage3_model_root),
        soundscape_files,
        row_id,
        class_names,
        args.n_folds,
        args.segment_batch_size,
        device,
    )
    preds["raw_wave"] = predict_raw_wave_folds(
        Path(args.raw_wave_model_root),
        soundscape_files,
        row_id,
        class_names,
        args.n_folds,
        args.segment_batch_size,
        device,
    )

    weights = normalize_weights(
        {
            "perch_lr": args.perch_lr_weight,
            "mamba": args.mamba_weight,
            "stage3": args.stage3_weight,
            "attention": args.attention_weight,
            "raw_wave": args.raw_wave_weight,
        }
    )
    pred_by_fold = blend_fold_predictions(preds, weights=weights, filename=filename, topk=args.file_scale_topk)
    oof_like = np.zeros_like(pred_by_fold[0])
    for fold_idx in range(args.n_folds):
        oof_like[row_folds == fold_idx] = pred_by_fold[fold_idx, row_folds == fold_idx]
    strict_auc = blend_util.macro_auc_and_class_scores(y_true.astype(np.float32), oof_like.astype(np.float32))[0]
    print(f"[INFO] Strict teacher self-check OOF-like CV = {strict_auc:.6f}")

    np.savez_compressed(
        output_dir / "strict_fold_teacher_predictions.npz",
        row_id=row_id,
        filename=filename,
        fold=row_folds.astype(np.int16),
        y_true=y_true.astype(np.uint8),
        pred_by_fold=pred_by_fold.astype(np.float32),
        pred_oof_like=oof_like.astype(np.float32),
        **{f"{name}_by_fold": value.astype(np.float32) for name, value in preds.items()},
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "rows": int(len(row_id)),
                "files": int(len(soundscape_files)),
                "classes": int(len(class_names)),
                "weights": weights,
                "file_scale_topk": int(args.file_scale_topk),
                "strict_oof_like_cv": float(strict_auc),
                "leakage_policy": (
                    "pred_by_fold[k] is produced only by teacher fold k models, which were trained without "
                    "student validation fold k labels. Student fold k should use pred_by_fold[k] for its train rows."
                ),
                "inputs": vars(args),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved strict teacher package to {output_dir / 'strict_fold_teacher_predictions.npz'}")


if __name__ == "__main__":
    main()
