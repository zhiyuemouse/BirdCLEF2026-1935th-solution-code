#!/usr/bin/env python3
"""Build fold-strict Perch ensemble teacher targets for labeled soundscapes.

For each outer fold, the teacher predictions for the student training files are
generated with the artifacts of that same fold.  Those teacher artifacts were
trained without the outer validation files, so the resulting student CV does
not receive information from its validation fold through soft targets.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import birdclef2026_perch_context_train as context_train
import birdclef2026_perch_spatial_mamba_train as spatial_train
from birdclef2026_perch_context_train import sigmoid_np


EPS = 1e-6


def install_numpy_core_compat() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    aliases = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.arrayprint": "numpy.core.arrayprint",
        "numpy._core.defchararray": "numpy.core.defchararray",
        "numpy._core.records": "numpy.core.records",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core._dtype_ctypes": "numpy.core._dtype_ctypes",
        "numpy._core._methods": "numpy.core._methods",
    }
    for alias, target in aliases.items():
        try:
            sys.modules.setdefault(alias, importlib.import_module(target))
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict Perch ensemble teacher targets.")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--fold-assignment-path", type=str, default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv")
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--spatial-cache-dir-mean", type=str, default="perch_spatial_cache_labeled_all")
    parser.add_argument("--spatial-cache-dir-flat64", type=str, default="perch_spatial_cache_labeled_all_flat64")
    parser.add_argument("--perch-lr-artifact", type=str, default="outputs/perch_context_deploy_labeled_all_cnn195634_folds_v1/perch_context_logreg_artifacts.joblib")
    parser.add_argument("--mamba-artifact", type=str, default="outputs/perch_spatial_mamba_mean_perchmambav1_conservative093_w025_cnn195634folds_nopca_noraw_v1/perch_spatial_mamba_artifacts.joblib")
    parser.add_argument("--attention-artifact", type=str, default="outputs/perch_spatial_attention_flat64_labeled_all_cnn195634folds_nopca_noraw_v1/perch_spatial_mamba_artifacts.joblib")
    parser.add_argument("--weights", type=str, default="perch_lr=0.30,mamba=0.35,attention=0.35")
    parser.add_argument("--output-path", type=str, default="outputs/strict_perch_teacher_20260524/teacher_targets.npz")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_class_names(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=1).columns.tolist()[1:]


def parse_weights(text: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return {key: value / total for key, value in weights.items()}


def logit_np(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(prob / (1.0 - prob)).astype(np.float32, copy=False)


def blend_logit(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    out = None
    for name, weight in weights.items():
        if name not in preds:
            raise KeyError(f"Missing prediction component: {name}")
        part = logit_np(preds[name]) * float(weight)
        out = part if out is None else out + part
    return sigmoid_np(out).astype(np.float32, copy=False)


def predict_binary_logreg_proba(model: object, x: np.ndarray) -> np.ndarray:
    """Compute binary LogisticRegression positive probability from weights.

    Some saved sklearn artifacts were produced by a newer sklearn than the
    local environment.  Calling predict_proba can fail on missing compatibility
    attributes, while the learned binary decision function is fully contained
    in coef_ and intercept_.
    """
    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)
    if coef.shape[0] != 1:
        raise ValueError(f"Expected binary LogisticRegression coef shape (1, n_features), got {coef.shape}")
    prob = sigmoid_np(x @ coef[0] + float(intercept[0])).astype(np.float32, copy=False)
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2 and int(classes[1]) != 1:
        prob = 1.0 - prob
    return prob.astype(np.float32, copy=False)


def predict_context_artifact_stable(
    fold_artifact: Dict[str, object],
    emb: np.ndarray,
    raw_scores: np.ndarray,
    context: np.ndarray,
    position_features: np.ndarray,
    metadata_features: np.ndarray,
    sigmoid_fallback: bool,
) -> np.ndarray:
    emb_proj = context_train.transform_embedding_projector(emb, fold_artifact=fold_artifact)
    base = context_train.build_base_features(
        emb_part=emb_proj,
        raw_scores=raw_scores,
        position_features=position_features,
        metadata_features=metadata_features,
    )
    base_scaled = fold_artifact["base_scaler"].transform(base).astype(np.float32)
    pred = sigmoid_np(raw_scores).astype(np.float32) if sigmoid_fallback else raw_scores.astype(np.float32, copy=True)

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


def load_fold_assignments(path: Path, row_ids: Sequence[str]) -> np.ndarray:
    fold_df = pd.read_csv(path)
    if "row_id" not in fold_df.columns or "fold" not in fold_df.columns:
        raise KeyError(f"{path} must contain row_id and fold columns")
    fold_map = fold_df.drop_duplicates("row_id").set_index("row_id")["fold"]
    folds = pd.Series(row_ids, dtype=str).map(fold_map)
    if folds.isna().any():
        missing = pd.Series(row_ids, dtype=str).loc[folds.isna()].head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {missing}")
    return folds.astype(int).to_numpy()


def load_spatial_tokens(cache_dir: Path, token_key: str) -> Tuple[pd.DataFrame, np.ndarray]:
    meta = context_train.load_meta(cache_dir / "perch_spatial_meta.parquet")
    arrays = np.load(cache_dir / "perch_spatial_arrays.npz")
    if token_key not in arrays:
        raise KeyError(f"{cache_dir}/perch_spatial_arrays.npz lacks {token_key}; keys={arrays.files}")
    return meta, arrays[token_key].astype(np.float32, copy=False)


def align_array_by_row_id(source_meta: pd.DataFrame, source_array: np.ndarray, row_ids: Sequence[str], name: str) -> np.ndarray:
    source_pos = pd.Series(np.arange(len(source_meta), dtype=np.int64), index=source_meta["row_id"].astype(str))
    idx = pd.Series(row_ids, dtype=str).map(source_pos)
    if idx.isna().any():
        missing = pd.Series(row_ids, dtype=str).loc[idx.isna()].head(5).tolist()
        raise ValueError(f"{name} missing {idx.isna().sum()} rows. Examples: {missing}")
    return source_array[idx.to_numpy(dtype=np.int64)]


def token_projector_from_artifact(obj: object) -> object:
    if isinstance(obj, list):
        return [token_projector_from_artifact(item) for item in obj]
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected token_projector artifact type: {type(obj)}")
    return spatial_train.TokenProjector(
        token_mean=obj["token_mean"],
        token_std=obj["token_std"],
        pca=obj["pca"],
        output_dim=int(obj["output_dim"]),
    )


def raw_projector_from_artifact(obj: object) -> object | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected raw_projector artifact type: {type(obj)}")
    return spatial_train.RawScoreProjector(
        mean=obj["mean"],
        std=obj["std"],
        pca=obj["pca"],
        output_dim=int(obj["output_dim"]),
    )


def predict_spatial_artifact(
    artifact: Dict[str, object],
    fold_idx: int,
    token_source: np.ndarray,
    raw_scores: np.ndarray,
    fallback_sigmoid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    fold_artifact = artifact["folds"][fold_idx]
    token_projector = token_projector_from_artifact(fold_artifact["token_projector"])
    raw_projector = raw_projector_from_artifact(fold_artifact.get("raw_projector"))
    model_artifact = fold_artifact["model"]
    tokens = spatial_train.transform_tokens_with_projector(token_source, token_projector)
    raw_features = None if raw_projector is None else raw_projector.transform(raw_scores)
    model = spatial_train.PerchSpatialMambaHead(
        token_dim=int(model_artifact["token_dim"]),
        num_classes=int(model_artifact["output_dim"]),
        num_blocks=int(model_artifact["num_blocks"]),
        kernel_size=int(model_artifact["kernel_size"]),
        hidden_dim=int(model_artifact["hidden_dim"]),
        dropout=float(model_artifact["dropout"]),
        raw_dim=int(model_artifact["raw_dim"]),
        freq_pool=str(model_artifact.get("freq_pool", "mean")),
        use_pos_embed=bool(model_artifact.get("use_pos_embed", False)),
        head_variant=str(model_artifact.get("head_variant", "generic")),
        prototype_per_class=int(model_artifact.get("prototype_per_class", 5)),
        prototype_temperature=float(model_artifact.get("prototype_temperature", 12.0)),
    ).to(device)
    model.load_state_dict(model_artifact["model_state"])
    pred_all = spatial_train.predict_model(
        model=model,
        tokens=tokens,
        raw_features=raw_features,
        device=device,
        batch_size=batch_size,
    )
    pred = fallback_sigmoid.copy()
    fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int64)
    pred[:, fitted] = pred_all[:, fitted]
    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)


def build_labels(labels_path: Path, class_names: Sequence[str], row_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(labels_path)
    grouped = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(context_train.union_labels)
        .reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)
    row_to_labels = grouped.set_index("row_id")["label_list"].to_dict()
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.zeros((len(row_ids), len(class_names)), dtype=np.uint8)
    mask = np.zeros(len(row_ids), dtype=bool)
    for i, row_id in enumerate(row_ids):
        labels = row_to_labels.get(str(row_id))
        if labels is None:
            continue
        mask[i] = True
        idxs = [label_to_idx[label] for label in labels if label in label_to_idx]
        if idxs:
            y[i, idxs] = 1
    return y, mask


def macro_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(context_train.macro_auc_skip_empty(y_true, y_score))


def main() -> None:
    install_numpy_core_compat()
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(args.sample_submission_path))
    weights = parse_weights(args.weights)
    base_meta, scores_full_raw, emb_full = context_train.load_cache(
        cache_dir=Path(args.base_cache_dir),
        meta_path_arg="",
        arrays_path_arg="",
    )
    row_ids = base_meta["row_id"].astype(str).to_numpy()
    filenames = base_meta["filename"].astype(str).to_numpy()
    row_folds = load_fold_assignments(Path(args.fold_assignment_path), row_ids)
    y_true, label_mask = build_labels(Path(args.labels_path), class_names, row_ids)
    raw_sigmoid = sigmoid_np(scores_full_raw).astype(np.float32, copy=False)

    position_features = context_train.build_position_features(context_train.parse_end_seconds(row_ids.tolist()))
    metadata_features, _ = context_train.build_metadata_features(base_meta, include_hour_features=False)
    context_tensor, _ = context_train.build_context_tensor(base_meta, scores_full_raw)

    mean_meta, mean_tokens = load_spatial_tokens(Path(args.spatial_cache_dir_mean), "spatial_tokens")
    flat_meta, flat_tokens = load_spatial_tokens(Path(args.spatial_cache_dir_flat64), "spatial_tokens_64")
    mean_tokens = align_array_by_row_id(mean_meta, mean_tokens, row_ids, "mean spatial")
    flat_tokens = align_array_by_row_id(flat_meta, flat_tokens, row_ids, "flat64 spatial")

    perch_lr_artifact = joblib.load(args.perch_lr_artifact)
    mamba_artifact = joblib.load(args.mamba_artifact)
    attention_artifact = joblib.load(args.attention_artifact)

    fold_values = sorted(pd.Index(row_folds).unique().tolist())
    teacher_by_fold = np.full((len(fold_values), len(row_ids), len(class_names)), np.nan, dtype=np.float32)
    fold_rows = []
    for fold_pos, fold in enumerate(fold_values):
        fold_idx = int(fold)
        train_mask = row_folds != fold_idx
        valid_mask = row_folds == fold_idx

        lr_pred = predict_context_artifact_stable(
            fold_artifact=perch_lr_artifact["folds"][fold_idx],
            emb=emb_full[train_mask],
            raw_scores=scores_full_raw[train_mask],
            context=context_tensor[train_mask],
            position_features=position_features[train_mask],
            metadata_features=metadata_features[train_mask],
            sigmoid_fallback=True,
        )
        mamba_pred = predict_spatial_artifact(
            artifact=mamba_artifact,
            fold_idx=fold_idx,
            token_source=mean_tokens[train_mask],
            raw_scores=scores_full_raw[train_mask],
            fallback_sigmoid=raw_sigmoid[train_mask],
            device=device,
            batch_size=args.batch_size,
        )
        attention_pred = predict_spatial_artifact(
            artifact=attention_artifact,
            fold_idx=fold_idx,
            token_source=flat_tokens[train_mask],
            raw_scores=scores_full_raw[train_mask],
            fallback_sigmoid=raw_sigmoid[train_mask],
            device=device,
            batch_size=args.batch_size,
        )
        pred = blend_logit(
            {"perch_lr": lr_pred, "mamba": mamba_pred, "attention": attention_pred},
            weights=weights,
        )
        teacher_by_fold[fold_pos, train_mask] = pred
        hard_auc = macro_auc_np(y_true[valid_mask & label_mask], raw_sigmoid[valid_mask & label_mask])
        train_teacher_auc = macro_auc_np(y_true[train_mask & label_mask], pred[label_mask[train_mask]])
        fold_rows.append(
            {
                "fold": int(fold_idx),
                "fold_pos": int(fold_pos),
                "teacher_rows_written": int(train_mask.sum()),
                "teacher_labeled_rows": int((train_mask & label_mask).sum()),
                "heldout_rows": int(valid_mask.sum()),
                "heldout_labeled_rows": int((valid_mask & label_mask).sum()),
                "heldout_raw_sigmoid_auc": float(hard_auc),
                "train_teacher_auc": float(train_teacher_auc),
            }
        )
        print(
            f"[FOLD {fold_idx}] wrote_teacher_rows={int(train_mask.sum())} "
            f"heldout_rows={int(valid_mask.sum())} heldout_labeled={int((valid_mask & label_mask).sum())} "
            f"train_teacher_auc={train_teacher_auc:.6f}",
            flush=True,
        )

    for fold_pos, fold in enumerate(fold_values):
        train_mask = row_folds != int(fold)
        if np.isnan(teacher_by_fold[fold_pos, train_mask]).any():
            raise RuntimeError(f"Teacher target has NaNs in train rows for fold {fold}.")

    raw_auc = macro_auc_np(y_true[label_mask], raw_sigmoid[label_mask])
    np.savez_compressed(
        output_path,
        row_id=row_ids.astype(object),
        filename=filenames.astype(object),
        fold=row_folds.astype(np.int16),
        fold_values=np.asarray(fold_values, dtype=np.int16),
        label_mask=label_mask.astype(bool),
        y_true=y_true.astype(np.uint8),
        teacher_by_fold=teacher_by_fold.astype(np.float32),
        raw_sigmoid=raw_sigmoid.astype(np.float32),
    )
    summary = {
        "output_path": str(output_path),
        "rows": int(len(row_ids)),
        "labeled_rows": int(label_mask.sum()),
        "files": int(base_meta["filename"].nunique()),
        "classes": int(len(class_names)),
        "weights": weights,
        "raw_sigmoid_labeled_auc": float(raw_auc),
        "mean_train_teacher_auc": float(np.mean([row["train_teacher_auc"] for row in fold_rows])),
        "leakage_policy": (
            "For outer fold f, teacher targets for student training rows are produced by fold_f "
            "artifacts, whose training data excludes fold f validation rows. Teacher targets for "
            "student validation rows are not used during that fold's training."
        ),
        "folds": fold_rows,
    }
    (output_path.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(output_path.parent / "fold_rows.csv", index=False)
    print(f"[INFO] raw_sigmoid_labeled_auc={raw_auc:.6f}")
    print(f"[INFO] mean_train_teacher_auc={summary['mean_train_teacher_auc']:.6f}")
    print(f"[INFO] Saved teacher targets to: {output_path}")


if __name__ == "__main__":
    main()
