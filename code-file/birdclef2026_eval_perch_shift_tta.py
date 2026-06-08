#!/usr/bin/env python3
"""Evaluate shifted-window Perch spatial TTA on local soundscape OOF folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

import birdclef2026_perch_spatial_mamba_train as spatial_train
from birdclef2026_perch_context_train import (
    build_aligned_labels,
    load_cache as load_base_cache,
    load_class_names,
    macro_auc_skip_empty,
    sigmoid_np,
)


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shifted Perch spatial TTA.")
    parser.add_argument("--artifact-path", type=str, required=True)
    parser.add_argument("--base-cache-dir", type=str, default="perch_cache_labeled_all")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument(
        "--fold-assignment-path",
        type=str,
        default="outputs/birdclef2026_gm/20260505_195634_convnextv2_atto.fcmae_ft_in1k/soundscape_segments_with_folds.csv",
    )
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        help="Named spatial cache in name=path form. Repeatable.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/perch_shift_tta_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_named_caches(items: Sequence[str]) -> Dict[str, Path]:
    if not items:
        items = [
            "exact=perch_spatial_cache_labeled_all",
            "m1=perch_spatial_cache_labeled_all_shift_m1",
            "p1=perch_spatial_cache_labeled_all_shift_p1",
        ]
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--cache must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty cache name in {item!r}")
        out[name] = Path(path)
    return out


def load_folds(path: Path, row_ids: Sequence[str]) -> np.ndarray:
    fold_df = pd.read_csv(path)
    fold_map = fold_df.drop_duplicates("row_id").set_index("row_id")["fold"]
    folds = pd.Series(row_ids, dtype=str).map(fold_map)
    if folds.isna().any():
        missing = pd.Series(row_ids, dtype=str).loc[folds.isna()].head(5).tolist()
        raise ValueError(f"Fold assignment misses {folds.isna().sum()} rows. Examples: {missing}")
    return folds.astype(int).to_numpy()


def load_spatial_source(cache_dir: Path, base_meta: pd.DataFrame, freq_pool: str) -> np.ndarray:
    meta, tokens, tokens_max, tokens_64 = spatial_train.load_spatial_cache_from_paths(str(cache_dir))
    if freq_pool == "flat64":
        if tokens_64 is None:
            raise KeyError(f"{cache_dir} lacks spatial_tokens_64")
        source = tokens_64
    elif freq_pool == "meanmax":
        if tokens_max is None:
            raise KeyError(f"{cache_dir} lacks spatial_tokens_max")
        source = np.stack([tokens, tokens_max], axis=1)
    else:
        source = tokens
    return spatial_train.align_spatial_to_base(base_meta=base_meta, spatial_meta=meta, spatial_tokens=source)


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


def predict_cache_oof(
    artifact: Dict[str, object],
    token_source: np.ndarray,
    raw_sigmoid: np.ndarray,
    row_folds: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    pred = raw_sigmoid.copy()
    for fold_artifact in artifact["folds"]:
        fold_name = str(fold_artifact.get("fold_name", ""))
        fold = int(fold_name.rsplit("_", 1)[-1])
        valid_idx = np.where(row_folds == fold)[0].astype(np.int64)
        model_artifact = fold_artifact["model"]
        raw_dim = int(model_artifact.get("raw_dim", 0))
        if raw_dim != 0:
            raise ValueError("Shift TTA evaluator currently supports spatial heads without raw features only.")
        projector = token_projector_from_artifact(fold_artifact["token_projector"])
        tokens = spatial_train.transform_tokens_with_projector(token_source[valid_idx], projector)
        model = spatial_train.PerchSpatialMambaHead(
            token_dim=int(model_artifact["token_dim"]),
            num_classes=int(model_artifact["output_dim"]),
            num_blocks=int(model_artifact["num_blocks"]),
            kernel_size=int(model_artifact["kernel_size"]),
            hidden_dim=int(model_artifact["hidden_dim"]),
            dropout=float(model_artifact["dropout"]),
            raw_dim=0,
            freq_pool=str(model_artifact.get("freq_pool", "mean")),
            use_pos_embed=bool(model_artifact.get("use_pos_embed", False)),
            head_variant=str(model_artifact.get("head_variant", "generic")),
            prototype_per_class=int(model_artifact.get("prototype_per_class", 5)),
            prototype_temperature=float(model_artifact.get("prototype_temperature", 12.0)),
        ).to(device)
        model.load_state_dict(model_artifact["model_state"])
        spatial_pred = spatial_train.predict_model(
            model=model,
            tokens=tokens,
            raw_features=None,
            device=device,
            batch_size=batch_size,
        )
        fitted = np.asarray(model_artifact["fitted_class_indices"], dtype=np.int64)
        pred[valid_idx[:, None], fitted[None, :]] = spatial_pred[:, fitted]
    return np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0)


def logit_np(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return np.log(prob / (1.0 - prob)).astype(np.float32, copy=False)


def sigmoid_np_local(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    class_names = load_class_names(Path(args.sample_submission_path))
    base_meta, raw_scores, _ = load_base_cache(cache_dir=Path(args.base_cache_dir), meta_path_arg="", arrays_path_arg="")
    y_true = build_aligned_labels(Path(args.labels_path), class_names, base_meta)
    row_folds = load_folds(Path(args.fold_assignment_path), base_meta["row_id"].astype(str).tolist())
    raw_sigmoid = sigmoid_np(raw_scores).astype(np.float32)
    artifact = joblib.load(args.artifact_path)
    freq_pool = str(artifact["folds"][0]["model"].get("freq_pool", artifact.get("config", {}).get("freq_pool", "mean")))

    cache_paths = parse_named_caches(args.cache)
    preds: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    for name, cache_dir in cache_paths.items():
        token_source = load_spatial_source(cache_dir=cache_dir, base_meta=base_meta, freq_pool=freq_pool)
        pred = predict_cache_oof(
            artifact=artifact,
            token_source=token_source,
            raw_sigmoid=raw_sigmoid,
            row_folds=row_folds,
            device=device,
            batch_size=args.batch_size,
        )
        preds[name] = pred
        rows.append({"name": name, "kind": "single", "auc": float(macro_auc_skip_empty(y_true, pred))})
        print(f"[INFO] {name}: auc={rows[-1]['auc']:.9f}", flush=True)

    names = list(preds)
    if len(names) >= 2:
        combos = [
            ("exact_m1", ["exact", "m1"]),
            ("exact_p1", ["exact", "p1"]),
            ("m1_p1", ["m1", "p1"]),
            ("exact_m1_p1", ["exact", "m1", "p1"]),
        ]
        for combo_name, combo_names in combos:
            if not all(name in preds for name in combo_names):
                continue
            stack = np.stack([preds[name] for name in combo_names], axis=0)
            prob_mean = stack.mean(axis=0)
            logit_mean = sigmoid_np_local(np.stack([logit_np(preds[name]) for name in combo_names], axis=0).mean(axis=0))
            rows.append({"name": combo_name, "kind": "prob_mean", "auc": float(macro_auc_skip_empty(y_true, prob_mean))})
            rows.append({"name": combo_name, "kind": "logit_mean", "auc": float(macro_auc_skip_empty(y_true, logit_mean))})
            print(
                f"[INFO] {combo_name}: prob_auc={rows[-2]['auc']:.9f} logit_auc={rows[-1]['auc']:.9f}",
                flush=True,
            )

    result_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    result_df.to_csv(output_dir / "shift_tta_results.csv", index=False)
    np.savez_compressed(
        output_dir / "shift_tta_predictions.npz",
        row_ids=base_meta["row_id"].astype(str).to_numpy(),
        **preds,
    )
    summary = {
        "artifact_path": str(args.artifact_path),
        "freq_pool": freq_pool,
        "base_cache_dir": str(args.base_cache_dir),
        "caches": {name: str(path) for name, path in cache_paths.items()},
        "best": result_df.iloc[0].to_dict(),
        "results": result_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] best: {summary['best']}")
    print(f"[INFO] saved: {output_dir}")


if __name__ == "__main__":
    main()
