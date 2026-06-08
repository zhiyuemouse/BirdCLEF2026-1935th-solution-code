#!/usr/bin/env python3
"""Second-stage site-aware stacker for BirdCLEF 2026.

This script is intentionally lightweight and conservative:

- It trains on existing soundscape OOF predictions from a stage2/stage3 run.
- It uses the original fold assignment saved by the base training pipeline.
- It learns a *residual* calibration on top of base logits, with site-specific
  biases per class.

Why this shape?
- It is much safer than hard-coding P(class|site) as a postprocess.
- It starts from the base model's own logits and only learns small corrections.
- It can later be applied to a submission-like CSV without touching the main
  CNN inference model.

Modes:
- `fit`: train/evaluate on OOF predictions and export a final stacker
- `apply`: apply an exported stacker to a submission-like CSV
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-stage site-aware stacker for BirdCLEF 2026.")
    parser.add_argument("--mode", type=str, choices=["fit", "apply"], default="fit")

    parser.add_argument(
        "--run-dir",
        type=str,
        default="outputs/birdclef2026_gm/20260428_164427_convnextv2_atto.fcmae_ft_in1k",
        help="Base model run directory containing soundscape OOF predictions.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_site_stacker")
    parser.add_argument("--limit-files", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--reg-log-scale", type=float, default=1e-3)
    parser.add_argument("--reg-bias", type=float, default=1e-3)
    parser.add_argument("--reg-site-bias", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument(
        "--stacker-path",
        type=str,
        default="",
        help="Path to a saved stacker .pt file. Defaults to <output-dir>/site_stacker.pt in apply mode.",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="",
        help="Path to a saved stacker meta .json file. Defaults to <output-dir>/site_stacker_meta.json in apply mode.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Submission-like CSV to transform in apply mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Output CSV path in apply mode. Defaults to <output-dir>/submission_stacked.csv",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def extract_site_code(name: str) -> Optional[str]:
    match = re.search(r"_(S\d{2})_", str(name))
    if match:
        return match.group(1)
    return None


def clip_probs(probs: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.clip(probs.astype(np.float32, copy=False), eps, 1.0 - eps)


def probs_to_logits(probs: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    probs = clip_probs(probs, eps=eps)
    return np.log(probs / (1.0 - probs)).astype(np.float32, copy=False)


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
        raise ValueError("No positive classes available for macro ROC-AUC.")
    return float(np.mean(scores)), int(len(scores))


def limit_by_files(df: pd.DataFrame, limit_files: int) -> pd.DataFrame:
    if limit_files <= 0:
        return df.reset_index(drop=True)
    keep_files = df["filename"].drop_duplicates().iloc[:limit_files].tolist()
    return df[df["filename"].isin(keep_files)].reset_index(drop=True)


def load_oof_frame(run_dir: Path, limit_files: int) -> tuple[pd.DataFrame, List[str]]:
    oof_path = run_dir / "soundscape_oof_predictions.csv"
    folds_path = run_dir / "soundscape_segments_with_folds.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"Fold metadata not found: {folds_path}")

    oof_df = pd.read_csv(oof_path)
    fold_df = pd.read_csv(folds_path, usecols=["row_id", "filename", "site", "fold"])
    merged = oof_df.merge(fold_df, on=["row_id", "site"], how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        missing = int(merged["fold"].isna().sum())
        raise ValueError(f"Failed to recover fold assignment for {missing} rows from {folds_path}")

    target_cols = [column for column in merged.columns if column.startswith("target_")]
    if not target_cols:
        raise ValueError("No target_* columns found in OOF predictions.")
    class_names = [column[len("target_") :] for column in target_cols]

    required_pred_cols = [column for column in class_names if column in merged.columns]
    if len(required_pred_cols) != len(class_names):
        missing = sorted(set(class_names) - set(required_pred_cols))
        raise ValueError(f"Missing prediction columns for classes: {missing[:10]}")

    merged = limit_by_files(merged, limit_files=limit_files)
    merged["fold"] = merged["fold"].astype(int)
    return merged, class_names


def build_site_mapping(sites: Sequence[str]) -> Dict[str, int]:
    unique_sites = sorted({str(site) for site in sites})
    return {site: idx for idx, site in enumerate(unique_sites)}


def map_sites_to_index(sites: Sequence[str], site_to_idx: Dict[str, int]) -> np.ndarray:
    indices = [site_to_idx.get(str(site), -1) for site in sites]
    return np.asarray(indices, dtype=np.int64)


class ResidualSiteStacker(nn.Module):
    def __init__(self, num_classes: int, num_sites: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        if num_sites > 0:
            self.site_bias = nn.Parameter(torch.zeros(num_sites, num_classes))
        else:
            self.register_parameter("site_bias", None)

    def forward(self, base_logits: torch.Tensor, site_idx: Optional[torch.Tensor]) -> torch.Tensor:
        scale = torch.exp(self.log_scale).unsqueeze(0)
        logits = base_logits * scale + self.bias.unsqueeze(0)

        if self.site_bias is not None and site_idx is not None:
            add = torch.zeros_like(logits)
            known_mask = site_idx >= 0
            if known_mask.any():
                add[known_mask] = self.site_bias[site_idx[known_mask]]
            logits = logits + add
        return logits


@dataclass
class TrainResult:
    pred_valid: np.ndarray
    train_loss: float
    final_scale_mean: float
    final_site_bias_abs_mean: float


def compute_pos_weight(y_true: np.ndarray, cap: float) -> torch.Tensor:
    y = torch.from_numpy(y_true.astype(np.float32, copy=False))
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    return (neg / (pos + 1.0)).clamp(max=float(cap))


def train_residual_site_stacker(
    base_logits_train: np.ndarray,
    y_train: np.ndarray,
    site_idx_train: np.ndarray,
    base_logits_valid: np.ndarray,
    site_idx_valid: np.ndarray,
    num_sites: int,
    epochs: int,
    lr: float,
    pos_weight_cap: float,
    reg_log_scale: float,
    reg_bias: float,
    reg_site_bias: float,
    device: torch.device,
    log_every: int,
) -> TrainResult:
    num_classes = y_train.shape[1]
    model = ResidualSiteStacker(num_classes=num_classes, num_sites=num_sites).to(device)

    x_train = torch.from_numpy(base_logits_train.astype(np.float32, copy=False)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32, copy=False)).to(device)
    site_train_t = torch.from_numpy(site_idx_train.astype(np.int64, copy=False)).to(device)

    x_valid = torch.from_numpy(base_logits_valid.astype(np.float32, copy=False)).to(device)
    site_valid_t = torch.from_numpy(site_idx_valid.astype(np.int64, copy=False)).to(device)

    pos_weight = compute_pos_weight(y_train, cap=pos_weight_cap).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = math.nan
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train, site_train_t)
        bce = criterion(logits, y_train_t)

        reg = (
            reg_log_scale * model.log_scale.pow(2).mean()
            + reg_bias * model.bias.pow(2).mean()
        )
        if model.site_bias is not None:
            reg = reg + reg_site_bias * model.site_bias.pow(2).mean()

        loss = bce + reg
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"[STACK] epoch={epoch:04d} loss={final_loss:.6f}")

    model.eval()
    with torch.no_grad():
        valid_logits = model(x_valid, site_valid_t)
        pred_valid = torch.sigmoid(valid_logits).cpu().numpy().astype(np.float32, copy=False)
        scale_mean = torch.exp(model.log_scale).mean().cpu().item()
        if model.site_bias is not None:
            site_bias_abs_mean = model.site_bias.abs().mean().cpu().item()
        else:
            site_bias_abs_mean = 0.0

    return TrainResult(
        pred_valid=pred_valid,
        train_loss=final_loss,
        final_scale_mean=float(scale_mean),
        final_site_bias_abs_mean=float(site_bias_abs_mean),
    )


def fit_final_model(
    base_logits: np.ndarray,
    y_true: np.ndarray,
    site_idx: np.ndarray,
    num_sites: int,
    epochs: int,
    lr: float,
    pos_weight_cap: float,
    reg_log_scale: float,
    reg_bias: float,
    reg_site_bias: float,
    device: torch.device,
    log_every: int,
) -> tuple[ResidualSiteStacker, float]:
    num_classes = y_true.shape[1]
    model = ResidualSiteStacker(num_classes=num_classes, num_sites=num_sites).to(device)

    x = torch.from_numpy(base_logits.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(y_true.astype(np.float32, copy=False)).to(device)
    site_t = torch.from_numpy(site_idx.astype(np.int64, copy=False)).to(device)

    pos_weight = compute_pos_weight(y_true, cap=pos_weight_cap).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = math.nan
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, site_t)
        bce = criterion(logits, y)
        reg = (
            reg_log_scale * model.log_scale.pow(2).mean()
            + reg_bias * model.bias.pow(2).mean()
        )
        if model.site_bias is not None:
            reg = reg + reg_site_bias * model.site_bias.pow(2).mean()
        loss = bce + reg
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"[FINAL] epoch={epoch:04d} loss={final_loss:.6f}")

    return model, final_loss


def save_stacker(
    output_dir: Path,
    model: ResidualSiteStacker,
    class_names: Sequence[str],
    site_to_idx: Dict[str, int],
    run_dir: Path,
    args: argparse.Namespace,
    final_train_loss: float,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stacker_path = output_dir / "site_stacker.pt"
    meta_path = output_dir / "site_stacker_meta.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
            "site_to_idx": dict(site_to_idx),
        },
        stacker_path,
    )

    meta = {
        "run_dir": str(run_dir),
        "class_names": list(class_names),
        "site_to_idx": dict(site_to_idx),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "pos_weight_cap": float(args.pos_weight_cap),
        "reg_log_scale": float(args.reg_log_scale),
        "reg_bias": float(args.reg_bias),
        "reg_site_bias": float(args.reg_site_bias),
        "final_train_loss": float(final_train_loss),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stacker_path, meta_path


def run_fit(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    df, class_names = load_oof_frame(run_dir=run_dir, limit_files=args.limit_files)
    site_to_idx = build_site_mapping(df["site"].tolist())
    site_idx = map_sites_to_index(df["site"].tolist(), site_to_idx=site_to_idx)

    y_true = df[[f"target_{label}" for label in class_names]].to_numpy(dtype=np.float32, copy=False)
    base_pred_raw = df[class_names].to_numpy(dtype=np.float32, copy=False)
    base_logits = probs_to_logits(base_pred_raw)

    unique_folds = sorted(df["fold"].unique().tolist())
    if len(unique_folds) < 2:
        raise ValueError(f"Need at least 2 folds for stacking CV, got {unique_folds}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Files: {df['filename'].nunique()}")
    print(f"[INFO] Classes: {len(class_names)}")
    print(f"[INFO] Sites: {site_to_idx}")
    print(f"[INFO] Folds: {unique_folds}")

    base_oof_auc, scored_classes = macro_auc_skip_missing(y_true, base_pred_raw)
    print(f"[INFO] Base OOF macro ROC-AUC: {base_oof_auc:.6f} | scored_classes={scored_classes}")

    oof_pred = np.zeros_like(base_pred_raw, dtype=np.float32)
    fold_rows: List[Dict[str, float | int]] = []

    for fold in unique_folds:
        train_mask = df["fold"].to_numpy() != fold
        valid_mask = df["fold"].to_numpy() == fold
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)
        if len(valid_idx) == 0 or len(train_idx) == 0:
            continue

        print(
            f"[INFO] Fold {fold} | train_rows={len(train_idx)} valid_rows={len(valid_idx)} "
            f"train_files={df.loc[train_mask, 'filename'].nunique()} valid_files={df.loc[valid_mask, 'filename'].nunique()}"
        )
        result = train_residual_site_stacker(
            base_logits_train=base_logits[train_idx],
            y_train=y_true[train_idx],
            site_idx_train=site_idx[train_idx],
            base_logits_valid=base_logits[valid_idx],
            site_idx_valid=site_idx[valid_idx],
            num_sites=len(site_to_idx),
            epochs=args.epochs,
            lr=args.lr,
            pos_weight_cap=args.pos_weight_cap,
            reg_log_scale=args.reg_log_scale,
            reg_bias=args.reg_bias,
            reg_site_bias=args.reg_site_bias,
            device=device,
            log_every=args.log_every,
        )
        oof_pred[valid_idx] = result.pred_valid

        base_fold_auc, _ = macro_auc_skip_missing(y_true[valid_idx], base_pred_raw[valid_idx])
        stacked_fold_auc, _ = macro_auc_skip_missing(y_true[valid_idx], result.pred_valid)
        print(
            f"[FOLD {fold}] base_auc={base_fold_auc:.6f} stacked_auc={stacked_fold_auc:.6f} "
            f"scale_mean={result.final_scale_mean:.4f} site_bias_abs_mean={result.final_site_bias_abs_mean:.4f}"
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "base_auc": float(base_fold_auc),
                "stacked_auc": float(stacked_fold_auc),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "train_files": int(df.loc[train_mask, "filename"].nunique()),
                "valid_files": int(df.loc[valid_mask, "filename"].nunique()),
                "train_loss": float(result.train_loss),
                "scale_mean": float(result.final_scale_mean),
                "site_bias_abs_mean": float(result.final_site_bias_abs_mean),
            }
        )

    stacked_oof_auc, scored_classes = macro_auc_skip_missing(y_true, oof_pred)
    print(f"[INFO] Stacked OOF macro ROC-AUC: {stacked_oof_auc:.6f} | scored_classes={scored_classes}")

    final_model, final_train_loss = fit_final_model(
        base_logits=base_logits,
        y_true=y_true,
        site_idx=site_idx,
        num_sites=len(site_to_idx),
        epochs=args.epochs,
        lr=args.lr,
        pos_weight_cap=args.pos_weight_cap,
        reg_log_scale=args.reg_log_scale,
        reg_bias=args.reg_bias,
        reg_site_bias=args.reg_site_bias,
        device=device,
        log_every=args.log_every,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    stacked_oof_path = output_dir / "stacked_oof_predictions.csv"
    summary_path = output_dir / "summary.json"

    pd.DataFrame(fold_rows).to_csv(fold_metrics_path, index=False)

    meta_df = df[["row_id", "filename", "site", "fold"]].copy()
    summary_df = pd.DataFrame(
        {
            "base_row_max": base_pred_raw.max(axis=1),
            "stacked_row_max": oof_pred.max(axis=1),
        }
    )
    base_pred_df = pd.DataFrame(base_pred_raw, columns=[f"base_{class_name}" for class_name in class_names])
    stacked_pred_df = pd.DataFrame(oof_pred, columns=list(class_names))
    target_df = pd.DataFrame(y_true, columns=[f"target_{class_name}" for class_name in class_names])
    out_df = pd.concat([meta_df, summary_df, base_pred_df, stacked_pred_df, target_df], axis=1)
    out_df.to_csv(stacked_oof_path, index=False)

    stacker_path, meta_path = save_stacker(
        output_dir=output_dir,
        model=final_model,
        class_names=class_names,
        site_to_idx=site_to_idx,
        run_dir=run_dir,
        args=args,
        final_train_loss=final_train_loss,
    )

    summary = {
        "run_dir": str(run_dir),
        "rows": int(len(df)),
        "files": int(df["filename"].nunique()),
        "classes": int(len(class_names)),
        "sites": dict(site_to_idx),
        "base_oof_auc": float(base_oof_auc),
        "stacked_oof_auc": float(stacked_oof_auc),
        "delta_auc": float(stacked_oof_auc - base_oof_auc),
        "scored_classes": int(scored_classes),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "pos_weight_cap": float(args.pos_weight_cap),
        "reg_log_scale": float(args.reg_log_scale),
        "reg_bias": float(args.reg_bias),
        "reg_site_bias": float(args.reg_site_bias),
        "stacker_path": str(stacker_path),
        "meta_path": str(meta_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved stacked OOF predictions to: {stacked_oof_path}")
    print(f"[INFO] Saved stacker to: {stacker_path}")
    print(f"[INFO] Saved stacker meta to: {meta_path}")
    print(f"[INFO] Saved summary to: {summary_path}")


def load_stacker(stacker_path: Path, meta_path: Path, device: torch.device) -> tuple[ResidualSiteStacker, dict]:
    checkpoint = torch.load(stacker_path, map_location="cpu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    class_names = checkpoint.get("class_names", meta["class_names"])
    site_to_idx = checkpoint.get("site_to_idx", meta["site_to_idx"])
    model = ResidualSiteStacker(num_classes=len(class_names), num_sites=len(site_to_idx))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    meta["class_names"] = class_names
    meta["site_to_idx"] = site_to_idx
    return model, meta


def predict_submission_df(
    model: ResidualSiteStacker,
    class_names: Sequence[str],
    site_to_idx: Dict[str, int],
    submission_df: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    probs = clip_probs(submission_df[list(class_names)].to_numpy(dtype=np.float32, copy=False))
    base_logits = probs_to_logits(probs)

    if "site" in submission_df.columns:
        sites = submission_df["site"].astype(str).tolist()
    else:
        sites = [extract_site_code(row_id) or "" for row_id in submission_df["row_id"].tolist()]
    site_idx = map_sites_to_index(sites, site_to_idx=site_to_idx)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(base_logits).to(device),
            torch.from_numpy(site_idx.astype(np.int64, copy=False)).to(device),
        )
        pred = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    return pred


def run_apply(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    stacker_path = Path(args.stacker_path) if args.stacker_path else (output_dir / "site_stacker.pt")
    meta_path = Path(args.meta_path) if args.meta_path else (output_dir / "site_stacker_meta.json")
    if not args.input_csv:
        raise ValueError("--input-csv is required in apply mode.")
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv else (output_dir / "submission_stacked.csv")

    if not stacker_path.exists():
        raise FileNotFoundError(f"Stacker file not found: {stacker_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Stacker meta file not found: {meta_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model, meta = load_stacker(stacker_path=stacker_path, meta_path=meta_path, device=device)
    class_names = meta["class_names"]
    site_to_idx = meta["site_to_idx"]

    submission_df = pd.read_csv(input_csv)
    missing = [column for column in class_names if column not in submission_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing class columns, e.g. {missing[:10]}")

    pred = predict_submission_df(
        model=model,
        class_names=class_names,
        site_to_idx=site_to_idx,
        submission_df=submission_df,
        device=device,
    )

    out_df = submission_df.copy()
    out_df.loc[:, class_names] = pred
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loaded stacker: {stacker_path}")
    print(f"[INFO] Loaded meta: {meta_path}")
    print(f"[INFO] Input CSV: {input_csv}")
    print(f"[INFO] Output CSV: {output_csv}")
    print(out_df.head())


def main() -> None:
    args = parse_args()
    if args.mode == "fit":
        run_fit(args)
    elif args.mode == "apply":
        run_apply(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
