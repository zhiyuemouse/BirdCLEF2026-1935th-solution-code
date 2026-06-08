#!/usr/bin/env python3
"""Blend Perch, Stage3 CNN, and base CNN submissions with OOF-selected settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend three BirdCLEF 2026 submission CSVs.")
    parser.add_argument("--perch-submission", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--stage3-submission", type=str, default="/kaggle/working/submission_stage3.csv")
    parser.add_argument("--base-cnn-submission", type=str, default="/kaggle/working/submission_cnn.csv")
    parser.add_argument("--output-path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--perch-weight", type=float, default=0.741275)
    parser.add_argument("--stage3-weight", type=float, default=0.253725)
    parser.add_argument("--base-cnn-weight", type=float, default=0.005)
    parser.add_argument("--file-scale-mode", choices=["none", "topk_mean", "max_power"], default="topk_mean")
    parser.add_argument("--file-scale-value", type=float, default=2.0)
    parser.add_argument("--smooth-mode", choices=["none", "plain", "adaptive"], default="plain")
    parser.add_argument("--smooth-alpha", type=float, default=0.10)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def parse_file_key(row_id: str) -> str:
    prefix, _, suffix = str(row_id).rpartition("_")
    if suffix.isdigit() and prefix:
        return prefix
    return str(row_id)


def file_level_scale(pred: np.ndarray, file_keys: np.ndarray, mode: str, value: float) -> np.ndarray:
    if mode == "none":
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for key in pd.Index(file_keys).unique():
        idx = np.where(file_keys == key)[0]
        p = pred[idx]
        if mode == "topk_mean":
            k = max(1, min(int(round(value)), len(idx)))
            scale = np.sort(p, axis=0)[-k:].mean(axis=0, keepdims=True)
        elif mode == "max_power":
            scale = np.power(np.maximum(p.max(axis=0, keepdims=True), EPS), float(value))
        else:
            raise ValueError(f"Unknown file scale mode: {mode}")
        out[idx] = p * scale
    return np.clip(out, 0.0, 1.0)


def temporal_smooth(pred: np.ndarray, file_keys: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    if mode == "none" or alpha <= 0:
        return pred.astype(np.float32, copy=True)
    out = pred.astype(np.float32, copy=True)
    for key in pd.Index(file_keys).unique():
        idx = np.where(file_keys == key)[0]
        if len(idx) <= 1:
            continue
        p = pred[idx]
        if mode == "plain":
            prev_p = np.concatenate([p[:1], p[:-1]], axis=0)
            next_p = np.concatenate([p[1:], p[-1:]], axis=0)
            out[idx] = (1.0 - alpha) * p + 0.5 * alpha * (prev_p + next_p)
        elif mode == "adaptive":
            new_p = p.copy()
            if len(idx) > 2:
                for pos in range(1, len(idx) - 1):
                    conf = float(p[pos].max())
                    a = alpha * (1.0 - conf)
                    new_p[pos] = (1.0 - a) * p[pos] + 0.5 * a * (p[pos - 1] + p[pos + 1])
            out[idx] = new_p
        else:
            raise ValueError(f"Unknown smooth mode: {mode}")
    return np.clip(out, 0.0, 1.0)


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    perch_path = Path(args.perch_submission)
    stage3_path = Path(args.stage3_submission)
    base_cnn_path = Path(args.base_cnn_submission)
    output_path = Path(args.output_path)

    weights = np.asarray([args.perch_weight, args.stage3_weight, args.base_cnn_weight], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError(f"Blend weights must be non-negative, got {weights.tolist()}")
    weight_sum = float(weights.sum())
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"[WARN] Blend weights sum to {weight_sum:.8f}; normalizing to 1.0")
        weights = weights / max(weight_sum, EPS)

    perch_df = load_submission(perch_path)
    stage3_df = load_submission(stage3_path)
    base_cnn_df = load_submission(base_cnn_path)

    if perch_df.columns.tolist() != stage3_df.columns.tolist() or perch_df.columns.tolist() != base_cnn_df.columns.tolist():
        raise ValueError("Submission columns differ.")
    if not perch_df["row_id"].equals(stage3_df["row_id"]) or not perch_df["row_id"].equals(base_cnn_df["row_id"]):
        raise ValueError("Submission row_id order differs.")

    class_cols = [col for col in perch_df.columns if col != "row_id"]
    perch = perch_df[class_cols].to_numpy(dtype=np.float32)
    stage3 = stage3_df[class_cols].to_numpy(dtype=np.float32)
    base_cnn = base_cnn_df[class_cols].to_numpy(dtype=np.float32)

    pred = sigmoid(weights[0] * logit(perch) + weights[1] * logit(stage3) + weights[2] * logit(base_cnn))
    file_keys = perch_df["row_id"].map(parse_file_key).to_numpy(dtype=object)
    pred = file_level_scale(pred, file_keys=file_keys, mode=args.file_scale_mode, value=args.file_scale_value)
    pred = temporal_smooth(pred, file_keys=file_keys, mode=args.smooth_mode, alpha=args.smooth_alpha)

    out_df = pd.DataFrame(pred.astype(np.float32), columns=class_cols)
    out_df.insert(0, "row_id", perch_df["row_id"].values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("[INFO] Saved three-way blended submission")
    print(f"[INFO] perch_submission: {perch_path}")
    print(f"[INFO] stage3_submission: {stage3_path}")
    print(f"[INFO] base_cnn_submission: {base_cnn_path}")
    print(f"[INFO] output_path: {output_path}")
    print(
        "[INFO] weights: "
        f"perch={weights[0]:.6f} stage3={weights[1]:.6f} base_cnn={weights[2]:.6f}"
    )
    print(f"[INFO] file_scale: {args.file_scale_mode} value={args.file_scale_value}")
    print(f"[INFO] smoothing: {args.smooth_mode} alpha={args.smooth_alpha}")
    print(f"[INFO] shape: {out_df.shape}")
    print(f"[INFO] prob range: {pred.min():.6f} to {pred.max():.6f}, mean={pred.mean():.6f}")


if __name__ == "__main__":
    main()
