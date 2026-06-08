#!/usr/bin/env python3
"""Honest local CV for Perch raw scores and a simple MLP probe.

This script is intentionally lightweight and leakage-aware:

- Uses cached local Perch outputs from `perch_cache/`
- Aligns labels by `row_id`
- Splits by `filename` with GroupKFold
- Reports:
  - raw Perch macro ROC-AUC
  - honest OOF MLP-probe macro ROC-AUC

The probe is trained on top of Perch features only. It does not use any
metadata priors, threshold tuning, or leaderboard-facing post-processing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Honest local CV for Perch raw scores + MLP probe.")
    parser.add_argument("--cache-dir", type=str, default="perch_cache")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--arrays-path", type=str, default="")
    parser.add_argument("--labels-path", type=str, default="input/train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission-path", type=str, default="input/sample_submission.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/perch_probe_cv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-files", type=int, default=-1)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["embedding", "embedding_plus_scores", "pca_embedding_plus_scores"],
        default="embedding_plus_scores",
    )
    parser.add_argument("--probe-type", type=str, choices=["mlp", "logreg"], default="mlp")
    parser.add_argument("--run-basic-suite", action="store_true")
    parser.add_argument("--hidden-dims", type=str, default="512,256")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument("--logreg-min-pos", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
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


def parse_label_cell(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def union_labels(series: Iterable[object]) -> List[str]:
    merged = set()
    for value in series:
        merged.update(parse_label_cell(value))
    return sorted(merged)


def load_class_names(sample_submission_path: Path) -> List[str]:
    sample_submission = pd.read_csv(sample_submission_path, nrows=0)
    return [column for column in sample_submission.columns if column != "row_id"]


def load_meta(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() == ".parquet":
        return pd.read_parquet(meta_path)
    if meta_path.suffix.lower() == ".csv":
        return pd.read_csv(meta_path)
    raise ValueError(f"Unsupported meta file suffix: {meta_path.suffix}")


def load_cache(cache_dir: Path, meta_path_arg: str, arrays_path_arg: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_candidates: List[Path] = []
    arrays_candidates: List[Path] = []

    if meta_path_arg:
        meta_candidates.append(Path(meta_path_arg))
    else:
        meta_candidates.extend(
            [
                cache_dir / "full_perch_meta.parquet",
                cache_dir / "full_perch_meta.csv",
            ]
        )

    if arrays_path_arg:
        arrays_candidates.append(Path(arrays_path_arg))
    else:
        arrays_candidates.append(cache_dir / "full_perch_arrays.npz")

    meta_path = next((path for path in meta_candidates if path.exists()), None)
    arrays_path = next((path for path in arrays_candidates if path.exists()), None)

    if meta_path is None:
        raise FileNotFoundError(f"Could not find Perch meta file under {cache_dir}")
    if arrays_path is None:
        raise FileNotFoundError(f"Could not find Perch arrays file under {cache_dir}")

    meta_df = load_meta(meta_path)
    arrays = np.load(arrays_path)
    scores_full_raw = arrays["scores_full_raw"].astype(np.float32, copy=False)
    emb_full = arrays["emb_full"].astype(np.float32, copy=False)
    return meta_df, scores_full_raw, emb_full


def build_aligned_labels(
    labels_path: Path,
    class_names: Sequence[str],
    meta_df: pd.DataFrame,
) -> np.ndarray:
    raw = pd.read_csv(labels_path)
    sc_clean = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean = sc_clean.reset_index(names="orig_index")
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.zeros((len(sc_clean), len(class_names)), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[label] for label in labels if label in label_to_idx]
        if idxs:
            y[i, idxs] = 1

    aligned = sc_clean.set_index("row_id").loc[meta_df["row_id"]].reset_index()
    if not np.all(aligned["filename"].values == meta_df["filename"].values):
        raise AssertionError("Meta and label filename order mismatch after row_id alignment.")
    return y[aligned["orig_index"].to_numpy()]


def limit_by_files(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
    limit_files: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if limit_files <= 0:
        return meta_df, y_true, scores_full_raw, emb_full
    keep_files = meta_df["filename"].drop_duplicates().iloc[:limit_files].tolist()
    keep_mask = meta_df["filename"].isin(keep_files).values
    return (
        meta_df.loc[keep_mask].reset_index(drop=True),
        y_true[keep_mask],
        scores_full_raw[keep_mask],
        emb_full[keep_mask],
    )


def build_features(
    emb_full: np.ndarray,
    scores_full_raw: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode == "embedding":
        return emb_full.astype(np.float32, copy=False)
    if feature_mode in {"embedding_plus_scores", "pca_embedding_plus_scores"}:
        return np.concatenate([emb_full, scores_full_raw], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def resolve_pca_dim(feature_mode: str, requested_pca_dim: int) -> int:
    return requested_pca_dim if feature_mode == "pca_embedding_plus_scores" else 0


def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true.sum(axis=0)
    neg = y_true.shape[0] - pos
    keep = (pos > 0) & (neg > 0)
    if keep.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FoldResult:
    fold: int
    n_train_rows: int
    n_valid_rows: int
    n_train_files: int
    n_valid_files: int
    raw_valid_auc: float
    probe_valid_auc: float
    best_epoch: int


def parse_hidden_dims(hidden_dims_text: str) -> List[int]:
    dims = [part.strip() for part in hidden_dims_text.split(",") if part.strip()]
    return [int(dim) for dim in dims]


def preprocess_train_valid_features(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_valid_scaled = scaler.transform(x_valid).astype(np.float32)

    actual_pca_dim = 0
    if pca_dim > 0:
        max_dim = min(pca_dim, x_train_scaled.shape[0] - 1, x_train_scaled.shape[1])
        if max_dim >= 1 and max_dim < x_train_scaled.shape[1]:
            pca = PCA(n_components=max_dim, random_state=seed)
            x_train_scaled = pca.fit_transform(x_train_scaled).astype(np.float32)
            x_valid_scaled = pca.transform(x_valid_scaled).astype(np.float32)
            actual_pca_dim = int(max_dim)

    return x_train_scaled, x_valid_scaled, actual_pca_dim


def train_probe_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    hidden_dims: Sequence[int],
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    device: torch.device,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    x_train_scaled, x_valid_scaled, _ = preprocess_train_valid_features(
        x_train=x_train,
        x_valid=x_valid,
        pca_dim=pca_dim,
        seed=seed,
    )

    train_dataset = NumpyDataset(x_train_scaled, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    model = ProbeMLP(
        input_dim=x_train_scaled.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    y_train_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
    pos = y_train_t.sum(dim=0)
    neg = y_train_t.shape[0] - pos
    pos_weight = (neg / (pos + 1.0)).clamp(max=50.0).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_valid_t = torch.from_numpy(x_valid_scaled).to(device)
    best_epoch = 0
    best_auc = -math.inf
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(x_valid_t)
            valid_pred = torch.sigmoid(valid_logits).cpu().numpy().astype(np.float32, copy=False)
        valid_auc = macro_auc_skip_empty(y_valid, valid_pred)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Probe training failed to produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = torch.sigmoid(model(x_valid_t)).cpu().numpy().astype(np.float32, copy=False)
    return final_pred, best_epoch


def train_logreg_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    raw_scores_valid: np.ndarray,
    c_value: float,
    max_iter: int,
    min_pos: int,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    x_train_scaled, x_valid_scaled, _ = preprocess_train_valid_features(
        x_train=x_train,
        x_valid=x_valid,
        pca_dim=pca_dim,
        seed=seed,
    )

    n_classes = y_train.shape[1]
    pred = raw_scores_valid.astype(np.float32, copy=True)

    for class_idx in range(n_classes):
        target = y_train[:, class_idx]
        pos = int(target.sum())
        neg = int(len(target) - pos)
        if pos < min_pos or neg == 0:
            continue

        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=2026,
        )
        model.fit(x_train_scaled, target)
        pred[:, class_idx] = model.predict_proba(x_valid_scaled)[:, 1].astype(np.float32, copy=False)

    return pred


@dataclass
class ExperimentConfig:
    feature_mode: str
    probe_type: str
    name: str


def get_experiment_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.run_basic_suite:
        return [
            ExperimentConfig(feature_mode="embedding", probe_type="mlp", name="embedding_mlp"),
            ExperimentConfig(feature_mode="embedding_plus_scores", probe_type="mlp", name="embedding_plus_scores_mlp"),
            ExperimentConfig(feature_mode="embedding", probe_type="logreg", name="embedding_logreg"),
            ExperimentConfig(feature_mode="embedding_plus_scores", probe_type="logreg", name="embedding_plus_scores_logreg"),
            ExperimentConfig(
                feature_mode="pca_embedding_plus_scores",
                probe_type="logreg",
                name="pca_embedding_plus_scores_logreg",
            ),
        ]
    return [
        ExperimentConfig(
            feature_mode=args.feature_mode,
            probe_type=args.probe_type,
            name=f"{args.feature_mode}_{args.probe_type}",
        )
    ]


def run_single_experiment(
    args: argparse.Namespace,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    scores_full_raw: np.ndarray,
    emb_full: np.ndarray,
    device: torch.device,
    hidden_dims: Sequence[int],
    output_dir: Path,
    exp_cfg: ExperimentConfig,
) -> Dict[str, float]:
    x = build_features(emb_full=emb_full, scores_full_raw=scores_full_raw, feature_mode=exp_cfg.feature_mode)
    exp_pca_dim = resolve_pca_dim(exp_cfg.feature_mode, args.pca_dim)
    groups = meta_df["filename"].to_numpy()
    unique_files = pd.Index(groups).unique()
    if len(unique_files) < args.n_folds:
        raise ValueError(
            f"Not enough unique filenames for GroupKFold: have {len(unique_files)}, need at least {args.n_folds}."
        )

    raw_perch_auc = macro_auc_skip_empty(y_true, scores_full_raw)
    gkf = GroupKFold(n_splits=args.n_folds)
    oof_probe = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_results: List[FoldResult] = []

    print(f"[INFO] Experiment: {exp_cfg.name}")
    print(f"[INFO] rows: {len(meta_df)}")
    print(f"[INFO] files: {len(unique_files)}")
    print(f"[INFO] feature_mode: {exp_cfg.feature_mode}")
    print(f"[INFO] probe_type: {exp_cfg.probe_type}")
    print(f"[INFO] input_dim: {x.shape[1]}")
    print(f"[INFO] pca_dim: {exp_pca_dim}")
    print(f"[INFO] raw_perch_auc: {raw_perch_auc:.6f}")
    print(f"[INFO] device: {device}")

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(x, groups=groups), start=1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)

        raw_valid_auc = macro_auc_skip_empty(y_true[valid_idx], scores_full_raw[valid_idx])
        if exp_cfg.probe_type == "mlp":
            fold_pred, best_epoch = train_probe_one_fold(
                x_train=x[train_idx],
                y_train=y_true[train_idx],
                x_valid=x[valid_idx],
                y_valid=y_true[valid_idx],
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                device=device,
                pca_dim=exp_pca_dim,
                seed=args.seed,
            )
        elif exp_cfg.probe_type == "logreg":
            fold_pred = train_logreg_one_fold(
                x_train=x[train_idx],
                y_train=y_true[train_idx],
                x_valid=x[valid_idx],
                raw_scores_valid=scores_full_raw[valid_idx],
                c_value=args.logreg_c,
                max_iter=args.logreg_max_iter,
                min_pos=args.logreg_min_pos,
                pca_dim=exp_pca_dim,
                seed=args.seed,
            )
            best_epoch = 0
        else:
            raise ValueError(f"Unsupported probe type: {exp_cfg.probe_type}")

        oof_probe[valid_idx] = fold_pred
        probe_valid_auc = macro_auc_skip_empty(y_true[valid_idx], fold_pred)

        fold_result = FoldResult(
            fold=fold,
            n_train_rows=len(train_idx),
            n_valid_rows=len(valid_idx),
            n_train_files=len(pd.Index(groups[train_idx]).unique()),
            n_valid_files=len(pd.Index(groups[valid_idx]).unique()),
            raw_valid_auc=raw_valid_auc,
            probe_valid_auc=probe_valid_auc,
            best_epoch=best_epoch,
        )
        fold_results.append(fold_result)
        print(
            f"[FOLD {fold}] raw_auc={raw_valid_auc:.6f} "
            f"probe_auc={probe_valid_auc:.6f} "
            f"best_epoch={best_epoch} "
            f"train_files={fold_result.n_train_files} "
            f"valid_files={fold_result.n_valid_files}"
        )

    probe_oof_auc = macro_auc_skip_empty(y_true, oof_probe)
    mean_fold_raw_auc = float(np.mean([result.raw_valid_auc for result in fold_results]))
    mean_fold_probe_auc = float(np.mean([result.probe_valid_auc for result in fold_results]))
    print(f"[INFO] probe_oof_auc: {probe_oof_auc:.6f}")
    print(f"[INFO] mean_fold_raw_auc: {mean_fold_raw_auc:.6f}")
    print(f"[INFO] mean_fold_probe_auc: {mean_fold_probe_auc:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df = pd.DataFrame([result.__dict__ for result in fold_results])
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y_true=y_true.astype(np.uint8, copy=False),
        raw_scores=scores_full_raw.astype(np.float32, copy=False),
        probe_oof_scores=oof_probe.astype(np.float32, copy=False),
    )

    summary = {
        "experiment_name": exp_cfg.name,
        "rows": int(len(meta_df)),
        "files": int(len(unique_files)),
        "n_folds": int(args.n_folds),
        "feature_mode": exp_cfg.feature_mode,
        "probe_type": exp_cfg.probe_type,
        "input_dim": int(x.shape[1]),
        "pca_dim": int(exp_pca_dim),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "logreg_c": float(args.logreg_c),
        "logreg_max_iter": int(args.logreg_max_iter),
        "logreg_min_pos": int(args.logreg_min_pos),
        "device": str(device),
        "raw_perch_auc": float(raw_perch_auc),
        "probe_oof_auc": float(probe_oof_auc),
        "mean_fold_raw_auc": mean_fold_raw_auc,
        "mean_fold_probe_auc": mean_fold_probe_auc,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Saved fold metrics to: {fold_metrics_path}")
    print(f"[INFO] Saved summary to: {summary_path}")
    return {
        "raw_perch_auc": float(raw_perch_auc),
        "probe_oof_auc": float(probe_oof_auc),
        "mean_fold_raw_auc": mean_fold_raw_auc,
        "mean_fold_probe_auc": mean_fold_probe_auc,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    labels_path = Path(args.labels_path)
    sample_submission_path = Path(args.sample_submission_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df, scores_full_raw, emb_full = load_cache(
        cache_dir=cache_dir,
        meta_path_arg=args.meta_path,
        arrays_path_arg=args.arrays_path,
    )
    class_names = load_class_names(sample_submission_path)
    y_true = build_aligned_labels(
        labels_path=labels_path,
        class_names=class_names,
        meta_df=meta_df,
    )

    meta_df, y_true, scores_full_raw, emb_full = limit_by_files(
        meta_df=meta_df,
        y_true=y_true,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
        limit_files=args.limit_files,
    )

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = resolve_device(args.device)
    experiment_configs = get_experiment_configs(args)
    suite_rows: List[Dict[str, float | str]] = []

    for exp_cfg in experiment_configs:
        exp_output_dir = output_dir / exp_cfg.name if args.run_basic_suite else output_dir
        metrics = run_single_experiment(
            args=args,
            meta_df=meta_df,
            y_true=y_true,
            scores_full_raw=scores_full_raw,
            emb_full=emb_full,
            device=device,
            hidden_dims=hidden_dims,
            output_dir=exp_output_dir,
            exp_cfg=exp_cfg,
        )
        suite_rows.append(
            {
                "experiment_name": exp_cfg.name,
                "feature_mode": exp_cfg.feature_mode,
                "probe_type": exp_cfg.probe_type,
                **metrics,
            }
        )

    if args.run_basic_suite:
        suite_df = pd.DataFrame(suite_rows)
        suite_path = output_dir / "suite_summary.csv"
        suite_df.to_csv(suite_path, index=False)
        print(f"[INFO] Saved suite summary to: {suite_path}")


if __name__ == "__main__":
    main()
