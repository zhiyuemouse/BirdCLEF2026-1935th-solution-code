from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import birdclef2026_gm_train as base_train


torch = base_train.torch
nn = base_train.nn
F = base_train.F
DataLoader = base_train.DataLoader
Dataset = base_train.Dataset
WeightedRandomSampler = base_train.WeightedRandomSampler
GradScaler = base_train.GradScaler
autocast = base_train.autocast


def should_disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return not sys.stderr.isatty()


@dataclass
class Stage3Config:
    root: str = "."
    input_dir: str = "input"
    output_dir: str = "outputs/birdclef2026_gm_stage3_pseudo"
    student_run_dir: str = ""
    pseudo_root: str = ""
    num_workers: int = 4
    stage3_epochs: int = 6
    stage3_batch_size: int = 16
    eval_batch_size: int = 16
    stage3_samples_per_epoch: int = 4096
    stage3_backbone_lr: float = 2e-5
    stage3_head_lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 5
    freeze_backbone_epochs: int = 0
    training_mode: str = "concat"
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.0
    noisy_student_lambda: float = 0.5
    noisy_student_pseudo_sample_power: float = 1.0
    pseudo_loss_weight: float = 0.5
    pseudo_sampler_weight: float = 0.5
    min_pseudo_max_prob: float = 0.55
    max_pseudo_rows: int = -1
    folds: Optional[List[int]] = None
    allow_global_pseudo: bool = False
    smoke_test: bool = False
    use_amp: bool = True


def parse_args() -> Stage3Config:
    parser = argparse.ArgumentParser(description="Stage 3 pseudo-label fine-tuning for BirdCLEF 2026.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_stage3_pseudo")
    parser.add_argument("--student-run-dir", type=str, required=True)
    parser.add_argument("--pseudo-root", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=max(2, (base_train.os.cpu_count() or 4) // 2))
    parser.add_argument("--stage3-epochs", type=int, default=6)
    parser.add_argument("--stage3-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--stage3-samples-per-epoch", type=int, default=4096)
    parser.add_argument("--stage3-backbone-lr", type=float, default=2e-5)
    parser.add_argument("--stage3-head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--training-mode", choices=["concat", "noisy-student"], default="concat")
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--noisy-student-lambda", type=float, default=0.5)
    parser.add_argument("--noisy-student-pseudo-sample-power", type=float, default=1.0)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-sampler-weight", type=float, default=0.5)
    parser.add_argument("--min-pseudo-max-prob", type=float, default=0.55)
    parser.add_argument("--max-pseudo-rows", type=int, default=-1)
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--allow-global-pseudo", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    cfg = Stage3Config(
        root=args.root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        student_run_dir=args.student_run_dir,
        pseudo_root=args.pseudo_root,
        num_workers=args.num_workers,
        stage3_epochs=args.stage3_epochs,
        stage3_batch_size=args.stage3_batch_size,
        eval_batch_size=args.eval_batch_size,
        stage3_samples_per_epoch=args.stage3_samples_per_epoch,
        stage3_backbone_lr=args.stage3_backbone_lr,
        stage3_head_lr=args.stage3_head_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        training_mode=args.training_mode,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        noisy_student_lambda=args.noisy_student_lambda,
        noisy_student_pseudo_sample_power=args.noisy_student_pseudo_sample_power,
        pseudo_loss_weight=args.pseudo_loss_weight,
        pseudo_sampler_weight=args.pseudo_sampler_weight,
        min_pseudo_max_prob=args.min_pseudo_max_prob,
        max_pseudo_rows=args.max_pseudo_rows,
        folds=parse_int_list(args.folds),
        allow_global_pseudo=args.allow_global_pseudo,
        smoke_test=args.smoke_test,
        use_amp=not args.disable_amp,
    )
    if cfg.smoke_test:
        cfg.stage3_epochs = 1
        cfg.stage3_samples_per_epoch = min(cfg.stage3_samples_per_epoch, 256)
        cfg.max_pseudo_rows = 128 if cfg.max_pseudo_rows < 0 else min(cfg.max_pseudo_rows, 128)
        cfg.num_workers = min(cfg.num_workers, 2)
        cfg.patience = 1
    if not 0.0 <= cfg.noisy_student_lambda <= 1.0:
        raise ValueError("--noisy-student-lambda must be in [0, 1].")
    if cfg.noisy_student_pseudo_sample_power < 0:
        raise ValueError("--noisy-student-pseudo-sample-power must be non-negative.")
    return cfg


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


def resolve_path(root: Path, path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_student_base_config(student_run_dir: Path) -> base_train.Config:
    with open(student_run_dir / "config.json", "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return base_train.Config(**payload)


def load_labeled_soundscape_df(
    student_cfg: base_train.Config,
    input_dir: Path,
    student_run_dir: Path,
    class_names: Sequence[str],
) -> pd.DataFrame:
    fold_path = student_run_dir / "soundscape_segments_with_folds.csv"
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    if fold_path.exists():
        df = pd.read_csv(fold_path)
        df["label_indices"] = df["label_indices"].apply(parse_label_indices)
        return df
    df = base_train.load_soundscape_segments(student_cfg, input_dir=input_dir, label_to_idx=label_to_idx)
    return base_train.build_soundscape_folds(df, num_classes=len(class_names), n_folds=student_cfg.n_folds, seed=student_cfg.seed)


def parse_label_indices(value) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = base_train.ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    text = text.strip("[]")
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def locate_pseudo_fold_dir(pseudo_root: Path, fold: int, allow_global_pseudo: bool) -> Path:
    fold_dir = pseudo_root / f"fold_{fold}"
    if fold_dir.exists():
        return fold_dir
    global_dir = pseudo_root / "global"
    if global_dir.exists():
        if not allow_global_pseudo:
            raise ValueError(
                f"Only global pseudo labels were found under {pseudo_root}, but fold-specific pseudo is required "
                "for trustworthy local CV. Pass --allow-global-pseudo only if you explicitly accept this leakage risk."
            )
        return global_dir
    if (pseudo_root / "pseudo_segments.csv").exists():
        if not allow_global_pseudo:
            raise ValueError(
                f"Pseudo root {pseudo_root} looks like a global pseudo package. "
                "Pass --allow-global-pseudo only if you explicitly accept the CV leakage risk."
            )
        return pseudo_root
    raise FileNotFoundError(f"Could not find pseudo labels for fold {fold} under {pseudo_root}")


def load_pseudo_package(
    pseudo_root: Path,
    input_dir: Path,
    fold: int,
    labeled_filenames: set,
    min_pseudo_max_prob: float,
    max_pseudo_rows: int,
    allow_global_pseudo: bool,
) -> Tuple[pd.DataFrame, np.ndarray, Path]:
    fold_dir = locate_pseudo_fold_dir(pseudo_root=pseudo_root, fold=fold, allow_global_pseudo=allow_global_pseudo)
    pseudo_df = pd.read_csv(fold_dir / "pseudo_segments.csv")
    pseudo_probs = np.load(fold_dir / "pseudo_probs.npy")
    if len(pseudo_df) != len(pseudo_probs):
        raise RuntimeError(f"Pseudo metadata/prob rows mismatch under {fold_dir}")

    keep_mask = (pseudo_df["max_prob"].to_numpy(dtype=np.float32) >= float(min_pseudo_max_prob))
    keep_mask &= ~pseudo_df["filename"].astype(str).isin(labeled_filenames).to_numpy()
    keep_mask &= (pseudo_df["positive_count"].to_numpy(dtype=np.int64) > 0)

    pseudo_df = pseudo_df.loc[keep_mask].reset_index(drop=True)
    pseudo_probs = pseudo_probs[keep_mask]
    if max_pseudo_rows > 0 and len(pseudo_df) > max_pseudo_rows:
        order = np.argsort(-pseudo_df["max_prob"].to_numpy(dtype=np.float32))
        order = order[:max_pseudo_rows]
        pseudo_df = pseudo_df.iloc[order].reset_index(drop=True)
        pseudo_probs = pseudo_probs[order]

    pseudo_df["audio_path"] = pseudo_df["filename"].map(lambda x: str((input_dir / "train_soundscapes" / x).resolve()))
    return pseudo_df, np.asarray(pseudo_probs, dtype=np.float32), fold_dir


def attach_labeled_audio_paths(labeled_df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    output = labeled_df.copy()
    output["audio_path"] = output["filename"].map(lambda x: str((input_dir / "train_soundscapes" / x).resolve()))
    return output


class MixedSoundscapeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pseudo_probs: Optional[np.ndarray],
        cfg: base_train.Config,
        renderer: base_train.SpectrogramRenderer,
        num_classes: int,
        train_mode: bool,
        pseudo_loss_weight: float,
    ):
        self.df = df.reset_index(drop=True)
        self.pseudo_probs = pseudo_probs
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.pseudo_loss_weight = float(pseudo_loss_weight)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = base_train.load_audio_clip(
            path=row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(row["start_sec"]),
        )
        if self.train_mode:
            audio = base_train.augment_waveform(audio)
        image = self.renderer(audio, train_mode=self.train_mode)

        if row["source"] == "pseudo":
            pseudo_idx = int(row["pseudo_idx"])
            target = torch.from_numpy(self.pseudo_probs[pseudo_idx].astype(np.float32))
            sample_weight = self.pseudo_loss_weight * float(row["max_prob"])
        else:
            target = torch.from_numpy(base_train.indices_to_multihot(row["label_indices"], self.num_classes))
            sample_weight = 1.0

        return {
            "image": image,
            "target": target,
            "loss_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "row_id": row["row_id"],
            "site": row.get("site", ""),
            "source": row["source"],
        }


class NoisyStudentSoundscapeDataset(Dataset):
    """Mix each real labeled window with one fold-safe pseudo-labeled window."""

    def __init__(
        self,
        real_df: pd.DataFrame,
        pseudo_df: pd.DataFrame,
        pseudo_probs: np.ndarray,
        cfg: base_train.Config,
        renderer: base_train.SpectrogramRenderer,
        num_classes: int,
        pseudo_mix_lambda: float,
        pseudo_sample_power: float,
    ):
        self.real_df = real_df.reset_index(drop=True)
        self.pseudo_df = pseudo_df.reset_index(drop=True)
        self.pseudo_probs = np.asarray(pseudo_probs, dtype=np.float32)
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.pseudo_mix_lambda = float(pseudo_mix_lambda)

        if len(self.pseudo_df) == 0:
            raise ValueError("Noisy Student training requires at least one pseudo row after filtering.")
        if len(self.pseudo_df) != len(self.pseudo_probs):
            raise RuntimeError("Pseudo metadata/prob row count mismatch in NoisyStudentSoundscapeDataset.")

        weights = self.pseudo_df["max_prob"].to_numpy(dtype=np.float64)
        if pseudo_sample_power != 1.0:
            weights = np.power(np.clip(weights, 0.0, None), float(pseudo_sample_power))
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(self.pseudo_df), dtype=np.float64)
        self.pseudo_cdf = np.cumsum(weights / weights.sum())
        self.pseudo_cdf[-1] = 1.0

    def __len__(self) -> int:
        return len(self.real_df)

    def _sample_pseudo_idx(self) -> int:
        return int(np.searchsorted(self.pseudo_cdf, np.random.random(), side="right"))

    def __getitem__(self, idx: int):
        real_row = self.real_df.iloc[idx]
        pseudo_idx = self._sample_pseudo_idx()
        pseudo_row = self.pseudo_df.iloc[pseudo_idx]

        real_audio = base_train.load_audio_clip(
            path=real_row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(real_row["start_sec"]),
        )
        pseudo_audio = base_train.load_audio_clip(
            path=pseudo_row["audio_path"],
            target_seconds=self.cfg.clip_seconds,
            sample_rate=self.cfg.sample_rate,
            train_mode=False,
            start_sec=float(pseudo_row["start_sec"]),
        )

        real_audio = base_train.augment_waveform(real_audio)
        pseudo_audio = base_train.augment_waveform(pseudo_audio)

        lam = self.pseudo_mix_lambda
        audio = np.clip((real_audio * lam) + (pseudo_audio * (1.0 - lam)), -1.0, 1.0).astype(np.float32)
        image = self.renderer(audio, train_mode=True)

        real_target = base_train.indices_to_multihot(real_row["label_indices"], self.num_classes)
        pseudo_target = self.pseudo_probs[pseudo_idx].astype(np.float32, copy=False)
        target = torch.from_numpy(((real_target * lam) + (pseudo_target * (1.0 - lam))).astype(np.float32))

        return {
            "image": image,
            "target": target,
            "loss_weight": torch.tensor(1.0, dtype=torch.float32),
            "row_id": real_row["row_id"],
            "site": real_row.get("site", ""),
            "source": "noisy_student",
        }


def build_mixed_sampler(df: pd.DataFrame, pseudo_probs: np.ndarray, num_classes: int, samples_per_epoch: int, pseudo_sampler_weight: float):
    class_counts = np.zeros(num_classes, dtype=np.float32)
    support_lists: List[np.ndarray] = []

    for _, row in df.iterrows():
        if row["source"] == "pseudo":
            support = np.flatnonzero(pseudo_probs[int(row["pseudo_idx"])] > 0).astype(np.int64)
        else:
            support = np.asarray(row["label_indices"], dtype=np.int64)
        support_lists.append(support)
        if len(support) > 0:
            class_counts[support] += 1.0

    weights = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        support = support_lists[row_idx]
        if len(support) > 0:
            weight = float(np.max(1.0 / np.sqrt(np.maximum(class_counts[support], 1.0))))
        else:
            weight = 0.05
        if row["source"] == "pseudo":
            weight *= float(pseudo_sampler_weight) * float(row["max_prob"])
        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=samples_per_epoch, replacement=True)


def build_real_sampler(df: pd.DataFrame, num_classes: int, samples_per_epoch: int):
    class_counts = np.zeros(num_classes, dtype=np.float32)
    support_lists: List[np.ndarray] = []

    for _, row in df.iterrows():
        support = np.asarray(row["label_indices"], dtype=np.int64)
        support_lists.append(support)
        if len(support) > 0:
            class_counts[support] += 1.0

    weights = []
    for support in support_lists:
        if len(support) > 0:
            weights.append(float(np.max(1.0 / np.sqrt(np.maximum(class_counts[support], 1.0)))))
        else:
            weights.append(0.05)

    weights = np.asarray(weights, dtype=np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=samples_per_epoch, replacement=True)


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = False, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def run_epoch_stage3(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    train_mode: bool,
    scaler: GradScaler,
    use_amp: bool,
    progress_desc: str,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
):
    model.train(train_mode)
    running_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []
    row_ids = []
    sites = []
    sources = []
    mixup_batches = 0
    cutmix_batches = 0

    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        desc=progress_desc,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        loss_weights = batch["loss_weight"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            images, targets, loss_weights, mix_info = base_train.apply_batch_mix_augmentation(
                images=images,
                targets=targets,
                mixup_alpha=mixup_alpha,
                mixup_prob=mixup_prob,
                cutmix_alpha=cutmix_alpha,
                cutmix_prob=cutmix_prob,
                sample_weights=loss_weights,
            )
            if mix_info is not None:
                if mix_info["mode"] == "mixup":
                    mixup_batches += 1
                elif mix_info["mode"] == "cutmix":
                    cutmix_batches += 1

        grad_context = torch.enable_grad() if train_mode else torch.inference_mode()
        with grad_context:
            with autocast(enabled=use_amp):
                logits = model(images)
                per_sample_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
                loss = (per_sample_loss * loss_weights).sum() / torch.clamp(loss_weights.sum(), min=1e-6)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{running_loss / max(sample_count, 1):.4f}")

        if not train_mode:
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
            row_ids.extend(batch["row_id"])
            sites.extend(batch["site"])
            sources.extend(batch["source"])

        del images, targets, loss_weights, logits, loss

    result = {
        "loss": running_loss / max(sample_count, 1),
        "y_true": None,
        "y_pred": None,
        "row_ids": row_ids,
        "sites": sites,
        "sources": sources,
        "mixup_batches": mixup_batches,
        "cutmix_batches": cutmix_batches,
    }
    if not train_mode and y_true:
        result["y_true"] = np.concatenate(y_true, axis=0)
        result["y_pred"] = np.concatenate(y_pred, axis=0)
    return result


def fit_stage3_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device,
    output_dir: Path,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    epochs: int,
    warmup_epochs: int,
    use_amp: bool,
    patience: int,
    freeze_backbone_epochs: int,
    mixup_alpha: float,
    mixup_prob: float,
    cutmix_alpha: float,
    cutmix_prob: float,
):
    optimizer = base_train.build_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    scheduler = base_train.build_scheduler(optimizer, steps_per_epoch=len(train_loader), epochs=epochs, warmup_epochs=warmup_epochs)
    scaler = GradScaler(enabled=use_amp)

    best_metric = -np.inf
    best_path = output_dir / "stage3_best.pth"
    history = []
    patience_left = patience

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0:
            base_train.set_backbone_trainable(model, trainable=epoch > freeze_backbone_epochs)

        train_result = run_epoch_stage3(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_mode=True,
            scaler=scaler,
            use_amp=use_amp,
            progress_desc=f"stage3/train/e{epoch:02d}",
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            cutmix_alpha=cutmix_alpha,
            cutmix_prob=cutmix_prob,
        )
        valid_result = run_epoch_stage3(
            model=model,
            loader=valid_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            train_mode=False,
            scaler=scaler,
            use_amp=use_amp,
            progress_desc=f"stage3/valid/e{epoch:02d}",
        )
        valid_metric = base_train.macro_auc_skip_missing(valid_result["y_true"], valid_result["y_pred"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "valid_loss": valid_result["loss"],
                "valid_auc": valid_metric,
            }
        )
        print(
            f"[stage3] epoch={epoch:02d} "
            f"train_loss={train_result['loss']:.4f} "
            f"valid_loss={valid_result['loss']:.4f} "
            f"valid_auc={valid_metric:.5f} "
            f"mixup_batches={train_result['mixup_batches']} "
            f"cutmix_batches={train_result['cutmix_batches']}"
        )
        if valid_metric > best_metric:
            best_metric = valid_metric
            patience_left = patience
            torch.save({"model": model.state_dict(), "history": history}, best_path)
            print(f"[stage3] saved best checkpoint -> {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[stage3] early stopping triggered.")
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    pd.DataFrame(history).to_csv(output_dir / "stage3_history.csv", index=False)
    return model


def evaluate_stage3(model: nn.Module, loader: DataLoader, device, use_amp: bool) -> pd.DataFrame:
    scaler = GradScaler(enabled=False)
    result = run_epoch_stage3(
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        device=device,
        train_mode=False,
        scaler=scaler,
        use_amp=use_amp,
        progress_desc="stage3_eval",
    )
    score = base_train.macro_auc_skip_missing(result["y_true"], result["y_pred"])
    print(f"[stage3-valid] fold_auc={score:.5f}")
    prediction_df = pd.DataFrame(result["y_pred"])
    prediction_df.insert(0, "row_id", result["row_ids"])
    prediction_df["site"] = result["sites"]
    prediction_df["fold_auc"] = score
    return prediction_df


def prepare_fold_datasets(
    cfg: Stage3Config,
    student_cfg: base_train.Config,
    input_dir: Path,
    labeled_df: pd.DataFrame,
    pseudo_root: Path,
    fold: int,
    renderer: base_train.SpectrogramRenderer,
    num_classes: int,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, np.ndarray, Path]:
    train_real_df = labeled_df[labeled_df["fold"] != fold].reset_index(drop=True)
    valid_df = labeled_df[labeled_df["fold"] == fold].reset_index(drop=True)
    labeled_filenames = set(labeled_df["filename"].astype(str).unique().tolist())
    pseudo_df, pseudo_probs, pseudo_fold_dir = load_pseudo_package(
        pseudo_root=pseudo_root,
        input_dir=input_dir,
        fold=fold,
        labeled_filenames=labeled_filenames,
        min_pseudo_max_prob=cfg.min_pseudo_max_prob,
        max_pseudo_rows=cfg.max_pseudo_rows,
        allow_global_pseudo=cfg.allow_global_pseudo,
    )
    pseudo_df["source"] = "pseudo"
    pseudo_df["pseudo_idx"] = np.arange(len(pseudo_df))

    train_real_df = train_real_df.copy()
    train_real_df["source"] = "real"
    train_real_df["pseudo_idx"] = -1
    train_real_df["max_prob"] = 1.0
    train_real_frame = train_real_df[["row_id", "site", "audio_path", "start_sec", "label_indices", "source", "pseudo_idx", "max_prob"]]
    pseudo_frame = pseudo_df[["row_id", "site", "audio_path", "start_sec", "source", "pseudo_idx", "max_prob"]].copy()

    valid_dataset = MixedSoundscapeDataset(
        df=valid_df.assign(source="real", pseudo_idx=-1, max_prob=1.0),
        pseudo_probs=None,
        cfg=student_cfg,
        renderer=renderer,
        num_classes=num_classes,
        train_mode=False,
        pseudo_loss_weight=cfg.pseudo_loss_weight,
    )

    if cfg.training_mode == "noisy-student":
        pseudo_frame["source"] = "pseudo_pool"
        train_df = pd.concat(
            [train_real_frame, pseudo_frame],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        train_df["label_indices"] = train_df["label_indices"].apply(lambda x: x if isinstance(x, list) else [])
        train_dataset = NoisyStudentSoundscapeDataset(
            real_df=train_real_df,
            pseudo_df=pseudo_df,
            pseudo_probs=pseudo_probs,
            cfg=student_cfg,
            renderer=renderer,
            num_classes=num_classes,
            pseudo_mix_lambda=cfg.noisy_student_lambda,
            pseudo_sample_power=cfg.noisy_student_pseudo_sample_power,
        )
        sampler = build_real_sampler(
            df=train_real_df,
            num_classes=num_classes,
            samples_per_epoch=cfg.stage3_samples_per_epoch,
        )
    else:
        train_df = pd.concat(
            [train_real_frame, pseudo_frame],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        train_df["label_indices"] = train_df["label_indices"].apply(lambda x: x if isinstance(x, list) else [])
        train_dataset = MixedSoundscapeDataset(
            df=train_df,
            pseudo_probs=pseudo_probs,
            cfg=student_cfg,
            renderer=renderer,
            num_classes=num_classes,
            train_mode=True,
            pseudo_loss_weight=cfg.pseudo_loss_weight,
        )
        sampler = build_mixed_sampler(
            df=train_df,
            pseudo_probs=pseudo_probs,
            num_classes=num_classes,
            samples_per_epoch=cfg.stage3_samples_per_epoch,
            pseudo_sampler_weight=cfg.pseudo_sampler_weight,
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.stage3_batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    return train_loader, valid_loader, valid_df, train_df, pseudo_probs, pseudo_fold_dir


def main() -> None:
    cfg = parse_args()
    base_train.require_training_dependencies()

    root = Path(cfg.root).resolve()
    input_dir = resolve_path(root, cfg.input_dir)
    student_run_dir = resolve_path(root, cfg.student_run_dir)
    pseudo_root = resolve_path(root, cfg.pseudo_root)
    output_root = resolve_path(root, cfg.output_dir)

    student_cfg = load_student_base_config(student_run_dir)
    base_train.seed_everything(student_cfg.seed)
    class_names = base_train.load_class_names(input_dir)
    num_classes = len(class_names)
    labeled_df = load_labeled_soundscape_df(student_cfg=student_cfg, input_dir=input_dir, student_run_dir=student_run_dir, class_names=class_names)
    labeled_df = attach_labeled_audio_paths(labeled_df, input_dir=input_dir)

    folds = cfg.folds or list(range(student_cfg.n_folds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{student_cfg.model_name.replace('/', '_')}_stage3_pseudo"
    base_train.ensure_dir(run_dir)
    run_config = asdict(cfg)
    for key in ["model_name", "sample_rate", "clip_seconds", "image_height", "image_width", "dropout", "drop_path", "head_type"]:
        run_config[key] = getattr(student_cfg, key)
    save_json(run_dir / "config.json", run_config)

    log_path = run_dir / "train.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with base_train.RunLogger(log_path):
        print(f"[INFO] Logging to {log_path}")
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Student run dir: {student_run_dir}")
        print(f"[INFO] Pseudo root: {pseudo_root}")
        print(f"[INFO] Folds: {folds}")
        print(f"[INFO] Training mode: {cfg.training_mode}")
        if cfg.training_mode == "noisy-student":
            print(
                f"[INFO] Noisy Student: lambda={cfg.noisy_student_lambda} "
                f"pseudo_sample_power={cfg.noisy_student_pseudo_sample_power}"
            )
        print(
            f"[INFO] Batch aug: mixup(alpha={cfg.mixup_alpha}, prob={cfg.mixup_prob}) | "
            f"cutmix(alpha={cfg.cutmix_alpha}, prob={cfg.cutmix_prob})"
        )

        renderer = base_train.SpectrogramRenderer(student_cfg)
        ckpt_dir = root / student_cfg.ckpt_dir
        backbone_weight_path = ckpt_dir / f"{student_cfg.model_name}.pth"

        oof_frames = []
        fold_scores = []
        source_summaries = []

        for fold in folds:
            print(f"[INFO] Stage 3 fold {fold + 1}/{student_cfg.n_folds}")
            fold_dir = run_dir / f"fold_{fold}"
            base_train.ensure_dir(fold_dir)

            train_loader, valid_loader, valid_df, train_df, _, pseudo_fold_dir = prepare_fold_datasets(
                cfg=cfg,
                student_cfg=student_cfg,
                input_dir=input_dir,
                labeled_df=labeled_df,
                pseudo_root=pseudo_root,
                fold=fold,
                renderer=renderer,
                num_classes=num_classes,
            )
            source_summary = (
                train_df["source"].value_counts().rename_axis("source").reset_index(name="rows")
            )
            source_summary["fold"] = fold
            source_summary["pseudo_dir"] = str(pseudo_fold_dir)
            source_summaries.append(source_summary)
            source_summary.to_csv(fold_dir / "train_source_summary.csv", index=False)

            model = base_train.BirdCLEFNet(cfg=student_cfg, num_classes=num_classes, backbone_weight_path=backbone_weight_path).to(device)
            checkpoint_path = student_run_dir / f"fold_{fold}" / f"stage2_fold{fold}_best.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)

            model = fit_stage3_fold(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device,
                output_dir=fold_dir,
                backbone_lr=cfg.stage3_backbone_lr,
                head_lr=cfg.stage3_head_lr,
                weight_decay=cfg.weight_decay,
                epochs=cfg.stage3_epochs,
                warmup_epochs=cfg.warmup_epochs,
                use_amp=cfg.use_amp and torch.cuda.is_available(),
                patience=cfg.patience,
                freeze_backbone_epochs=cfg.freeze_backbone_epochs,
                mixup_alpha=cfg.mixup_alpha,
                mixup_prob=cfg.mixup_prob,
                cutmix_alpha=cfg.cutmix_alpha,
                cutmix_prob=cfg.cutmix_prob,
            )

            prediction_df = evaluate_stage3(model, valid_loader, device=device, use_amp=cfg.use_amp and torch.cuda.is_available())
            prediction_df.insert(1, "fold", fold)
            prediction_df.to_csv(fold_dir / "valid_predictions.csv", index=False)
            fold_scores.append(float(prediction_df["fold_auc"].iloc[0]))

            prediction_df = prediction_df.rename(columns={i: class_names[i] for i in range(num_classes)})
            truth_df = valid_df[["row_id", "site", "label_indices"]].copy()
            truth_matrix = np.stack(valid_df["label_indices"].map(lambda x: base_train.indices_to_multihot(x, num_classes)).to_numpy())
            truth_frame = pd.DataFrame(truth_matrix, columns=[f"target_{label}" for label in class_names])
            truth_full = pd.concat([truth_df.reset_index(drop=True), truth_frame.reset_index(drop=True)], axis=1)
            pred_full = prediction_df[["row_id"] + class_names].copy()
            merged = truth_full.merge(pred_full, on="row_id", how="left", validate="one_to_one")
            if merged[class_names].isna().any().any():
                missing = merged.loc[merged[class_names].isna().any(axis=1), "row_id"].head(10).tolist()
                raise RuntimeError(f"Missing predictions for validation rows: {missing}")
            oof_frames.append(merged)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
        y_true = oof_df[[f"target_{label}" for label in class_names]].to_numpy(dtype=np.float32)
        y_pred = oof_df[class_names].to_numpy(dtype=np.float32)
        final_cv = base_train.macro_auc_skip_missing(y_true, y_pred)
        print(f"[INFO] Final Stage3 OOF local CV = {final_cv:.6f}")

        oof_df.to_csv(run_dir / "soundscape_oof_predictions.csv", index=False)
        pd.DataFrame({"fold": folds, "fold_auc": fold_scores}).to_csv(run_dir / "fold_scores.csv", index=False)
        pd.concat(source_summaries, axis=0, ignore_index=True).to_csv(run_dir / "train_source_summary.csv", index=False)
        save_json(
            run_dir / "metrics.json",
            {
                "final_oof_cv": final_cv,
                "fold_scores": fold_scores,
                "student_run_dir": str(student_run_dir),
                "pseudo_root": str(pseudo_root),
                "folds": folds,
            },
        )
        print(f"[INFO] Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
