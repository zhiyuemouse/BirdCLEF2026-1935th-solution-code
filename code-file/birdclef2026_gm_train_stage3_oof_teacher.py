from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import birdclef2026_gm_train as base_train
from birdclef2026_teacher_oof import load_teacher_oof_predictions, load_teacher_predictions_for_fold


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
class Stage3OOFTeacherConfig:
    root: str = "."
    input_dir: str = "input"
    output_dir: str = "outputs/birdclef2026_gm_stage3_oof_teacher"
    student_run_dir: str = ""
    teacher_oof_path: str = ""
    teacher_loss_weight: float = 0.25
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
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.0
    folds: Optional[List[int]] = None
    smoke_test: bool = False
    use_amp: bool = True


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


def parse_args() -> Stage3OOFTeacherConfig:
    parser = argparse.ArgumentParser(description="Stage3 fold-safe OOF-teacher distillation for BirdCLEF 2026.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="input")
    parser.add_argument("--output-dir", type=str, default="outputs/birdclef2026_gm_stage3_oof_teacher")
    parser.add_argument("--student-run-dir", type=str, required=True)
    parser.add_argument("--teacher-oof-path", type=str, required=True)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.25)
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
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    cfg = Stage3OOFTeacherConfig(
        root=args.root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        student_run_dir=args.student_run_dir,
        teacher_oof_path=args.teacher_oof_path,
        teacher_loss_weight=args.teacher_loss_weight,
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
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        folds=parse_int_list(args.folds),
        smoke_test=args.smoke_test,
        use_amp=not args.disable_amp,
    )
    if cfg.smoke_test:
        cfg.stage3_epochs = 1
        cfg.stage3_samples_per_epoch = min(cfg.stage3_samples_per_epoch, 256)
        cfg.num_workers = min(cfg.num_workers, 2)
        cfg.patience = 1
    if cfg.teacher_loss_weight < 0:
        raise ValueError("--teacher-loss-weight must be non-negative.")
    if cfg.teacher_loss_weight > 0 and any(
        value > 0 for value in [cfg.mixup_alpha, cfg.mixup_prob, cfg.cutmix_alpha, cfg.cutmix_prob]
    ):
        raise ValueError(
            "Batch mixup/cutmix is disabled for OOF-teacher distillation so hard and soft targets stay aligned."
        )
    return cfg


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


def parse_label_indices(value) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    text = text.strip("[]")
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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
    else:
        df = base_train.load_soundscape_segments(student_cfg, input_dir=input_dir, label_to_idx=label_to_idx)
        df = base_train.build_soundscape_folds(df, num_classes=len(class_names), n_folds=student_cfg.n_folds, seed=student_cfg.seed)
    output = df.copy()
    output["audio_path"] = output["filename"].map(lambda x: str((input_dir / "train_soundscapes" / x).resolve()))
    return output


class OOFTeacherSoundscapeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        teacher_probs: Optional[np.ndarray],
        cfg: base_train.Config,
        renderer: base_train.SpectrogramRenderer,
        num_classes: int,
        train_mode: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.teacher_probs = None if teacher_probs is None else np.asarray(teacher_probs, dtype=np.float32)
        self.cfg = cfg
        self.renderer = renderer
        self.num_classes = num_classes
        self.train_mode = train_mode
        if self.teacher_probs is not None and len(self.teacher_probs) != len(self.df):
            raise ValueError("teacher_probs must have the same row count as df")

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
        target = torch.from_numpy(base_train.indices_to_multihot(row["label_indices"], self.num_classes)).float()
        if self.teacher_probs is None:
            teacher_target = torch.zeros(self.num_classes, dtype=torch.float32)
            has_teacher = torch.tensor(False, dtype=torch.bool)
        else:
            teacher_target = torch.from_numpy(self.teacher_probs[idx].astype(np.float32, copy=False)).float()
            has_teacher = torch.tensor(True, dtype=torch.bool)
        item = {
            "image": image,
            "target": target,
            "teacher_target": teacher_target,
            "has_teacher": has_teacher,
            "row_id": str(row["row_id"]),
            "site": str(row.get("site", "")),
        }
        if self.cfg.use_waveform_branch:
            item["waveform"] = torch.from_numpy(audio).float()
        return item


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
    return WeightedRandomSampler(
        weights=torch.from_numpy(np.asarray(weights, dtype=np.float64)),
        num_samples=samples_per_epoch,
        replacement=True,
    )


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


def run_epoch_stage3_oof_teacher(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    train_mode: bool,
    scaler: GradScaler,
    use_amp: bool,
    progress_desc: str,
    teacher_loss_weight: float,
):
    model.train(train_mode)
    running_loss = 0.0
    running_hard_loss = 0.0
    running_teacher_loss = 0.0
    sample_count = 0
    y_true = []
    y_pred = []
    row_ids = []
    sites = []

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
        waveforms = batch["waveform"].to(device, non_blocking=True) if "waveform" in batch else None
        targets = batch["target"].to(device, non_blocking=True)
        teacher_targets = batch["teacher_target"].to(device, non_blocking=True)
        has_teacher = batch["has_teacher"].to(device, non_blocking=True).bool()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if train_mode else torch.inference_mode()
        with grad_context:
            with autocast(enabled=use_amp):
                outputs = model(images, waveform=waveforms)
                hard_loss = base_train.compute_training_loss(
                    outputs=outputs,
                    targets=targets,
                    criterion=nn.BCEWithLogitsLoss(),
                    target_mask=None,
                    global_loss_weight=0.0,
                    sed_frame_loss_weight=0.5,
                )
                loss = hard_loss
                teacher_loss = torch.zeros((), dtype=hard_loss.dtype, device=hard_loss.device)
                if train_mode and teacher_loss_weight > 0 and bool(has_teacher.any().item()):
                    metric_logits = base_train.model_metric_logits(outputs)
                    if metric_logits.ndim != 2:
                        raise ValueError(f"OOF teacher loss expects [B,C] logits, got {tuple(metric_logits.shape)}")
                    teacher_loss = F.binary_cross_entropy_with_logits(
                        metric_logits[has_teacher],
                        teacher_targets[has_teacher],
                    )
                    loss = hard_loss + (float(teacher_loss_weight) * teacher_loss)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        running_hard_loss += float(hard_loss.item()) * batch_size
        running_teacher_loss += float(teacher_loss.item()) * batch_size
        sample_count += batch_size
        progress.set_postfix(
            loss=f"{running_loss / max(sample_count, 1):.4f}",
            hard=f"{running_hard_loss / max(sample_count, 1):.4f}",
            teacher=f"{running_teacher_loss / max(sample_count, 1):.4f}",
        )

        if not train_mode:
            metric_logits = base_train.model_metric_logits(outputs)
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(torch.sigmoid(metric_logits.float()).detach().cpu().numpy())
            row_ids.extend(batch["row_id"])
            sites.extend(batch["site"])

        del images, waveforms, targets, teacher_targets, has_teacher, outputs, hard_loss, teacher_loss, loss

    result = {
        "loss": running_loss / max(sample_count, 1),
        "hard_loss": running_hard_loss / max(sample_count, 1),
        "teacher_loss": running_teacher_loss / max(sample_count, 1),
        "y_true": None,
        "y_pred": None,
        "row_ids": row_ids,
        "sites": sites,
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
    teacher_loss_weight: float,
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

        train_result = run_epoch_stage3_oof_teacher(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_mode=True,
            scaler=scaler,
            use_amp=use_amp,
            progress_desc=f"stage3_oof_teacher/train/e{epoch:02d}",
            teacher_loss_weight=teacher_loss_weight,
        )
        valid_result = run_epoch_stage3_oof_teacher(
            model=model,
            loader=valid_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            train_mode=False,
            scaler=scaler,
            use_amp=use_amp,
            progress_desc=f"stage3_oof_teacher/valid/e{epoch:02d}",
            teacher_loss_weight=0.0,
        )
        valid_metric = base_train.macro_auc_skip_missing(valid_result["y_true"], valid_result["y_pred"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_hard_loss": train_result["hard_loss"],
                "train_teacher_loss": train_result["teacher_loss"],
                "valid_loss": valid_result["loss"],
                "valid_auc": valid_metric,
            }
        )
        print(
            f"[stage3_oof_teacher] epoch={epoch:02d} "
            f"train_loss={train_result['loss']:.4f} "
            f"hard={train_result['hard_loss']:.4f} "
            f"teacher={train_result['teacher_loss']:.4f} "
            f"valid_loss={valid_result['loss']:.4f} "
            f"valid_auc={valid_metric:.5f}"
        )
        if valid_metric > best_metric:
            best_metric = valid_metric
            patience_left = patience
            torch.save({"model": model.state_dict(), "history": history}, best_path)
            print(f"[stage3_oof_teacher] saved best checkpoint -> {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[stage3_oof_teacher] early stopping triggered.")
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    pd.DataFrame(history).to_csv(output_dir / "stage3_history.csv", index=False)
    return model


def evaluate_stage3(model: nn.Module, loader: DataLoader, device, use_amp: bool) -> pd.DataFrame:
    scaler = GradScaler(enabled=False)
    result = run_epoch_stage3_oof_teacher(
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        device=device,
        train_mode=False,
        scaler=scaler,
        use_amp=use_amp,
        progress_desc="stage3_oof_teacher_eval",
        teacher_loss_weight=0.0,
    )
    score = base_train.macro_auc_skip_missing(result["y_true"], result["y_pred"])
    print(f"[stage3-valid] fold_auc={score:.5f}")
    prediction_df = pd.DataFrame(result["y_pred"])
    prediction_df.insert(0, "row_id", result["row_ids"])
    prediction_df["site"] = result["sites"]
    prediction_df["fold_auc"] = score
    return prediction_df


def prepare_fold_datasets(
    cfg: Stage3OOFTeacherConfig,
    student_cfg: base_train.Config,
    labeled_df: pd.DataFrame,
    teacher_probs: np.ndarray,
    fold: int,
    renderer: base_train.SpectrogramRenderer,
    num_classes: int,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    train_mask = (labeled_df["fold"] != fold).to_numpy()
    valid_mask = (labeled_df["fold"] == fold).to_numpy()
    train_df = labeled_df.loc[train_mask].reset_index(drop=True)
    valid_df = labeled_df.loc[valid_mask].reset_index(drop=True)
    train_teacher_probs = teacher_probs[train_mask]

    train_dataset = OOFTeacherSoundscapeDataset(
        df=train_df,
        teacher_probs=train_teacher_probs,
        cfg=student_cfg,
        renderer=renderer,
        num_classes=num_classes,
        train_mode=True,
    )
    valid_dataset = OOFTeacherSoundscapeDataset(
        df=valid_df,
        teacher_probs=None,
        cfg=student_cfg,
        renderer=renderer,
        num_classes=num_classes,
        train_mode=False,
    )
    sampler = build_real_sampler(train_df, num_classes=num_classes, samples_per_epoch=cfg.stage3_samples_per_epoch)
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
    return train_loader, valid_loader, valid_df, train_df


def main() -> None:
    cfg = parse_args()
    base_train.require_training_dependencies()

    root = Path(cfg.root).resolve()
    input_dir = resolve_path(root, cfg.input_dir)
    student_run_dir = resolve_path(root, cfg.student_run_dir)
    teacher_oof_path = resolve_path(root, cfg.teacher_oof_path)
    output_root = resolve_path(root, cfg.output_dir)

    student_cfg = load_student_base_config(student_run_dir)
    base_train.seed_everything(student_cfg.seed)
    class_names = base_train.load_class_names(input_dir)
    num_classes = len(class_names)
    labeled_df = load_labeled_soundscape_df(
        student_cfg=student_cfg,
        input_dir=input_dir,
        student_run_dir=student_run_dir,
        class_names=class_names,
    )
    expected_targets = np.stack(
        labeled_df["label_indices"].map(lambda x: base_train.indices_to_multihot(x, num_classes)).to_numpy()
    ).astype(np.float32)
    teacher_probs = load_teacher_oof_predictions(
        teacher_path=teacher_oof_path,
        row_ids=labeled_df["row_id"].astype(str).tolist(),
        class_names=class_names,
        expected_targets=expected_targets,
    )
    if teacher_oof_path.suffix.lower() == ".npz" and "pred_by_fold" in np.load(teacher_oof_path, allow_pickle=True).files:
        print("[INFO] Teacher package contains pred_by_fold; each stage3 fold will use its matching strict teacher slice.")

    folds = cfg.folds or list(range(student_cfg.n_folds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{student_cfg.model_name.replace('/', '_')}_stage3_oof_teacher_w{cfg.teacher_loss_weight:g}"
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
        print(f"[INFO] Teacher OOF path: {teacher_oof_path}")
        print(f"[INFO] Teacher OOF shape: {teacher_probs.shape}")
        print(f"[INFO] Teacher loss weight: {cfg.teacher_loss_weight}")
        print(f"[INFO] Folds: {folds}")
        print("[INFO] Leakage policy: train rows use aligned OOF teacher only; validation loss/CV use hard labels only.")

        renderer = base_train.SpectrogramRenderer(student_cfg)
        ckpt_dir = root / student_cfg.ckpt_dir
        backbone_weight_path = ckpt_dir / f"{student_cfg.model_name}.pth"

        oof_frames = []
        fold_scores = []
        source_summaries = []

        for fold in folds:
            print(f"[INFO] Stage3 OOF-teacher fold {fold + 1}/{student_cfg.n_folds}")
            fold_dir = run_dir / f"fold_{fold}"
            base_train.ensure_dir(fold_dir)

            train_loader, valid_loader, valid_df, train_df = prepare_fold_datasets(
                cfg=cfg,
                student_cfg=student_cfg,
                labeled_df=labeled_df,
                teacher_probs=load_teacher_predictions_for_fold(
                    teacher_path=teacher_oof_path,
                    row_ids=labeled_df["row_id"].astype(str).tolist(),
                    fold=fold,
                    class_names=class_names,
                    expected_targets=expected_targets,
                ),
                fold=fold,
                renderer=renderer,
                num_classes=num_classes,
            )
            source_summary = pd.DataFrame(
                {
                    "fold": [fold],
                    "train_real_rows": [int(len(train_df))],
                    "valid_rows": [int(len(valid_df))],
                    "teacher_rows": [int(len(train_df))],
                }
            )
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
                teacher_loss_weight=cfg.teacher_loss_weight,
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
        print(f"[INFO] Final Stage3 OOF-teacher local CV = {final_cv:.6f}")

        oof_df.to_csv(run_dir / "soundscape_oof_predictions.csv", index=False)
        pd.DataFrame({"fold": folds, "fold_auc": fold_scores}).to_csv(run_dir / "fold_scores.csv", index=False)
        pd.concat(source_summaries, axis=0, ignore_index=True).to_csv(run_dir / "train_source_summary.csv", index=False)
        save_json(
            run_dir / "metrics.json",
            {
                "final_oof_cv": final_cv,
                "fold_scores": fold_scores,
                "student_run_dir": str(student_run_dir),
                "teacher_oof_path": str(teacher_oof_path),
                "teacher_loss_weight": cfg.teacher_loss_weight,
                "folds": folds,
            },
        )
        print(f"[INFO] Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
