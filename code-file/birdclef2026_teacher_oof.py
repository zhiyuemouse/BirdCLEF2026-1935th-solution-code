from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_teacher_oof_predictions(
    teacher_path: str | Path,
    row_ids: Sequence[str],
    class_names: Sequence[str] | None = None,
    expected_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Load teacher OOF probabilities and align them to ``row_ids``.

    Supported formats:
    - ``.npz`` with ``row_id`` and either ``pred`` or ``oof_pred``.
    - ``.csv`` with ``row_id`` and columns listed by ``class_names``.

    If ``expected_targets`` is provided and the npz contains ``y_true``, the
    targets are checked after row alignment.  This catches accidental use of a
    teacher file from a different fold split or label ordering.
    """

    path = Path(teacher_path)
    target_row_ids = np.asarray([str(row_id) for row_id in row_ids], dtype=object)
    if not path.exists():
        raise FileNotFoundError(f"Teacher OOF path does not exist: {path}")

    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=True)
        if "row_id" not in npz:
            raise KeyError(f"{path} must contain row_id")
        pred_key = (
            "pred"
            if "pred" in npz
            else "oof_pred"
            if "oof_pred" in npz
            else "pred_oof_like"
            if "pred_oof_like" in npz
            else ""
        )
        if not pred_key:
            raise KeyError(f"{path} must contain pred or oof_pred")
        source_row_ids = npz["row_id"].astype(str)
        source_pred = npz[pred_key].astype(np.float32, copy=False)
        source_targets = npz["y_true"].astype(np.float32, copy=False) if "y_true" in npz else None
    elif path.suffix.lower() == ".csv":
        if class_names is None:
            raise ValueError("class_names are required when loading teacher OOF from csv")
        df = pd.read_csv(path)
        missing = [col for col in ["row_id", *class_names] if col not in df.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing[:10]}")
        source_row_ids = df["row_id"].astype(str).to_numpy()
        source_pred = df[list(class_names)].to_numpy(dtype=np.float32)
        source_targets = None
    else:
        raise ValueError(f"Unsupported teacher OOF suffix: {path.suffix}")

    order = pd.DataFrame({"row_id": target_row_ids}).merge(
        pd.DataFrame({"row_id": source_row_ids, "_pos": np.arange(len(source_row_ids), dtype=np.int64)}),
        on="row_id",
        how="left",
        validate="one_to_one",
    )["_pos"]
    if order.isna().any():
        examples = pd.Series(target_row_ids)[order.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Teacher OOF misses {order.isna().sum()} rows. Examples: {examples}")

    order_arr = order.to_numpy(dtype=np.int64)
    aligned_pred = np.clip(source_pred[order_arr].astype(np.float32, copy=False), 0.0, 1.0)
    if expected_targets is not None and source_targets is not None:
        aligned_targets = source_targets[order_arr]
        if not np.array_equal(aligned_targets.astype(np.float32), expected_targets.astype(np.float32)):
            raise ValueError("Teacher OOF y_true does not match expected targets after row_id alignment.")
    return aligned_pred


def load_teacher_predictions_for_fold(
    teacher_path: str | Path,
    row_ids: Sequence[str],
    fold: int,
    class_names: Sequence[str] | None = None,
    expected_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Load teacher probabilities for a specific outer student fold.

    ``load_teacher_oof_predictions`` is safe per row, but a student fold can
    still be indirectly leaked if its training rows use teacher folds that saw
    the current validation fold.  Strict teacher packages store
    ``pred_by_fold[student_fold, row, class]`` so fold ``k`` always uses a
    teacher model that did not train on fold ``k`` labels.
    """

    path = Path(teacher_path)
    if path.suffix.lower() != ".npz":
        return load_teacher_oof_predictions(
            teacher_path=path,
            row_ids=row_ids,
            class_names=class_names,
            expected_targets=expected_targets,
        )

    npz = np.load(path, allow_pickle=True)
    if "pred_by_fold" not in npz:
        return load_teacher_oof_predictions(
            teacher_path=path,
            row_ids=row_ids,
            class_names=class_names,
            expected_targets=expected_targets,
        )
    if "row_id" not in npz:
        raise KeyError(f"{path} must contain row_id")

    pred_by_fold = npz["pred_by_fold"].astype(np.float32, copy=False)
    if pred_by_fold.ndim != 3:
        raise ValueError(f"{path} pred_by_fold must have shape [fold,row,class], got {pred_by_fold.shape}")
    fold = int(fold)
    if fold < 0 or fold >= pred_by_fold.shape[0]:
        raise ValueError(f"Requested fold {fold}, but {path} has {pred_by_fold.shape[0]} folds")

    source_row_ids = npz["row_id"].astype(str)
    target_row_ids = np.asarray([str(row_id) for row_id in row_ids], dtype=object)
    order = pd.DataFrame({"row_id": target_row_ids}).merge(
        pd.DataFrame({"row_id": source_row_ids, "_pos": np.arange(len(source_row_ids), dtype=np.int64)}),
        on="row_id",
        how="left",
        validate="one_to_one",
    )["_pos"]
    if order.isna().any():
        examples = pd.Series(target_row_ids)[order.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Teacher fold package misses {order.isna().sum()} rows. Examples: {examples}")

    order_arr = order.to_numpy(dtype=np.int64)
    aligned_pred = np.clip(pred_by_fold[fold, order_arr].astype(np.float32, copy=False), 0.0, 1.0)
    if expected_targets is not None and "y_true" in npz:
        aligned_targets = npz["y_true"].astype(np.float32, copy=False)[order_arr]
        if not np.array_equal(aligned_targets.astype(np.float32), expected_targets.astype(np.float32)):
            raise ValueError("Teacher fold package y_true does not match expected targets after row_id alignment.")
    return aligned_pred
