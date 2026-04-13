#!/usr/bin/env python3
"""
Huấn luyện HybridFallTransformer — chia val theo nhóm (groups.npy) nếu có;
metrics đầy đủ; lưu ngưỡng tối ưu trên validation.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import FEATURE_DIM, SEQ_LEN


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def stratified_train_val_indices(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_flat = np.asarray(y).reshape(-1)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for c in np.unique(y_flat):
        idx = np.where(y_flat == c)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            train_parts.append(idx)
            continue
        n_val = max(1, int(round(len(idx) * val_ratio)))
        if n_val >= len(idx):
            n_val = len(idx) - 1
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ưu tiên GroupShuffleSplit khi có groups.npy (cùng subject không lệch train/val)."""
    y_flat = np.asarray(y).reshape(-1)
    if groups is not None and len(groups) == len(X):
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            train_idx, val_idx = next(gss.split(X, y_flat, groups))
            return train_idx, val_idx
        except ValueError:
            print("[warn] GroupShuffleSplit thất bại — dùng stratified theo nhãn.")
    return stratified_train_val_indices(y, val_ratio=val_ratio, seed=seed)


@torch.no_grad()
def collect_val_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_l: list[np.ndarray] = []
    y_l: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        logits_l.append(logits)
        y_l.append(yb.numpy())
    return np.concatenate(logits_l, axis=0).ravel(), np.concatenate(y_l, axis=0).ravel()


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    pred = (probs >= thr).astype(np.int32)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def print_classification_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    thr: float,
) -> None:
    y_true = np.asarray(y_true).astype(np.int32)
    pred = (probs >= thr).astype(np.int32)
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_true, pred))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, probs):.4f}")
    except ValueError:
        print("ROC-AUC: n/a (một lớp trên val)")
    try:
        print(f"PR-AUC (avg precision): {average_precision_score(y_true, probs):.4f}")
    except ValueError:
        print("PR-AUC: n/a")


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Quét ngưỡng trên val để tối đa F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = float(f1_score(y_true, (probs >= t).astype(np.int32), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HybridFallTransformer")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("best_hybrid_transformer.pth"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_path = args.data_dir / "X_train.npy"
    y_path = args.data_dir / "y_train.npy"
    g_path = args.data_dir / "groups.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise SystemExit(f"Thiếu {x_path} hoặc {y_path}. Chạy data_extractor.py trước.")

    X = np.load(x_path)
    y = np.load(y_path)
    groups: np.ndarray | None = None
    if g_path.is_file():
        groups = np.load(g_path, allow_pickle=True)

    if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, FEATURE_DIM):
        raise SystemExit(
            f"Cần X.shape == (N, {SEQ_LEN}, {FEATURE_DIM}), nhận {X.shape}"
        )
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    train_idx, val_idx = split_train_val(
        X, y, groups, val_ratio=args.val_ratio, seed=args.seed
    )
    if len(val_idx) == 0:
        raise SystemExit("Tập validation rỗng.")

    full_ds = SequenceDataset(X, y)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridFallTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            train_loss += loss.item() * bs
            n_tr += bs
        train_loss /= max(n_tr, 1)

        val_logits, val_y = collect_val_predictions(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_f1 = float(
            f1_score(val_y, (val_probs >= 0.5).astype(np.int32), zero_division=0)
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_f1@0.5={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping (patience={args.patience}), "
                    f"best val_f1@0.5={best_f1:.4f}"
                )
                break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    val_logits, val_y = collect_val_predictions(model, val_loader, device)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    best_thr, best_f1_thr = tune_threshold(val_y, val_probs)

    print("\n--- Validation (best weights, sau tinh chỉnh ngưỡng) ---")
    print(f"best_threshold={best_thr:.4f}  val_f1={best_f1_thr:.4f}")
    print_classification_report(val_y, val_probs, best_thr)
    m = metrics_at_threshold(val_y, val_probs, best_thr)
    print(f"precision={m['precision']:.4f} recall={m['recall']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "best_val_f1": best_f1,
            "best_val_f1_tuned": best_f1_thr,
            "best_threshold": best_thr,
            "d_model": 256,
            "seq_len": SEQ_LEN,
            "in_features": FEATURE_DIM,
        },
        args.out,
    )
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
