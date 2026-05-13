#!/usr/bin/env python3
"""
Huấn luyện HybridFallTransformer — chia val theo nhóm (groups.npy) nếu có;
metrics đầy đủ; lưu ngưỡng tối ưu trên validation.

Features:
    - Data Augmentation: Temporal Shift, Gaussian Noise, Horizontal Flip
    - Validation Loop: F1-Score calculation with tqdm progress bar
    - Early Stopping: Based on validation F1-score (patience=25)
    - Optimizer: AdamW with weight_decay=1e-5
    - Loss: BCEWithLogitsLoss

Based on methodology from:
    - Benabdennour et al. (2026) - IEEE Access Fall Detection
    - PLOS ONE Fall Detection Studies (2026)
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
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import FEATURE_DIM, SEQ_LEN
from scripts.augmentation import SequenceAugmenter


class SequenceDataset(Dataset):
    """PyTorch Dataset with optional augmentation for training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        augmenter: SequenceAugmenter | None = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            X: Features array shape (N, seq_len, feature_dim).
            y: Labels array shape (N, 1) or (N,).
            augment: Whether to apply augmentation during training.
            augmenter: SequenceAugmenter instance (required if augment=True).
        """
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

        self.augment = augment
        self.augmenter = augmenter or SequenceAugmenter(seed=None)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item with optional augmentation.

        Augmentation is ONLY applied during training (when self.augment=True).
        Validation set should have augment=False.
        """
        x = self.X[i].numpy().copy()
        y = self.y[i]

        # Apply augmentation ONLY during training
        if self.augment:
            x = self.augmenter.apply(x)

        return torch.from_numpy(x).float(), y


class ValidationDataset(Dataset):
    """Dataset wrapper for validation (no augmentation)."""

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
    """Collect all validation predictions and labels."""
    model.eval()
    logits_l: list[np.ndarray] = []
    y_l: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        logits_l.append(logits)
        y_l.append(yb.numpy())
    return np.concatenate(logits_l, axis=0).ravel(), np.concatenate(y_l, axis=0).ravel()


def compute_val_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute comprehensive validation metrics.

    Returns dict with:
        - loss: BCE loss
        - f1: F1-score at threshold 0.5
        - precision: Precision at threshold 0.5
        - recall: Recall at threshold 0.5
        - roc_auc: ROC-AUC score
        - pr_auc: Average precision score
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_logits = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)
    all_logits = np.concatenate(all_logits, axis=0).ravel()
    all_labels = np.concatenate(all_labels, axis=0).ravel()
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(np.int32)

    metrics = {
        "loss": avg_loss,
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, probs))
    except ValueError:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = float(average_precision_score(all_labels, probs))
    except ValueError:
        metrics["pr_auc"] = 0.0

    return metrics


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


def resolve_device(device_arg: str) -> torch.device:
    """
    Chọn thiết bị train an toàn.
    - cpu: ép CPU
    - auto: thử CUDA trước, nếu không chạy được thì fallback CPU
    - các giá trị khác: chuyển thẳng cho torch.device
    """
    d = (device_arg or "auto").strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "auto":
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception as e:
                print(f"[warn] CUDA không dùng được ({e}). Fallback CPU.")
                return torch.device("cpu")
        return torch.device("cpu")
    try:
        return torch.device(d)
    except Exception:
        print(f"[warn] device '{device_arg}' không hợp lệ. Dùng CPU.")
        return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HybridFallTransformer with Augmentation and Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_transformer.py --data-dir data/processed --epochs 100 --batch-size 64
  python train_transformer.py --augment --aug-noise-prob 0.5 --aug-hflip-prob 0.5
  python train_transformer.py --device cuda --epochs 200 --patience 30
        """,
    )
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Thiết bị train: "auto", "cpu", "cuda", "cuda:0"...',
    )

    # Augmentation arguments
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation during training",
    )
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=None,
        help="Random seed for augmentation",
    )
    parser.add_argument(
        "--aug-temporal-shift-prob",
        type=float,
        default=0.5,
        help="Probability of temporal shift [0, 1]",
    )
    parser.add_argument(
        "--aug-noise-prob",
        type=float,
        default=0.5,
        help="Probability of Gaussian noise [0, 1]",
    )
    parser.add_argument(
        "--aug-hflip-prob",
        type=float,
        default=0.5,
        help="Probability of horizontal flip [0, 1]",
    )
    parser.add_argument(
        "--aug-temporal-shift-max",
        type=int,
        default=5,
        help="Max frames to shift ±",
    )
    parser.add_argument(
        "--aug-noise-sigma",
        type=float,
        default=0.01,
        help="Gaussian noise std dev (recommended: 0.01 per PLOS ONE 2026)",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─── Load Data ───
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

    print(f"\n{'='*60}")
    print(f"DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Fall samples (train): {y[train_idx].sum():.0f}")
    print(f"No-Fall samples (train): {len(train_idx) - y[train_idx].sum():.0f}")

    # ─── Initialize Augmenter ───
    augmenter = None
    if args.augment:
        augmenter = SequenceAugmenter(
            temporal_shift_prob=args.aug_temporal_shift_prob,
            noise_prob=args.aug_noise_prob,
            hflip_prob=args.aug_hflip_prob,
            temporal_shift_max=args.aug_temporal_shift_max,
            noise_sigma=args.aug_noise_sigma,
            seed=args.aug_seed,
        )
        print(f"\n{'='*60}")
        print(f"AUGMENTATION ENABLED")
        print(f"{'='*60}")
        print(f"  Temporal Shift: prob={args.aug_temporal_shift_prob}, max_shift=±{args.aug_temporal_shift_max}")
        print(f"  Gaussian Noise: prob={args.aug_noise_prob}, sigma={args.aug_noise_sigma}")
        print(f"  Horizontal Flip: prob={args.aug_hflip_prob}")
        print(f"  Augmenter: {augmenter}")

    # ─── Create Datasets ───
    # Training dataset WITH augmentation
    train_ds = SequenceDataset(
        X[train_idx],
        y[train_idx],
        augment=args.augment,
        augmenter=augmenter,
    )
    # Validation dataset WITHOUT augmentation
    val_ds = ValidationDataset(X[val_idx], y[val_idx])

    # ─── Setup Device, Model, Optimizer ───
    device = resolve_device(args.device)
    print(f"\n[info] Train device: {device}")

    model = HybridFallTransformer().to(device)

    # AdamW optimizer as per Benabdennour et al. (2026)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # BCEWithLogitsLoss as validated by Benabdennour et al. (2026)
    criterion = nn.BCEWithLogitsLoss()

    # ─── DataLoaders ───
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ─── Training Loop ───
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Loss: BCEWithLogitsLoss")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Early Stopping: patience={args.patience}")
    print(f"  Data Augmentation: {args.augment}")
    print(f"\n{'='*60}")
    print(f"TRAINING LOG")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val F1':>8} | {'ROC-AUC':>8} | {'Best F1':>8} | Status")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        # ─── Training Phase ───
        model.train()
        train_loss = 0.0
        n_tr = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{args.epochs} [Train]",
            leave=False,
            disable=True,  # Disable inner progress to avoid clutter
        )

        for xb, yb in train_pbar:
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

        # ─── Validation Phase ───
        val_metrics = compute_val_metrics(model, val_loader, device)

        # Check for improvement
        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            status = "★ SAVE"
        else:
            epochs_no_improve += 1
            status = ""

        # Print epoch summary
        print(
            f"{epoch:>5} | {train_loss:>12.4f} | {val_metrics['loss']:>12.4f} | "
            f"{val_metrics['f1']:>8.4f} | {val_metrics['roc_auc']:>8.4f} | "
            f"{best_f1:>8.4f} | {status}"
        )

        # ─── Early Stopping ───
        if epochs_no_improve >= args.patience:
            print(
                f"\n{'='*60}"
                f"\n[EARLY STOPPING] No improvement for {args.patience} epochs."
                f"\n[INFO] Best validation F1: {best_f1:.4f}"
                f"\n{'='*60}"
            )
            break

    # ─── Final Evaluation ───
    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (Best Model)")
    print(f"{'='*60}")

    # Get final validation predictions
    val_logits, val_y = collect_val_predictions(model, val_loader, device)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    # Tune threshold for optimal F1
    best_thr, best_f1_thr = tune_threshold(val_y, val_probs)

    print(f"best_threshold={best_thr:.4f}  val_f1={best_f1_thr:.4f}")
    print_classification_report(val_y, val_probs, best_thr)

    m = metrics_at_threshold(val_y, val_probs, best_thr)
    print(f"precision={m['precision']:.4f} recall={m['recall']:.4f}")

    # ─── Save Model ───
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
            "training_config": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs_trained": epoch,
                "augmentation_enabled": args.augment,
            },
        },
        args.out,
    )
    print(f"\n[INFO] Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
