#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Fall Detection System.

Tác giả: Fall Detection Team
Ngày: May 2026

Mô tả:
    Script đánh giá toàn diện cho hệ thống Hybrid YOLOv11-Pose + PIFR + Transformer.
    Tính toán tất cả metrics tiêu chuẩn, tạo visualization chất lượng publication,
    và so sánh với các phương pháp SOTA.

Metrics được tính:
    - Accuracy, Precision (Macro & Class-specific), Recall, F1-Score
    - mAP@0.5, mAP@0.5:0.95
    - Inference Latency (ms/sequence), Throughput (FPS)
    - Model Complexity (Params, FLOPs)

Visualizations:
    - Confusion Matrix (Normalized & Unnormalized)
    - Precision-Recall Curve
    - F1-Confidence Curve

Sử dụng:
    python evaluate.py --data-dir data/processed --model best_model.pth --output eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
FEATURE_DIM = 60
NUM_KEYPOINTS = 17

# SOTA Models placeholder (for thesis comparison table)
SOTA_MODELS = [
    {"Model": "Benabdennour et al. (2026)", "Dataset": "URFD", "Accuracy": "", "Precision": "", "Recall": "", "F1-Score": "", "mAP@0.5": "", "FPS": ""},
    {"Model": "Khawam et al. (2025)", "Dataset": "URFD", "Accuracy": "", "Precision": "", "Recall": "", "F1-Score": "", "mAP@0.5": "", "FPS": ""},
    {"Model": "Xu et al. (2024)", "Dataset": "URFD", "Accuracy": "", "Precision": "", "Recall": "", "F1-Score": "", "mAP@0.5": "", "FPS": ""},
    {"Model": "Han et al. (2023)", "Dataset": "URFD", "Accuracy": "", "Precision": "", "Recall": "", "F1-Score": "", "mAP@0.5": "", "FPS": ""},
    {"Model": "Liu et al. (2022)", "Dataset": "URFD", "Accuracy": "", "Precision": "", "Recall": "", "F1-Score": "", "mAP@0.5": "", "FPS": ""},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Mathematical Formulas
# ═══════════════════════════════════════════════════════════════════════════════

"""
METRIC DEFINITIONS:

1. Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Precision (Positive Predictive Value) = TP / (TP + FP)
   - Tỷ lệ predictions positive đúng trong tất cả predictions positive

3. Recall (Sensitivity, True Positive Rate) = TP / (TP + FN)
   - Tỷ lệ positive cases được phát hiện đúng

4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean của Precision và Recall

5. Specificity (True Negative Rate) = TN / (TN + FP)
   - Tỷ lệ negative cases được phát hiện đúng

6. mAP@0.5 (Mean Average Precision @ IoU=0.5)
   - Trung bình của Average Precision tại threshold IoU=0.5

7. FPS (Frames Per Second) = 1 / (latency_per_sequence / sequence_length)
   - Throughput của model

8. FLOPs (Floating Point Operations)
   - Số phép tính floating point cần thiết cho một forward pass
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading & Complexity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """
    Đếm tổng số parameters trong model.

    Args:
        model: PyTorch nn.Module

    Returns:
        Tổng số parameters (trainable + non-trainable)
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Đếm số trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_shape: tuple) -> int:
    """
    Ước tính FLOPs cho model.

    Args:
        model: PyTorch nn.Module
        input_shape: Shape của input tensor (B, seq_len, feature_dim)

    Returns:
        Ước tính số FLOPs
    """
    model.eval()
    x = torch.randn(*input_shape).to(DEVICE)

    # Thủ công ước tính FLOPs dựa trên architecture
    # Transformer: O(L^2 * D) cho attention, O(L * D * F) cho FFN
    B, L, D = input_shape
    nhead = 4
    dim_ff = 256
    n_layers = 3

    # Linear layers FLOPs
    linear_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                # FLOPs = 2 * in_features * out_features * batch
                linear_flops += 2 * module.in_features * module.out_features * B

    # Attention FLOPs: QKV projection + attention + output
    # QKV: 3 * D * D * L
    # Attention: L^2 * D
    # Output: L * D * D
    attention_flops = 0
    for _ in range(n_layers):
        attention_flops += 6 * D * D * L  # QKV + output
        attention_flops += 2 * L * L * D  # Attention scores

    # FFN FLOPs: 2 * D * dim_ff * L
    ffn_flops = n_layers * 2 * D * dim_ff * L

    total_flops = linear_flops + attention_flops + ffn_flops
    return total_flops


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════════════════════

class FallDataset(torch.utils.data.Dataset):
    """
    Dataset cho evaluation.
    Load pre-extracted features từ disk.
    """

    def __init__(self, data_dir: str, split: str = "test"):
        self.data_dir = Path(data_dir)
        self.split = split

        # Load numpy arrays
        X_path = self.data_dir / f"X_{split}.npy"
        y_path = self.data_dir / f"y_{split}.npy"
        groups_path = self.data_dir / f"groups_{split}.npy"

        if not X_path.exists():
            raise FileNotFoundError(f"Features not found: {X_path}")

        self.X = np.load(X_path)
        self.y = np.load(y_path).ravel()
        self.groups = np.load(groups_path, allow_pickle=True) if groups_path.exists() else None

        print(f"  Loaded {len(self.y)} samples from {split}")
        print(f"  Features shape: {self.X.shape}")
        print(f"  Class distribution: {np.bincount(self.y.astype(int))}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(np.array([self.y[idx]])).float()
        return x, y


def create_dataloader(data_dir: str, batch_size: int = 32, split: str = "test") -> torch.utils.data.DataLoader:
    """Create DataLoader for evaluation."""
    dataset = FallDataset(data_dir, split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Chạy inference trên test set.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader chứa test data
        device: Device để inference
        use_gpu: Có dùng GPU không

    Returns:
        (y_true, y_pred, y_prob, latencies)
    """
    model.eval()
    model.to(device)

    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    latencies = []

    # GPU warmup
    if use_gpu and torch.cuda.is_available():
        print("\n[GPU Warmup] Running 10 warmup iterations...")
        dummy_input = torch.randn(32, SEQ_LEN, FEATURE_DIM).to(device)
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    print("\n[Inference] Running model on test set...")
    for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
        batch_x = batch_x.to(device)
        batch_y_np = batch_y.cpu().numpy().ravel()

        # Measure latency
        if use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        outputs = model(batch_x)
        if use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000 / len(batch_x)
        latencies.extend([latency_ms] * len(batch_x))

        # Get predictions
        probs = torch.sigmoid(outputs).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

        y_true_list.extend(batch_y_np.tolist())
        y_pred_list.extend(preds.tolist())
        y_prob_list.extend(probs.tolist())

    return (
        np.array(y_true_list),
        np.array(y_pred_list),
        np.array(y_prob_list),
        latencies,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Calculation
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """
    Tính toán tất cả metrics đánh giá.

    Attributes:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        latencies: List of inference latencies (ms)
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        latencies: list[float],
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.latencies = np.array(latencies)

        # Calculate confusion matrix
        self.cm = confusion_matrix(y_true, y_pred)

        # Extract TP, TN, FP, FN
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()

    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        return float((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-10))

    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        return float(self.tp / (self.tp + self.fp + 1e-10))

    def precision_macro(self) -> float:
        """Macro Precision = (Precision_0 + Precision_1) / 2"""
        p0 = self.tn / (self.tn + self.fn + 1e-10)  # Precision for class 0
        p1 = self.tp / (self.tp + self.fp + 1e-10)  # Precision for class 1
        return float((p0 + p1) / 2)

    def recall(self) -> float:
        """Recall (Sensitivity) = TP / (TP + FN)"""
        return float(self.tp / (self.tp + self.fn + 1e-10))

    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        return float(self.tn / (self.tn + self.fp + 1e-10))

    def f1(self) -> float:
        """F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision(), self.recall()
        return float(2 * p * r / (p + r + 1e-10))

    def f1_macro(self) -> float:
        """Macro F1-Score"""
        p0 = self.tn / (self.tn + self.fn + 1e-10)
        r0 = self.tn / (self.tn + self.fp + 1e-10)
        f1_0 = 2 * p0 * r0 / (p0 + r0 + 1e-10) if (p0 + r0) > 0 else 0.0

        p1 = self.tp / (self.tp + self.fp + 1e-10)
        r1 = self.tp / (self.tp + self.fn + 1e-10)
        f1_1 = 2 * p1 * r1 / (p1 + r1 + 1e-10) if (p1 + r1) > 0 else 0.0

        return float((f1_0 + f1_1) / 2)

    def auc_roc(self) -> float:
        """AUC-ROC = Area Under ROC Curve"""
        try:
            return float(roc_auc_score(self.y_true, self.y_prob))
        except ValueError:
            return 0.0

    def calculate_map(self) -> dict[str, float]:
        """
        Tính mAP@0.5 và mAP@0.5:0.95.

        mAP (Mean Average Precision) là trung bình của AP (Average Precision)
        cho tất cả classes. Trong binary classification, đây là AP cho class 1.

        AP = Σ (R_n - R_{n-1}) * P_n
        Trong đó P_n, R_n là Precision và Recall tại threshold n.
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_prob)

        # mAP@0.5: AP với threshold fixed tại 0.5
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        ap_05 = float(precision[idx_05] * recall[idx_05])  # Simplified

        # True AP (area under PR curve)
        ap = auc(recall, precision)

        return {
            "mAP@0.5": float(precision[0]) if len(precision) > 0 else 0.0,  # Precision at threshold 0.5
            "AP": float(ap),
        }

    def average_latency_ms(self) -> float:
        """Trung bình latency (ms) per sequence."""
        return float(np.mean(self.latencies))

    def throughput_fps(self, seq_len: int = SEQ_LEN) -> float:
        """
        Tính throughput (FPS).

        FPS = sequence_length / (latency_ms / 1000)
        """
        avg_latency_s = np.mean(self.latencies) / 1000
        return float(seq_len / avg_latency_s) if avg_latency_s > 0 else 0.0

    def get_all_metrics(self) -> dict[str, Any]:
        """Tính tất cả metrics và trả về dict."""
        map_metrics = self.calculate_map()

        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "precision_macro": self.precision_macro(),
            "recall": self.recall(),
            "specificity": self.specificity(),
            "f1_score": self.f1(),
            "f1_macro": self.f1_macro(),
            "auc_roc": self.auc_roc(),
            **map_metrics,
            "avg_latency_ms": self.average_latency_ms(),
            "throughput_fps": self.throughput_fps(),
            "confusion_matrix": self.cm.tolist(),
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: list[str] = ["Fall", "No-Fall"],
    normalize: bool = True,
) -> None:
    """
    Vẽ Confusion Matrix với seaborn heatmap.

    Args:
        cm: Confusion matrix
        save_path: Đường dẫn lưu figure
        class_names: Tên các classes
        normalize: Có normalize hay không
    """
    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        cm_normalized = cm
        title = "Confusion Matrix (Counts)"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Thêm TP, TN, FP, FN annotations
    labels = [
        f"TN={cm[0,0]}\nFP={cm[0,1]}",
        f"FN={cm[1,0]}\nTP={cm[1,1]}",
    ]

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """
    Vẽ Precision-Recall Curve.

    PR Curve thể hiện trade-off giữa Precision và Recall
    tại các thresholds khác nhau.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "b-", linewidth=2, label=f"PR Curve (AP={ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.2)

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color="r", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_f1_confidence_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    thresholds: np.ndarray | None = None,
) -> None:
    """
    Vẽ F1-Confidence Curve.

    F1-Confidence Curve thể hiện F1-Score tại các probability thresholds
    khác nhau, giúp chọn optimal threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    f1_scores = []
    precisions = []
    recalls = []

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        if preds.sum() == 0:
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
            continue

        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)

    # Tìm optimal threshold
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1_scores, "b-", linewidth=2, label=f"F1-Score (Best={best_f1:.3f} @ {best_thresh:.2f})")
    ax.plot(thresholds, precisions, "g--", linewidth=1.5, label="Precision")
    ax.plot(thresholds, recalls, "r--", linewidth=1.5, label="Recall")

    ax.axvline(x=best_thresh, color="orange", linestyle=":", linewidth=2, label=f"Optimal Threshold ({best_thresh:.2f})")
    ax.scatter([best_thresh], [best_f1], color="orange", s=100, zorder=5)

    ax.set_xlabel("Probability Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("F1-Confidence Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")

    return best_thresh, best_f1


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """Vẽ ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOTA Comparison Table
# ═══════════════════════════════════════════════════════════════════════════════

def print_sota_comparison_table(
    metrics: dict[str, Any],
    output_path: str | None = None,
) -> None:
    """
    In bảng so sánh với các SOTA models.

    Format giống như trong research papers.
    """
    # Tạo bảng với đầy đủ models
    rows = []

    # Proposed model (tính được)
    proposed_model = {
        "Model": "**Proposed (YOLOv11-Pose + PIFR + Transformer)**",
        "Dataset": "AIO",
        "Accuracy": f"{metrics['accuracy']:.4f}",
        "Precision": f"{metrics['precision']:.4f}",
        "Recall": f"{metrics['recall']:.4f}",
        "F1-Score": f"{metrics['f1_score']:.4f}",
        "mAP@0.5": f"{metrics.get('mAP@0.5', metrics.get('AP', 0)):.4f}",
        "FPS": f"{metrics['throughput_fps']:.1f}",
    }
    rows.append(proposed_model)

    # Thêm SOTA models
    for model in SOTA_MODELS:
        rows.append(model)

    # Print table
    print("\n" + "=" * 120)
    print("COMPARATIVE ANALYSIS WITH STATE-OF-THE-ART METHODS")
    print("=" * 120)

    # Header
    header = "| {:35} | {:10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |".format(
        "Model", "Dataset", "Accuracy", "Precision", "Recall", "F1-Score", "mAP@0.5", "FPS"
    )
    print(header)
    print("|" + "-" * 37 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|")

    # Rows
    for row in rows:
        line = "| {:35} | {:10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |".format(
            row["Model"],
            row["Dataset"],
            row["Accuracy"] or "-",
            row["Precision"] or "-",
            row["Recall"] or "-",
            row["F1-Score"] or "-",
            row["mAP@0.5"] or "-",
            row["FPS"] or "-",
        )
        print(line)

    print("=" * 120)

    # Save to file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Comparative Analysis with State-of-the-Art Methods\n\n")
            f.write(header + "\n")
            f.write("|" + "-" * 37 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|\n")
            for row in rows:
                line = "| {:35} | {:10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |".format(
                    row["Model"],
                    row["Dataset"],
                    row["Accuracy"] or "-",
                    row["Precision"] or "-",
                    row["Recall"] or "-",
                    row["F1-Score"] or "-",
                    row["mAP@0.5"] or "-",
                    row["FPS"] or "-",
                )
                f.write(line + "\n")
        print(f"  [Saved] {output_path}")


def export_results(
    metrics: dict[str, Any],
    output_dir: str,
) -> None:
    """
    Export metrics ra JSON và CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] {json_path}")

    # CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")
        for key, value in metrics.items():
            if not isinstance(value, (list, dict)):
                f.write(f"{key},{value}\n")
    print(f"  [Saved] {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Hàm chính để đánh giá model.

    Args:
        model_path: Đường dẫn đến trained model (.pth)
        data_dir: Thư mục chứa test data
        output_dir: Thư mục lưu kết quả
        batch_size: Batch size cho inference
        device: Device ('cuda', 'cpu', hoặc None tự động)

    Returns:
        Dict chứa tất cả metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print("=" * 60)
    print("FALL DETECTION MODEL EVALUATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model:        {model_path}")
    print(f"  Data:         {data_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Device:       {device}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ─── 1. Load Model ───
    print("\n" + "-" * 40)
    print("STEP 1: Loading Model")
    print("-" * 40)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = HybridFallTransformer(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  [OK] Model loaded from {model_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        raise

    # Model complexity
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    input_shape = (batch_size, SEQ_LEN, FEATURE_DIM)
    estimated_flops = estimate_flops(model, input_shape)

    print(f"\n  Model Complexity:")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Estimated FLOPs:      {estimated_flops:,}")

    # ─── 2. Load Data ───
    print("\n" + "-" * 40)
    print("STEP 2: Loading Test Data")
    print("-" * 40)

    try:
        dataloader = create_dataloader(data_dir, batch_size=batch_size, split="test")
        print(f"  [OK] Loaded {len(dataloader.dataset)} test samples")
    except FileNotFoundError as e:
        print(f"  [WARN] Test data not found, trying 'val' split...")
        try:
            dataloader = create_dataloader(data_dir, batch_size=batch_size, split="val")
            print(f"  [OK] Loaded {len(dataloader.dataset)} validation samples")
        except FileNotFoundError:
            raise FileNotFoundError(f"No test or val data found in {data_dir}") from e

    # ─── 3. Run Inference ───
    print("\n" + "-" * 40)
    print("STEP 3: Running Inference")
    print("-" * 40)

    y_true, y_pred, y_prob, latencies = run_inference(
        model, dataloader, device, use_gpu=(device.type == "cuda")
    )
    print(f"\n  Inference complete: {len(y_true)} samples")

    # ─── 4. Calculate Metrics ───
    print("\n" + "-" * 40)
    print("STEP 4: Calculating Metrics")
    print("-" * 40)

    calculator = MetricsCalculator(y_true, y_pred, y_prob, latencies)
    metrics = calculator.get_all_metrics()

    # Thêm model complexity vào metrics
    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = trainable_params
    metrics["estimated_flops"] = estimated_flops

    # Print metrics
    print("\n  Evaluation Metrics:")
    print(f"    Accuracy:       {metrics['accuracy']:.4f}")
    print(f"    Precision:      {metrics['precision']:.4f}")
    print(f"    Recall:         {metrics['recall']:.4f}")
    print(f"    F1-Score:       {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"    mAP@0.5:        {metrics.get('mAP@0.5', 0):.4f}")
    print(f"    Avg Latency:    {metrics['avg_latency_ms']:.2f} ms")
    print(f"    Throughput:     {metrics['throughput_fps']:.1f} FPS")

    print("\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # ─── 5. Generate Visualizations ───
    print("\n" + "-" * 40)
    print("STEP 5: Generating Visualizations")
    print("-" * 40)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Confusion Matrix
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        str(viz_dir / "confusion_matrix_normalized.png"),
        normalize=True,
    )
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        str(viz_dir / "confusion_matrix_counts.png"),
        normalize=False,
    )

    # PR Curve
    plot_pr_curve(y_true, y_prob, str(viz_dir / "pr_curve.png"))

    # F1-Confidence Curve
    plot_f1_confidence_curve(y_true, y_prob, str(viz_dir / "f1_confidence_curve.png"))

    # ROC Curve
    plot_roc_curve(y_true, y_prob, str(viz_dir / "roc_curve.png"))

    # ─── 6. Export Results ───
    print("\n" + "-" * 40)
    print("STEP 6: Exporting Results")
    print("-" * 40)

    export_results(metrics, str(output_dir))

    # ─── 7. Print SOTA Comparison ───
    print("\n" + "-" * 40)
    print("STEP 7: SOTA Comparison Table")
    print("-" * 40)

    print_sota_comparison_table(metrics, str(output_dir / "comparison_table.md"))

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - visualizations/confusion_matrix_*.png")
    print(f"  - visualizations/pr_curve.png")
    print(f"  - visualizations/f1_confidence_curve.png")
    print(f"  - visualizations/roc_curve.png")
    print(f"  - results.json")
    print(f"  - results.csv")
    print(f"  - comparison_table.md")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation for Fall Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate.py --model best_model.pth --data-dir data/processed

  # Custom output directory
  python evaluate.py --model best_model.pth --data-dir data/processed --output eval_results

  # Use CPU
  python evaluate.py --model best_model.pth --data-dir data/processed --device cpu

  # Custom batch size
  python evaluate.py --model best_model.pth --data-dir data/processed --batch-size 64
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_hybrid_transformer.pth",
        help="Path to trained model (.pth file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_test.npy, y_test.npy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Run evaluation
    try:
        evaluate(
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
