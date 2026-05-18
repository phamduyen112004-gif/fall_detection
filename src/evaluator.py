"""
Benchmark / Evaluation Pipeline for Fall Detection Transformer.
========================================================
Generates comprehensive accuracy and efficiency metrics for thesis.

Outputs:
    results/benchmark_comparison.csv
    results/pr_curve.png
    results/roc_curve.png
    results/confusion_matrix.png

Usage:
    python main.py --mode evaluate
    # or
    from src.evaluator import run_evaluation; run_evaluation()
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import MODEL_SAVE_DIR, OUTPUT_DIR, RESULTS_DIR
from src.hybrid_transformer import HybridFallTransformer
from src.utils import calculate_metrics, get_device, load_npy_files

# =============================================================================
# EPSILON FOR NUMERICAL STABILITY
# =============================================================================

EPSILON: float = 1e-8
"""Small constant added to denominators to prevent division by zero."""

# Lazy thop import
try:
    from thop import profile
except ImportError:
    import subprocess
    logging.warning("Installing thop package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop", "-q"])
    from thop import profile


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure module-level logging to file and console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("evaluator")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


LOGGER = setup_logging()


# =============================================================================
# MODEL LOADING & INFERENCE
# =============================================================================

def load_model(weights_path: str | os.PathLike, device: torch.device) -> nn.Module:
    """
    Load trained HybridFallTransformer from a checkpoint.

    Args:
        weights_path: Path to .pth checkpoint.
        device: Target device.

    Returns:
        nn.Module: Loaded model in eval mode.

    Raises:
        RuntimeError: If model cannot be loaded.
    """
    try:
        model = HybridFallTransformer().to(device)

        try:
            ckpt = torch.load(weights_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            LOGGER.warning(f"Strict loading failed, trying non-strict: %s", e)
            ckpt = torch.load(weights_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
            model.load_state_dict(state, strict=False)

        model.eval()
        LOGGER.info("Model loaded successfully from %s", weights_path)
        return model

    except Exception as e:
        LOGGER.error("Failed to load model from %s: %s", weights_path, e)
        raise


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Get fall probability predictions for test set.

    Args:
        model: Trained model.
        X: Feature array of shape (N, 60, 60).
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        np.ndarray: Array of probabilities.
    """
    model.eval()
    probs: list[float] = []

    try:
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
                logits = model(batch).squeeze()
                prob = torch.sigmoid(logits).cpu().numpy()

                if prob.ndim == 0:
                    probs.append(float(prob))
                else:
                    probs.extend(float(p) for p in prob)

                del batch, logits, prob
                torch.cuda.empty_cache()

    except Exception as e:
        LOGGER.error("Error during inference: %s", e)
        raise

    return np.array(probs)


# =============================================================================
# ACCURACY METRICS
# =============================================================================

def compute_accuracy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """
    Calculate all accuracy metrics with division-by-zero protection.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        y_proba: Prediction probabilities.

    Returns:
        dict: Metric name → value mapping.
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.size == 1:
        tn = int(cm[0, 0]) if cm.shape == (1, 1) else 0
        fp, fn, tp = 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()

    precision_denom = tp + fp + EPSILON
    recall_denom = tp + fn + EPSILON
    specificity_denom = tn + fp + EPSILON
    accuracy_denom = tp + tn + fp + fn + EPSILON

    metrics: dict[str, float] = {
        "precision": float(tp / precision_denom),
        "recall": float(tp / recall_denom),
        "specificity": float(tn / specificity_denom),
        "accuracy": float((tp + tn) / accuracy_denom),
        "f1_score": float(2 * tp / (2 * tp + fp + fn + EPSILON)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError as e:
        LOGGER.warning("Could not compute AUC: %s", e)
        metrics["auc"] = 0.0

    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics["avg_precision"] = 0.0

    metrics["confusion_matrix"] = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return metrics


def evaluate_yolo_map(yolo_model_path: str = "yolo11n-pose.pt") -> dict[str, float]:
    """
    Evaluate YOLOv11n-Pose mAP using Ultralytics val API.

    Args:
        yolo_model_path: Path to YOLO pose weights.

    Returns:
        dict: yolo_map50 and yolo_map50_95.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(yolo_model_path)
        results = model.val(verbose=False)
        return {
            "yolo_map50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
            "yolo_map50_95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
        }
    except Exception as e:
        LOGGER.warning("YOLO mAP evaluation failed: %s", e)
        return {"yolo_map50": 0.0, "yolo_map50_95": 0.0}


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

def compute_model_params(model: nn.Module) -> float:
    """
    Calculate total trainable parameters in millions.

    Args:
        model: PyTorch model.

    Returns:
        float: Parameters in millions (M).
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_model_size_mb(weights_path: str | os.PathLike) -> float:
    """
    Calculate model file size in megabytes.

    Args:
        weights_path: Path to model weights file.

    Returns:
        float: Size in MB.
    """
    if os.path.exists(weights_path):
        return os.path.getsize(weights_path) / (1024 * 1024)
    LOGGER.warning("Model file not found: %s", weights_path)
    return 0.0


def compute_gflops(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (1, 60, 60),
    device: torch.device | None = None,
) -> float:
    """
    Calculate Giga Floating-Point Operations using thop profile.

    Args:
        model: The PyTorch model.
        input_shape: Input tensor shape (batch, seq_len, features).
        device: Compute device.

    Returns:
        float: GFLOPs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(flops / 1e9)
    except Exception as e:
        LOGGER.warning("GFLOPs computation failed: %s", e)
        return 0.0


def measure_inference_speed(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Measure inference latency and throughput.

    Args:
        model: Trained model.
        X: Feature array for benchmarking.
        device: Compute device.
        num_runs: Number of benchmark iterations.

    Returns:
        dict: avg_latency_ms, std_latency_ms, fps, etc.
    """
    model.eval()

    warmup_runs = min(10, len(X))
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model(torch.FloatTensor(X[i : i + 1]).to(device))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies: list[float] = []

    with torch.no_grad():
        for i in range(min(num_runs, len(X))):
            start = time.perf_counter()
            _ = model(torch.FloatTensor(X[i : i + 1]).to(device))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000.0)

    avg_latency = float(np.mean(latencies))
    return {
        "avg_latency_ms": avg_latency,
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "fps": 1000.0 / avg_latency if avg_latency > EPSILON else 0.0,
    }


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | os.PathLike,
    avg_precision: float | None = None,
) -> None:
    """
    Plot Precision-Recall Curve.

    Args:
        y_true: Ground truth labels.
        y_proba: Prediction probabilities.
        save_path: Path to save the figure.
        avg_precision: Pre-computed average precision (optional).
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = avg_precision if avg_precision is not None else float(average_precision_score(y_true, y_proba))

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, "b-", linewidth=2, label=f"PR Curve (AP = {ap:.4f})")
        plt.fill_between(recall, precision, alpha=0.2, color="blue")
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline (P = {baseline:.4f})")

        f1_scores = 2.0 * (precision * recall) / (precision + recall + EPSILON)
        optimal_idx = int(np.argmax(f1_scores))
        plt.scatter(
            recall[optimal_idx], precision[optimal_idx],
            color="green", s=100, zorder=5,
            label=f"Optimal (R={recall[optimal_idx]:.2f}, P={precision[optimal_idx]:.2f})",
        )

        plt.xlabel("Recall (Sensitivity)", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve\nFall Detection Transformer", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved PR curve: %s", save_path)

    except Exception as e:
        LOGGER.error("Failed to plot PR curve: %s", e)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | os.PathLike,
    auc_score: float | None = None,
) -> None:
    """
    Plot ROC Curve.

    Args:
        y_true: Ground truth labels.
        y_proba: Prediction probabilities.
        save_path: Path to save the figure.
        auc_score: Pre-computed AUC score (optional).
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc_score if auc_score is not None else float(auc(fpr, tpr))

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.fill_between(fpr, tpr, alpha=0.2, color="blue")
        plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random (AUC = 0.5)")

        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        plt.scatter(
            fpr[optimal_idx], tpr[optimal_idx],
            color="green", s=100, zorder=5,
            label=f"Optimal threshold",
        )

        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
        plt.title("ROC Curve\nFall Detection Transformer", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved ROC curve: %s", save_path)

    except Exception as e:
        LOGGER.error("Failed to plot ROC curve: %s", e)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | os.PathLike,
) -> None:
    """
    Plot and save the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: Path to save the figure.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Non-Fall", "Fall"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix\nFall Detection Transformer")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=18)

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved confusion matrix: %s", save_path)

    except Exception as e:
        LOGGER.error("Failed to plot confusion matrix: %s", e)


# =============================================================================
# REPORT EXPORT
# =============================================================================

def export_benchmark_csv(metrics: dict[str, Any], output_path: str | os.PathLike) -> None:
    """
    Export all metrics to CSV for thesis / reproducibility.

    Args:
        metrics: Dictionary of metrics to export.
        output_path: Path to save the CSV file.
    """
    import csv

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Accuracy", f"{metrics['accuracy']:.4f}"])
            writer.writerow(["Precision", f"{metrics['precision']:.4f}"])
            writer.writerow(["Recall (Sensitivity)", f"{metrics['recall']:.4f}"])
            writer.writerow(["Specificity", f"{metrics['specificity']:.4f}"])
            writer.writerow(["F1-Score", f"{metrics['f1_score']:.4f}"])
            writer.writerow(["AUC-ROC", f"{metrics['auc']:.4f}"])
            writer.writerow(["Average Precision", f"{metrics['avg_precision']:.4f}"])

            if "yolo_map50" in metrics:
                writer.writerow(["YOLO mAP@0.5", f"{metrics['yolo_map50']:.4f}"])
                writer.writerow(["YOLO mAP@0.5:0.95", f"{metrics['yolo_map50_95']:.4f}"])

            writer.writerow([])
            writer.writerow(["Confusion Matrix"])
            cm = metrics["confusion_matrix"]
            writer.writerow([f"  TP={cm['TP']}  FP={cm['FP']}"])
            writer.writerow([f"  FN={cm['FN']}  TN={cm['TN']}"])
            writer.writerow([])
            writer.writerow(["Efficiency"])
            writer.writerow(["Parameters (M)", f"{metrics['params_m']:.2f}"])
            writer.writerow(["Model Size (MB)", f"{metrics['model_size_mb']:.2f}"])
            writer.writerow(["GFLOPs", f"{metrics['gflops']:.2f}"])
            writer.writerow(["Avg Latency (ms)", f"{metrics['avg_latency_ms']:.4f}"])
            writer.writerow(["Latency Std (ms)", f"{metrics['std_latency_ms']:.4f}"])
            writer.writerow(["FPS", f"{metrics['fps']:.2f}"])

        LOGGER.info("Saved benchmark CSV: %s", output_path)

    except Exception as e:
        LOGGER.error("Failed to export benchmark CSV: %s", e)


def print_report(metrics: dict[str, Any]) -> None:
    """
    Print a detailed benchmark report to console.

    Args:
        metrics: Dictionary of metrics to display.
    """
    separator = "=" * 70
    LOGGER.info("\n%s", separator)
    LOGGER.info("FALL DETECTION TRANSFORMER - BENCHMARK REPORT")
    LOGGER.info("%s", separator)

    LOGGER.info("\nACCURACY METRICS")
    LOGGER.info("-" * 70)
    LOGGER.info("  Accuracy:             %.4f (%.2f%%)", metrics['accuracy'], metrics['accuracy'] * 100)
    LOGGER.info("  Precision:            %.4f (%.2f%%)", metrics['precision'], metrics['precision'] * 100)
    LOGGER.info("  Recall (Sensitivity): %.4f (%.2f%%)", metrics['recall'], metrics['recall'] * 100)
    LOGGER.info("  Specificity:          %.4f (%.2f%%)", metrics['specificity'], metrics['specificity'] * 100)
    LOGGER.info("  F1-Score:             %.4f (%.2f%%)", metrics['f1_score'], metrics['f1_score'] * 100)
    LOGGER.info("  AUC-ROC:              %.4f", metrics['auc'])
    LOGGER.info("  Average Precision:    %.4f", metrics['avg_precision'])

    if "yolo_map50" in metrics:
        LOGGER.info("  YOLO mAP@0.5:        %.4f", metrics['yolo_map50'])
        LOGGER.info("  YOLO mAP@0.5:0.95:   %.4f", metrics['yolo_map50_95'])

    cm = metrics["confusion_matrix"]
    LOGGER.info("\n  Confusion Matrix:")
    LOGGER.info("    TP=%4d  FP=%4d", cm['TP'], cm['FP'])
    LOGGER.info("    FN=%4d  TN=%4d", cm['FN'], cm['TN'])

    LOGGER.info("\nEFFICIENCY METRICS")
    LOGGER.info("-" * 70)
    LOGGER.info("  Parameters:           %.2f M", metrics['params_m'])
    LOGGER.info("  Model Size:          %.2f MB", metrics['model_size_mb'])
    LOGGER.info("  GFLOPs:              %.2f G", metrics['gflops'])
    LOGGER.info("  Avg Latency:         %.4f ms", metrics['avg_latency_ms'])
    LOGGER.info("  Latency Std:         %.4f ms", metrics['std_latency_ms'])
    LOGGER.info("  FPS:                 %.2f fps", metrics['fps'])
    LOGGER.info("\n%s", separator)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_evaluation() -> None:
    """Run full evaluation: load model, evaluate, generate plots and CSV."""
    LOGGER.info("=" * 70)
    LOGGER.info("FALL DETECTION BENCHMARK")
    LOGGER.info("=" * 70)

    try:
        device = get_device()
        LOGGER.info("\nDevice: %s", device)

        model_path = Path(MODEL_SAVE_DIR) / "best_model.pth"

        if not model_path.exists():
            LOGGER.error("Model not found at %s", model_path)
            LOGGER.error("Please train first: python main.py --mode train")
            return

        LOGGER.info("\nLoading test data...")
        X, y = load_npy_files(OUTPUT_DIR)
        LOGGER.info("  Total: %d | Fall: %d | Non-Fall: %d",
                    len(X), int(np.sum(y == 1)), int(np.sum(y == 0)))

        LOGGER.info("\nLoading model...")
        model = load_model(model_path, device)

        LOGGER.info("Running inference...")
        y_proba = predict_proba(model, X, device)
        y_pred = (y_proba >= 0.5).astype(int)

        accuracy_metrics = compute_accuracy_metrics(y, y_pred, y_proba)
        yolo_metrics = evaluate_yolo_map()
        accuracy_metrics.update(yolo_metrics)

        efficiency_metrics = {
            "params_m": compute_model_params(model),
            "model_size_mb": compute_model_size_mb(model_path),
            "gflops": compute_gflops(model, (1, 60, 60), device),
            **measure_inference_speed(model, X, device),
        }

        all_metrics = {**accuracy_metrics, **efficiency_metrics}

        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("\nGenerating visualizations...")
        plot_pr_curve(y, y_proba, save_path=results_dir / "pr_curve.png",
                      avg_precision=accuracy_metrics["avg_precision"])
        plot_roc_curve(y, y_proba, save_path=results_dir / "roc_curve.png",
                       auc_score=accuracy_metrics["auc"])
        plot_confusion_matrix(y, y_pred, save_path=results_dir / "confusion_matrix.png")

        export_benchmark_csv(all_metrics, output_path=results_dir / "benchmark_comparison.csv")
        print_report(all_metrics)

        del model, X, y, y_proba, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("BENCHMARK COMPLETE")
        LOGGER.info("=" * 70)
        LOGGER.info("\nOutput files in %s/:", results_dir)
        for name in ["benchmark_comparison.csv", "pr_curve.png", "roc_curve.png", "confusion_matrix.png"]:
            LOGGER.info("  %s", name)
        LOGGER.info("=" * 70)

    except FileNotFoundError as e:
        LOGGER.error("Required file not found: %s", e)
        raise
    except ValueError as e:
        LOGGER.error("Invalid data or configuration: %s", e)
        raise
    except Exception as e:
        LOGGER.error("Evaluation failed with unexpected error: %s", e)
        raise


if __name__ == "__main__":
    run_evaluation()
