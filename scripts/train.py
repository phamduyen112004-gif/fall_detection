#!/usr/bin/env python
"""
Train HybridFallTransformer for Fall Detection.

This script supports both single training and ablation study modes.

Usage (Single Training):
    python train.py --data /path/to/processed --output /path/to/output

Usage (Ablation Study):
    python train.py --mode ablation --data /path/to/processed

Ablation Study Configurations (per Benabdennour et al., 2026):
    1. Baseline:  d_model=256, num_layers=3, nhead=4
    2. Variant 1: d_model=256, num_layers=3, nhead=8
    3. Variant 2: d_model=256, num_layers=4, nhead=4
    4. Final:     d_model=256, num_layers=4, nhead=8 (Proposed)
"""

import os
import sys
import json
import random
import shutil
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# Use centralized config and trainer from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid_transformer import HybridFallTransformer
from src.config import (
    DATA_DIR as DEFAULT_DATA,
    MODEL_SAVE_DIR as DEFAULT_OUT,
    RESULTS_DIR as DEFAULT_RES,
    TRAINING_HYPERPARAMS,
    RANDOM_SEED as SEED,
)
from src.trainer import (
    FallDataset,
    load_data,
    compute_metrics,
    train_one_epoch,
    evaluate,
    setup_logging,
)


# =============================================================================
# THRESHOLD CALCULATION
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple:
    """
    Find optimal threshold based on F1-score maximization.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_probs: Predicted probabilities for positive class

    Returns:
        Tuple of (optimal_threshold, max_f1_score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # Calculate F1-scores for all thresholds
    f1_scores = np.zeros_like(precision[:-1])
    for i in range(len(precision) - 1):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0.0

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# =============================================================================
# ABLATION STUDY CONFIGURATIONS
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation study variant."""
    name: str
    d_model: int
    num_layers: int
    nhead: int
    dropout: float = 0.1


ABLATION_VARIANTS = [
    AblationConfig(name="Baseline",    d_model=256, num_layers=3, nhead=4),
    AblationConfig(name="Variant1",    d_model=256, num_layers=3, nhead=8),
    AblationConfig(name="Variant2",    d_model=256, num_layers=4, nhead=4),
    AblationConfig(name="Final",       d_model=256, num_layers=4, nhead=8),
]


@dataclass
class AblationResult:
    """Results from a single ablation study run."""
    config: AblationConfig
    num_params: float
    gflops: Optional[float]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: list
    epochs_trained: int
    training_time_seconds: float
    best_val_f1: float


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HybridFallTransformer (with optional Ablation Study)"
    )
    parser.add_argument("--mode", "-m", type=str, default="single",
                        choices=["single", "ablation"],
                        help="Training mode: single run or ablation study (default: single)")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help=f"Processed data directory (default: {DEFAULT_DATA})")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help=f"Model output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--results", "-r", type=str, default=None,
                        help=f"Results directory (default: {DEFAULT_RES})")
    parser.add_argument("--epochs", "-e", type=int, default=TRAINING_HYPERPARAMS.get("epochs", 100),
                        help=f"Max epochs (default: {TRAINING_HYPERPARAMS.get('epochs', 100)})")
    parser.add_argument("--batch-size", "-b", type=int, default=TRAINING_HYPERPARAMS.get("batch_size", 64),
                        help=f"Batch size (default: {TRAINING_HYPERPARAMS.get('batch_size', 64)})")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda (default: auto)")
    parser.add_argument("--skip-ablation", nargs="+", default=[],
                        help="Skip specific ablation variants by name (e.g., --skip-ablation Variant1 Variant2)")
    return parser.parse_args()


# =============================================================================
# MODEL PROFILING UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> float:
    """Count total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def calculate_gflops(model: nn.Module, input_size: tuple = (1, 60, 60),
                     device: torch.device = None) -> Optional[float]:
    """
    Calculate GFLOPs using thop library.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, frames, features)
        device: Device to run on
        
    Returns:
        GFLOPs as float, or None if thop not available
    """
    if not THOP_AVAILABLE:
        return None
    
    try:
        dummy_input = torch.randn(input_size).to(device)
        gflops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return gflops / 1e9  # Convert to GFLOPs
    except Exception as e:
        print(f"  Warning: Could not calculate GFLOPs ({e})")
        return None


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_single_variant(
    config: AblationConfig,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    base_out_dir: str,
    base_res_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    logger=None
) -> AblationResult:
    """
    Train a single model variant.
    
    Args:
        config: Ablation configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        base_out_dir: Base model output directory
        base_res_dir: Base results directory
        epochs, batch_size, lr, weight_decay, patience: Training hyperparameters
        device: torch device
        logger: Optional logger
        
    Returns:
        AblationResult with metrics
    """
    # Create variant-specific directories
    variant_dir = config.name.replace(" ", "_").lower()
    variant_out = os.path.join(base_out_dir, f"ablation_{variant_dir}")
    variant_res = os.path.join(base_res_dir, f"ablation_{variant_dir}")
    os.makedirs(variant_out, exist_ok=True)
    os.makedirs(variant_res, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VARIANT: {config.name}")
    print(f"{'='*60}")
    print(f"  d_model={config.d_model}, num_layers={config.num_layers}, nhead={config.nhead}")
    print(f"  Output: {variant_out}")
    
    # Initialize model
    model = HybridFallTransformer(
        input_dim=60,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    # Count parameters and FLOPs
    num_params = count_parameters(model)
    gflops = calculate_gflops(model, device=device)
    
    print(f"\n  Model Statistics:")
    print(f"    Parameters: {num_params:.2f}M")
    if gflops is not None:
        print(f"    GFLOPs: {gflops:.2f}G")
    else:
        print(f"    GFLOPs: N/A (thop not available)")
    
    # Create data loaders
    num_workers = 4
    train_loader = DataLoader(
        FallDataset(X_train, y_train, augment=True),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        FallDataset(X_val, y_val, augment=False),
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        FallDataset(X_test, y_test, augment=False),
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_f1, patience_counter = 0.0, 0
    best_model_state = None
    training_start = time.time()
    epochs_trained = 0
    
    print(f"\n  Training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_f1 = val_metrics.get("f1", 0.0)
        scheduler.step(val_f1)
        
        epochs_trained = epoch + 1
        
        if (epoch + 1) % 5 == 0 or val_f1 > best_f1:
            log_msg = f"  Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(variant_out, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - training_start
    print(f"  Training completed in {training_time:.1f}s ({epochs_trained} epochs)")
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    test_metrics = evaluate(model, test_loader, criterion, device, return_probs=True)

    # Calculate optimal threshold based on F1-score maximization
    y_true = test_metrics["y_true"]
    y_probs = test_metrics["y_probs"]
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_probs)

    acc = test_metrics.get("accuracy", 0.0)
    prec = test_metrics.get("precision", 0.0)
    rec = test_metrics.get("recall", 0.0)
    f1 = test_metrics.get("f1", 0.0)
    auc = test_metrics.get("auc", 0.0)
    cm = test_metrics.get("confusion_matrix", [[0,0],[0,0]])

    print(f"\n  Test Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:   {rec:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"    AUC-ROC:  {auc:.4f}")
    print(f"\n  Optimal Threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")
    
    # Save variant metrics with optimal threshold
    metrics = {
        'variant': config.name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'optimal_threshold': float(optimal_threshold),
        'threshold_f1_score': float(optimal_f1),
        'confusion_matrix': cm,
        'best_val_f1': float(best_f1),
        'num_params_M': float(num_params),
        'gflops': float(gflops) if gflops else None,
        'training_time_seconds': float(training_time),
        'epochs_trained': epochs_trained,
        'hyperparameters': {
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
        }
    }

    with open(os.path.join(variant_res, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save threshold config file for GUI
    threshold_config = {
        'optimal_threshold': float(optimal_threshold),
        'threshold_f1_score': float(optimal_f1),
        'model_path': 'best_model.pth',
    }
    with open(os.path.join(variant_res, 'threshold_config.json'), 'w') as f:
        json.dump(threshold_config, f, indent=2)
    
    # Copy model to results
    shutil.copy(
        os.path.join(variant_out, 'best_model.pth'),
        os.path.join(variant_res, 'best_model.pth')
    )
    
    # Cleanup to avoid OOM
    del model
    del best_model_state
    del train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return AblationResult(
        config=config,
        num_params=num_params,
        gflops=gflops,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        auc_roc=auc,
        confusion_matrix=cm,
        epochs_trained=epochs_trained,
        training_time_seconds=training_time,
        best_val_f1=best_f1
    )


def run_ablation_study(
    variants: list,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    out_dir: str,
    res_dir: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    skip_variants: list = None
) -> list:
    """
    Run full ablation study across all variants.
    
    Args:
        variants: List of AblationConfig to test
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        out_dir: Model output directory
        res_dir: Results directory
        epochs: Max training epochs
        batch_size: Batch size
        device: torch device
        skip_variants: List of variant names to skip
        
    Returns:
        List of AblationResult for each variant
    """
    if skip_variants is None:
        skip_variants = []
    
    # Filter variants
    active_variants = [v for v in variants if v.name not in skip_variants]
    
    print(f"\n{'#'*70}")
    print(f"# ABLATION STUDY - {len(active_variants)} variants")
    print(f"{'#'*70}")
    print(f"# Configurations:")
    for i, v in enumerate(active_variants, 1):
        print(f"#   {i}. {v.name}: d_model={v.d_model}, L={v.num_layers}, H={v.nhead}")
    print(f"{'#'*70}")
    
    results: list[AblationResult] = []
    overall_start = time.time()
    
    for i, variant in enumerate(active_variants, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(active_variants)}] Running: {variant.name}")
        print(f"{'='*70}")
        
        variant_start = time.time()
        
        result = train_single_variant(
            config=variant,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            base_out_dir=out_dir,
            base_res_dir=res_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=TRAINING_HYPERPARAMS.get("learning_rate", 5e-4),
            weight_decay=TRAINING_HYPERPARAMS.get("weight_decay", 1e-5),
            patience=TRAINING_HYPERPARAMS.get("early_stopping_patience", 25),
            device=device
        )
        
        results.append(result)
        variant_time = time.time() - variant_start
        
        print(f"\n  {variant.name} completed in {variant_time:.1f}s")
        print(f"  Best Val F1: {result.best_val_f1:.4f} | Test F1: {result.f1_score:.4f}")
        
        # Clear cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    total_time = time.time() - overall_start
    
    return results


def generate_ablation_report(results: list[AblationResult], output_dir: str) -> str:
    """
    Generate markdown and CSV ablation study reports.
    
    Args:
        results: List of AblationResult
        output_dir: Directory to save reports
        
    Returns:
        Path to the generated markdown report
    """
    print(f"\n{'='*70}")
    print("GENERATING ABLATION STUDY REPORT")
    print(f"{'='*70}")
    
    # Sort results by F1 score (descending)
    sorted_results = sorted(results, key=lambda r: r.f1_score, reverse=True)
    
    # Determine best variant
    best_result = sorted_results[0]
    
    # Generate markdown table
    md_lines = [
        "# Ablation Study Results",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration Summary",
        "",
        "| Variant | d_model | Layers (L) | Heads (H) | Dropout |",
        "|---------|---------|-----------|-----------|---------|",
    ]
    for r in results:
        md_lines.append(f"| {r.config.name} | {r.config.d_model} | {r.config.num_layers} | {r.config.nhead} | {r.config.dropout} |")
    
    md_lines.extend([
        "",
        "## Performance Comparison",
        "",
        "| Variant | Params (M) | GFLOPs | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Time (s) |",
        "|---------|-----------|--------|----------|-----------|--------|----------|---------|----------|",
    ])
    
    for r in results:
        gflops_str = f"{r.gflops:.2f}" if r.gflops else "N/A"
        md_lines.append(
            f"| {r.config.name} | {r.num_params:.2f} | {gflops_str} | "
            f"{r.accuracy:.4f} | {r.precision:.4f} | {r.recall:.4f} | "
            f"**{r.f1_score:.4f}** | {r.auc_roc:.4f} | {r.training_time_seconds:.0f} |"
        )
    
    md_lines.extend([
        "",
        "## Key Findings",
        "",
        f"- **Best Variant:** {best_result.config.name} (F1={best_result.f1_score:.4f})",
        f"- **Total Training Time:** {sum(r.training_time_seconds for r in results):.0f}s",
        f"- **Total Parameters Range:** {min(r.num_params for r in results):.2f}M - {max(r.num_params for r in results):.2f}M",
    ])
    
    # Add analysis if Final model is present
    final_results = [r for r in results if r.config.name == "Final"]
    if final_results:
        final = final_results[0]
        baseline_results = [r for r in results if r.config.name == "Baseline"]
        if baseline_results:
            baseline = baseline_results[0]
            f1_improvement = final.f1_score - baseline.f1_score
            param_increase = (final.num_params - baseline.num_params) / baseline.num_params * 100
            md_lines.extend([
                "",
                "## Ablation Analysis (vs Baseline)",
                "",
                f"- **F1 Improvement (Final vs Baseline):** {f1_improvement:+.4f}",
                f"- **Parameter Increase:** {param_increase:+.1f}%",
                f"- **Final Model Efficiency:** {final.f1_score / final.num_params:.4f} F1/M params",
            ])
    
    md_lines.extend([
        "",
        "## Confusion Matrices",
        "",
    ])
    for r in results:
        cm = r.confusion_matrix
        md_lines.extend([
            f"### {r.config.name}",
            "",
            f"- TN: {cm[0][0]}, FP: {cm[0][1]}",
            f"- FN: {cm[1][0]}, TP: {cm[1][1]}",
            "",
        ])
    
    md_content = "\n".join(md_lines)
    md_path = os.path.join(output_dir, "ablation_study_report.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  Markdown report: {md_path}")
    
    # Generate CSV
    csv_lines = [
        "Variant,d_model,num_layers,nhead,dropout,params_M,gflops,accuracy,precision,recall,f1_score,auc_roc,training_time_s,epochs_trained"
    ]
    for r in results:
        gflops_str = f"{r.gflops:.4f}" if r.gflops else ""
        csv_lines.append(
            f"{r.config.name},{r.config.d_model},{r.config.num_layers},{r.config.nhead},{r.config.dropout},"
            f"{r.num_params:.4f},{gflops_str},{r.accuracy:.4f},{r.precision:.4f},{r.recall:.4f},"
            f"{r.f1_score:.4f},{r.auc_roc:.4f},{r.training_time_seconds:.0f},{r.epochs_trained}"
        )
    
    csv_content = "\n".join(csv_lines)
    csv_path = os.path.join(output_dir, "ablation_study_results.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    print(f"  CSV report: {csv_path}")
    
    # Print summary table to console
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<12} {'Params(M)':<10} {'GFLOPs':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print(f"{'-'*70}")
    for r in sorted_results:
        gflops_str = f"{r.gflops:.2f}" if r.gflops else "N/A"
        marker = " <-- BEST" if r == best_result else ""
        print(f"{r.config.name:<12} {r.num_params:<10.2f} {gflops_str:<8} "
              f"{r.accuracy:<8.4f} {r.precision:<8.4f} {r.recall:<8.4f} "
              f"{r.f1_score:<8.4f} {r.auc_roc:<8.4f}{marker}")
    print(f"{'-'*70}")
    print(f"\nBest Model: {best_result.config.name} with F1-Score = {best_result.f1_score:.4f}")
    
    return md_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    args = parse_args()

    DATA = args.data or DEFAULT_DATA
    OUT = args.output or DEFAULT_OUT
    RES = args.results or DEFAULT_RES

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    # Hyperparameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = TRAINING_HYPERPARAMS.get("learning_rate", 5e-4)
    WEIGHT_DECAY = TRAINING_HYPERPARAMS.get("weight_decay", 1e-5)
    PATIENCE = TRAINING_HYPERPARAMS.get("early_stopping_patience", 25)

    print(f"Data:   {DATA}")
    print(f"Output: {OUT}")
    print(f"Results: {RES}")
    print(f"Device: {args.device}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading data...")
    X, y = load_data(DATA)
    print(f"Loaded: {X.shape}, Classes: {np.bincount(y.astype(int))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Run ablation study or single training
    if args.mode == "ablation":
        print("\n" + "="*70)
        print("RUNNING ABLATION STUDY MODE")
        print("="*70)
        
        # Check thop availability
        if not THOP_AVAILABLE:
            print("\nWarning: 'thop' library not installed. GFLOPs will not be calculated.")
            print("Install with: pip install thop")
        
        # Run ablation study
        results = run_ablation_study(
            variants=ABLATION_VARIANTS,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            out_dir=OUT,
            res_dir=RES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            device=device,
            skip_variants=args.skip_ablation
        )
        
        # Generate report
        report_path = generate_ablation_report(results, RES)
        
        print(f"\n{'='*70}")
        print("Ablation study completed!")
        print(f"Report saved to: {report_path}")
        print("="*70)
        
    else:
        # Single training mode (original behavior)
        print("\n" + "="*50)
        print("SINGLE TRAINING MODE")
        print("="*50)
        
        # Use default config from TRAINING_HYPERPARAMS
        D_MODEL = TRAINING_HYPERPARAMS.get("d_model", 256)
        NHEAD = TRAINING_HYPERPARAMS.get("nhead", 4)
        NLAYER = TRAINING_HYPERPARAMS.get("num_layers", 3)
        
        config = AblationConfig(
            name="Single",
            d_model=D_MODEL,
            num_layers=NLAYER,
            nhead=NHEAD,
            dropout=TRAINING_HYPERPARAMS.get("dropout", 0.1)
        )
        
        result = train_single_variant(
            config=config,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            base_out_dir=OUT,
            base_res_dir=RES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            device=device
        )
        
        print(f"\nTraining completed!")
        print(f"  F1-Score: {result.f1_score:.4f}")
        print(f"  Results saved to: {RES}")

    print("\nDONE!")


if __name__ == "__main__":
    main()
