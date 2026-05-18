#!/usr/bin/env python
"""
Train HybridFallTransformer for Fall Detection.

This script wraps src.trainer for CLI usage with additional command-line arguments.

Usage:
    python train.py --data /path/to/processed --output /path/to/output
    python train.py -e 50 -b 32
"""

import os
import sys
import json
import random
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train HybridFallTransformer")
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
    return parser.parse_args()


def main():
    args = parse_args()

    DATA = args.data or DEFAULT_DATA
    OUT = args.output or DEFAULT_OUT
    RES = args.results or DEFAULT_RES

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    # Hyperparameters
    D_MODEL = TRAINING_HYPERPARAMS.get("d_model", 256)
    NHEAD = TRAINING_HYPERPARAMS.get("nhead", 4)
    NLAYER = TRAINING_HYPERPARAMS.get("num_layers", 3)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = TRAINING_HYPERPARAMS.get("learning_rate", 5e-4)
    WEIGHT_DECAY = TRAINING_HYPERPARAMS.get("weight_decay", 1e-5)
    PATIENCE = TRAINING_HYPERPARAMS.get("early_stopping_patience", 25)

    print(f"Data:   {DATA}")
    print(f"Output: {OUT}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")

    # Set seeds
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

    # Create dataloaders (use centralized FallDataset from src.trainer)
    num_workers = 4
    train_loader = DataLoader(
        FallDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        FallDataset(X_val, y_val, augment=False),
        batch_size=BATCH_SIZE,
        num_workers=num_workers, pin_memory=True
    )

    # Model
    model = HybridFallTransformer(
        input_dim=60, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NLAYER, dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_f1, patience = 0, 0

    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)

    for epoch in range(EPOCHS):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        f1 = val_metrics.get("f1", 0.0)

        scheduler.step(f1)

        if (epoch + 1) % 5 == 0 or f1 > best_f1:
            print(f"Epoch {epoch+1:3d}: Loss={train_metrics.get('loss', 0):.4f}, Val F1={f1:.4f}")

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(OUT, 'best_model.pth'))
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Test evaluation
    print("\n" + "="*50)
    print("TEST EVALUATION")
    print("="*50)

    model.load_state_dict(torch.load(os.path.join(OUT, 'best_model.pth'), weights_only=True))
    model.eval()

    test_loader = DataLoader(
        Dataset.__new__(Dataset),  # Placeholder, we evaluate directly
        batch_size=BATCH_SIZE
    )
    # Create simple test dataset
    class _TestDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    test_loader = DataLoader(_TestDataset(X_test, y_test), batch_size=BATCH_SIZE)
    test_metrics = evaluate(model, test_loader, criterion, device)

    acc = test_metrics.get("accuracy", 0.0)
    pre = test_metrics.get("precision", 0.0)
    rec = test_metrics.get("recall", 0.0)
    f1 = test_metrics.get("f1", 0.0)
    auc = test_metrics.get("auc", 0.0)
    cm = test_metrics.get("confusion_matrix", [[0,0],[0,0]])

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    metrics = {
        'accuracy': float(acc),
        'precision': float(pre),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm,
        'best_val_f1': float(best_f1),
        'num_params_M': float(num_params),
        'hyperparameters': {
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_layers': NLAYER,
            'epochs_trained': epoch + 1,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'weight_decay': WEIGHT_DECAY,
        }
    }

    with open(os.path.join(RES, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Copy model to results
    shutil.copy(os.path.join(OUT, 'best_model.pth'), os.path.join(RES, 'best_model.pth'))

    # Auto zip
    shutil.make_archive(RES, 'zip', RES)

    print(f"\nResults saved to: {RES}")
    print("\nDONE!")


if __name__ == "__main__":
    main()
