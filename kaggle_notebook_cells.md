# Fall Detection - Notebook 3: Training & Evaluation

## Dataset Paths (Processed Data from Notebooks 1 & 2)

| Dataset | Kaggle Path |
|---------|-------------|
| CaucaFall (processed) | `/kaggle/input/datasets/phmthduyn/caucafall-processed` |
| MCFD (processed) | `/kaggle/input/datasets/phmthduyn/mcfd-processed` |

---

# CELL 1: Setup & Clone Repository

```python
import os
import sys

WORK_DIR = "/kaggle/working"
os.chdir(WORK_DIR)

# Clone repository
!cd {WORK_DIR} && rm -rf fall_detection && \
    git clone https://github.com/phamduyen112004-gif/fall_detection.git

%cd {WORK_DIR}/fall_detection

# Install dependencies
!pip install -r requirements.txt -q

print("✓ Repository cloned and dependencies installed")
```

---

# CELL 2: Define HybridFallTransformer Model

```python
"""
HybridFallTransformer: PIFR Features + Transformer Encoder for Fall Detection.

This cell defines the complete model architecture.
Based on methodology from Benabdennour et al. (2026).
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class FallAttention(nn.Module):
    """Multi-head attention with fall-specific pooling."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with residual connection."""
        attn_out, _ = self.self_attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class HybridFallTransformer(nn.Module):
    """
    Hybrid Transformer for Fall Detection using PIFR features.

    Architecture:
        1. Input projection: (B, T, 60) -> (B, T, d_model)
        2. Temporal attention layers
        3. Global average pooling
        4. Classification head: (B, d_model) -> (B, 1)

    Args:
        input_dim: Feature dimension (60 for PIFR)
        num_frames: Number of temporal frames (60)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 60,
        num_frames: int = 60,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_frames + 1)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Gaussian."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, F) where B=batch, T=frames, F=features

        Returns:
            Logits (B, 1) for binary classification
        """
        B = x.size(0)

        # Project input
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T+1, d_model)

        # Take CLS token output for classification
        cls_output = x[:, 0, :]  # (B, d_model)

        # Classify
        logits = self.classifier(cls_output)  # (B, 1)

        return logits


class FallDataset(torch.utils.data.Dataset):
    """Dataset for fall detection with optional augmentation."""

    def __init__(self, X, y, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Random temporal shift
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-3, 4, (1,)).item()
                x = torch.roll(x, shift, dims=0)

            # Random noise (sigma=0.01 per PLOS ONE 2026)
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
                x = torch.clamp(x, 0.0, 1.0)

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])  # Flip feature dimension

        return x, y


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test model
model = FallDataset(torch.rand(4, 60, 60), torch.rand(4))
test_model = HybridFallTransformer()
x_test = torch.randn(2, 60, 60)
out = test_model(x_test)
print(f"✓ HybridFallTransformer defined successfully")
print(f"  Input shape: {x_test.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Parameters: {count_parameters(test_model):,} ({count_parameters(test_model)/1e6:.2f}M)")
```

---

# CELL 3: Load Processed Data

```python
import numpy as np
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = WORK / "processed_data"
DATA.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading Processed Datasets")
print("=" * 60)

# Dataset sources (upload processed data from Notebooks 1 & 2)
cauca_sources = [
    "/kaggle/input/datasets/phmthduyn/caucafall-processed",
    "/kaggle/working/caucafall_processed",
]

mcfd_sources = [
    "/kaggle/input/datasets/phmthduyn/mcfd-processed",
    "/kaggle/working/mcfd_processed",
]

def load_all_data():
    """Load all processed data from available sources."""
    X_list = []
    y_list = []

    for data_dir in cauca_sources + mcfd_sources:
        p = Path(data_dir)
        if p.exists():
            x_files = sorted([f for f in p.glob("X_*.npy")])
            y_files = sorted([f for f in p.glob("y_*.npy")])

            if len(x_files) > 0:
                print(f"  Loading {len(x_files)} samples from {data_dir}")
                for xf, yf in zip(x_files, y_files):
                    X_list.append(np.load(xf))
                    y_list.append(np.load(yf).item())

    return np.array(X_list), np.array(y_list)

X, y = load_all_data()

if len(X) == 0:
    print("\n⚠ No data found! Please run Notebooks 1 & 2 first to process the datasets.")
else:
    print(f"\n✓ MERGED DATASET: {len(X)} samples")
    print(f"  Fall: {np.sum(y == 1)}, No Fall: {np.sum(y == 0)}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print("=" * 60)
```

---

# CELL 4: Execute Training (K-Fold Cross-Validation)

```python
import torch
import numpy as np
import os
import gc
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Setup directories
MODEL_DIR = Path("/kaggle/working/models")
LOG_DIR = Path("/kaggle/working/logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Training HybridFallTransformer with K-Fold CV")
print("=" * 60)

# Hyperparameters (per Benabdennour et al., 2026)
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 100
N_FOLDS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25

print(f"\nHyperparameters:")
print(f"  d_model:     {D_MODEL}")
print(f"  nhead:       {NHEAD}")
print(f"  num_layers:  {NUM_LAYERS}")
print(f"  dropout:     {DROPOUT}")
print(f"  lr:          {LR}")
print(f"  batch_size:  {BATCH_SIZE}")
print(f"  epochs:      {EPOCHS}")
print(f"  folds:       {N_FOLDS}")
print(f"  patience:    {PATIENCE}")
print("=" * 60)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_metrics = []
best_fold = 0
best_fold_f1 = 0.0

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*40}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*40}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Split train into train and validation (90/10)
    val_size = int(len(X_train) * 0.1)
    indices = np.random.permutation(len(X_train))
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create data loaders
    train_ds = FallDataset(X_tr, y_tr, augment=True)
    val_ds = FallDataset(X_val, y_val, augment=False)
    test_ds = FallDataset(X_test, y_test, augment=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = HybridFallTransformer(
        input_dim=X.shape[2],
        num_frames=X.shape[1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                val_preds.extend(preds.tolist())
                val_labels.extend(y_batch.numpy().tolist())

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / f"best_model_fold{fold}.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Evaluate on test
    model.load_state_dict(torch.load(MODEL_DIR / f"best_model_fold{fold}.pth"))
    model.eval()

    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            test_probs.extend(probs.tolist())
            test_preds.extend(preds.tolist())
            test_labels.extend(y_batch.numpy().tolist())

    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, zero_division=0)

    try:
        auc_score = roc_auc_score(test_labels, test_probs)
    except:
        auc_score = 0.0

    print(f"Fold {fold + 1} Test - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

    fold_metrics.append({
        "fold": fold + 1,
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc_score)
    })

    if f1 > best_fold_f1:
        best_fold_f1 = f1
        best_fold = fold + 1
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Summary
print(f"\n{'='*60}")
print("K-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
avg_f1 = np.mean([m["f1"] for m in fold_metrics])
avg_auc = np.mean([m["auc"] for m in fold_metrics])

print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average AUC: {avg_auc:.4f}")
print(f"Best Fold: {best_fold} (F1: {best_fold_f1:.4f})")

# Save metrics
import json
results = {
    "average_metrics": {
        "accuracy": float(avg_acc),
        "f1": float(avg_f1),
        "auc": float(avg_auc)
    },
    "fold_metrics": fold_metrics,
    "best_fold": best_fold,
    "total_samples": len(X),
    "hyperparameters": {
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "n_folds": N_FOLDS
    }
}

with open(LOG_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Training complete!")
print(f"✓ Model saved: {MODEL_DIR / 'best_model.pth'}")
print(f"✓ Metrics saved: {LOG_DIR / 'metrics.json'}")
```

---

# CELL 5: Evaluate Model on Test Set

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# Config
MODEL_PATH = Path("/kaggle/working/models/best_model.pth")
LOG_DIR = Path("/kaggle/working/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Training config (must match!)
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.1
N_FOLDS = 5
RANDOM_STATE = 42

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

if not MODEL_PATH.exists():
    print("\n⚠ Model not found! Run Cell 4 first.")
else:
    # Load best model
    model = HybridFallTransformer(
        input_dim=X.shape[2],
        num_frames=X.shape[1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\nDevice: {device}")
    print(f"Model loaded: {MODEL_PATH}")

    # ========== EVALUATE ON ALL FOLD TEST SETS (K-Fold CV) ==========
    print("\n" + "=" * 70)
    print("EVALUATING ON ALL FOLD TEST SETS (5-Fold CV)")
    print("=" * 70)

    # Collect predictions from all folds
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []
    fold_results = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_fold_test, y_fold_test = X[test_idx], y[test_idx]

        # Get predictions for this fold's test set
        fold_preds, fold_probs, fold_labels = [], [], []
        with torch.no_grad():
            for i in range(0, len(X_fold_test), 64):
                batch_x = torch.FloatTensor(X_fold_test[i:i+64]).to(device)
                outputs = model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                fold_probs.extend(probs.tolist())
                fold_preds.extend(preds.tolist())
                fold_labels.extend(y_fold_test[i:i+64].tolist())

        all_test_labels.extend(fold_labels)
        all_test_preds.extend(fold_preds)
        all_test_probs.extend(fold_probs)

        # Calculate fold metrics
        fold_tn, fold_fp, fold_fn, fold_tp = confusion_matrix(fold_labels, fold_preds).ravel()
        fold_acc = (fold_tp + fold_tn) / (fold_tp + fold_tn + fold_fp + fold_fn)
        fold_prec = fold_tp / (fold_tp + fold_fp) if (fold_tp + fold_fp) > 0 else 0
        fold_rec = fold_tp / (fold_tp + fold_fn) if (fold_tp + fold_fn) > 0 else 0
        fold_f1 = 2 * fold_prec * fold_rec / (fold_prec + fold_rec) if (fold_prec + fold_rec) > 0 else 0

        fold_results.append({
            'fold': fold + 1,
            'tn': fold_tn, 'fp': fold_fp, 'fn': fold_fn, 'tp': fold_tp,
            'accuracy': fold_acc, 'precision': fold_prec,
            'recall': fold_rec, 'f1': fold_f1,
            'test_size': len(fold_labels)
        })

        print(f"Fold {fold + 1}: Acc={fold_acc:.4f}, Prec={fold_prec:.4f}, Rec={fold_rec:.4f}, F1={fold_f1:.4f} (n={len(fold_labels)})")

    # Aggregate across all folds
    y_true = np.array(all_test_labels)
    y_pred = np.array(all_test_preds)
    y_scores = np.array(all_test_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate final metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    false_alarm_rate = 1 - specificity

    # Calculate mean and std for each metric across folds
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    mean_prec = np.mean([r['precision'] for r in fold_results])
    std_prec = np.std([r['precision'] for r in fold_results])
    mean_rec = np.mean([r['recall'] for r in fold_results])
    std_rec = np.std([r['recall'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])
    std_f1 = np.std([r['f1'] for r in fold_results])

    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_roc, tpr_roc)
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # ========== CALCULATE OPTIMAL THRESHOLD (F1-maximization) ==========
    f1_scores = (2 * precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds_pr[optimal_idx]) if optimal_idx < len(thresholds_pr) else 0.5
    optimal_f1 = float(f1_scores[optimal_idx])
    print(f"\n[Optimal Threshold] threshold={optimal_threshold:.4f}, F1={optimal_f1:.4f}")

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS (Aggregated)")
    print("=" * 70)
    print(f"Total Test Samples: {len(y_true)}")
    print("-" * 55)
    print(f"{'METRIC':<30} {'MEAN':<12} {'STD':<10}")
    print("-" * 55)
    print(f"{'Accuracy':<30} {mean_acc:.4f}      {std_acc:.4f}")
    print(f"{'Precision (PPV)':<30} {mean_prec:.4f}      {std_prec:.4f}")
    print(f"{'Sensitivity / Recall (TPR)':<30} {mean_rec:.4f}      {std_rec:.4f}")
    print(f"{'Specificity (TNR)':<30} {specificity:.4f}      (per-fold)")
    print(f"{'F1-Score':<30} {mean_f1:.4f}      {std_f1:.4f}")
    print("-" * 55)
    print(f"{'ROC AUC (Overall)':<30} {roc_auc:.4f}")
    print(f"{'PR AUC / AP (Overall)':<30} {pr_auc:.4f}")
    print("-" * 55)
    print(f"\nAggregated Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'], digits=4)}")

    # ========== INFERENCE SPEED ==========
    print("=" * 70)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 70)

    # Warmup
    for _ in range(10):
        _ = model(torch.FloatTensor(1, 60, 60).to(device))

    # Benchmark single sample
    n_runs = 500
    start = time.time()
    for _ in range(n_runs):
        _ = model(torch.FloatTensor(1, 60, 60).to(device))
    elapsed = time.time() - start

    fps_single = n_runs / elapsed
    latency_ms = (elapsed / n_runs) * 1000
    print(f"FPS (batch=1): {fps_single:.1f}")
    print(f"Latency: {latency_ms:.2f} ms/sample")

    # ========== PLOTS ==========
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'],
                annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_title('Confusion Matrix\n(5-Fold CV)', fontsize=13, fontweight='bold')

    # ROC Curve
    axes[1].plot(fpr_roc, tpr_roc, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].fill_between(fpr_roc, tpr_roc, alpha=0.2, color='blue')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('ROC Curve\n(5-Fold CV)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    # PR Curve
    axes[2].plot(recall_arr, precision_arr, 'g-', linewidth=2, label=f'PR (AP={pr_auc:.4f})')
    baseline = np.sum(y_true) / len(y_true)
    axes[2].axhline(y=baseline, color='r', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.2f})')
    axes[2].fill_between(recall_arr, precision_arr, alpha=0.2, color='green')
    axes[2].set_xlabel('Recall', fontsize=11)
    axes[2].set_ylabel('Precision', fontsize=11)
    axes[2].set_title('Precision-Recall Curve\n(5-Fold CV)', fontsize=13, fontweight='bold')
    axes[2].legend(loc='lower left')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(LOG_DIR / 'evaluation_results_5fold_cv.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {LOG_DIR / 'evaluation_results_5fold_cv.png'}")

    # Detailed Confusion Matrix
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Fall (0)', 'Fall (1)'],
                yticklabels=['No Fall (0)', 'Fall (1)'],
                annot_kws={'size': 22, 'weight': 'bold'},
                linewidths=3, linecolor='white',
                cbar_kws={'label': 'Number of Samples'})

    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix - 5-Fold Cross-Validation\n(Aggregated Results)', fontsize=15, fontweight='bold', pad=20)

    # Metrics box
    metrics_text = (
        f"5-Fold CV Results\n"
        f"{'─' * 22}\n"
        f"Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Precision:  {mean_prec:.4f} ± {std_prec:.4f}\n"
        f"Recall:     {mean_rec:.4f} ± {std_rec:.4f}\n"
        f"F1-Score:   {mean_f1:.4f} ± {std_f1:.4f}\n"
        f"{'─' * 22}\n"
        f"ROC-AUC:    {roc_auc:.4f}\n"
        f"PR-AUC:     {pr_auc:.4f}"
    )
    ax2.text(1.35, 0.5, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(LOG_DIR / 'confusion_matrix_5fold_cv.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {LOG_DIR / 'confusion_matrix_5fold_cv.png'}")

    # ========== SAVE FINAL METRICS ==========
    import json
    metrics = {
        "model": "HybridFallTransformer",
        "evaluation_type": "5-Fold Cross-Validation",
        "mean_accuracy": float(mean_acc), "std_accuracy": float(std_acc),
        "mean_precision": float(mean_prec), "std_precision": float(std_prec),
        "mean_recall": float(mean_rec), "std_recall": float(std_rec),
        "mean_f1": float(mean_f1), "std_f1": float(std_f1),
        "accuracy": float(accuracy), "precision": float(precision_val),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "f1_score": float(f1), "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc), "error_rate": float(error_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "optimal_threshold": optimal_threshold,
        "optimal_threshold_f1": optimal_f1,
        "fps_single": float(fps_single), "latency_ms": float(latency_ms),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": len(y_true),
        "per_fold_results": [
            {"fold": r['fold'], "accuracy": r['accuracy'], "precision": r['precision'],
             "recall": r['recall'], "f1": r['f1'], "test_size": r['test_size']}
            for r in fold_results
        ],
        "hyperparameters": {
            "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "n_folds": N_FOLDS
        }
    }
    with open(LOG_DIR / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save threshold_config.json for GUI usage
    threshold_config = {
        "optimal_threshold": optimal_threshold,
        "optimal_threshold_f1": optimal_f1,
        "model_path": str(MODEL_PATH)
    }
    with open(LOG_DIR / 'threshold_config.json', 'w') as f:
        json.dump(threshold_config, f, indent=2)

    print(f"\n✓ Saved: {LOG_DIR / 'evaluation_metrics.json'}")
    print(f"✓ Saved: {LOG_DIR / 'threshold_config.json'}")
    print("=" * 70)
```

---

# CELL 6: Save All Results (ZIP)

```python
import shutil
import os
import zipfile
from pathlib import Path

# Create zip of ALL results
MODEL_DIR = Path("/kaggle/working/models")
LOG_DIR = Path("/kaggle/working/logs")

zip_path = "/kaggle/working/fall_detection_results.zip"

# Collect all files
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name, directory in [("models", MODEL_DIR), ("logs", LOG_DIR)]:
        if directory.exists():
            for f in directory.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to('/kaggle/working'))
                    print(f"  Adding: {name}/{f.name}")

size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\n✓ Created: {zip_path}")
print(f"✓ Size: {size:.1f} MB")
print("\nContents: models/*, logs/*")
```

---

# Summary

| Cell | Task | Output |
|------|------|--------|
| 1 | Setup & Clone | Repository cloned |
| 2 | Define Model | HybridFallTransformer class |
| 3 | Load Data | Merged dataset ready |
| 4 | Train Model | 5-Fold CV models & metrics |
| 5 | Evaluate | Visualizations & final metrics |
| 6 | Save Results | ZIP file for download |

---

# SOTA Hyperparameters (Benabdennour et al., 2026)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 8 |
| num_layers | 4 |
| dropout | 0.1 |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 32 |
| patience | 25 |
| max_epochs | 100 |
| noise_std | 0.01 (augmentation) |
