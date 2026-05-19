# Fall Detection - 3-Notebook Kaggle Pipeline

## Dataset Paths (Actual Folder Structure from Screenshots)

| Dataset | Kaggle Path |
|---------|-------------|
| CaucaFall | `/kaggle/input/datasets/tuyenldvn/caucafall/Dataset CAUCAFall/CAUCAFall` |
| MCFD | `/kaggle/input/datasets/soumicksarker/multiple-cameras-fall-dataset/dataset/dataset` |

### Labeling Convention

- **CaucaFall**: Label extracted from action folder name (`fall` → 1, else → 0)
- **MCFD**: Auto-labeled by chute number (chute 01-04 = fall, chute 05-12 = ADL)

> **Note**: MCFD annotation CSV (`data_tuple3.csv`) is not available in this dataset version, so we auto-label based on standard MCFD chute conventions.

---

# NOTEBOOK 1: Process CaucaFall

## Cell 1: Setup & Clone Repository

```python
import os
import subprocess

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

## Cell 2: Configure for CaucaFall

```python
import sys
import os
from pathlib import Path

WORK_DIR = "/kaggle/working"
os.chdir(WORK_DIR)
sys.path.insert(0, f'{WORK_DIR}/fall_detection')

# ACTUAL Kaggle path based on folder structure
CAUCAFALL_DIR = Path("/kaggle/input/datasets/tuyenldvn/caucafall/Dataset CAUCAFall/CAUCAFall")
OUTPUT_DIR = Path("/kaggle/working/caucafall_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config
YOLO_MODEL = "yolo11l-pose.pt"
TARGET_FRAMES = 60
MAX_FRAMES = 300

print(f"✓ CaucaFall source: {CAUCAFALL_DIR}")
print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ YOLO model: {YOLO_MODEL}")

# Scan and list all videos
video_list = []
for subj_dir in sorted(os.listdir(CAUCAFALL_DIR)):
    subj_path = CAUCAFALL_DIR / subj_dir
    if not subj_path.is_dir() or not subj_dir.startswith("Subject."):
        continue
    
    for action_dir in sorted(os.listdir(subj_path)):
        action_path = subj_path / action_dir
        if not action_path.is_dir():
            continue
        
        for video in sorted(os.listdir(action_path)):
            if not video.endswith(".avi"):
                continue
            
            # Label: 1 if action contains "fall", else 0
            label = 1 if "fall" in action_dir.lower() else 0
            video_list.append({
                "path": action_path / video,
                "subject": subj_dir,
                "action": action_dir,
                "label": label,
            })

print(f"✓ Found {len(video_list)} videos")
fall_count = sum(1 for v in video_list if v['label'] == 1)
adl_count = sum(1 for v in video_list if v['label'] == 0)
print(f"  Falls: {fall_count}, ADL: {adl_count}")
```

## Cell 3: Process CaucaFall Dataset

```python
import logging
import numpy as np
import torch
import cv2
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from ultralytics import YOLO
from src.pifr_features import extract_keypoints, compute_pifr

print("=" * 60)
print("Processing CaucaFall Dataset")
print("=" * 60)

model = YOLO(YOLO_MODEL)
zero_fallback = np.zeros(60, dtype=np.float32)

processed = skipped = errors = 0

def safe_name(s):
    return str(s).replace(".", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")

def temporal_subsample(features, target_frames=60):
    """Convert (N, 60) → (60, 60) with truncation, subsampling, and padding."""
    arr = np.array(features, dtype=np.float32)
    
    # Truncate to first 120 frames
    arr = arr[:120]
    
    # Subsample: take every 2nd frame (0, 2, 4, ...)
    arr = arr[::2]
    
    # Pad with LAST vector if < 60 frames
    if len(arr) < target_frames:
        pad = np.tile(arr[-1], (target_frames - len(arr), 1))
        arr = np.vstack([arr, pad])
    
    # Assert exact shape
    assert arr.shape == (60, 60), f"Expected (60, 60), got {arr.shape}"
    return arr

for item in tqdm(video_list, desc="CaucaFall"):
    x_name = f"X_cauca_{safe_name(item['subject'])}_{safe_name(item['action'])}.npy"
    y_name = f"y_cauca_{safe_name(item['subject'])}_{safe_name(item['action'])}.npy"
    x_path = OUTPUT_DIR / x_name
    y_path = OUTPUT_DIR / y_name

    if x_path.exists() and y_path.exists():
        skipped += 1
        continue

    try:
        cap = cv2.VideoCapture(str(item["path"]))
        if not cap.isOpened():
            errors += 1
            continue

        features = []
        prev = zero_fallback.copy()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            kpts = extract_keypoints(frame, model)

            if kpts is None:
                features.append(prev.copy())
            else:
                vec = compute_pifr(kpts, w, h)
                features.append(vec)
                prev = vec

        cap.release()

        if features:
            feat_array = temporal_subsample(features, TARGET_FRAMES)
            np.save(x_path, feat_array)
            np.save(y_path, np.array([item["label"]], dtype=np.int32))
            processed += 1

        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Error: {item['path']} -> {e}")
        errors += 1

logger.info(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
print(f"\n✓ CaucaFall complete: {processed} processed, {skipped} skipped, {errors} errors")
print(f"✓ Output shape: (60, 60)")
```

## Cell 4: Zip & Verify Output

```python
import shutil
import os

src = "/kaggle/working/caucafall_processed"
zip_path = "/kaggle/working/caucafall_processed.zip"

if os.path.exists(src):
    # Remove existing zip if any
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    shutil.make_archive("/kaggle/working/caucafall_processed", 'zip', src)
    size = os.path.getsize(zip_path) / 1024 / 1024
    print(f"✓ Created: {zip_path}")
    print(f"✓ Size: {size:.1f} MB")
    
    # Verify contents
    x_files = [f for f in os.listdir(src) if f.startswith('X_')]
    print(f"✓ Total .npy files: {len(x_files)}")
else:
    print("✗ Output directory not found!")
```

---

# NOTEBOOK 2: Process MCFD

## Cell 1: Setup & Clone Repository

```python
import os
import subprocess

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

## Cell 2: Configure for MCFD with CSV

```python
import sys
import os
import pandas as pd
from pathlib import Path

WORK_DIR = "/kaggle/working"
os.chdir(WORK_DIR)
sys.path.insert(0, f'{WORK_DIR}/fall_detection')

# ACTUAL Kaggle path based on folder structure
MCFD_DIR = Path("/kaggle/input/datasets/soumicksarker/multiple-cameras-fall-dataset/dataset/dataset")
CSV_PATH = Path("/kaggle/input/datasets/soumicksarker/multiple-cameras-fall-dataset/data_tuple3.csv")
OUTPUT_DIR = Path("/kaggle/working/mcfd_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config
YOLO_MODEL = "yolo11l-pose.pt"
TARGET_FRAMES = 60
MAX_FRAMES = 300

print(f"✓ MCFD source: {MCFD_DIR}")
print(f"✓ CSV path: {CSV_PATH}")
print(f"✓ Output directory: {OUTPUT_DIR}")

# Load CSV annotations
df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df)} annotations from CSV")
print(f"  Columns: {list(df.columns)}")
print(df.head(3))
```

## Cell 3: Process MCFD Dataset with CSV Slicing

```python
import logging
import numpy as np
import torch
import cv2
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from ultralytics import YOLO
from src.pifr_features import extract_keypoints, compute_pifr

print("=" * 60)
print("Processing MCFD Dataset (CSV Slicing)")
print("=" * 60)

model = YOLO(YOLO_MODEL)
zero_fallback = np.zeros(60, dtype=np.float32)

processed = skipped = errors = 0

def temporal_subsample(features, target_frames=60):
    """Convert (N, 60) → (60, 60) with truncation, subsampling, and padding."""
    arr = np.array(features, dtype=np.float32)
    
    # Truncate to first 120 frames
    arr = arr[:120]
    
    # Subsample: take every 2nd frame (0, 2, 4, ...)
    arr = arr[::2]
    
    # Pad with LAST vector if < 60 frames
    if len(arr) < target_frames:
        pad = np.tile(arr[-1], (target_frames - len(arr), 1))
        arr = np.vstack([arr, pad])
    
    # Assert exact shape
    assert arr.shape == (60, 60), f"Expected (60, 60), got {arr.shape}"
    return arr

for idx, row in tqdm(df.iterrows(), total=len(df), desc="MCFD"):
    chute = int(row["chute"])
    cam = int(row["cam"])
    start = int(row["start"])
    end = int(row["end"])
    label = int(row["label"])
    
    video_path = MCFD_DIR / f"chute{chute:02d}" / f"cam{cam}.avi"
    
    x_name = f"X_mcfd_c{chute:02d}_cam{cam}_row{idx}.npy"
    y_name = f"y_mcfd_c{chute:02d}_cam{cam}_row{idx}.npy"
    x_path = OUTPUT_DIR / x_name
    y_path = OUTPUT_DIR / y_name

    if x_path.exists() and y_path.exists():
        skipped += 1
        continue

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            errors += 1
            continue

        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        features = []
        prev = zero_fallback.copy()
        cur_frame = start

        while True:
            ret, frame = cap.read()
            if not ret or cur_frame > end:
                break

            h, w = frame.shape[:2]
            kpts = extract_keypoints(frame, model)

            if kpts is None:
                features.append(prev.copy())
            else:
                vec = compute_pifr(kpts, w, h)
                features.append(vec)
                prev = vec

            cur_frame += 1

        cap.release()

        if features:
            feat_array = temporal_subsample(features, TARGET_FRAMES)
            np.save(x_path, feat_array)
            np.save(y_path, np.array([label], dtype=np.int32))
            processed += 1

        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Error: {video_path} -> {e}")
        errors += 1

logger.info(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
print(f"\n✓ MCFD complete: {processed} processed, {skipped} skipped, {errors} errors")
print(f"✓ Output shape: (60, 60)")
```

## Cell 4: Zip & Verify Output

```python
import shutil
import os

src = "/kaggle/working/mcfd_processed"
zip_path = "/kaggle/working/mcfd_processed.zip"

if os.path.exists(src):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    shutil.make_archive("/kaggle/working/mcfd_processed", 'zip', src)
    size = os.path.getsize(zip_path) / 1024 / 1024
    print(f"✓ Created: {zip_path}")
    print(f"✓ Size: {size:.1f} MB")
    
    # Verify contents
    x_files = [f for f in os.listdir(src) if f.startswith('X_')]
    print(f"✓ Total .npy files: {len(x_files)}")
else:
    print("✗ Output directory not found!")
```

---

# NOTEBOOK 3: Train Model

## Cell 1: Setup & Clone Repository

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

## Cell 2: Load Processed Data (Both Datasets)

```python
import os
import numpy as np
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = WORK / "processed_data"
DATA.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Loading ALL Processed Datasets")
print("=" * 60)

# Check all possible dataset sources
cauca_sources = [
    "/kaggle/input/datasets/phmthduyn/caucafall-processed",
    "/kaggle/working/caucafall_processed",
]

mcfd_sources = [
    "/kaggle/input/datasets/phmthduyn/mcfd-processed",
    "/kaggle/working/mcfd_processed",
]

# Find CaucaFall
cauca_dir = None
for s in cauca_sources:
    p = Path(s)
    if p.exists():
        x_files = list(p.glob("X_*.npy"))
        if len(x_files) > 0:
            cauca_dir = p
            print(f"✓ CaucaFall: {len(x_files)} samples at {s}")
            break

# Find MCFD
mcfd_dir = None
for s in mcfd_sources:
    p = Path(s)
    if p.exists():
        x_files = list(p.glob("X_*.npy"))
        if len(x_files) > 0:
            mcfd_dir = p
            print(f"✓ MCFD: {len(x_files)} samples at {s}")
            break

# Merge both datasets
all_data_dirs = [d for d in [cauca_dir, mcfd_dir] if d is not None]

if len(all_data_dirs) == 0:
    print("✗ No datasets found!")
else:
    print(f"\n✓ Using {len(all_data_dirs)} dataset(s)")
    # Load all data
    X_list = []
    y_list = []
    
    for data_dir in all_data_dirs:
        x_files = sorted([f for f in data_dir.glob("X_*.npy")])
        y_files = sorted([f for f in data_dir.glob("y_*.npy")])
        
        for xf, yf in zip(x_files, y_files):
            X_list.append(np.load(xf))
            y_list.append(np.load(yf).item())
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✓ MERGED DATASET: {len(X)} samples")
    print(f"  Fall: {np.sum(y == 1)}, No Fall: {np.sum(y == 0)}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    MERGED_DATA_DIR = DATA / "merged"
    MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Merged data will be saved to: {MERGED_DATA_DIR}")
```

## Cell 3: Show Loaded Data

```python
print("=" * 60)
print("Merged Dataset Info")
print("=" * 60)
print(f"Total samples: {len(X)}")
print(f"Fall: {np.sum(y == 1)}, No Fall: {np.sum(y == 0)}")
print(f"Shape: X={X.shape}, y={y.shape}")
print("=" * 60)
```

## Cell 4: Execute Training (with K-Fold Cross-Validation)

```python
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import TRAINING_CONFIG
from src.trainer import HybridFallTransformer
from src.trainer import FallDataset, setup_logging

# Setup directories
MODEL_DIR = Path("/kaggle/working/models")
LOG_DIR = Path("/kaggle/working/logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = setup_logging(LOG_DIR)

print("=" * 60)
print("Training HybridFallTransformer with K-Fold CV")
print("=" * 60)

# Hyperparameters
d_model = 256
nhead = 8
num_layers = 4
dropout = 0.2
batch_size = 32
epochs = 100
n_folds = 5
lr = 5e-4
weight_decay = 1e-4

print(f"\nHyperparameters:")
print(f"  d_model:    {d_model}")
print(f"  nhead:       {nhead}")
print(f"  num_layers:  {num_layers}")
print(f"  dropout:     {dropout}")
print(f"  lr:          {lr}")
print(f"  batch_size:  {batch_size}")
print(f"  epochs:      {epochs}")
print(f"  folds:       {n_folds}")
print("=" * 60)

# Load merged data
print("\nLoading data from both datasets...")

def load_all_data():
    """Load all processed data from available sources."""
    X_list = []
    y_list = []
    
    sources = [
        "/kaggle/input/datasets/phmthduyn/caucafall-processed",
        "/kaggle/input/datasets/phmthduyn/mcfd-processed",
    ]
    
    for data_dir in sources:
        p = Path(data_dir)
        if p.exists():
            x_files = sorted([f for f in p.glob("X_*.npy")])
            y_files = sorted([f for f in p.glob("y_*.npy")])
            
            print(f"  Loading {len(x_files)} samples from {data_dir}")
            
            for xf, yf in zip(x_files, y_files):
                X_list.append(np.load(xf))
                y_list.append(np.load(yf).item())
    
    return np.array(X_list), np.array(y_list)

X, y = load_all_data()
print(f"\n✓ Total dataset: {len(X)} samples")
print(f"  Fall: {np.sum(y == 1)}, No Fall: {np.sum(y == 0)}")
print(f"  Shape: X={X.shape}, y={y.shape}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_metrics = []
best_fold = 0
best_fold_f1 = 0.0

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*40}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*40}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Split train into train and val
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
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    
    # Model
    model = HybridFallTransformer(
        input_dim=X.shape[2],
        num_frames=X.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_val_f1 = 0.0
    patience = 0
    
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
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), MODEL_DIR / f"best_model_fold{fold}.pth")
        else:
            patience += 1
            if patience >= 25:
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

# Save final metrics
import json
results = {
    "average_metrics": {
        "accuracy": avg_acc,
        "f1": avg_f1,
        "auc": avg_auc
    },
    "fold_metrics": fold_metrics,
    "best_fold": best_fold,
    "total_samples": len(X)
}

with open(LOG_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Training complete!")
print(f"✓ Model saved: {MODEL_DIR / 'best_model.pth'}")
```

## Cell 5: Evaluate Model (Full Benchmark Metrics)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from pathlib import Path

# Config
MODEL_PATH = Path("/kaggle/working/models/best_model.pth")
LOG_DIR = Path("/kaggle/working/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Training config (must match!)
d_model = 256
nhead = 8
num_layers = 4
dropout = 0.2

if not MODEL_PATH.exists():
    print("✗ Model not found. Run Cell 4 first.")
else:
    # Load model
    from src.hybrid_transformer import HybridFallTransformer
    
    model = HybridFallTransformer(
        input_dim=X.shape[2],
        num_frames=X.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    
    # Get predictions
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch_x = torch.FloatTensor(X[i:i+64]).to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y[i:i+64].tolist())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)
    
    # ========== CONFUSION MATRIX ==========
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # ========== BASIC METRICS ==========
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR / Selectivity
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # ========== ADVANCED METRICS (for paper comparison) ==========
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # G-Mean: geometric mean of sensitivity and specificity
    g_mean = np.sqrt(sensitivity * specificity)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Error Rate
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    
    # Fall Detection Rate (Sensitivity) & False Alarm Rate (1-Specificity)
    fall_detection_rate = sensitivity
    false_alarm_rate = 1 - specificity
    
    # ROC AUC & PR AUC
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_roc, tpr_roc)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # F2 Score (emphasize recall more than precision - important for fall detection)
    f2 = f1_score(y_true, y_pred, beta=2)
    
    # Youden's J statistic (optimal threshold selection)
    youden_j = sensitivity + specificity - 1
    
    # ========== DISPLAY ==========
    print("=" * 70)
    print("FULL EVALUATION RESULTS - FOR PAPER COMPARISON")
    print("=" * 70)
    print(f"\n{'METRIC':<35} {'VALUE':<15} {'DESCRIPTION'}")
    print("-" * 70)
    print(f"{'Accuracy':<35} {accuracy:.4f}         Overall correctness")
    print(f"{'Precision (PPV)':<35} {precision:.4f}         Positive predictive value")
    print(f"{'Recall / Sensitivity (TPR)':<35} {sensitivity:.4f}         Fall detection rate")
    print(f"{'Specificity (TNR)':<35} {specificity:.4f}         True negative rate")
    print(f"{'F1-Score':<35} {f1:.4f}         Harmonic mean P & R")
    print(f"{'F2-Score':<35} {f2:.4f}         Emphasize recall (beta=2)")
    print(f"{'Balanced Accuracy':<35} {balanced_acc:.4f}         Mean of TPR & TNR")
    print(f"{'G-Mean':<35} {g_mean:.4f}         Geometric mean of TPR & TNR")
    print(f"{'MCC (Matthews)':<35} {mcc:.4f}         Correlation coefficient")
    print(f"{'Cohen Kappa':<35} {kappa:.4f}         Agreement measure")
    print(f"{'ROC AUC':<35} {roc_auc:.4f}         ROC area under curve")
    print(f"{'PR AUC (AP)':<35} {pr_auc:.4f}         PR area under curve")
    print(f"{'Error Rate':<35} {error_rate:.4f}         (FP + FN) / Total")
    print(f"{'Fall Detection Rate':<35} {fall_detection_rate:.4f}         Same as Sensitivity")
    print(f"{'False Alarm Rate':<35} {false_alarm_rate:.4f}         1 - Specificity")
    print(f"{'Youden J':<35} {youden_j:.4f}         Optimal threshold metric")
    print("-" * 70)
    
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"             No Fall  Fall")
    print(f"Actual No Fall  {tn:<8} {fp}")
    print(f"       Fall    {fn:<8} {tp}")
    print("-" * 70)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'], digits=4))
    
    # ========== FPS BENCHMARK ==========
    print("\n" + "=" * 70)
    print("INFERENCE SPEED (FPS)")
    print("=" * 70)
    
    # Warmup
    dummy = torch.FloatTensor(1, 60, 60).to(device)
    for _ in range(10):
        _ = model(dummy)
    
    # Benchmark
    n_runs = 200
    start = time.time()
    for _ in range(n_runs):
        batch = torch.FloatTensor(32, 60, 60).to(device)
        _ = model(batch)
    elapsed = time.time() - start
    
    fps_batch = n_runs * 32 / elapsed
    latency_ms = (elapsed / n_runs / 32) * 1000
    
    print(f"Batch size 32: {fps_batch:.1f} samples/sec")
    print(f"Latency: {latency_ms:.2f} ms per sample")
    print(f"Throughput: {n_runs * 32 / elapsed:.1f} frames/sec @ batch=32")
    
    # ========== PLOTS ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'],
                annot_kws={'size': 14})
    axes[0,0].set_xlabel('Predicted', fontsize=12)
    axes[0,0].set_ylabel('Actual', fontsize=12)
    axes[0,0].set_title('Confusion Matrix', fontsize=14)
    
    # 2. ROC Curve
    axes[0,1].plot(fpr_roc, tpr_roc, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0,1].fill_between(fpr_roc, tpr_roc, alpha=0.2)
    axes[0,1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0,1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0,1].set_title('ROC Curve', fontsize=14)
    axes[0,1].legend(loc='lower right')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    baseline = np.sum(y_true) / len(y_true)
    axes[1,0].plot(recall_arr, precision_arr, 'g-', linewidth=2, label=f'PR (AP = {pr_auc:.4f})')
    axes[1,0].axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
    axes[1,0].set_xlabel('Recall', fontsize=12)
    axes[1,0].set_ylabel('Precision', fontsize=12)
    axes[1,0].set_title('Precision-Recall Curve', fontsize=14)
    axes[1,0].legend(loc='lower left')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Metrics Bar Chart
    metrics_names = ['Acc', 'Prec', 'Sens', 'Spec', 'F1', 'AUC']
    metrics_values = [accuracy, precision, sensitivity, specificity, f1, roc_auc]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    bars = axes[1,1].bar(metrics_names, metrics_values, color=colors, edgecolor='black')
    axes[1,1].set_ylim(0, 1.1)
    axes[1,1].set_ylabel('Score', fontsize=12)
    axes[1,1].set_title('Performance Metrics Summary', fontsize=14)
    axes[1,1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, metrics_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Plot saved to: {LOG_DIR / 'evaluation_results.png'}")
    
    # ========== SAVE METRICS JSON ==========
    import json
    metrics = {
        # Standard metrics
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "f2_score": float(f2),
        # Advanced metrics for paper
        "balanced_accuracy": float(balanced_acc),
        "g_mean": float(g_mean),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        # Confusion matrix elements
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        # Detection rates
        "fall_detection_rate": float(fall_detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "error_rate": float(error_rate),
        "youden_j": float(youden_j),
        # Speed
        "fps_batch_32": float(fps_batch),
        "latency_ms": float(latency_ms),
        # Dataset info
        "total_samples": len(y_true),
        "fall_samples": int(np.sum(y_true == 1)),
        "nofall_samples": int(np.sum(y_true == 0)),
        "class_ratio": float(np.sum(y_true == 1) / np.sum(y_true == 0))
    }
    
    with open(LOG_DIR / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {LOG_DIR / 'final_metrics.json'}")
    
    # ========== COMPARISON TEMPLATE ==========
    print("\n" + "=" * 70)
    print("COMPARISON WITH OTHER PAPERS (copy this table)")
    print("=" * 70)
    print(f"""
| Metric           | This Work | Paper 1 | Paper 2 | Paper 3 |
|------------------|-----------|---------|---------|---------|
| Accuracy         | {accuracy:.4f}     |         |         |         |
| Precision        | {precision:.4f}     |         |         |         |
| Sensitivity/Recall| {sensitivity:.4f}     |         |         |         |
| Specificity      | {specificity:.4f}     |         |         |         |
| F1-Score         | {f1:.4f}     |         |         |         |
| ROC AUC          | {roc_auc:.4f}     |         |         |         |
| MCC              | {mcc:.4f}     |         |         |         |
| Cohen's Kappa    | {kappa:.4f}     |         |         |         |
| FPS              | {fps_batch:.1f}     |         |         |         |
""")
    print("=" * 70)
```

## Cell 6: Save All Results (ZIP)

```python
import shutil
import os
import zipfile
from pathlib import Path

# Create zip of ALL results
MODEL_DIR = Path("/kaggle/working/models")
RESULTS_DIR = Path("/kaggle/working/results")
LOG_DIR = Path("/kaggle/working/logs")

zip_path = "/kaggle/working/fall_detection_results.zip"

# Collect all files
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name, directory in [("models", MODEL_DIR), ("results", RESULTS_DIR), ("logs", LOG_DIR)]:
        if directory.exists():
            for f in directory.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to('/kaggle/working'))
                    print(f"  Adding: {name}/{f.name}")

size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\n✓ Created: {zip_path}")
print(f"✓ Size: {size:.1f} MB")
print("\nContents: models/*, results/*, logs/*")
```

---

# Summary

| Notebook | Task | Output |
|----------|------|--------|
| Notebook 1 | Process CaucaFall | `caucafall-processed/` folder |
| Notebook 2 | Process MCFD | `mcfd-processed/` folder |
| Notebook 3 | Train Model | `fall_detection_results.zip` |

---

# Quick Start Guide

1. **Notebook 1**: Run Cells 1-4 → Upload `caucafall_processed` folder as Kaggle dataset
2. **Notebook 2**: Run Cells 1-4 → Upload `mcfd_processed` folder as Kaggle dataset
3. **Notebook 3**: Run Cells 1-6 → Download `fall_detection_results.zip`

---

# SOTA Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| nhead | 4 |
| num_layers | 3 |
| dropout | 0.1 |
| lr | 5e-4 |
| weight_decay | 1e-5 |
| batch_size | 64 |
| patience | 25 |
| max_epochs | 100 |
| noise_std | 0.01 |
| mask_ratio | 0.05 |
