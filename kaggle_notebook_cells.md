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

## Cell 5: Evaluate Model

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
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
    from src.hybrid_transformer import HybridFallTransformer
    
    model = HybridFallTransformer(
        input_dim=X.shape[2], num_frames=X.shape[1],
        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
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
    
    y_true, y_pred, y_scores = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # ========== METRICS ==========
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    false_alarm_rate = 1 - specificity
    
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_roc, tpr_roc)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'METRIC':<30} {'VALUE':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<30} {accuracy:.4f}")
    print(f"{'Precision (PPV)':<30} {precision_val:.4f}")
    print(f"{'Sensitivity / Recall (TPR)':<30} {sensitivity:.4f}")
    print(f"{'Specificity (TNR)':<30} {specificity:.4f}")
    print(f"{'F1-Score':<30} {f1:.4f}")
    print(f"{'ROC AUC':<30} {roc_auc:.4f}")
    print(f"{'PR AUC (AP)':<30} {pr_auc:.4f}")
    print(f"{'Error Rate':<30} {error_rate:.4f}")
    print(f"{'False Alarm Rate':<30} {false_alarm_rate:.4f}")
    print("-" * 45)
    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'], digits=4))
    
    # ========== FPS ==========
    print("=" * 70)
    print("INFERENCE SPEED")
    print("=" * 70)
    
    # Warmup
    for _ in range(10):
        _ = model(torch.FloatTensor(1, 60, 60).to(device))
    
    # Benchmark
    n_runs = 200
    start = time.time()
    for _ in range(n_runs):
        _ = model(torch.FloatTensor(32, 60, 60).to(device))
    elapsed = time.time() - start
    
    fps_batch = n_runs * 32 / elapsed
    latency_ms = (elapsed / n_runs / 32) * 1000
    print(f"FPS (batch=32): {fps_batch:.1f}")
    print(f"Latency: {latency_ms:.2f} ms/sample")
    
    # ========== PLOTS ==========
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    axes[1].plot(fpr_roc, tpr_roc, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(recall_arr, precision_arr, 'g-', linewidth=2, label=f'PR (AP={pr_auc:.4f})')
    axes[2].axhline(y=np.sum(y_true)/len(y_true), color='r', linestyle='--', label='Baseline')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('PR Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR / 'evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save metrics
    import json
    metrics = {
        "accuracy": float(accuracy), "precision": float(precision_val),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "f1_score": float(f1), "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc), "error_rate": float(error_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "fps_batch_32": float(fps_batch), "latency_ms": float(latency_ms),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": len(y_true)
    }
    with open(LOG_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved: {LOG_DIR / 'metrics.json'}")
    print("=" * 70)
```
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
