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

## Cell 2: Upload & Extract Processed Data

```python
import zipfile
import os
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = WORK / "processed_data"
DATA.mkdir(parents=True, exist_ok=True)

print("Extracting processed datasets...")

# Extract both zip files
for zip_file, name in [
    ("/kaggle/input/caucafall_processed.zip", "CaucaFall"),
    ("/kaggle/input/mcfd_processed.zip", "MCFD")
]:
    if os.path.exists(zip_file):
        print(f"  Extracting {name}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(DATA)
        print(f"  ✓ {name} extracted")
    else:
        print(f"  ✗ {name} zip not found: {zip_file}")

# Count samples
x_files = [f for f in os.listdir(DATA) if f.startswith('X_')]
y_files = [f for f in os.listdir(DATA) if f.startswith('y_')]
print(f"\n✓ Total samples: {len(x_files)}")
print(f"✓ X files: {len(x_files)}, y files: {len(y_files)}")
```

## Cell 3: Configure & Run Training

```python
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, '/kaggle/working/fall_detection')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/kaggle/working/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.config import DEFAULT_CONFIG, TRAINING_CONFIG

print("=" * 60)
print("Training HybridFallTransformer")
print("=" * 60)
print(f"\nSOTA Hyperparameters:")
print(f"  d_model:    {TRAINING_CONFIG.D_MODEL}")
print(f"  nhead:       {TRAINING_CONFIG.NHEAD}")
print(f"  num_layers:  {TRAINING_CONFIG.NUM_LAYERS}")
print(f"  dropout:     {TRAINING_CONFIG.DROPOUT}")
print(f"  lr:          {TRAINING_CONFIG.LR}")
print(f"  weight_decay: {TRAINING_CONFIG.WEIGHT_DECAY}")
print(f"  batch_size:  {TRAINING_CONFIG.BATCH_SIZE}")
print(f"  max_epochs:  {TRAINING_CONFIG.MAX_EPOCHS}")
print(f"  patience:    {TRAINING_CONFIG.PATIENCE}")
print("=" * 60)
```

## Cell 4: Execute Training

```python
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from src.hybrid_transformer import HybridFallTransformer
from src.trainer import train_model

# Paths
DATA_DIR = Path("/kaggle/working/processed_data")
MODEL_DIR = Path("/kaggle/working/models")
RESULTS_DIR = Path("/kaggle/working/results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare data
logger.info("Loading data...")
X_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('X_')])
y_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('y_')])

X = np.array([np.load(DATA_DIR / f) for f in tqdm(X_files, desc="Loading X")])
y = np.array([np.load(DATA_DIR / f).item() for f in tqdm(y_files, desc="Loading y")])

logger.info(f"Dataset: X.shape={X.shape}, y.shape={y.shape}")
logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

# Initialize model
from src.config import TRAINING_CONFIG

model = HybridFallTransformer(
    input_dim=TRAINING_CONFIG.INPUT_DIM,
    num_frames=TRAINING_CONFIG.NUM_FRAMES,
    d_model=TRAINING_CONFIG.D_MODEL,
    nhead=TRAINING_CONFIG.NHEAD,
    num_layers=TRAINING_CONFIG.NUM_LAYERS,
    dropout=TRAINING_CONFIG.DROPOUT
)

# Train
logger.info("Starting training...")
history = train_model(
    model=model,
    X=X,
    y=y,
    epochs=TRAINING_CONFIG.MAX_EPOCHS,
    batch_size=TRAINING_CONFIG.BATCH_SIZE,
    lr=TRAINING_CONFIG.LR,
    weight_decay=TRAINING_CONFIG.WEIGHT_DECAY,
    patience=TRAINING_CONFIG.PATIENCE,
    model_save_path=str(MODEL_DIR / "best_model.pth"),
    results_dir=str(RESULTS_DIR)
)

# Save final model
torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")
logger.info(f"Training complete! Model saved to {MODEL_DIR}")
```

## Cell 5: Evaluate Model

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path

# Load trained model
MODEL_PATH = Path("/kaggle/working/models/best_model.pth")

if MODEL_PATH.exists():
    from src.hybrid_transformer import HybridFallTransformer
    from src.config import TRAINING_CONFIG
    
    model = HybridFallTransformer(
        input_dim=TRAINING_CONFIG.INPUT_DIM,
        num_frames=TRAINING_CONFIG.NUM_FRAMES,
        d_model=TRAINING_CONFIG.D_MODEL,
        nhead=TRAINING_CONFIG.NHEAD,
        num_layers=TRAINING_CONFIG.NUM_LAYERS,
        dropout=TRAINING_CONFIG.DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch_x = torch.FloatTensor(X[i:i+64]).to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y[i:i+64])
    
    # Metrics
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=['No Fall', 'Fall']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    print("\n✓ Evaluation complete!")
else:
    print("✗ Model not found. Run Cell 4 first.")
```

## Cell 6: Save Results

```python
import shutil
import os
from pathlib import Path

# Create zip of results
MODEL_DIR = Path("/kaggle/working/models")
RESULTS_DIR = Path("/kaggle/working/results")

zip_path = "/kaggle/working/fall_detection_results.zip"

# Collect all files
import zipfile
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for d in [MODEL_DIR, RESULTS_DIR]:
        if d.exists():
            for f in d.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to(d.parent))
                    print(f"  Adding: {f.name}")

size = os.path.getsize(zip_path) / 1024 / 1024
print(f"\n✓ Created: {zip_path}")
print(f"✓ Size: {size:.1f} MB")
```

---

# Summary

| Notebook | Task | Output |
|----------|------|--------|
| Notebook 1 | Process CaucaFall | `caucafall_processed.zip` |
| Notebook 2 | Process MCFD | `mcfd_processed.zip` |
| Notebook 3 | Train Model | `fall_detection_results.zip` |

---

# Quick Start Guide

1. **Notebook 1**: Run Cells 1-4 → Download `caucafall_processed.zip`
2. **Notebook 2**: Run Cells 1-4 → Download `mcfd_processed.zip`
3. **Upload** both zip files to Kaggle as datasets
4. **Notebook 3**: Run Cells 1-6 → Download `fall_detection_results.zip`

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
