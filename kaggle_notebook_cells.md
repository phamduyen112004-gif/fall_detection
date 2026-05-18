# Fall Detection - 3-Notebook Kaggle Pipeline

## Dataset Paths

| Dataset | Kaggle Path |
|---------|-------------|
| CaucaFall | `/kaggle/input/caucafall/Dataset CAUCAFall/CAUCAFall` |
| MCFD | `/kaggle/input/multiple-cameras-fall-dataset/dataset/dataset` |
| MCFD CSV | `/kaggle/input/multiple-cameras-fall-dataset/data_tuple3.csv` |

---

# NOTEBOOK 1: Process CaucaFall

## Cell 1: Setup

```python
# Clone repository
!cd /kaggle/working && rm -rf fall_detection && \
    git clone https://github.com/phamduyen112004-gif/fall_detection.git
%cd /kaggle/working/fall_detection
!pip install -r requirements.txt -q
```

## Cell 2: Run

```python
%cd /kaggle/working/fall_detection

# Process CaucaFall - output: /kaggle/working/processed/
!python scripts/process_caucafall.py \
    --input "/kaggle/input/caucafall/Dataset CAUCAFall/CAUCAFall" \
    --output "/kaggle/working/caucafall_processed"
```

## Cell 3: Zip

```python
import shutil
import os

# Move files to one place and zip
src = "/kaggle/working/caucafall_processed"
zip_path = "/kaggle/working/caucafall_processed.zip"

if os.path.exists(src):
    shutil.make_archive(src, 'zip', src)
    print(f"Created: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
```

---

# NOTEBOOK 2: Process MCFD

## Cell 1: Setup

```python
# Clone repository
!cd /kaggle/working && rm -rf fall_detection && \
    git clone https://github.com/phamduyen112004-gif/fall_detection.git
%cd /kaggle/working/fall_detection
!pip install -r requirements.txt -q
```

## Cell 2: Run

```python
%cd /kaggle/working/fall_detection

# Process MCFD - output: /kaggle/working/processed/
!python scripts/process_mcfd.py \
    --input "/kaggle/input/multiple-cameras-fall-dataset/dataset/dataset" \
    --csv "/kaggle/input/multiple-cameras-fall-dataset/data_tuple3.csv" \
    --output "/kaggle/working/mcfd_processed"
```

## Cell 3: Zip

```python
import shutil
import os

src = "/kaggle/working/mcfd_processed"
zip_path = "/kaggle/working/mcfd_processed.zip"

if os.path.exists(src):
    shutil.make_archive(src, 'zip', src)
    print(f"Created: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
```

---

# NOTEBOOK 3: Train Model

## Cell 1: Setup

```python
# Clone repository
!cd /kaggle/working && rm -rf fall_detection && \
    git clone https://github.com/phamduyen112004-gif/fall_detection.git
%cd /kaggle/working/fall_detection
!pip install -r requirements.txt -q
```

## Cell 2: Merge Data

```python
# Upload caucafall_processed.zip and mcfd_processed.zip to Kaggle input
# Then extract and merge

import zipfile, os

WORK = "/kaggle/working"
DATA = "/kaggle/working/processed_data"
os.makedirs(DATA, exist_ok=True)

# Extract both zip files
for zip_file in [
    "/kaggle/input/caucafall_processed.zip",
    "/kaggle/input/mcfd_processed.zip"
]:
    if os.path.exists(zip_file):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(DATA)

# Count
x_files = [f for f in os.listdir(DATA) if f.startswith('X_')]
print(f"Total samples: {len(x_files)}")
```

## Cell 3: Train

```python
%cd /kaggle/working/fall_detection

# Train model
!python scripts/train.py \
    --data "/kaggle/working/processed_data" \
    --output "/kaggle/working/models" \
    --results "/kaggle/working/results" \
    --epochs 100 \
    --batch-size 64
```

## Cell 4: Save Results

```python
import shutil
import os

# Zip final results
src = "/kaggle/working/results"
zip_path = "/kaggle/working/fall_detection_results.zip"

if os.path.exists(src):
    shutil.make_archive(src, 'zip', src)
    print(f"Created: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
```

---

# Summary

| Notebook | Task | Output File |
|----------|------|-------------|
| Notebook 1 | Process CaucaFall | `caucafall_processed.zip` |
| Notebook 2 | Process MCFD | `mcfd_processed.zip` |
| Notebook 3 | Train Model | `fall_detection_results.zip` |

---

# Quick Guide

1. **Notebook 1**: Run Cells 1-3 → Download `caucafall_processed.zip`
2. **Notebook 2**: Run Cells 1-3 → Download `mcfd_processed.zip`
3. **Upload** both zip files to Kaggle (as datasets for Notebook 3)
4. **Notebook 3**: Run Cells 1-4 → Download `fall_detection_results.zip`

---

# SOTA Hyperparameters Used

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
