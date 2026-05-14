# Hybrid YOLOv11-Pose + Transformer — Kaggle Full Pipeline
## Dataset Paths (3 datasets gộp chung)

| Dataset | Kaggle Path |
|---------|-------------|
| URFD | `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD` |
| GMDCSA24 | `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24` |
| LE2I | `/kaggle/input/datasets/tuyenldvn/falldataset-imvia` |

---

## Cell 0: Clone Repository

```python
GIT_URL = "https://github.com/phamduyen112004-gif/fall_detection.git"
REPO = GIT_URL.rstrip("/").split("/")[-1].replace(".git", "")

%cd /kaggle/working
!rm -rf "/kaggle/working/{REPO}"
!git clone "{GIT_URL}"
%cd "/kaggle/working/{REPO}"
print("Cloned:", REPO)
```

---

## Cell 1: Install Dependencies

```bash
%cd /kaggle/working/fall_detection
!pip install -r requirements.txt
```

---

## Cell 2: Environment Setup

```python
import os
import sys

WORK_DIR = "/kaggle/working/fall_detection"
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# Dataset paths - 3 datasets gộp chung
URFD_ROOT   = "/kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD"
GMDCSA_ROOT = "/kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24"
LE2I_ROOT   = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"

# Output paths
AIO_ROOT   = os.path.join(WORK_DIR, "AIO_Dataset")
PROCESSED  = os.path.join(WORK_DIR, "data", "processed")
LE2I_PROC  = os.path.join(WORK_DIR, "data", "le2i_processed")
MODEL_OUT  = os.path.join(WORK_DIR, "best_hybrid_transformer.pth")

print(f"WORK_DIR:    {WORK_DIR}")
print(f"URFD:        {URFD_ROOT}")
print(f"GMDCSA24:    {GMDCSA_ROOT}")
print(f"LE2I:        {LE2I_ROOT}")
print(f"Output:      {MODEL_OUT}")
```

---

## Cell 3: Prepare Dataset (URFD + GMDCSA + LE2I — GỘP CHUNG)

```bash
cd /kaggle/working/fall_detection

python prepare_dataset.py \
    --urfd-root /kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD \
    --gmdcsa-root /kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24 \
    --le2i-root /kaggle/input/datasets/tuyenldvn/falldataset-imvia \
    --out /kaggle/working/fall_detection/AIO_Dataset
```

> Tự động gộp cả 3 dataset vào `AIO_Dataset/{fall,nofall}/`. LE2I sẽ tạo file `_le2i_annotations.json` chứa start_fall/end_fall.

---

## Cell 4: Extract PIFR Features (URFD + GMDCSA)

```bash
cd /kaggle/working/fall_detection

python data_extractor.py \
    --aio-dir /kaggle/working/fall_detection/AIO_Dataset \
    --out-dir /kaggle/working/fall_detection/data/processed \
    --model yolo11n-pose.pt \
    --device cpu
```

---

## Cell 5: Zone-based Extraction cho LE2I

```bash
cd /kaggle/working/fall_detection

python le2i_zone_based_extractor.py \
    --aio-dir /kaggle/working/fall_detection/AIO_Dataset \
    --annotation-json /kaggle/working/fall_detection/AIO_Dataset/_le2i_annotations.json \
    --out-dir /kaggle/working/fall_detection/data/le2i_processed \
    --val-subjects 0.2 \
    --stride 15 \
    --save-format merged \
    --device cpu
```

---

## Cell 6: Merge Datasets

```python
import numpy as np
import os

# Load URFD + GMDCSA
X1 = np.load("/kaggle/working/fall_detection/data/processed/X_train.npy")
y1 = np.load("/kaggle/working/fall_detection/data/processed/y_train.npy")
g1 = np.load("/kaggle/working/fall_detection/data/processed/groups.npy", allow_pickle=True)

# Load LE2I Zone-based
X2 = np.load("/kaggle/working/fall_detection/data/le2i_processed/X_train.npy")
y2 = np.load("/kaggle/working/fall_detection/data/le2i_processed/y_train.npy")
g2 = np.load("/kaggle/working/fall_detection/data/le2i_processed/groups_train.npy", allow_pickle=True)

# Merge
X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)
g = np.concatenate([g1, np.array([f"le2i_{x}" for x in g2])], axis=0)

# Save merged
MERGED = "/kaggle/working/fall_detection/data/merged"
os.makedirs(MERGED, exist_ok=True)
np.save(f"{MERGED}/X_train.npy", X)
np.save(f"{MERGED}/y_train.npy", y)
np.save(f"{MERGED}/groups.npy", g, allow_pickle=True)

print(f"Merged: N={len(y)} ({int(np.sum(y>=0.5))} fall, {int(np.sum(y<0.5))} nofall)")
```

---

## Cell 7: Train Transformer

```bash
cd /kaggle/working/fall_detection

python train_transformer.py \
    --data-dir /kaggle/working/fall_detection/data/merged \
    --out /kaggle/working/fall_detection/best_hybrid_transformer.pth \
    --device cpu \
    --epochs 100
```

---

## Cell 8: Final Evaluation

```bash
cd /kaggle/working/fall_detection

python final_evaluation.py \
    --model /kaggle/working/fall_detection/best_hybrid_transformer.pth \
    --data-dir /kaggle/working/fall_detection/data/merged \
    --output /kaggle/working/fall_detection/final_results \
    --batch-size 64 \
    --device cpu
```

---

## Lưu ý

### Dataset paths chính xác
- **URFD:** `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD`
- **GMDCSA24:** `/kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24`
- **LE2I:** `/kaggle/input/datasets/tuyenldvn/falldataset-imvia`

### Output artifacts
| File | Mục đích |
|------|----------|
| `best_hybrid_transformer.pth` | Model checkpoint |
| `AIO_Dataset/_le2i_annotations.json` | LE2I metadata |
| `data/merged/X_train.npy` | Merged features |
| `final_results/results.json` | Metrics + SOTA |
