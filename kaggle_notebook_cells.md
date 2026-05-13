# Hybrid YOLOv11-Pose + Transformer — Kaggle Full Pipeline
## Kiểm tra cấu trúc dataset

```bash
# Dataset đã gộp URFD + GMDCSA24 tại:
# /kaggle/input/datasets/phmthduyn/fall-detection-dataset
# LE2I tại: /kaggle/input/datasets/tuyenldvn/falldataset-imvia

# Kiểm tra cấu trúc
ls /kaggle/input/datasets/phmthduyn/fall-detection-dataset/
echo "---"
ls /kaggle/working/fall_detection/
```

---

## Cell 1: Environment Setup

```python
import os
import sys

# WORK_DIR = thư mục chứa code (fall_detection repo)
WORK_DIR = "/kaggle/working/fall_detection"
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# Dataset paths
FALL_DATASET_ROOT = "/kaggle/input/datasets/phmthduyn/fall-detection-dataset"  # URFD + GMDCSA
LE2I_ROOT = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"             # LE2I (optional)
WORK_ROOT = "/kaggle/working"

# Output paths
AIO_ROOT = os.path.join(WORK_DIR, "AIO_Dataset")
PROCESSED = os.path.join(WORK_DIR, "data", "processed")
MODEL_OUT = os.path.join(WORK_DIR, "best_hybrid_transformer.pth")

print(f"WORK_DIR:    {WORK_DIR}")
print(f"Dataset:     {FALL_DATASET_ROOT}")
print(f"Output:      {MODEL_OUT}")
print(f"Processed:   {PROCESSED}")
```

---

## Cell 2: Prepare Dataset (URFD + GMDCSA — ĐÃ GỘP SẴN)

```bash
cd /kaggle/working/fall_detection

python prepare_dataset.py \
    --urfd-root /kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD \
    --gmdcsa-root /kaggle/input/datasets/phmthduyn/fall-detection-dataset/GMDCSA24 \
    --out /kaggle/working/fall_detection/AIO_Dataset
```

> Dataset URFD và GMDCSA24 đã gộp sẵn tại link `phmthduyn/fall-detection-dataset`. KHÔNG dùng link `tuyenldvn/falldataset-imvia` cho URFD/GMDCSA24.

---

## Cell 3: Extract PIFR Features

```bash
cd /kaggle/working/fall_detection

python data_extractor.py \
    --aio-dir /kaggle/working/fall_detection/AIO_Dataset \
    --out-dir /kaggle/working/fall_detection/data/processed \
    --model yolo11n-pose.pt \
    --device cpu
```

---

## Cell 4: Train Transformer

```bash
cd /kaggle/working/fall_detection

python train_transformer.py \
    --data-dir /kaggle/working/fall_detection/data/processed \
    --out /kaggle/working/fall_detection/best_hybrid_transformer.pth \
    --device cpu \
    --epochs 100
```

---

## Cell 5: Final Evaluation + SOTA Comparison (CHẠY SAU KHI TRAIN XONG)

```bash
cd /kaggle/working/fall_detection

python final_evaluation.py \
    --model /kaggle/working/fall_detection/best_hybrid_transformer.pth \
    --data-dir /kaggle/working/fall_detection/data/processed \
    --output /kaggle/working/fall_detection/final_results \
    --batch-size 64 \
    --device cpu
```

**Output files trong `final_results/`:**
- `results.json` — Full metrics + confusion matrix + SOTA data
- `report.md` — Markdown summary report
- `sota_comparison.csv` — SOTA table (CSV)
- `sota_table.tex` — LaTeX table cho bài báo
- `visualizations/confusion_matrix.png`
- `visualizations/roc_curve.png`
- `visualizations/pr_curve.png`

---

## Cell 6: FPS Benchmark (tùy chọn — cần test videos)

```bash
cd /kaggle/working/fall_detection

# Tìm video từ dataset để benchmark
mkdir -p /kaggle/working/fall_detection/test_videos
# Copy một số video fall từ URFD (đã extract ZIP)
find /kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD/Fall/ -name "*.zip" 2>/dev/null | head -5 || true

# Hoặc nếu đã extract video, chỉ định thư mục chứa video:
python benchmark_fps.py \
    --video-dir /kaggle/input/datasets/phmthduyn/fall-detection-dataset/URFD/Fall/ \
    --pose-weights yolo11n-pose.pt \
    --cls-weights /kaggle/working/fall_detection/best_hybrid_transformer.pth \
    --output /kaggle/working/fall_detection/fps_results.csv \
    --device cpu
```

---

## Cell 7: Nén tất cả artifacts về local

```bash
cd /kaggle/working/fall_detection

tar -czvf fall_detection_artifacts.tar.gz \
    best_hybrid_transformer.pth \
    yolo11n-pose.pt \
    data/processed/X_train.npy \
    data/processed/y_train.npy \
    data/processed/groups.npy \
    final_results/results.json \
    final_results/sota_comparison.csv \
    final_results/sota_table.tex \
    final_results/report.md \
    final_results/visualizations/

ls -lh fall_detection_artifacts.tar.gz
```

---

## Lưu ý quan trọng

### Dataset path
- **URFD + GMDCSA24:** `/kaggle/input/datasets/phmthduyn/fall-detection-dataset`
  - URFD: `URFD/` folder trong dataset
  - GMDCSA24: `GMDCSA24/` folder trong dataset
- **LE2I:** `/kaggle/input/datasets/tuyenldvn/falldataset-imvia` (tùy chọn — zone-based bị lỗi, cần fix riêng)

### Files cần thiết
| File | Mục đích |
|------|-----------|
| `best_hybrid_transformer.pth` | Model checkpoint |
| `data/processed/X_train.npy` | Feature data |
| `data/processed/y_train.npy` | Labels |
| `data/processed/groups.npy` | Subject groups |
| `final_results/results.json` | Metrics + SOTA |
| `REPORT_TEMPLATE.md` | Template báo cáo (local) |

### Sửa báo cáo
Sau khi chạy `final_evaluation.py`, copy giá trị từ `final_results/results.json` vào `REPORT_TEMPLATE.md`:
- `accuracy`, `sensitivity`, `specificity`, `precision`, `f1_score`, `gmean`, `roc_auc`, `pr_auc`
- `confusion_matrix`: TN, FP, FN, TP
- `fps_benchmark`: avg_fps, pose_only_fps, pose_ms_avg, tfm_ms_avg
