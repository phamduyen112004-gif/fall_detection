# Hybrid YOLOv11-Pose + Transformer — Fall Detection

Hai hướng suy luận:

| Cách chạy | Mô tả | Lệnh |
|-----------|--------|------|
| **Pipeline góc + thời gian** | Tkinter, quy tắc tư thế nằm ngang + thời gian, không cần model đã học Transformer | `python main.py --gui` |
| **Transformer** | PyQt5, chuỗi 60×60 đặc trưng PIFR + `HybridFallTransformer`, cần `best_hybrid_transformer.pth` | `python main.py --gui-transformer` hoặc `python app_inference.py` |

**Chuẩn bị dữ liệu**

#### AIO Dataset (URFD + GMDCSA-24)
1. `python prepare_dataset.py` — gộp URFD + GMDCSA vào `AIO_Dataset/` (tùy đường dẫn). URFD: zip trong `Fall`/`fall` và `ADL`/`adl` dưới cùng một thư mục gốc, ví dụ `--urfd-root data/raw/URFD` (file kiểu `ADL/adl-13-cam0-rgb.zip`). GMDCSA-24: `Subject N/Fall`, `Subject N/ADL` hoặc `fall`/`adl`. Ví dụ GMDCSA: `--gmdcsa-root data/raw/GMDCSA24 --skip-urfd`.
2. `python data_extractor.py --aio-dir AIO_Dataset --out-dir data/processed` — sinh `X_train.npy`, `y_train.npy`, `groups.npy`.

#### LE2I Fall Detection Dataset (với Zone-based Protocol - IEEE 2026)

LE2I dataset yêu cầu annotation `start_fall`/`end_fall` để phân tách vùng Fall/ADL một cách nghiêm ngặt.

1. **Chuẩn bị LE2I clips** vào `AIO_Dataset/`:
   ```bash
   # Nếu có annotation CSV (khuyến nghị)
   python prepare_le2i_dataset.py --le2i-root data/raw/LE2I \
       --out AIO_Dataset \
       --annotation-csv data/raw/LE2I/LE2I_Fall_Annotation.csv

   # Nếu không có annotation CSV (tự động phân loại fall/adl bằng tên file/folder)
   python prepare_le2i_dataset.py --le2i-root data/raw/LE2I --out AIO_Dataset
   ```
   Script tự động nhận diện cấu trúc LE2I (Kaggle, Zenodo, hoặc raw folder).

2. **Trích keypoint + sinh sliding window theo Zone-based Protocol**:
   ```bash
   python le2i_zone_based_extractor.py \
       --aio-dir AIO_Dataset \
       --annotation-json AIO_Dataset/_le2i_annotations.json \
       --out-dir data/le2i_processed \
       --val-subjects 5 \
       --device cuda
   ```

   **Zone-based Protocol (IEEE 2026) - Phân loại nghiêm ngặt:**

   | Class | Label | Rule |
   |-------|-------|------|
   | **Fall** | 0 | Window bao trùm hoàn toàn `[start_fall, end_fall]` |
   | **Non-Fall (ADL)** | 1 | Window kết thúc >= 30 frames **trước** `start_fall` |
   | **Discarded** | - | Buffer zone (30 frame trước fall), Post-fall zone, ambiguous overlaps |

   **Output:** `data/le2i_processed/{train,val}/{fall,nofall}/*.npy` hoặc `X_train.npy`, `y_train.npy`, `groups.npy`

**Huấn luyện**

```bash
python train_transformer.py --data-dir data/processed
```

Checkpoint lưu `best_threshold` (ngưỡng tối ưu F1 trên validation) — `app_inference.py` đọc để so sánh xác suất.

**Đặc trưng 60 chiều** (một nguồn: `src/pifr_features.py`): 51 keypoint + 9 hình học gồm `shoulder_nose_angle` (không dùng `bbox_aspect_ratio` trong vector).

**Chạy test**

```bash
pip install pytest scikit-learn
pytest tests/ -q
```

Trên GitHub, workflow `.github/workflows/ci.yml` chạy `pytest tests/` khi push hoặc PR vào `main` / `master`.

**Kaggle:** dùng `kaggle_notebook_cells.md` (6 cell) theo flow **clone repo → cài dependencies → set `FALL_DATASET_ROOT` (trỏ Kaggle Input) → chạy `python -m src.kaggle_pipeline --strict`**.

**Lưu ý CI:** workflow CI chỉ cài `requirements-ci.txt` (nhẹ) để tránh lỗi cài `PyQt5`/`torch` trên runner; chạy app/train vẫn dùng `requirements.txt`.

Kaggle quickstart (tóm tắt):

```python
GIT_URL = "https://github.com/<username>/<repo>.git"
REPO = GIT_URL.rstrip("/").split("/")[-1].replace(".git", "")
%cd /kaggle/working
!rm -rf "/kaggle/working/{REPO}"
!git clone "{GIT_URL}"
%cd "/kaggle/working/{REPO}"
!pip -q install -r requirements.txt
import os; os.environ["FALL_DATASET_ROOT"] = "/kaggle/input/fall-detection-dataset"
!python -m src.kaggle_pipeline --strict
```

**Export ONNX (tùy chọn)** — cần `pip install onnx`.

```bash
python scripts/export_onnx.py --weights best_hybrid_transformer.pth --out model.onnx
```
